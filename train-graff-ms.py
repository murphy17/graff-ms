import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics.mass import Composition
from pytorch_lightning import seed_everything
from pandarallel import pandarallel
from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.smiles import from_mol
from src.gnn import add_virtual_node, graph_laplacian
from src.graff import GrAFF
from src.graff import precursor_types, instruments, isotope_types

################################################################
# greedy product/loss vocabulary selection
################################################################

def learn_vocabulary(df, vocab_size=None):
    annots = df[['InChIKey','products','losses','intensities','peaks']].copy()
    # equally split peak height among all compatible annotations
    annots['intensities'] = annots[['intensities','peaks']].apply(
        lambda item: item.intensities[item.peaks] / item.intensities.sum() \
                     / np.bincount(item.peaks)[item.peaks], axis=1)
    annots = annots.drop(columns=['peaks']).explode(['products','losses','intensities'])
    annots = annots.groupby(['InChIKey','products','losses']).sum()
    annots['intensities'] /= annots['intensities'].sum()
    annots = annots.reset_index()

    # separately rank products and losses by how much peak height each explains
    products = annots.groupby('products')['intensities'].sum().sort_values()[::-1].to_frame()
    products['kind'] = 'product'
    losses = annots.groupby('losses')['intensities'].sum().sort_values()[::-1].to_frame()
    losses['kind'] = 'loss'

    # take top vocab_size of either type
    vocab = pd.concat([products,losses],axis=0).sort_values('intensities',ascending=False)
    vocab.index.name = 'formula'
    vocab = vocab.reset_index()
    
    if vocab_size:
        vocab = vocab.head(vocab_size)
    
    pt = Chem.GetPeriodicTable()
    mws = {pt.GetElementSymbol(n): pt.GetMostCommonIsotopeMass(n) for n in range(1,119)}
    formulas = vocab['formula'].map(lambda x: Composition(formula=x))
    vocab['mz'] = formulas.map(lambda x: sum(mws[a] * n for a,n in x.items()))
    
    return vocab

################################################################
# load hyperparameters from command line
################################################################

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('df_path')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--vocab_size', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=int, default=5e-4)
parser.add_argument('--grad_clipping', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--encoder_dim', type=int, default=512)
parser.add_argument('--decoder_dim', type=int, default=1024)
parser.add_argument('--encoder_depth', type=int, default=6)
parser.add_argument('--decoder_depth', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_eigs', type=int, default=8)
parser.add_argument('--eig_dim', type=int, default=32)
parser.add_argument('--eig_depth', type=int, default=2)
parser.add_argument('--min_probability', type=float, default=0)
parser.add_argument('--min_mz', type=float, default=0)
parser.add_argument('--subsample', type=int, default=0)
parser.add_argument('--cache_path', type=str, default=None)
args = parser.parse_args()

seed_everything(args.seed, workers=True)
pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=args.num_workers)

################################################################
# load prepared dataframe
################################################################

df = pd.read_pickle(args.df_path)
df = df.query('split!=""')

if args.subsample:
    df = df.sample(n=args.subsample, random_state=args.seed)

################################################################
# compute the fixed vocabulary from training spectra
################################################################

print(f'Selecting vocabulary (K={args.vocab_size})... ',end='')
vocab = learn_vocabulary(df, args.vocab_size)
assert len(vocab) == args.vocab_size, 'Requested vocabulary is larger than training data!'
print('done')

################################################################
# generate indices into the fixed vocabulary for each peak
################################################################

# reserve zero index for the pad
product_lut = {f:i+1 for i,f in vocab.query('kind=="product"')['formula'].items()}
loss_lut = {l:i+1 for i,l in vocab.query('kind=="loss"')['formula'].items()}
iso_lut = {v:i for i,v in enumerate(sorted(isotope_types))}

def index_peaks(item):
    peaks = pd.DataFrame(item[['products','losses','isotopes','peaks']]).T
    peaks = peaks.explode([*peaks.columns])
    peaks['product_idx'] = peaks['products'].map(product_lut)
    peaks['loss_idx'] = peaks['losses'].map(loss_lut)
    peaks['isotope_idx'] = peaks['isotopes'].map(iso_lut)
    # don't predict rare higher isotope peaks
    peaks = peaks.dropna(subset=['isotope_idx'])
    peaks = peaks[['product_idx','loss_idx','isotope_idx','peaks']]
    # drop peaks not in either product or loss vocabulary
    peaks = peaks.loc[~(peaks['product_idx'].isna() & peaks['loss_idx'].isna())]
    # if a product cannot be explained as a loss, or vice-versa,
    # (which should usually be the case), assign the missing index to the pad
    peaks = peaks.fillna(0).astype(int)
    return peaks.values.T

# this is not necessary if cached
print(f'Indexing peaks... ',end='')
df['product_idx'], df['loss_idx'], df['isotope_idx'], df['peak_idx'] = (
    zip(*df.parallel_apply(index_peaks, axis=1))
)
print('done')

# remove any (rare!) spectra from training or validation that have no explanations
df.loc[(df['product_idx'].str.len()==0)&(df['split'].isin(['train','val'])),'split'] = ''

################################################################
# precompute graph features
################################################################

max_peaks = df['intensities'].str.len().max()
max_annots = df['product_idx'].str.len().max()

def pad1d(x, n):
    return F.pad(x, (0, n-len(x)))

def featurize_spectrum(item):
    # graph features
    mol = Chem.MolFromSmiles(item.SMILES)
    g = from_mol(mol)
    g = graph_laplacian(g, args.num_eigs)
    g = add_virtual_node(g)
    # must pad the eigenfeatures for the virtual node
    eig_pad = torch.zeros(g.num_nodes-g.eigvecs.shape[0],g.eigvecs.shape[1],
                         dtype=g.eigvecs.dtype,device=g.eigvecs.device)
    g.eigvecs = torch.cat([g.eigvecs,eig_pad],0)

    instrument = [0] * len(instruments)
    instrument[instruments.index(item.Instrument)] = 1

    precursor_type = [0] * len(precursor_types)
    precursor_type[precursor_types.index(item.Precursor_type)] = 1

    covariates = np.array([*instrument, *precursor_type, item.NCE, item.has_isotopes])
    
    # if an annotation matched both the product and loss vocabularies, it's doubled
    double_counted = np.zeros(args.vocab_size+1,dtype=bool)
    double_counted[item.product_idx] |= (item.loss_idx > 0)
    double_counted[item.loss_idx] |= (item.product_idx > 0)
    double_counted[0] = False

    g.spectrum = str(item.Spectrum)
    g.split = item.split
    g.precursor_mz = item.PrecursorMZ
    g.covariates = torch.FloatTensor(covariates).view(1,-1)
    g.product_idx = pad1d(torch.LongTensor(item.product_idx), max_annots).view(1,-1)
    g.loss_idx = pad1d(torch.LongTensor(item.loss_idx), max_annots).view(1,-1)
    g.peak_idx = pad1d(torch.LongTensor(item.peak_idx), max_annots).view(1,-1)
    g.isotope_idx = pad1d(torch.LongTensor(item.isotope_idx), max_annots).view(1,-1)
    g.mzs = pad1d(torch.FloatTensor(item.mzs), max_peaks).view(1,-1)
    g.intensities = pad1d(torch.FloatTensor(item.intensities), max_peaks).view(1,-1)
    g.double_counted = torch.BoolTensor(double_counted).view(1,-1)

    return g

print('Featurizing spectra... ',end='')
if args.cache_path:
    if os.path.exists(args.cache_path):
        items = pd.read_pickle(args.cache_path)
    else:
        items = df.parallel_apply(featurize_spectrum, axis=1)
        items.to_pickle(args.cache_path)
else:
    items = df.parallel_apply(featurize_spectrum, axis=1)
print('done')

################################################################
# split data
################################################################

datasets = {}
for split in ['train','val','test']:
    datasets[split] = [x for x in items if x.split == split]

loaders = {}
for split in datasets:
    loaders[split] = DataLoader(
        datasets[split],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=split=='train',
        drop_last=split=='train'
    )

################################################################
# fit model
################################################################

trainer = pl.Trainer(
    accelerator='gpu' if args.gpus else 'cpu', 
    devices=args.gpus if args.gpus else None,
    strategy='ddp' if args.gpus>1 else None,
    precision=args.precision,
    max_epochs=args.max_epochs,
    gradient_clip_val=args.grad_clipping,
    logger=TensorBoardLogger(
        'lightning_logs', 
        default_hp_metric=False,
        name='graff'
    ),
    callbacks=[
        ModelCheckpoint(
            monitor='val/loss', 
            mode='min',
            save_top_k=1
        )
    ],
)

model = GrAFF(
    vocab=vocab,
    precursor_types=precursor_types,
    instruments=instruments,
    **args.__dict__
)

trainer.fit(model, loaders['train'], loaders['val'])
