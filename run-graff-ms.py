import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
from time import time
import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric as pyg
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from pandarallel import pandarallel
from multiprocessing import cpu_count
from pytorch_lightning import seed_everything
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from pyteomics.mass import Composition
from torch.utils.data import Dataset

################################################################
# read hyperparameters from command line
################################################################

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('model_path')
parser.add_argument('input_path')
parser.add_argument('output_path')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--instrument', type=str, default='Thermo Finnigan Elite Orbitrap')
parser.add_argument('--has_isotopes', type=int, default=0)
parser.add_argument('--min_probability', type=float, default=1e-4)
parser.add_argument('--min_mz', type=float, default=50)
parser.add_argument('--subsample', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ignore_errors', type=int, default=1)
args = parser.parse_args()

pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=args.num_workers)
seed_everything(args.seed, workers=True)

# start timing now
t = time()

################################################################
# load SMILES strings to predict
################################################################

print('Reading queries... ',end='')
df = pd.read_csv(args.input_path,sep='\t',header=None)
print('done')

cols = ['Spectrum','SMILES','Precursor_type','NCE']
if df.shape[1] == 5:
    cols += ['Instrument']
df.columns = cols

if args.subsample:
    df = df.sample(n=args.subsample, random_state=args.seed)

################################################################
# fill in metadata
################################################################

print('Generating metadata... ',end='')

if 'Instrument' not in cols:
    df['Instrument'] = args.instrument
df['has_isotopes'] = args.has_isotopes

smiles = df['SMILES'].drop_duplicates()
smiles.index = smiles
mols = smiles.parallel_apply(lambda x: Chem.MolFromSmiles(x))
inchikeys = mols.parallel_apply(lambda x: Chem.MolToInchiKey(x))
formulas = mols.map(CalcMolFormula)
mws = mols.map(ExactMolWt)

df['mol'] = df['SMILES'].map(mols.to_dict())
df = df.dropna(subset='mol')

hydrogen_mass = ExactMolWt(Chem.MolFromSmiles('[H]'))
df['PrecursorMZ'] = df['SMILES'].map(mws.to_dict())
df['PrecursorMZ'] += df['Precursor_type'].map({
    '[M+H]+': hydrogen_mass,
    '[M-H]-': -hydrogen_mass
})
df['Formula'] = df['SMILES'].map(formulas.to_dict())
df['InChIKey'] = df['SMILES'].map(inchikeys.to_dict())

print('done')

################################################################
# load model
################################################################

from src.graff import GrAFF

print(f'Loading model {args.model_path}... ',end='')
model = GrAFF.load_from_checkpoint(args.model_path)
# inference-time parameters, override them in training checkpoint
model.min_probability = args.min_probability
model.min_mz = args.min_mz
print('done')

#################################################################
# featurize molecules
#################################################################

from src.smiles import from_mol
from src.gnn import graph_laplacian, add_virtual_node

print('Precomputing graphs... ')

# featurize for inference
class MoleculeFeaturizer:
    def __init__(self, vocab, hparams):
        self.vocab = vocab
        self.hparams = hparams
        
        self.atom_types = sorted(['C','H','N','O','P','S','F','Cl','Br','I'])
        self.atom_mzs = {
            a: ExactMolWt(Chem.MolFromSmiles(f'[{a}]')) 
            for a in self.atom_types
        }

        self.mzs = (
            self.vocab['formula']
            .map(lambda x: Composition(formula=x.rstrip('+-')))
            .map(lambda x: sum(self.atom_mzs[a]*n for a,n in x.items()))
        ).values
        
        self.is_product = self.vocab['kind'].values == 'product'
        
        self.hydrogen_mass = ExactMolWt(Chem.MolFromSmiles('[H]'))
        
    def __call__(self, item):
        try:
            # graph features

            g = item['graph'].clone()

            instrument = [0] * len(self.hparams.instruments)
            instrument[self.hparams.instruments.index(item['Instrument'])] = 1

            precursor_type = [0] * len(self.hparams.precursor_types)
            precursor_type[self.hparams.precursor_types.index(item['Precursor_type'])] = 1

            covariates = np.array([*instrument, *precursor_type, item['NCE'], item['has_isotopes']])

            # compute double-counting correction
            
            precursor = Composition(formula=item['Formula'].rstrip('+-'))
            if item['Precursor_type'] == '[M+H]+':
                precursor['H'] += 1
            elif item['Precursor_type'] == '[M-H]-':
                precursor['H'] -= 1
            else:
                raise NotImplementedError
            precursor_mz = sum(self.atom_mzs[a]*n for a,n in precursor.items())
            
            # hash formulas using theoretical m/zs
            mzs = np.where(self.is_product, self.mzs, precursor_mz - self.mzs)
            # account for pad
            mzs = np.concatenate([[np.nan], mzs])
            _, inverse, counts = np.unique(mzs, return_inverse=True, return_counts=True)
            double_counted = counts[inverse] > 1

            g.spectrum = str(item['Spectrum'])
            g.precursor_mz = float(precursor_mz)
            g.has_isotopes = bool(item['has_isotopes'])
            g.covariates = torch.FloatTensor(covariates).view(1,-1)
            g.double_counted = torch.BoolTensor(double_counted).view(1,-1)
            g.mzs = torch.FloatTensor(mzs).view(1,-1)

            return g
        except Exception as e:
            if args.ignore_errors:
                print('Error on ' + item['SMILES'])
                return None
            else:
                raise e

# usually fewer molecular graphs than spectra, so cache them
num_eigs = model.hparams.num_eigs
def mol_to_graph(mol):
    try:
        g = from_mol(mol)
        g = graph_laplacian(g, num_eigs)
        g = add_virtual_node(g)
        # must pad the eigenfeatures for the virtual node
        eig_pad = torch.zeros(g.num_nodes-g.eigvecs.shape[0],g.eigvecs.shape[1],
                             dtype=g.eigvecs.dtype,device=g.eigvecs.device)
        g.eigvecs = torch.cat([g.eigvecs,eig_pad],0)
        return g
    except Exception as e:
        if args.ignore_errors:
            print('Error on ' + g.smiles)
            return None
        else:
            raise e

graphs = mols.parallel_apply(mol_to_graph)
df['graph'] = df['SMILES'].map(graphs.to_dict())
df = df.dropna(subset='graph')

print('done')

class MoleculeFeaturizerDataset(Dataset):
    def __init__(self, df, feat):
        self.items = df.to_dict('records')
        self.feat = feat
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.feat(self.items[idx])

feat = MoleculeFeaturizer(model.vocab, model.hparams)
dataset = MoleculeFeaturizerDataset(df, feat)

#################################################################
# run inference
#################################################################

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

trainer = pl.Trainer(
    accelerator='gpu' if args.gpus else 'cpu', 
    devices=args.gpus if args.gpus else None,
    strategy='ddp' if args.gpus>1 else None,
    precision=args.precision
)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=True
)
# catch errors from bad smiles
loader.collate_fn = lambda xs: loader.collate_fn([x for x in xs if x is not None])

print('Predicting spectra... ',end='')
result = trainer.predict(model, loader)
print('done')

#################################################################
# postprocess predictions
#################################################################

spectra, mzs, intensities = zip(*result)
spectra = sum(spectra,[])
mzs = [x.numpy() for x in sum(mzs,[])]
intensities = [x.numpy() for x in sum(intensities,[])]
result = pd.DataFrame({
    'mzs': mzs, 
    'intensities': intensities
})
result.index = spectra
result.index = result.index.astype(str)
df.index = df['Spectrum'].astype(str) 
result = result.join(df)

print(f'Time elapsed: {time() - t} seconds')

#################################################################
# output to text format
#################################################################

from src.io import write_msp

cols = ['Spectrum','SMILES','Precursor_type','NCE',
        'Instrument','InChIKey','Formula','PrecursorMZ']

print('Exporting to MSP... ',end='')
write_msp(
    args.output_path,
    mzs=result.mzs.tolist(),
    intensities=result.intensities.tolist(),
    **{c: result[c].tolist() for c in cols}
)
print('done')
