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

from src.smiles import from_mol
from src.gnn import graph_laplacian, add_virtual_node
from src.graff import GrAFF

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

from src.neims import NEIMS

print(f'Loading model {args.model_path}... ',end='')
model = NEIMS.load_from_checkpoint(args.model_path)
# inference-time parameters, override them in training checkpoint
model.min_probability = args.min_probability
model.min_mz = args.min_mz
print('done')

#################################################################
# featurize molecules
#################################################################

from src.neims import NEIMSFeaturizer

class MoleculeFeaturizerDataset(Dataset):
    def __init__(self, df, feat):
        self.items = df.to_dict('records')
        self.feat = feat
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.feat(self.items[idx])

feat = NEIMSFeaturizer()
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
# catch errors
loader.collate_fn = lambda xs: loader.collate_fn([x for x in xs if x is not None])

print('Predicting spectra... ',end='')
result = trainer.predict(model, loader)
print('done')

# use the spectrum index instead of relying on ordering, since errors can happen
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
