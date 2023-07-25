import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import pandas as pd
from tqdm import tqdm
import os

import torch
import torch_geometric as pyg
from torch import nn
from torch.nn import functional as F

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics.mass import Composition

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('df_path')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--precision', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--subsample', type=int, default=0)
args = parser.parse_args()

from pytorch_lightning import seed_everything
seed_everything(args.seed, workers=True)

from pandarallel import pandarallel
from multiprocessing import cpu_count
pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=args.num_workers)

df = pd.read_pickle(args.df_path)
df = df.query('split!=""')

if args.subsample:
    df = df.sample(n=args.subsample, random_state=args.seed)

from src.neims import NEIMSFeaturizer

featurizer = NEIMSFeaturizer()

items = df.parallel_apply(featurizer, axis=1)

datasets = {}
for split in ['train','val','test']:
    datasets[split] = [x for x in items if x['split'] == split]

from torch_geometric.loader import DataLoader

loaders = {}
for split in datasets:
    loaders[split] = DataLoader(
        datasets[split],
        batch_size=100,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=split=='train',
        drop_last=split=='train'
    )

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

trainer = pl.Trainer(
    accelerator='gpu' if args.gpus else 'cpu', 
    devices=args.gpus if args.gpus else None,
    strategy='ddp' if args.gpus>1 else None,
    precision=args.precision,
    max_epochs=args.max_epochs,
    val_check_interval=0.5,
    gradient_clip_val=1.,
    logger=TensorBoardLogger(
        'lightning_logs', 
        default_hp_metric=False,
        name='neims'
    ),
    callbacks=[
        ModelCheckpoint(
            monitor='val/cosine', 
            mode='max',
            save_top_k=1
        ),
        EarlyStopping(
            monitor='val/cosine',
            mode='max',
            patience=10
        ),
    ],
)

from src.neims import NEIMS

model = NEIMS(
    input_dim=4096,
    covariate_dim=7,
    model_dim=2000,
    model_depth=7,
    lr=1e-3,
    dropout=0.25,
    bottleneck=0.5,
    delta_mz=0.1,
    min_mz=0,
    min_probability=0,
    max_mz=1000,
    intensity_power=1,
    **args.__dict__
)

trainer.fit(model, loaders['train'], loaders['val'])
