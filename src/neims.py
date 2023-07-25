import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
import pytorch_lightning as pl

from scipy.sparse import csr_array
from rdkit.Chem.AllChem import GetHashedMorganFingerprint
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem.Descriptors import ExactMolWt

from .graff import precursor_types, instruments

class NEIMSFeaturizer:
    def __init__(
        self, 
        delta_mz=0.1,
        max_mz=1000,
        fp_bits=4096,
        fp_radius=2,
        precursor_types=precursor_types,
        instruments=instruments
    ):
        self.delta_mz = delta_mz
        self.max_mz = max_mz
        self.mzs = np.arange(0,max_mz,delta_mz)
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.precursor_types = precursor_types
        self.instruments = instruments
        self.hydrogen_mass = ExactMolWt(Chem.MolFromSmiles('[H]'))
    
    def __call__(self, item):
        if 'mzs' in item:
            bin_idxs = (item['mzs'] / self.delta_mz).round().clip(0,len(self.mzs)-1).astype(int)
            binned_heights = csr_array((item['intensities'],(0 * bin_idxs, bin_idxs)),
                                       shape=(1,len(self.mzs)))
            binned_heights = binned_heights.toarray().squeeze()
        else:
            binned_heights = np.zeros_like(self.mzs)
        
        mol = Chem.MolFromSmiles(item['SMILES'])
        fp = GetHashedMorganFingerprint(mol, radius=self.fp_radius, nBits=self.fp_bits)
        fingerprint = np.zeros(1)
        ConvertToNumpyArray(fp, fingerprint)
        
        instrument = [0] * len(self.instruments)
        instrument[self.instruments.index(item['Instrument'])] = 1

        precursor_type = [0] * len(self.precursor_types)
        precursor_type[self.precursor_types.index(item['Precursor_type'])] = 1

        covariates = np.array([*instrument, *precursor_type, item['NCE'], item['has_isotopes']])
        
        precursor_mz = ExactMolWt(mol)
        if item['Precursor_type'] == '[M+H]+':
            precursor_mz += self.hydrogen_mass
        elif item['Precursor_type'] == '[M-H]-':
            precursor_mz -= self.hydrogen_mass
        else:
            raise NotImplementedError

        return {
            'spectrum': str(item['Spectrum']),
            'split': item['split'] if hasattr(item,'split') else '',
            'precursor_mz': precursor_mz,
            'covariates': torch.FloatTensor(covariates),
            'binned_heights': torch.FloatTensor(binned_heights),
            'fingerprint': torch.FloatTensor(fingerprint),
        }

class NEIMSResidualBlock(nn.Module):
    def __init__(self, model_dim, dropout, bottleneck):
        super().__init__()
        self.dense1 = nn.Linear(model_dim, int(bottleneck*model_dim))
        self.dense2 = nn.Linear(int(bottleneck*model_dim), model_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(model_dim)
        self.norm2 = nn.BatchNorm1d(int(bottleneck*model_dim))
        
    def forward(self, x):
        dx = self.norm1(x)
        dx = self.relu1(dx)
        dx = self.dropout(dx)
        dx = self.dense1(dx)
        dx = self.norm2(dx)
        dx = self.relu2(dx)
        dx = self.dense2(dx)
        return x + dx

class NEIMS(pl.LightningModule):
    def __init__(
        self, 
        input_dim=4096,
        covariate_dim=7,
        model_dim=2000,
        model_depth=7,
        lr=1e-3,
        dropout=0.25,
        bottleneck=0.5,
        delta_mz=0.1,
        min_mz=0,
        max_mz=1000,
        min_probability=0,
        intensity_power=1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.covariate_dim = covariate_dim
        self.model_dim = model_dim
        self.model_depth = model_depth
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_probability = min_probability
        self.delta_mz = delta_mz
        self.dropout = dropout
        self.lr = lr
        self.intensity_power = intensity_power
        self.bottleneck = bottleneck
        
        self.hydrogen_mass = ExactMolWt(Chem.MolFromSmiles('[H]'))
        
        mzs = torch.arange(0,max_mz,delta_mz)
        self.register_buffer('mzs',mzs)
        self.output_dim = len(mzs)
        
        mz_weights = mzs / mzs.sum()
        self.register_buffer('mz_weights',mz_weights)
        
        self.proj = nn.Linear(input_dim + covariate_dim, model_dim)
        layers = []
        for _ in range(self.model_depth):
            layer = NEIMSResidualBlock(self.model_dim, self.dropout, self.bottleneck)
            layers.append(layer)
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(self.model_dim)
        self.fwd = nn.Linear(self.model_dim, self.output_dim)
        self.rev = nn.Linear(self.model_dim, self.output_dim)
        self.gate = nn.Linear(self.model_dim, self.output_dim)
            
    def configure_optimizers(self): 
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        return opt
    
    def forward(self, batch):
        fingerprint = batch['fingerprint'].float()
        precursor_mz = batch['precursor_mz'].view(-1,1)
        covariates = batch['covariates'].float()
        
        x = torch.cat([fingerprint,covariates],dim=1)
        x = self.proj(x)
        x = self.mlp(x)
        x = self.norm(x)
        
        # bidirectional prediction
        x_fwd = self.fwd(x)
        rev_idx = ((precursor_mz.view(-1,1) - self.mzs.view(1,-1)) / self.delta_mz).long()
        x_rev = self.rev(x)
        x_rev[rev_idx<0] = 0
        rev_idx[rev_idx<0] = 0
        x_rev = torch.zeros_like(x_fwd).scatter_add_(1, rev_idx, x_rev)
        x_gate = torch.sigmoid(self.gate(x))
        x = x_gate * x_fwd + (1 - x_gate) * x_rev
        
        return x
    
    def step(self, batch, step=None):
        y = batch['binned_heights']
        y = y / y.max(dim=1,keepdims=True).values.clamp(1e-6)
        batch_size = y.shape[0]

        y_pred_pow = self(batch)
        y_pow = y ** self.intensity_power
        
        loss = ((y_pow - y_pred_pow).square() @ self.mz_weights).mean()
        self.log(f'{step}/loss', loss, batch_size=batch_size, sync_dist=step=='val')
            
        with torch.no_grad():
            y_pred = y_pred_pow.clamp(0) ** (1/self.intensity_power)
            sim = torch.cosine_similarity(y, y_pred, dim=1)
            self.log(f'{step}/cosine', sim.mean(), batch_size=batch_size, sync_dist=step=='val')

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, 'val')
    
    def predict_step(self, batch, batch_idx):
        # this should only be called at inference time - uses different batch structure
        
        y_pred_pow = self(batch)
        y_pred = y_pred_pow.clamp(0) ** (1/self.intensity_power)
        y_pred = y_pred / y_pred.sum(1,keepdim=True).clamp(1e-6)
        
        # sparsify predictions
        xs = []
        ys = []
        for precursor_mz, y in zip(batch['precursor_mz'], y_pred):
            mask = self.mzs > self.min_mz - self.delta_mz
            mask &= self.mzs < precursor_mz + self.delta_mz + 2 * self.hydrogen_mass # isotopes
            mask &= y > self.min_probability
            
            x = self.mzs[mask].flatten()
            y = y[mask].flatten() / y[mask].sum().clamp(1e-6)
            
            xs.append(x)
            ys.append(y)

        return batch['spectrum'], xs, ys
