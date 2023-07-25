import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torch_geometric as pyg
from torch_scatter import scatter_logsumexp

from .gnn import GINE, ResBlock, CanonicalOneHot, SignNet

atom_types = sorted(['C','H','N','O','P','S','F','Cl','Br','I'])
isotope_types = [0,1,2]
neutron_mass = 1.008665
precursor_types = ['[M+H]+','[M-H]-']
instruments = [
    'Orbitrap Fusion Lumos',
    'Thermo Finnigan Elite Orbitrap',
    'Thermo Finnigan Velos Orbitrap'
]

class GrAFF(pl.LightningModule):
    def __init__(
        self,
        *,
        vocab,
        encoder_dim,
        decoder_dim,
        encoder_depth,
        decoder_depth,
        num_eigs,
        eig_dim,
        eig_depth,
        dropout,
        learning_rate,
        weight_decay,
        precursor_types,
        instruments,
        min_probability,
        min_mz,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab = vocab
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.num_eigs = num_eigs
        self.eig_dim = eig_dim
        self.eig_depth = eig_depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.precursor_types = precursor_types
        self.instruments = instruments
        
        covariates_dim = len(precursor_types) + len(instruments) + 1 + 1
        vocab_size = len(vocab)
        self.vocab_size = vocab_size
        self.covariates_dim = covariates_dim
        
        # inference time only
        self.min_probability = min_probability
        self.min_mz = min_mz
        
        vocab_mzs = vocab['mz'].values
        self.register_buffer('vocab_mzs', torch.FloatTensor(vocab_mzs))
        vocab_kinds = vocab['kind'].values
        self.register_buffer('vocab_kinds', torch.BoolTensor(vocab_kinds == 'product'))
        self.register_buffer('isotope_mzs', torch.FloatTensor(isotope_types) * neutron_mass)
        
        self.log_epsilon = -10
        
        # embedding layers
        extra_node_feats = {
            'is_virtual_node': {False: 0, True: 1},
        }
        extra_edge_feats = {
            'is_virtual_in_edge': {False: 0, True: 1},
            'is_virtual_out_edge': {False: 0, True: 1},
        }
        self.onehot = CanonicalOneHot(extra_node_feats, extra_edge_feats)
        self.node_emb = nn.Sequential(
            nn.Linear(self.onehot.node_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim)
        )
        self.edge_emb = nn.Sequential(
            nn.Linear(self.onehot.edge_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim)
        )
        
        if num_eigs > 0:
            self.signnet = SignNet(
                num_eigs=num_eigs,
                embed_dim=encoder_dim,
                rho_dim=encoder_dim,
                rho_depth=eig_depth,
                phi_dim=eig_dim,
                phi_depth=eig_depth,
                dropout=dropout
            )
        else:
            self.signnet = None
        
        self.encoder = GINE(
            node_dim=encoder_dim,
            edge_dim=encoder_dim,
            model_dim=encoder_dim, 
            model_depth=encoder_depth,
            dropout=dropout
        )
        
        self.cov_emb = nn.Sequential(
            nn.Linear(self.covariates_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, encoder_dim)
        )
        
        self.attn = nn.Linear(encoder_dim, 1)
        
        layers = []
        layers.append(nn.Linear(encoder_dim, decoder_dim))
        for _ in range(decoder_depth):
            layers.append(ResBlock(decoder_dim, dropout))
        self.decoder = nn.Sequential(*layers)
        
        self.isotope_shift = nn.Linear(decoder_dim, len(isotope_types))
        self.clf = nn.Linear(decoder_dim, vocab_size + 1)
                
    def configure_optimizers(self): 
        opt = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return opt
    
    def forward(self, g):
        batch_size = len(g.ptr) - 1
        device = g.x.device
        
        # embed node, edge, eigenfeatures
        x_atom, x_bond = self.onehot(g.x, g.edge_attr)
        x_atom = self.node_emb(x_atom)
        x_bond = self.edge_emb(x_bond)
        if self.num_eigs > 0:
            x_eig = self.signnet(g.eigvecs[:,:self.num_eigs],
                                 g.eigvals[:,:self.num_eigs][g.batch])
        else:
            x_eig = 0

        # run message passing
        x_mol = self.encoder(g, x_atom + x_eig, x_bond)
        # and attention-pool across atoms
        w = pyg.utils.softmax(self.attn(x_mol), g.batch)
        z = pyg.nn.global_add_pool(x_mol * w, g.batch)
        
        # condition molecule representation on covariates
        z = z + self.cov_emb(g.covariates)
        # transform to spectrum representation
        z = self.decoder(z)
        # and predict logits
        log_y_pred = self.clf(z)
        
        # add the (approximated) isotopic envelope
        log_y_pred = log_y_pred.view(batch_size, self.vocab_size+1, 1)
        isotope_shift = self.isotope_shift(z).view(batch_size, 1, len(isotope_types))
        log_y_pred = log_y_pred + isotope_shift
        
        # correct intensities of double-counted formulas
        log_y_pred = log_y_pred - g.double_counted.unsqueeze(-1) * np.log(2)
        
        return log_y_pred
        
    def step(self, batch, step):
        batch_size = len(batch.ptr) - 1
        device = batch.x.device
        
        log_y_pred = self(batch)
        
        y_pred = torch.softmax(log_y_pred.flatten(1), dim=1).view(log_y_pred.shape)
        
        ################################################################
        # calculate peak-marginal cross-entropy
        ################################################################
        
        # recall this has zero entries, which point to the pad element
        # they exactly should line up with the padding of y
        product_idx = batch.product_idx * len(isotope_types) + batch.isotope_idx
        loss_idx = batch.loss_idx * len(isotope_types) + batch.isotope_idx

        log_y_pred = torch.log_softmax(log_y_pred.flatten(1), dim=1).view(log_y_pred.shape)
        pad_mask = torch.zeros(1, self.vocab_size+1, 1,
                               device=device, dtype=log_y_pred.dtype)
        pad_mask[:,0] = self.log_epsilon
        log_y_pred = log_y_pred + pad_mask
        log_y_pred = torch.logsumexp(torch.stack([
            torch.gather(log_y_pred.flatten(1), 1, product_idx),
            torch.gather(log_y_pred.flatten(1), 1, loss_idx)
        ], dim=-1), dim=-1)
        
        is_pad = torch.maximum(batch.product_idx, batch.loss_idx) == 0
        peak_idx = batch.peak_idx + 1
        peak_idx[is_pad] = 0
        num_peaks = batch.intensities.shape[1]
        log_y_pred = scatter_logsumexp(log_y_pred, peak_idx, 1, dim_size=num_peaks+1)[:,1:]
        
        # predict a height of zero for peaks in intensities that weren't annotated / in vocab
        mask = torch.isinf(log_y_pred)
        log_y_pred = torch.where(mask, torch.zeros_like(log_y_pred) + self.log_epsilon, log_y_pred)
        
        loss = -((batch.intensities * log_y_pred).sum(1)).mean()
        
        self.log(f'{step}/loss', loss, batch_size=batch_size, sync_dist=step=='val')
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')
        
    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')
    
    def predict_step(self, batch, batch_idx):
        # this should only be called at inference time - uses different batch structure
        
        log_y_pred = self(batch)
        
        x_pred = batch.mzs.unsqueeze(-1) + self.isotope_mzs.view(1,1,-1)
        
        y_pred = torch.softmax(log_y_pred.flatten(1), dim=1).view(log_y_pred.shape)
        
        # sparsify predictions
        xs = []
        ys = []
        for precursor_mz, has_isotopes, x, y in zip(
            batch.precursor_mz, batch.has_isotopes, x_pred, y_pred
        ):
            mask = (x >= self.min_mz)
            mask &= (x <= precursor_mz + self.isotope_mzs.max())
            mask &= (y > self.min_probability)
            mask[:,1:] &= has_isotopes # if we didn't ask for isotopes, don't predict them
            
            x = x[mask].flatten()
            y = y[mask].flatten() / y[mask].sum()
            
            # fast (approximate) way to deduplicate
            x = torch.round(x, decimals=4)
            x, idx = torch.unique(x, return_inverse=True)
            y = torch.zeros_like(x).scatter_add_(0, idx, y)
            
            xs.append(x)
            ys.append(y)

        return batch.spectrum, xs, ys
