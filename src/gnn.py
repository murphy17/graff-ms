import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as pyg
from torch_geometric.nn import GraphNorm
import pytorch_lightning as pl
import numpy as np
from rdkit import Chem

class ResBlock(nn.Module):
    def __init__(self, ch, dr):
        super().__init__()
        self.linear = nn.Linear(ch, ch)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dr)
        self.norm = nn.LayerNorm(ch)
    
    def forward(self, x):
        x = x + self.dropout(self.act(self.linear(x)))
        x = self.norm(x)
        return x

class GINEConv(pyg.GINEConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act = nn.SiLU(inplace=True)
    def message(self, x_j, edge_attr):
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return self.act(x_j + edge_attr)

class GINELayer(nn.Module):
    def __init__(self, model_dim, dropout=0, bottleneck=1):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        identity = nn.Sequential(nn.Identity())
        identity[0].in_features = model_dim // bottleneck
        self.conv = GINEConv(
            nn=identity,
            edge_dim=model_dim // bottleneck,
            eps=0,
            train_eps=True
        )
        self.edge_lin = nn.Linear(model_dim, model_dim // bottleneck)
        self.lin1 = nn.Linear(model_dim, model_dim // bottleneck)
        self.lin2 = nn.Linear(model_dim // bottleneck, model_dim) if bottleneck>1 else nn.Identity()
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = GraphNorm(model_dim)
        self.act = nn.SiLU(inplace=True)
        
    def forward(self, x, e, batch, edge_index):
        dx = self.lin1(x)
        dx = self.conv(dx, edge_index, e)
        dx = self.dropout(dx)
        dx = self.lin2(dx)
        dx = self.act(dx)
        x = self.norm(x + dx, batch)
        return x

class GINEEdgeLayer(nn.Module):
    def __init__(self, model_dim, dropout=0, bottleneck=1):
        super().__init__()
        self.model_dim = model_dim
        self.dropout = dropout
        self.lin1 = nn.Linear(3 * model_dim, model_dim // bottleneck)
        self.lin2 = nn.Linear(model_dim // bottleneck, model_dim) if bottleneck>1 else nn.Identity()
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.norm = GraphNorm(model_dim)

    def forward(self, x, e, batch, edge_index):
        de = torch.cat([e, x[edge_index[0]], x[edge_index[1]]], 1)
        de = self.lin1(de)
        de = self.act(de)
        de = self.dropout(de)
        de = self.lin2(de)
        e = self.norm(e + de, batch[edge_index[0]])
        return e

class GINE(nn.Module):
    def __init__(self, node_dim, edge_dim, model_dim, model_depth, 
                 jumping_knowledge=True,
                 bottleneck=1,
                 dropout=0, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.model_dim = model_dim
        self.model_depth = model_depth
        self.dropout = dropout
        self.bottleneck = bottleneck
        self.jumping_knowledge = jumping_knowledge
        
        if node_dim == model_dim:
            self.node_emb = nn.Identity()
        else:
            self.node_emb = nn.Sequential(
                nn.Linear(node_dim, model_dim),
            )
        if edge_dim == model_dim:
            self.edge_emb = nn.Identity()
        else:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_dim, model_dim),
            )

        layers = []
        edge_layers = []
        for _ in range(model_depth):
            layers.append(GINELayer(model_dim, dropout, bottleneck))
            edge_layers.append(GINEEdgeLayer(model_dim, dropout, bottleneck))
        self.layers = nn.ModuleList(layers)
        self.edge_layers = nn.ModuleList(edge_layers)
        
        self.linear = nn.Linear(model_dim, model_dim)
        
    def forward(self, g, x, e, e_mask=None):
        x = self.node_emb(x)
        e = self.edge_emb(e)
        for layer, edge_layer in zip(self.layers, self.edge_layers):
        # for layer in self.layers:
            x = layer(x, e, g.batch, g.edge_index)
            if e_mask is not None:
                e = torch.where(e_mask, 
                                edge_layer(x, e[e_mask], g.batch, g.edge_index[:,e_mask]),
                                e)
            else:
                e = edge_layer(x, e, g.batch, g.edge_index)
        x = self.linear(x)
        return x

from src.smiles import from_smiles, x_map, e_map
class CanonicalOneHot(nn.Module):
    def __init__(self, node_feats={}, edge_feats={}, mask_value=-1, use_bool=True):
        super().__init__()
        self.node_feats = {**x_map, **node_feats}
        self.edge_feats = {**e_map, **edge_feats}
        self.node_dim = sum(map(len,self.node_feats.values()))
        self.edge_dim = sum(map(len,self.edge_feats.values()))
        self.mask_value = -1
        self.use_bool = use_bool
    
    def forward(self, x, e):
        x_onehot = torch.zeros(x.shape[0],self.node_dim,device=x.device)
        j = 0
        for i, (feat, levels) in enumerate(self.node_feats.items()):
            if self.use_bool and [*levels] == [False, True]:
                # set masks to False
                mask = x[:,i] == self.mask_value
                x_onehot[~mask,j] = x[~mask,i].float()
                j += 1
            else:
                d = len(levels)
                mask = x[:,i] == self.mask_value
                x_onehot[~mask,j:j+d] = F.one_hot(x[~mask,i].long(),d).float()
                j += d
            
        e_onehot = torch.zeros(e.shape[0],self.edge_dim,device=x.device)
        j = 0
        for i, (feat, levels) in enumerate(self.edge_feats.items()):
            if self.use_bool and [*levels] == [False, True]:
                # set masks to False
                mask = e[:,i] == self.mask_value
                e_onehot[~mask,j] = e[~mask,i].float()
                j += 1
            else:
                d = len(levels)
                mask = e[:,i] == self.mask_value
                e_onehot[~mask,j:j+d] = F.one_hot(e[~mask,i].long(),d).float()
                j += d

        return x_onehot, e_onehot

from .smiles import from_mol
def mol_to_graph(mol):
    graph = from_mol(mol)
    graph.x = graph.x.byte()
    graph.edge_attr = graph.edge_attr.byte()
    return graph

def smiles_to_graph(smiles):
    return mol_to_graph(Chem.MolFromSmiles(smiles))

def add_virtual_node(g, inplace=False, add_flags=True, mask_value=0):
    edge_index = []
    edge_index.extend([(i,g.num_nodes) for i in range(g.num_nodes)])
    edge_index.extend([(g.num_nodes,i) for i in range(g.num_nodes)])
    edge_index = torch.LongTensor(edge_index).view(-1,2).T 
    edge_attr = torch.zeros(edge_index.shape[1], g.edge_attr.shape[1],
                            dtype=g.edge_attr.dtype,
                            device=g.edge_attr.device) + mask_value
    num_nodes = g.num_nodes
    num_edges = g.num_edges
    
    x = torch.zeros(1, g.x.shape[1], 
                    dtype=g.x.dtype, device=g.x.device) + mask_value
    
    if not inplace:
        g = g.clone()
    g.x = torch.cat([g.x,x],0)
    g.edge_index = torch.cat([g.edge_index,edge_index],1)
    g.edge_attr = torch.cat([g.edge_attr,edge_attr],0)
    if add_flags:
        n_in = n_out = edge_index.shape[1] // 2
        is_virt_node = torch.BoolTensor([0]*num_nodes+[1]).unsqueeze(-1)
        is_virt_in_edge = torch.BoolTensor([0]*num_edges+[1]*n_in+[0]*n_out).unsqueeze(-1)
        is_virt_out_edge = torch.BoolTensor([0]*num_edges+[0]*n_in+[1]*n_out).unsqueeze(-1)
        g.x = torch.cat([g.x,is_virt_node],1)
        g.edge_attr = torch.cat([g.edge_attr,is_virt_in_edge,is_virt_out_edge],1)
        
    return g

from scipy.linalg import eigh
from torch_geometric.utils import (
    get_laplacian,
    to_scipy_sparse_matrix
)
def graph_laplacian(graph, k, padding_value=0):
    num_nodes = graph.num_nodes
    
    V = torch.empty(num_nodes, k)
    D = torch.empty(k)
    
    if num_nodes > 0:
        edge_index, edge_weight = get_laplacian(
            graph.edge_index,
            normalization='sym',
            num_nodes=num_nodes,
        )

        L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

        eig_vals, eig_vecs = eigh(L.toarray())

        idx = eig_vals.argsort()
        eig_vecs = np.real(eig_vecs[:,idx])
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:,1:k+1]
        eig_vals = eig_vals[1:k+1]

        eig_vecs = eig_vecs / np.linalg.norm(eig_vecs,axis=0)
        
        V[:] = padding_value
        V[:,:eig_vecs.shape[1]] = torch.from_numpy(eig_vecs)

        D[:] = padding_value
        D[:eig_vals.shape[0]] = torch.from_numpy(eig_vals)

    graph.eigvecs = V
    graph.eigvals = D.unsqueeze(0)

    return graph

class SignNet(nn.Module):
    def __init__(self,
                 num_eigs, embed_dim,
                 phi_dim, phi_depth,
                 rho_dim, rho_depth,
                 dropout=0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_eigs = num_eigs
        self.phi_dim = phi_dim
        self.phi_depth = phi_depth
        self.rho_dim = rho_dim
        self.rho_depth = rho_depth
        self.dropout = dropout
        
        layers = []
        layers.append(nn.Linear(2, phi_dim))
        for _ in range(phi_depth):
            layers.extend([
                nn.Linear(phi_dim, phi_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(phi_dim)
            ])
        self.phi = nn.Sequential(*layers)
        
        layers = []
        if phi_dim != rho_dim:
            layers.append(nn.Linear(phi_dim, rho_dim))
        for _ in range(rho_depth):
            layers.extend([
                nn.Linear(rho_dim, rho_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.LayerNorm(rho_dim)
            ])
        if embed_dim != rho_dim:
            layers.append(nn.Linear(rho_dim, embed_dim))
        self.rho = nn.Sequential(*layers)
    
    def forward(self, v, l):
        x = self.phi(torch.stack([ v,l],-1)) + \
            self.phi(torch.stack([-v,l],-1))
        x = self.rho(x.sum(1))
        return x
