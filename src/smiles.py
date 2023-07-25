# this is largely copied and pasted from DGLLife canonical atom + bond featurizers

import torch
from rdkit import Chem, RDLogger
import numpy as np
from torch_geometric.data import Data
RDLogger.DisableLog('rdApp.*')

x_map = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

x_map = {k: {v: i for i, v in enumerate(vs)} for k, vs in x_map.items()}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

e_map = {k: {v: i for i, v in enumerate(vs)} for k, vs in e_map.items()}

def from_smiles(smiles, with_hydrogen=False, kekulize=False):
    mol = Chem.MolFromSmiles(smiles)
    return from_mol(mol, with_hydrogen, kekulize)

def from_mol(mol, with_hydrogen=False, kekulize=False):
    smiles = Chem.MolToSmiles(mol)
    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = [
            x_map['atomic_num'][atom.GetAtomicNum()],
            x_map['chirality'][str(atom.GetChiralTag())],
            x_map['degree'][atom.GetTotalDegree()],
            x_map['formal_charge'][atom.GetFormalCharge()],
            x_map['num_hs'][atom.GetTotalNumHs()],
            x_map['num_radical_electrons'][atom.GetNumRadicalElectrons()],
            x_map['hybridization'][str(atom.GetHybridization())],
            x_map['is_aromatic'][atom.GetIsAromatic()],
            x_map['is_in_ring'][atom.IsInRing()]
        ]
        xs.append(x)

    x = torch.tensor(np.array(xs), dtype=torch.long).view(-1, len(x_map))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        idx = bond.GetIdx()

        e = [
            e_map['bond_type'][str(bond.GetBondType())],
            e_map['stereo'][str(bond.GetStereo())],
            e_map['is_conjugated'][bond.GetIsConjugated()],
        ]
        
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(np.array(edge_indices))
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.long).view(-1, len(e_map))

    if edge_index.numel() > 0:
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
