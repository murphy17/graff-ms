import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
import re
import os

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from pyteomics.mass import Composition

from src.io import read_msp, write_msp

from src.graff import atom_types

################################################################
# Helper functions for parsing annotations
# ###############################################################

def composition_to_string(x):
    return ''.join([a+str(x[a]) for a in sorted(x)])

def parse_collision_energy(s):
    if s.startswith('NCE='):
        s = s[4:]
        if s.endswith('eV'):
            s, ev = s.split(' ')
            ev = float(ev[:-2])
            if ev == 0:
                ev = np.nan
        else:
            ev = np.nan
        s = s[:-1]
        nce = float(s)
    else:
        ev = float(s)
        nce = np.nan
    return ev, nce

def parse_formula(annot, precursor):
    formula = Composition('')
    precursor = Composition(formula=precursor)
    adducts = []
    losses = []
    isotope = 0
    charge = 1
    chunks = [''] + re.split(r'([\+-])',annot)
    for prefix, chunk in zip(chunks[::2],chunks[1::2]):
        if len(chunk) == 0:
            continue
        elif prefix == '^':
            charge = int(chunk)
        else:
            if chunk[-1] == 'i':
                chunk = chunk[:-1]
                isotope = int(chunk) if chunk else 1
                if prefix == '-':
                    isotope = -isotope
            else:
                if chunk[0].isdigit():
                    n = int(chunk[0])
                    chunk = chunk[1:]
                else:
                    n = 1
                if chunk == 'p':
                    chunk = precursor
                else:
                    chunk = Composition(formula=chunk) * n
                if prefix == '':
                    formula += chunk
                elif prefix == '+':
                    adducts.append(chunk)
                    formula += chunk
                elif prefix == '-':
                    losses.append(chunk)
                    formula -= chunk
    return formula, adducts, losses, isotope, charge

def extract_annotations(item):
    annots = item['annots']
    precursor = Composition(formula=item.Formula)
    if item.Precursor_type == '[M+H]+':
        precursor['H'] += 1
    elif item.Precursor_type == '[M-H]-':
        precursor['H'] -= 1
    else:
        raise NotImplementedError
    precursor = composition_to_string(precursor)

    peaks = []
    products = []
    losses = []
    isotopes = []

    all_good = True

    for peak, xs in enumerate(annots):
        if not all_good:
            break

        # split the formula hypotheses
        xs = xs.split(';')

        for j, x in enumerate(xs):
            # strip the peak count
            x = re.sub(r' ?\d+/\d+$','',x)
            if len(x) == 0:
                all_good = False
                break
            if x in ('?','more'):
                continue
            # glycan parsing not implemented
            if '|' in x:
                all_good = False
                break
            if '/' in x:
                x, ppm = x.split('/')
            if '=' in x:
                x, _ = x.split('=')

            product, _, _, isotope, _ = parse_formula(x, precursor)
            loss = Composition(formula=precursor) - product

            product = composition_to_string(product)
            loss = composition_to_string(loss)

            peaks.append(peak)
            products.append(product)
            losses.append(loss)
            isotopes.append(isotope)

    if all_good:
        return peaks, products, losses, isotopes
    else:
        return None

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('nist_path')
    parser.add_argument('inchi_path')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_frac', type=float, default=0.8)
    parser.add_argument('--test_frac', type=float, default=0.1)
    args = parser.parse_args()
    
    from pandarallel import pandarallel
    from multiprocessing import cpu_count
    pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=cpu_count()//2)

    print('Parsing MSP... ',end='')
    df = read_msp(args.nist_path, parallel=True)
    print('done')
    
    ################################################################
    # filter compounds
    ################################################################
    
    df = df.query('Instrument_type == "HCD"')
    df = df.query('(Precursor_type == "[M+H]+") or (Precursor_type == "[M-H]-")')
    df['PrecursorMZ'] = df['PrecursorMZ'].str.split(',').str[0].astype(float)
    df = df.query('PrecursorMZ <= 1000')
    df = df.loc[~df['Notes'].str.lower().str.contains('peptide')]
    df = df.loc[~df['Notes'].str.lower().str.contains('glycan')]
    df = df.loc[df['Formula'].parallel_apply(lambda x: set(Composition(formula=x)) <= set(atom_types))]

    # convert eV <-> NCE
    # see Thermo Scientific PSB104, Normalized Collision Energy Technology
    # we use NCE in the model as it is directly provided for more spectra than eV is
    df['eV'], df['NCE'] = zip(*df['Collision_energy'].map(parse_collision_energy))
    df['eV'] = df['eV'].fillna(df['NCE']*df['PrecursorMZ']/500)
    df['NCE'] = df['NCE'].fillna(df['eV']*500/df['PrecursorMZ'])
    
    # uniqu identifier
    df['Spectrum'] = df['NISTNO']

    ################################################################
    # query UniChem for inchis from keys - slow
    ################################################################
    
    print('Loading InChIs... ',end='')
    if not os.path.exists(args.inchi_path):
        from bioservices import UniChem
        unichem = UniChem()
        inchis = {}
        for k in tqdm(df['InChIKey'].unique()):
            result = unichem.get_inchi_from_inchikey(k)
            if 'error' in result:
                continue
            inchis[k] = result[0]['standardinchi']
        inchis = pd.Series(inchis)
        inchis.to_pickle(args.inchi_path)
    else:
        inchis = pd.read_pickle(args.inchi_path)
    print('done')

    df['InChI'] = df['InChIKey'].map(inchis)
    df = df.dropna(subset=['InChI'])
    df['mol'] = df['InChI'].parallel_apply(lambda x: Chem.MolFromInchi(x))
    df = df.dropna(subset='mol')
    df['SMILES'] = df['mol'].parallel_apply(lambda x: Chem.MolToSmiles(x))
    df = df.dropna(subset='SMILES')
    # remove disconnected molecular graphs (e.g. organometallics)
    df = df.loc[~df['SMILES'].str.contains(r'\.')]
    
    df = df.reset_index(drop=True)

    ################################################################
    # parse annotations
    ################################################################
    
    print('Parsing annotations... ',end='')
    annots = df.parallel_apply(extract_annotations, axis=1)
    annots = pd.DataFrame([*annots])
    annots.index = df.index
    annots = annots.dropna() # remove spectra that failed parsing
    annots.columns = ['peaks','products','losses','isotopes']
    df = df.join(annots,how='inner')
    print('done')

    # fill in some helper fields
    df['InChIKey2D'] = df['InChIKey'].str.split('-').str[0]
    df['has_isotopes'] = df['isotopes'].map(any)
    df['intensities'] = df['intensities'] / df['intensities'].map(sum)
    
    ################################################################
    # structure-disjoint uniform splitting
    ################################################################
    
    inchis = df['InChIKey2D'].unique()

    npr.seed(args.seed)
    npr.shuffle(inchis)
    train_inchis = inchis[:int(args.train_frac*len(inchis))]
    val_inchis = inchis[int(args.train_frac*len(inchis)):-int(args.test_frac*len(inchis))]
    test_inchis = inchis[-int(args.test_frac*len(inchis)):]

    df['split'] = ''
    df.loc[df['InChIKey2D'].isin(train_inchis),'split'] = 'train'
    df.loc[df['InChIKey2D'].isin(val_inchis),'split'] = 'val'
    df.loc[df['InChIKey2D'].isin(test_inchis),'split'] = 'test'

    # restrict test split to monoisotopic spectra at 20/35/50 NCE (for CFM-ID)
    df.loc[df.query('split=="test" and (has_isotopes or NCE!=20 and NCE!=35 and NCE!=50)').index,'split'] = ''

    ################################################################
    # export
    ################################################################
    
    print('Saving dataset... ',end='')
    df_path = os.path.splitext(args.nist_path)[0] + '.pkl'
    df.to_pickle(df_path)
    print('done')

    # export a filtered MSP and a CSV for inference
    print('Exporting splits... ',end='')
    for split in ['train','val','test']:
        tsv_path = os.path.splitext(args.nist_path)[0] + '_' + split + '.tsv'
        cols = ['Spectrum','SMILES','Precursor_type','NCE','Instrument']
        tsv_df = df.query(f'split=="{split}"')[cols]
        tsv_df.to_csv(tsv_path, sep='\t', header=False, index=False)

        msp_path = os.path.splitext(args.nist_path)[0] + '_' + split + '.msp'
        cols += ['InChIKey','Formula','PrecursorMZ']
        msp_df = df.query(f'split=="{split}"')[cols+['mzs','intensities']]
        write_msp(
            msp_path,
            msp_df['mzs'].tolist(), 
            msp_df['intensities'].tolist(),
            **{c: msp_df[c].tolist() for c in cols}
        )
    print('done')
    
    # log split sizes to console
    num_spectra = df['split'].value_counts()
    num_structures = df.groupby('split')['InChIKey'].nunique()
    for split in ['train','val','test']:
        print(f'{split}: {num_spectra[split]} spectra, {num_structures[split]} structures')
