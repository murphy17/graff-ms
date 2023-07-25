import re
import numpy as np
from collections import defaultdict
import pandas as pd

def read_cfm(path):
    with open(path,'r') as f:
        text = f.read()
    filename = path.split('/')[-1]
    smiles = re.findall(r'SMILES=(.+)',text)[0]
    inchikey = re.findall(r'InChiKey=(.+)',text)[0]
    spectra = []
    for l in text.split('\n'):
        if not l:
            break
        elif l.startswith('#'):
            continue
        elif l.startswith('energy'):
            energy = l
            spectra.append({
                'filename': filename,
                'smiles': smiles,
                'inchikey': inchikey,
                'energy': energy,
                'mzs': [],
                'intensities': []
            })
        elif l[0].isdigit():
            l = l.split(' ')
            spectra[-1]['mzs'].append(float(l[0]))
            spectra[-1]['intensities'].append(float(l[1]))
    for s in range(len(spectra)):
        spectra[s]['mzs'] = np.array(spectra[s]['mzs'])
        spectra[s]['intensities'] = np.array(spectra[s]['intensities'])
    return spectra

def read_mgf(path):
    with open(path,'r') as f:
        text = f.read()
    filename = path.split('/')[-1]
    spectra = []
    for block in text.split('\n\n'):
        precursor_mz = float(re.findall('PEPMASS=(.+)',block)[0])
        precursor_charge = int(re.findall('CHARGE=(\d+)',block)[0])
        peaks = re.findall(r'(\d+\.?\d*)\t(\d+\.?\d*)',block)
        if len(peaks)==0:
            continue
        mzs, intensities = zip(*peaks)
        mzs = np.array([*map(float,mzs)])
        intensities = np.array([*map(float,intensities)])
        spectra.append({
            'filename': filename,
            'precursor_mz': precursor_mz,
            'precursor_charge': precursor_charge,
            'mzs': mzs,
            'intensities': intensities
        })
    return spectra

from collections import Counter
from tqdm import tqdm

def write_msp(path,mzs,intensities,annots=None,verbose=False,**metadata):
    metadata = {k: [*v] for k,v in metadata.items()}
    with open(path,'w') as f:
        for i in tqdm(range(len(mzs)),disable=not verbose):
            for k in metadata.keys():
                v = metadata[k][i]
                f.write(f'{k}: {v}\n')
            x = mzs[i]
            y = intensities[i] / max(intensities[i]) * 999
            if len(x)>1:
                idx = np.argsort(x)
                x = x[idx]
                y = y[idx]
            if annots:
                a = np.array(annots[i])
                a = a[idx]
            for j in range(len(x)):
                line = f'{x[j]:.4f} {y[j]:.2f}'
                if annots is not None:
                    line = line + ' ' + a[j]
                f.write(line+'\n')
            f.write('\n')

def read_msp(path,parallel=False,sample=None):
    with open(path,'r') as f:
        text = f.read()
    blocks = text.strip().split('\n\n')
    
    if parallel:
        from pandarallel import pandarallel
        from os import cpu_count
        pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=cpu_count()//2)
    
    # shuffle for load balancing
    blocks = pd.Series(blocks).sample(frac=1, random_state=0)
    
    if sample:
        blocks = blocks[:sample]
    
    def func(block):
        data = defaultdict(list)
        mzs = []
        intens = []
        annots = []
        for line in block.split('\n'): 
            if len(line) == 0: continue
            if ':' in line:
                k, v = line.split(':',maxsplit=1)
                data[k.strip()].append(v.strip())
            elif line[0].isdigit():
                fields = line.split(maxsplit=2)
                x, y = fields[:2]
                if len(fields) == 3:
                    a = fields[-1].strip('"')
                else:
                    a = ''
                mzs.append(x)
                intens.append(y)
                annots.append(a)
            else:
                continue
                
        data = {k: v[0] if len(v)==1 else v for k, v in data.items()}

        data['mzs'] = np.array(mzs,dtype=np.float32)
        data['intensities'] = np.array(intens,dtype=np.float32)
        data['annots'] = np.array(annots,dtype=np.object_)
        
        return data
        
    if parallel:
        df = pd.DataFrame.from_records(blocks.parallel_apply(func)).sort_index()
    else:
        df = pd.DataFrame.from_records([*map(func,blocks)]).sort_index()
    df = df.reset_index(drop=True)
    df['filename'] = path.split('/')[-1]
    
    df = df.infer_objects()
    
    return df