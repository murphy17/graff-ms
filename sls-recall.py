import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from pandarallel import pandarallel
from multiprocessing import cpu_count
from sklearn.neighbors import BallTree

from src.io import read_msp
from src.metrics import ms_cosine_similarity, ci95

################################################################
# load hyperparameters from command line
################################################################

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('query_path')
parser.add_argument('library_path')
parser.add_argument('--k', type=int, nargs='+', default=[1,5,10])
parser.add_argument('--precursor_tol', type=float, default=0.1)
parser.add_argument('--matchms_tol', type=float, default=0.1)
parser.add_argument('--min_matches', type=int, default=2)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--subsample', type=int, default=0)
args = parser.parse_args()

pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=cpu_count()//2)

################################################################
# load query spectra
################################################################

print('Loading queries... ',end='')
df_query = read_msp(args.query_path)
print('done')

################################################################
# load library spectra
################################################################

print('Loading library... ',end='')
df_library = read_msp(args.library_path)
print('done')

num_queries = df_query['Spectrum'].nunique()
num_targets = df_library['Spectrum'].nunique()
print(f'Querying {num_queries} against {num_targets} spectra')

if args.subsample:
    df_library = df_library.sample(n=args.subsample, random_state=args.seed)

# extract connectivity substring for subsequent recall evaluation
df_query['InChIKey2D'] = df_query['InChIKey'].str.split('-').str[0]
df_library['InChIKey2D'] = df_library['InChIKey'].str.split('-').str[0]

################################################################
# prefilter matches using sensible MS1, NCE criteria
################################################################

df_query['PrecursorMZ'] = df_query['PrecursorMZ'].astype(float)
df_library['PrecursorMZ'] = df_library['PrecursorMZ'].astype(float)
df_query['NCE'] = df_query['NCE'].astype(float)
df_library['NCE'] = df_library['NCE'].astype(float)

print('Prefiltering matches... ',end='')
query_idxs = []
library_idxs = []
for (precursor_type, nce), sub_library in df_library.groupby(['Precursor_type','NCE']):
    # use a ball-tree for fast lookup within float tolerance
    tree = BallTree(sub_library['PrecursorMZ'].values[:,None])
    sub_query = df_query.query(f'Precursor_type=="{precursor_type}" and NCE=={nce}')
    radii = tree.query_radius(sub_query['PrecursorMZ'].values[:,None], args.precursor_tol)
    for i, r in enumerate(radii):
        for j in r:
            query_idxs.append(sub_query.index[i])
            library_idxs.append(sub_library.index[j])
df = pd.concat([
    df_query.loc[query_idxs].add_suffix('_query').reset_index(drop=True),
    df_library.loc[library_idxs].add_suffix('_library').reset_index(drop=True)
], axis=1)
print('done')

################################################################
# calculate cosine similarity in parallel
################################################################

print('Calculating cosine similarity... ',end='')
func = lambda item: ms_cosine_similarity(
    mzs_x=item.mzs_query,
    intensities_x=item.intensities_query,
    precursor_x=item.PrecursorMZ_query,
    mzs_y=item.mzs_library,
    intensities_y=item.intensities_library,
    precursor_y=item.PrecursorMZ_library,
    tol=args.matchms_tol
)
df['score'] = df.parallel_apply(func,axis=1)
print('done')

################################################################
# remove any duplicated structures (keeping best match)
################################################################

print('Deduplicating library matches... ',end='')
df = df.sort_values('score',ascending=False)
df = df.drop_duplicates(subset=['Spectrum_query','InChIKey2D_library'],keep='first')
print('done')

################################################################
# compute recall-at-k
################################################################

print('Calculating recall... ',end='')
structure_rank = {}
formula_rank = {}
num_queries = 0
for spectrum, matches in df.groupby('Spectrum_query'):
    # two assumptions: (1) the target is actually in the library
    if matches['InChIKey2D_query'].iat[0] not in set(matches['InChIKey2D_library']): continue
    # # and (2) it's not the only thing
    if len(matches) < args.min_matches: continue
    structure_rank[spectrum] = (matches['InChIKey2D_query'] == matches['InChIKey2D_library']).argmax() + 1
    formula_rank[spectrum] = (matches['Formula_query'] == matches['Formula_library']).argmax() + 1
    num_queries += 1
structure_rank = pd.Series(structure_rank)
formula_rank = pd.Series(formula_rank)
print('done')

################################################################
# report 95% CIs via nonparametric bootstrap
################################################################

for k in args.k:
    print(f'Structure recall @ {k} = %.2f +- %.2f' % ci95(structure_rank<=k))

for k in args.k:
    print(f'Formula recall @ {k} = %.2f +- %.2f' % ci95(formula_rank<=k))
