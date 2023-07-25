import numpy as np
import numpy.random as npr
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from multiprocessing import cpu_count

from src.io import read_msp
from src.metrics import ms_cosine_similarity, ci95

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('prediction_path')
parser.add_argument('target_path')
parser.add_argument('--matchms_tol', type=float, default=0.1)
args = parser.parse_args()

pandarallel.initialize(progress_bar=False, verbose=0, nb_workers=cpu_count()//2)

################################################################
# load predictions
################################################################

print('Loading predicted spectra... ',end='')
df_pred = read_msp(args.prediction_path)
print('done')

################################################################
# load ground truth
################################################################

print('Loading ground truth spectra... ',end='')
df_true = read_msp(args.target_path)
print('done')

################################################################
# merge on shared Spectrum key
################################################################

df = df_true.merge(df_pred, on='Spectrum', how='inner', suffixes=('_true','_pred'))

################################################################
# parallelize cosine similarity scoring
################################################################

print('Calculating cosine similarity... ',end='')
func = lambda item: ms_cosine_similarity(
    mzs_x=item.mzs_pred,
    intensities_x=item.intensities_pred,
    precursor_x=item.PrecursorMZ_pred,
    mzs_y=item.mzs_true,
    intensities_y=item.intensities_true,
    precursor_y=item.PrecursorMZ_true,
    tol=args.matchms_tol
)
df['score'] = df.parallel_apply(func, axis=1)
print('done')

################################################################
# report nonparametric 95% CIs via bootstrap
################################################################

print(f'Mean cosine similarity = %.2f +- %.2f (N={len(df)})' % ci95(df['score']))
