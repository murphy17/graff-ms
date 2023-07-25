import numpy as np
import numpy.random as npr
from matchms import Spectrum
from matchms.similarity import CosineHungarian, CosineGreedy

def ms_cosine_similarity(
    mzs_x, intensities_x, precursor_x,
    mzs_y, intensities_y, precursor_y,
    tol
):  
    mzs_x = np.array(mzs_x, dtype=float)
    intensities_x = np.array(intensities_x, dtype=float)
    precursor_x = float(precursor_x)
    mzs_y = np.array(mzs_y, dtype=float)
    intensities_y = np.array(intensities_y, dtype=float)
    precursor_y = float(precursor_y)
    
    idx_x = np.argsort(mzs_x)
    idx_y = np.argsort(mzs_y)
    mzs_x, intensities_x = mzs_x[idx_x], intensities_x[idx_x]
    mzs_y, intensities_y = mzs_y[idx_y], intensities_y[idx_y]
    
    spectrum_x = Spectrum(
        mz=mzs_x,
        intensities=intensities_x,
        metadata={'precursor_mz': float(precursor_x)}
    )
    spectrum_y = Spectrum(
        mz=mzs_y,
        intensities=intensities_y,
        metadata={'precursor_mz': float(precursor_y)}
    )
    
    cos_xy = float(CosineHungarian(tolerance=tol).pair(spectrum_x, spectrum_y)['score'])
        
    return cos_xy

def bootstrap(x,B=1000,a=0.05,random_state=0):
    R = npr.RandomState(random_state)
    stats = [np.mean(R.choice(x,len(x),replace=True)) for _ in range(B)]
    return np.mean(x), np.quantile(stats, a/2), np.quantile(stats, 1-a/2)

def ci95(x):
    m, l, u = bootstrap(x)
    return m, max(u-m,m-l)
