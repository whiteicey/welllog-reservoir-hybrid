"""Fixed monotone SW coordinate, no fitted parameters or validation calibration."""
import numpy as np
from scipy.special import expit

def transform(value):
    value=np.asarray(value,dtype=np.float64)
    bounded=np.clip(value,1e-6,100.-1e-6)
    return np.log(bounded)-np.log(100.-bounded)

def inverse(value):
    return 100.*expit(np.asarray(value,dtype=np.float64))
