"""Conservative feature-only treatment in the baseline's transformed input space.

Fixed before looking at validation scores: 11-point local median, 6 robust
deviations and 0.5 training-IQR minimum excursion, only runs <=2 points;
then training-fold 0.1/99.9-percentile clipping. No target-based filtering.
"""
import numpy as np
from scipy.ndimage import median_filter
import pandas as pd

CURVES=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']


def fit_treatment(train_arrays, columns):
    data=np.concatenate(train_arrays)
    q=np.nanpercentile(data,[.1,25,75,99.9],axis=0)
    return {'columns':columns,'treated_columns':CURVES,'lower':q[0].tolist(),'upper':q[3].tolist(),
            'iqr':np.maximum(q[2]-q[1],1e-3).tolist(),'window':11,'sigma':6.,'iqr_floor':.5,
            'max_run':2,'quantile_percent':[.1,99.9]}


def transform(a,spec):
    out=a.copy();audit={}
    for c in spec['treated_columns']:
        j=spec['columns'].index(c);original=a[:,j];finite=np.isfinite(original)
        if not finite.any():
            audit[c]={'observed':0,'spikes':0,'clipped':0};continue
        # Filled values only define local robust statistics; original missing values stay missing.
        filled=pd.Series(original).interpolate(limit_direction='both').to_numpy()
        center=median_filter(filled,size=spec['window'],mode='nearest')
        residual=np.abs(filled-center)
        local_scale=1.4826*median_filter(residual,size=spec['window'],mode='nearest')
        threshold=np.maximum(spec['sigma']*local_scale,spec['iqr_floor']*spec['iqr'][j])
        candidate=finite & (residual>threshold)
        changes=np.diff(np.r_[False,candidate,False].astype(np.int8))
        starts=np.flatnonzero(changes==1);ends=np.flatnonzero(changes==-1)
        spike=np.zeros(len(a),bool)
        for start,end in zip(starts,ends):
            if end-start<=spec['max_run'] and start>0 and end<len(a):spike[start:end]=True
        out[spike,j]=np.nan
        clip=np.isfinite(out[:,j]) & ((out[:,j]<spec['lower'][j]) | (out[:,j]>spec['upper'][j]))
        out[:,j]=np.clip(out[:,j],spec['lower'][j],spec['upper'][j])
        audit[c]={'observed':int(finite.sum()),'spikes':int(spike.sum()),'clipped':int(clip.sum())}
    return out,audit
