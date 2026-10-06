"""Input-only local increment texture; not an anomaly labeling rule."""
import numpy as np
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
FEATURE_NAMES=tuple(c+'_'+s for c in CURVES for s in
                    ('DIRECTION_PERSISTENCE51','ENERGY_FRACTION9_51','SPIKE_CONCENTRATION51','INCREMENT_COVERAGE51'))

def rolling_sum(x,width):
    n=len(x);i=np.arange(n);left=np.maximum(i-width//2,0);right=np.minimum(i+width//2+1,n)
    acc=np.r_[0.,np.cumsum(x,dtype=np.float64)]
    return acc[right]-acc[left],right-left

def texture_features(frame):
    result=[]
    for c in CURVES:
        values=frame[c].to_numpy(np.float64).copy()
        if c in ('RT','RXO'):values=np.log1p(np.maximum(values,0))
        ok=np.isfinite(values);z=np.zeros(len(values))
        if ok.any():
            lo,med,hi=np.percentile(values[ok],[25,50,75])
            z[ok]=np.clip((values[ok]-med)/max(hi-lo,1e-6),-4,4)
        valid=np.r_[False,ok[1:]&ok[:-1]]
        diff=np.r_[0.,np.diff(z)];diff[~valid]=0
        signed,n=rolling_sum(diff,51)
        absolute,_=rolling_sum(np.abs(diff),51)
        energy,_=rolling_sum(diff**2,51)
        local_energy,_=rolling_sum(diff**2,9)
        fourth,_=rolling_sum(diff**4,51)
        count,_=rolling_sum(valid.astype(float),51)
        persistence=np.abs(signed)/(absolute+1e-8)
        fraction=local_energy/(energy+1e-8)
        concentration=fourth/(energy**2+1e-8)
        coverage=count/np.maximum(n,1)
        result.extend([persistence,fraction,concentration,coverage])
    return np.clip(np.column_stack(result),0,1).astype(np.float32)
