"""Input-only directional contrasts over physical depth intervals."""
import numpy as np
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
RADII=(1.,4.)
FEATURE_NAMES=tuple(n for r in RADII for n in
    tuple(f'{c}_DEPTH_BELOW_ABOVE_{r:g}M' for c in CURVES)+
    (f'JOINT_CHANGE_{r:g}M',f'CHANGE_PARTICIPATION_{r:g}M',f'PAIRED_COVERAGE_{r:g}M'))

def directional_features(frame):
    depth=frame['DEPTH'].to_numpy(np.float64)
    if not np.isfinite(depth).all() or (np.diff(depth)<0).any():
        raise ValueError('Finite monotonic depth is required; do not silently reorder output rows.')
    values=frame[list(CURVES)].to_numpy(np.float64).copy()
    for j,c in enumerate(CURVES):
        if c in ('RT','RXO'):values[:,j]=np.log1p(np.maximum(values[:,j],0))
    observed=np.isfinite(values); z=np.zeros_like(values)
    for j in range(9):
        ok=observed[:,j]
        if ok.any():
            lo,med,hi=np.percentile(values[ok,j],[25,50,75])
            z[ok,j]=np.clip((values[ok,j]-med)/max(hi-lo,1e-6),-4,4)
    sums=np.vstack([np.zeros((1,9)),np.cumsum(z,axis=0)])
    counts=np.vstack([np.zeros((1,9)),np.cumsum(observed,axis=0)])
    left_end=np.searchsorted(depth,depth,side='left')
    right_start=np.searchsorted(depth,depth,side='right')
    parts=[]
    for radius in RADII:
        left_start=np.searchsorted(depth,depth-radius-1e-7,side='left')
        right_end=np.searchsorted(depth,depth+radius+1e-7,side='right')
        lc=counts[left_end]-counts[left_start];rc=counts[right_end]-counts[right_start]
        lm=(sums[left_end]-sums[left_start])/np.maximum(lc,1)
        rm=(sums[right_end]-sums[right_start])/np.maximum(rc,1)
        paired=(lc>0)&(rc>0)
        diff=np.tanh(rm-lm)*paired
        magnitude=np.sqrt((diff*diff).sum(1)/np.maximum(paired.sum(1),1))
        lf=lc/np.maximum((left_end-left_start)[:,None],1)
        rf=rc/np.maximum((right_end-right_start)[:,None],1)
        coverage=np.minimum(lf,rf).mean(1)
        participation=np.abs(diff).sum(1)**2/(9*(diff*diff).sum(1)+1e-12)
        parts.append(np.column_stack([diff,magnitude,participation,coverage]))
    return np.concatenate(parts,axis=1).astype(np.float32)
