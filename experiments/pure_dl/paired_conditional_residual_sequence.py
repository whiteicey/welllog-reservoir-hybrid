"""Within-well input-only conditional residuals across nine curves."""
import numpy as np
import torch
from torch import nn
from paired_well_view_sequence import RAW_CURVES
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES

RESIDUAL_NAMES=tuple(c+'_CONDITIONAL_RESIDUAL' for c in RAW_CURVES)

def conditional_features(frame):
    values=frame[list(RAW_CURVES)].to_numpy(np.float64).copy()
    for j,c in enumerate(RAW_CURVES):
        if c in ('RT','RXO'):
            values[:,j]=np.log1p(np.maximum(values[:,j],0))
    observed=np.isfinite(values)
    z=np.zeros_like(values)
    for j in range(9):
        ok=observed[:,j]
        if not ok.any():continue
        q25,median,q75=np.percentile(values[ok,j],[25,50,75])
        z[ok,j]=np.clip((values[ok,j]-median)/max(q75-q25,1e-6),-4,4)
    # Missing observations remain zero for input-only ridge covariance;
    # original nine missing indicators are retained in the original206.
    center=z.mean(0)
    z=z-center
    covariance=z.T@z/max(len(z),1)
    precision=np.linalg.inv(covariance+.2*np.eye(9))
    residual=(z@precision)/np.diag(precision)[None,:]
    residual[~observed]=0
    return np.clip(residual,-4,4).astype(np.float32)

class OriginalInputs(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
    def forward(self,x,relative_depth,valid,auxiliary=False,absolute_depth=None):
        return self.original(x[...,:206],relative_depth,valid,auxiliary,absolute_depth)

class ResidualInputs(nn.Module):
    def __init__(self,original,numeric_indices):
        super().__init__();self.original=original
        self.register_buffer('numeric_indices',torch.tensor(numeric_indices))
        with torch.random.fork_rng(devices=[]):
            self.projection=nn.Linear(9,len(numeric_indices),bias=False)
            nn.init.zeros_(self.projection.weight)
    def forward(self,x,relative_depth,valid,auxiliary=False,absolute_depth=None):
        clean=x.masked_fill(~valid[...,None],0)
        original=clean[...,:206]
        value=original.index_select(-1,self.numeric_indices)+self.projection(clean[...,206:])
        original=original.index_copy(-1,self.numeric_indices,value.to(original.dtype))
        return self.original(original,relative_depth,valid,auxiliary,absolute_depth)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,conditioning_indices):
        super().__init__(206,initial_means,retain_indices=retain_indices,numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0]=OriginalInputs(self.members[0])
        self.members[1]=ResidualInputs(self.members[1],numeric_indices)
