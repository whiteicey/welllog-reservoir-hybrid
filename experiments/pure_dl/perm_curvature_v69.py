"""PERM quantile-spacing curvature penalty; reuse the unchanged v68 sum control."""
import numpy as np
import torch
from perm_gradient_v68 import ROOT,Network as Parent,Evaluator,load_data,PARENT_PRED,assign_gradients
from quantile_density_v48 import channel_objective,knots
STUDY=ROOT/'results/deep_learning/perm_curvature_v69_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_gradient_v68_20260929'
CONTROL=ROOT/'outputs/pure_dl_perm_gradient_v68_PERM_sum_fold5'
MODES={'curvature':dict(target='PERM',c=1,length=1024)}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class Network(Parent):
    def __init__(self,mode,metadata):
        if mode!='curvature':raise ValueError(mode)
        super().__init__('sum',metadata);self.study_mode=mode

def curvature_loss(output,default,mask,quality):
    raw=output[1][mask].float();d=default[mask][:,1]
    logwidth=knots(raw).diff(dim=-1).clamp_min(1e-8).log()
    second=logwidth[:,2:]-2*logwidth[:,1:-1]+logwidth[:,:-2]
    roughness=second.square().mean(-1).masked_fill(d,0)
    weighted=(quality[mask].detach().float()*roughness).mean()
    return weighted,dict(numeric_logspacing_curvature=float(roughness[~d].detach().mean()) if bool((~d).any()) else None,requires_grad=weighted.requires_grad)
