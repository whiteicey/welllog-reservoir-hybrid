"""Same PERM RQS predictor, point versus finite-resolution observation loss."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from perm_gradient_v68 import ROOT,Network as Parent,Evaluator,load_data,PARENT_PRED
from quantile_density_v48 import cdf_logpdf

STUDY=ROOT/'results/deep_learning/perm_observation_v72_20260929'
PREVIOUS=ROOT/'results/deep_learning/sw_depth_attention_v71_20260929'
MODES={m:dict(target='PERM',c=1,length=1024) for m in ('point','interval')}
NODES=(-.906179845938664,-.538469310105683,0.,.538469310105683,.906179845938664)
WEIGHTS=(.236926885056189,.478628670499366,.568888888888889,.478628670499366,.236926885056189)

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('sum',metadata)
        self.study_mode=mode

def channel_objective(output,truth,default,mask,quality,target_index,mode):
    gate,raw=output;d=default[mask][:,target_index]
    g=gate[mask][:,0].float();r=raw[mask].float()
    y=truth[mask][:,target_index].masked_fill(d,0)
    # Fixed modelling assumption, not inferred measurement accuracy or score tolerance.
    halfwidth=.05 if mode=='interval' else 0.
    nodes=r.new_tensor(NODES);weights=r.new_tensor(WEIGHTS)/2
    _,logpdf=cdf_logpdf(r,y[:,None]+halfwidth*nodes[None,:],'rqs')
    nll=-torch.logsumexp(logpdf+weights.log()[None,:],-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    w=quality[mask].detach().float()
    return (w*(bce+nll)).mean()/3,(w*bce).mean(),(w*nll).mean()
