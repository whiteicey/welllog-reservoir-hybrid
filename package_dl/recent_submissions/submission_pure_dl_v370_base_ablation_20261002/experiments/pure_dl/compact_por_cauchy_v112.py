"""Compact POR location training with a fixed Cauchy predictive family."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import t
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_mean_scope_v101 import Network as MeanNetwork
from train_stage_fullcv_v97 import Evaluator as BaseEvaluator
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/compact_por_cauchy_v112_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_cauchy_v112_fold5'
STAGES={'cauchy':('POR',256,False,0.,3e-5)}
# Student-t3 scale=reference_sd/sqrt3; its 75% quantile is the Cauchy scale
# needed for identical component IQR. Cauchy has no finite mean or variance.
IQR_SCALE=float(t.ppf(.75,3)/math.sqrt(3))

class Network(MeanNetwork):
    def initialize_priors(self,data):
        return dict(family='Cauchy',location_not_expectation=True,finite_variance=False,
                    scale_conversion=IQR_SCALE,initial_component_IQR_matches_source_t3=True,
                    source_raw_parameters_identical=True,initial_predictive_function_differs=True)

def load_data(stage):return original_data('por')
def make_model(stage,master,metadata):return Network(master)

def channel_objective(output,truth,default,mask,quality,target_index):
    assert target_index==0
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float();d=default[mask][:,0]
    y=truth[mask][:,0].masked_fill(d,0);location=r[:,1]
    gamma=(F.softplus(r[:,2])+.025)*IQR_SCALE
    logp=-math.log(math.pi)-gamma.log()-torch.log1p(((y[:,None]-location)/gamma).square())
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');w=quality[mask].detach().float()
    return (w*(bce+nll)).mean()/3,(w*bce).mean(),(w*nll).mean()

def probability(index,weight,location,reference_sd,edges):
    assert index==0
    gamma=reference_sd.float()*IQR_SCALE
    cdf=.5+torch.atan((edges[None,:,None]-location.float()[:,None,:])/gamma[:,None,:])/math.pi
    mass=(cdf.diff(dim=1)*weight.float()[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class Evaluator(BaseEvaluator):
    # The original evaluation loop is copied below with only its probability
    # function resolved to this module's Cauchy family.
    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),self.batch):
            cc=torch.tensor(self.centers[begin:begin+self.batch],device='cuda');anchor=torch.tensor(self.anchors[begin:begin+self.batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=self.length,anchors=anchor)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchor,None])/10)
            context=d['well_bank'][d['row_well'][anchor]]
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float()
            cp=probability(self.c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        # Calculate this target score without treating another channel's labels
        # as its loss or using a fabricated full-fold TOTAL for stage selection.
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,self.c]=result
        score=base.metric(y,p)[base.TS[self.c]]
        return result,float(score)
