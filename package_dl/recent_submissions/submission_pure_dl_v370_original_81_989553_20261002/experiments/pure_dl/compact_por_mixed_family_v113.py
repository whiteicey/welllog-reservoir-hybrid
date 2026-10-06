"""Two fixed expert-family assignments, jointly trained compact POR readout."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data
from compact_mean_scope_v101 import Network as MeanNetwork
from compact_por_cauchy_v112 import IQR_SCALE
from train_stage_fullcv_v97 import Evaluator as BaseEvaluator
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/compact_por_mixed_family_v113_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5'
STAGES={name:('POR',256,False,0.,3e-5) for name in ('t3_cauchy','cauchy_t3')}

class Network(MeanNetwork):
    def __init__(self,master,mode):
        super().__init__(master);self.family_mode=mode
    def initialize_priors(self,data):
        return dict(families=self.family_mode,component_identity='first8 expert0, next8 expert1',
            fixed_assignment=True,component_IQR_matching=True,cauchy_scale_conversion=IQR_SCALE,
            whole_mixture_has_no_finite_mean_or_variance=True,initial_raw_parameters_identical=True,
            initial_predictive_functions_differ=True,no_external_models_or_voting=True)

def load_data(stage):return original_data('por')
def make_model(stage,master,metadata):return Network(master,stage)

def choose(mode,t3,cauchy):
    assert t3.shape[-1]==cauchy.shape[-1]==16
    if mode=='t3_cauchy':return torch.cat((t3[...,:8],cauchy[...,8:]),-1)
    if mode=='cauchy_t3':return torch.cat((cauchy[...,:8],t3[...,8:]),-1)
    raise ValueError(mode)

def channel_objective(output,truth,default,mask,quality,target_index,mode):
    assert target_index==0
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float();d=default[mask][:,0]
    y=truth[mask][:,0].masked_fill(d,0);location=r[:,1];sd=F.softplus(r[:,2])+.025
    scale=sd/math.sqrt(3)
    t3=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((y[:,None]-location)/scale).square()/3)
    gamma=sd*IQR_SCALE
    cauchy=-math.log(math.pi)-gamma.log()-torch.log1p(((y[:,None]-location)/gamma).square())
    logp=choose(mode,t3,cauchy)
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');w=quality[mask].detach().float()
    return (w*(bce+nll)).mean()/3,(w*bce).mean(),(w*nll).mean()

def probability(mode,index,weight,location,reference_sd,edges):
    assert index==0
    delta=edges[None,:,None]-location.float()[:,None,:];sd=reference_sd.float()[:,None,:]
    phi=torch.atan(delta/sd)
    t3=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
    cauchy=.5+torch.atan(delta/(sd*IQR_SCALE))/math.pi
    cdf=choose(mode,t3,cauchy)
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf).clamp(0,1)
    mass=(cdf.diff(dim=1)*weight.float()[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class Evaluator(BaseEvaluator):
    def __init__(self,data,master,c,length,mode):
        super().__init__(data,master,c,length);self.family_mode=mode
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
            cp=probability(self.family_mode,self.c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        # Calculate this target score without treating another channel's labels
        # as its loss or using a fabricated full-fold TOTAL for stage selection.
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,self.c]=result
        score=base.metric(y,p)[base.TS[self.c]]
        return result,float(score)
