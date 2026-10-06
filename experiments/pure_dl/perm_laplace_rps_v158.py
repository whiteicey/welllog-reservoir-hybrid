"""PERM-specific Laplace mixture readout with proper likelihood and joint RPS."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Parent,load_data
from train_stage_fullcv_v97 import Evaluator as BaseEvaluator
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/perm_laplace_rps_v158_20261001'
OUT=ROOT/'outputs/pure_dl_perm_laplace_rps_v158_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

def component_cdf(delta,scale):
    positive=1-.5*torch.exp(-delta.clamp_min(0)/scale)
    negative=.5*torch.exp(delta.clamp_max(0)/scale)
    return torch.where(delta>=0,positive,negative)

def probability(weight,location,sd,edges):
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges))
    cdf=component_cdf(safe[None,:,None]-location[:,None,:],sd[:,None,:]/math.sqrt(2))
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    mass=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class Network(Parent):
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(family_mode='Laplace',scale_conversion='reference_sd/sqrt(2)',initial_function_matches_Gaussian=False,same_initial_predictive_function=False,
            same_raw_initialization_and_trainable_capacity=True,initial_family_change_not_training_gain=True,
            trained_v154_weights_not_loaded=True)
        return info
def make_model(stage,master,metadata):return Network(master,stage)

def ranked_loss(g,r,y,d,edges,default_z):
    sd=F.softplus(r[:,2])+.025
    cdf=component_cdf(edges[None,:,None]-r[:,1,None,:],sd[:,None,:]/math.sqrt(2))
    conditional=(cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
    gate=g.sigmoid()[:,None];joint=gate*(default_z<=edges).float()[None,:]+(1-gate)*conditional
    return (joint-(torch.where(d,default_z,y)[:,None]<=edges[None,:]).float()).square().mean(-1)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    d=default[mask][:,1];y=truth[mask][:,1].float();q=quality[mask].detach().float()
    scale=(F.softplus(r[:,2])+.025)/math.sqrt(2)
    logp=-(y.masked_fill(d,0)[:,None]-r[:,1]).abs()/scale-(2*scale).log()
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');parts=[]
    for begin in range(0,len(g),1024):
        take=slice(begin,begin+1024)
        parts.append(checkpoint(ranked_loss,g[take],r[take],y[take],d[take],model.rps_edges,model.default_z,use_reentrant=False,preserve_rng_state=False))
    rps=torch.cat(parts)
    return (q*(bce+nll+rps)).mean()/3,(q*bce).mean(),(q*(nll+rps)).mean()

class Evaluator(BaseEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length)
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
            cp=probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        # Calculate this target score without treating another channel's labels
        # as its loss or using a fabricated full-fold TOTAL for stage selection.
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,self.c]=result
        score=base.metric(y,p)[base.TS[self.c]]
        return result,float(score)
