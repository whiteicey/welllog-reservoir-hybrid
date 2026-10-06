"""Conditional monotone Bernstein warp of the mixture CDF; exact density derivative."""
import math
import numpy as np
from multiscale_sequence import window_indices
import train_invariant as base
import torch
from torch import nn
from torch.nn import functional as F
from por_joint_density_width_v140 import ROOT,SOURCE,PARENT,Network as Parent,load_data
from compact_por_mixed_family_v113 import choose,channel_objective as base_loss,Evaluator as ParentEvaluator,probability
from compact_por_cauchy_v112 import IQR_SCALE

STUDY=ROOT/'results/deep_learning/por_smooth_cdf_readout_v150_20261001'
OUT=ROOT/'outputs/pure_dl_por_smooth_cdf_readout_v150_fold5'
STAGES={'bernstein8':('POR',256,False,0.,3e-4)}
K=8

def mixture_cdf(values,raw):
    location=raw[:,1];sd=F.softplus(raw[:,2])+.025
    if values.ndim==1:delta=values[:,None]-location
    else:delta=values[...,None]-location[:,None,:];sd=sd[:,None,:]
    phi=torch.atan(delta/sd)
    cdf=choose('t3_cauchy',.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi,
               .5+torch.atan(delta/(sd*IQR_SCALE))/math.pi).clamp(0,1)
    weights=raw[:,0].softmax(-1)
    if values.ndim>1:weights=weights[:,None,:]
    return (cdf*weights).sum(-1).clamp(0,1)

def warp_density(u,logits):
    # Beta(i+1,K-i) densities partition unity when averaged uniformly.
    basis=torch.stack([K*math.comb(K-1,i)*u.pow(i)*(1-u).pow(K-1-i) for i in range(K)],-1)
    return (1+((logits.softmax(-1)-1/K)*basis).sum(-1)).clamp_min(1e-8)

def warp_cdf_residual(u,logits):
    # Beta CDF = binomial tail; finite polynomials on [0,1], no quantile inversion.
    terms=torch.stack([math.comb(K,j)*u.pow(j)*(1-u).pow(K-j) for j in range(K+1)],-1)
    tail=terms.flip(-1).cumsum(-1).flip(-1)[...,1:]
    return ((logits.softmax(-1)-1/K)[:,None,:]*tail).sum(-1)

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.context_head.output=nn.Conv1d(256,32+K,1)
        nn.init.zeros_(self.context_head.output.weight);nn.init.zeros_(self.context_head.output.bias)
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),old_predictor_fully_frozen=True,same_initial_predictive_function=True,
            hidden_width=256,warp_basis_count=K,bounded_warp_logits=.5,monotone_conditional_CDF=True,
            derivative_of_CDF_used_in_training_density=True,gate_scales_families_frozen=True,
            no_action_grid_or_interval_resolution_change=True,no_trained_v140_head_loaded=True,
            executed_training_stages_including_source=5)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        from compact_por_mixed_family_v113 import Network as Frozen
        self.local_captured.clear()
        with torch.no_grad():gate,raw=Frozen.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid);own=raw[:,:,0]
        reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:32]),
            own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2).unsqueeze(2)
        return gate,corrected,.5*torch.tanh(u[...,32:]/.5)

def make_model(stage,master,metadata):return Network(master,stage)

def channel_objective(output,truth,default,mask,quality,target_index,mode):
    gate,raw,logits=output;original=base_loss((gate,raw),truth,default,mask,quality,0,'t3_cauchy')
    use=mask&~default[...,0]
    if not bool(use.any()):return original
    r=raw[use][:,0].float();u=mixture_cdf(truth[...,0][use].float(),r)
    correction=-warp_density(u,logits[use].float()).log()
    extra=(quality[use].detach().float()*correction).sum()/mask.sum().clamp_min(1)/3
    return original[0]+extra,original[1],original[2]+3*extra

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length,'t3_cauchy')
    # Evaluation loop reused from the frozen original-grid evaluator in v148.
    @torch.no_grad()
    def corrected_probability(self,raw,logits):
        cp=probability('t3_cauchy',0,raw[:,0].softmax(-1),raw[:,1],F.softplus(raw[:,2])+.025,self.edges)
        safe=torch.where(torch.isfinite(self.edges),self.edges,torch.zeros_like(self.edges))
        u=mixture_cdf(safe[None,:].expand(len(raw),-1),raw)
        u=torch.where(torch.isneginf(self.edges)[None,:],torch.zeros_like(u),u)
        u=torch.where(torch.isposinf(self.edges)[None,:],torch.ones_like(u),u)
        cp=(cp+warp_cdf_residual(u,logits).diff(dim=1)).clamp_min(0)
        return cp/cp.sum(-1,keepdim=True).clamp_min(1e-12)
    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),self.batch):
            cc=torch.tensor(self.centers[begin:begin+self.batch],device='cuda');anchor=torch.tensor(self.anchors[begin:begin+self.batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=self.length,anchors=anchor)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchor,None])/10)
            context=d['well_bank'][d['row_well'][anchor]]
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw,tilt=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],well_inputs=context)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float();t=tilt[:,lo:hi][take].float()
            cp=self.corrected_probability(r,t)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,0]=result
        return result,float(base.metric(y,p)['POR'])
