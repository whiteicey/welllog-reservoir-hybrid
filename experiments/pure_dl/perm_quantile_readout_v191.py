"""SOURCE-referenced IQF32; unchanged128 trunk and32 output parameters."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_joint_density_v136 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator as ParentEvaluator
from compact_five_stage_v100 import CompactReadout
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/perm_quantile_readout_v191_20261001'
OUT=ROOT/'outputs/pure_dl_perm_quantile_readout_v191_fold5'
STAGES={'iqf':('PERM',256,True,0.,3e-4)}

def mixture_cdf(raw,points):
    sd=F.softplus(raw[:,2])+.025
    z=(points[...,None]-raw[:,1,None,:])/(sd[:,None,:]*math.sqrt(2))
    return (.5*(1+torch.erf(z))*raw[:,0].softmax(-1)[:,None,:]).sum(-1)

@torch.no_grad()
def reference_nodes(raw,edges,taus):
    pieces=[]
    for start in range(0,len(raw),1024):
        r=raw[start:start+1024].float();sd=F.softplus(r[:,2])+.025
        low=torch.minimum((r[:,1]-8*sd).amin(-1),edges[0]-1e-3)
        high=torch.maximum((r[:,1]+8*sd).amax(-1),edges[-1]+1e-3)
        grid=torch.cat((low[:,None],edges[None,:].expand(len(r),-1),high[:,None]),1)
        cdf=mixture_cdf(r,grid)
        index=torch.searchsorted(cdf.contiguous(),taus[None,:].expand(len(r),-1).contiguous()).clamp(1,grid.shape[1]-1)
        left=grid.gather(1,index-1);right=grid.gather(1,index)
        for _ in range(4):
            mid=(left+right)*.5;below=mixture_cdf(r,mid)<taus[None,:]
            left=torch.where(below,mid,left);right=torch.where(below,right,mid)
        q=(left+right)*.5
        delta=(q[:,1:]-q[:,:-1]).clamp_min(.002)
        pieces.append(torch.cat((q[:,:1],q[:,:1]+delta.cumsum(-1)),-1))
    return torch.cat(pieces)

def iqf_cdf(q,z,taus):
    if z.ndim==1:z=z[None,:].expand(len(q),-1)
    index=((z[...,None]>=q[:,None,:]).sum(-1)-1).clamp(0,30)
    lo=q.gather(1,index);hi=q.gather(1,index+1)
    tlo=taus[index];slope=(hi-lo)/(taus[index+1]-tlo)
    middle=tlo+(z-lo)/slope
    bleft=taus[0]*(q[:,1:2]-q[:,:1])/(taus[1]-taus[0])
    bright=(1-taus[-1])*(q[:,-1:]-q[:,-2:-1])/(taus[-1]-taus[-2])
    left=taus[0]*torch.exp(((z-q[:,:1])/bleft).clamp_max(0))
    right=1-(1-taus[-1])*torch.exp(((q[:,-1:]-z)/bright).clamp_max(0))
    return torch.where(z<q[:,:1],left,torch.where(z>=q[:,-1:],right,middle)).clamp(0,1)

def iqf_logpdf(q,y,taus):
    index=((y[:,None]>=q).sum(-1)-1).clamp(0,30)
    lo=q.gather(1,index[:,None]).squeeze(1);hi=q.gather(1,(index+1)[:,None]).squeeze(1)
    slope=(hi-lo)/(taus[index+1]-taus[index])
    bleft=taus[0]*(q[:,1]-q[:,0])/(taus[1]-taus[0])
    bright=(1-taus[-1])*(q[:,-1]-q[:,-2])/(taus[-1]-taus[-2])
    left=taus[0].log()+(y-q[:,0])/bleft-bleft.log()
    right=(1-taus[-1]).log()+(q[:,-1]-y)/bright-bright.log()
    return torch.where(y<q[:,0],left,torch.where(y>=q[:,-1],right,-slope.log()))

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('rps_edges',torch.as_tensor(master['density_edges'][1][1:-1],dtype=torch.float32))
        self.register_buffer('taus',torch.linspace(.02,.98,32))
        self.register_buffer('physical_centers',torch.as_tensor(master['centers'][1],dtype=torch.float32))
        self.register_buffer('target_median',torch.tensor(float(master['regression_median'][1])))
        self.register_buffer('target_scale',torch.tensor(float(master['regression_scale'][1])))
        self._frozen_step_cache=None
    def clear_source_cache(self):self._frozen_step_cache=None
    def train(self,mode=True):
        super().train(mode);self.clear_source_cache();return self
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(readout='32monotoneIQFquantiles',Gaussian_output_family_replaced=True,
                    same_initial_predictive_function=False,source_reference='frozen original conditional Gaussian mixture, no labels',
                    reference_inversion='original255 finite grid plus dynamic8sigma bounds,4 local bisections',
                    fixed_taus=[.02,.98,32],minimum_increment=.001,reference_minimum_increment=.002,
                    new_quantile_residual_bound='q0 plus tanh; increments exp(tanh), positive epsilon',
                    tail_slopes='first/last interior slopes, exponential standardized tails',
                    source_features_cached_only_within_same_SAM_step=True,cache_not_persisted=True,
                    default_gate_frozen=True,conditionalRPS1=True,utility_weight=.1,SAM_rho=.05,
                    no_trained_v174_weights_loaded=True,
                    training_NLL='standardizedlog10 target space; physicalJacobian independent of parameters')
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if self.training and self._frozen_step_cache is not None:
            gate,reference,features=self._frozen_step_cache
        else:
            self.local_captured.clear()
            with torch.no_grad():
                gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
                features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1).detach()
                self.local_captured.clear()
                reference=reference_nodes(raw[:,:,0].reshape(-1,3,16),self.rps_edges,self.taus).reshape(*x.shape[:2],32)
            if self.training:self._frozen_step_cache=(gate.detach(),reference.detach(),features)
        correction=self.context_head(features,valid)
        q0=reference[...,:1]+correction[...,:1].tanh()
        increments=.001+(reference[...,1:]-reference[...,:-1]-.001)*correction[...,1:].tanh().exp()
        nodes=torch.cat((q0,q0+increments.cumsum(-1)),-1)
        return gate,nodes.unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)

def extra_score(g,q,y,d,edges,taus,centers,median,scale):
    cdf=iqf_cdf(q,edges,taus)
    rps=(cdf-(y[:,None]<=edges[None,:]).float()).square().mean(-1).masked_fill(d,0)
    mass=torch.cat((torch.zeros_like(cdf[:,:1]),cdf,torch.ones_like(cdf[:,:1])),-1).diff(dim=-1).clamp_min(0)
    mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
    target_log=torch.where(d,torch.full_like(y,-2.),y*scale+median)
    reward=(1-(centers.clamp_min(1e-12).log10()[None,:]-target_log[:,None]).abs()).clamp_min(0)
    atom=(1-(-2.-target_log).abs()).clamp_min(0);p=g.sigmoid()
    utility=1-p*atom-(1-p)*(mass*reward).sum(-1)
    return rps+.1*utility

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    gate,nodes=output;g=gate[mask][:,0].float();q=nodes[mask][:,0].float()
    d=default[mask][:,1];y=truth[mask][:,1].float().masked_fill(d,0);weight=quality[mask].detach().float()
    nll=-iqf_logpdf(q,y,model.taus).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');parts=[]
    for start in range(0,len(g),1024):
        take=slice(start,start+1024)
        parts.append(checkpoint(extra_score,g[take],q[take],y[take],d[take],model.rps_edges,model.taus,
                               model.physical_centers,model.target_median,model.target_scale,
                               use_reentrant=False,preserve_rng_state=False))
    extra=torch.cat(parts)
    return (weight*(bce+nll+extra)).mean()/3,(weight*bce).mean(),(weight*(nll+extra)).mean()

class Evaluator(ParentEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        d=self.data;model.eval();lo=self.length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for start in range(0,len(self.centers),self.batch):
            centers=torch.tensor(self.centers[start:start+self.batch],device='cuda');anchors=torch.tensor(self.anchors[start:start+self.batch],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],length=self.length,anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                gate,nodes=model(d['x'][ix],relative,valid,absolute_depth=d['absolute'][ix],well_inputs=d['well_bank'][d['row_well'][anchors]])
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda');g=gate[:,lo:hi][take][:,0].float().sigmoid();q=nodes[:,lo:hi][take][:,0].float()
            cdf=iqf_cdf(q,self.edges[1:-1],model.taus)
            mass=torch.cat((torch.zeros_like(cdf[:,:1]),cdf,torch.ones_like(cdf[:,:1])),-1).diff(dim=-1).clamp_min(0)
            mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
            joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,1]=result
        return result,float(base.metric(y,p)['PERM'])
