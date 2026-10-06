"""Adaptive categorical density in frozen SOURCE probability coordinates."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_joint_density_v136 import ROOT,SOURCE,PARENT,Network as Parent,load_data,Evaluator as ParentEvaluator
from compact_five_stage_v100 import CompactReadout
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/perm_adaptive_mass_v193_20261001'
OUT=ROOT/'outputs/pure_dl_perm_adaptive_mass_v193_fold5'
STAGES={'adaptive':('PERM',256,True,0.,3e-4)}

def source_cdf(raw,z):
    if z.ndim==1:z=z[None,:].expand(len(raw),-1)
    sd=(F.softplus(raw[:,2])+.025)[:,None,:]
    cdf=.5*(1+torch.erf((z[...,None]-raw[:,1,None,:])/(sd*math.sqrt(2))))
    return (cdf*raw[:,0].softmax(-1)[:,None,:]).sum(-1)

def transport(u,correction):
    widths=.001+.984*correction[:,:16].tanh().softmax(-1)
    masses=.001+.984*correction[:,16:].tanh().softmax(-1)
    zero=torch.zeros_like(widths[:,:1])
    edges=torch.cat((zero,widths.cumsum(-1)),-1)
    cumulative=torch.cat((zero,masses.cumsum(-1)),-1)
    index=((u[...,None]>=edges[:,None,1:]).sum(-1)).clamp_max(15)
    left=edges.gather(1,index);width=widths.gather(1,index)
    mass=masses.gather(1,index);before=cumulative.gather(1,index)
    value=before+mass*(u-left)/width
    return value.clamp(0,1),(mass/width).log()

class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.register_buffer('rps_edges',torch.as_tensor(master['density_edges'][1][1:-1],dtype=torch.float32))
        self.register_buffer('physical_centers',torch.as_tensor(master['centers'][1],dtype=torch.float32))
        self.register_buffer('target_median',torch.tensor(float(master['regression_median'][1])))
        self.register_buffer('target_scale',torch.tensor(float(master['regression_scale'][1])))
        self._frozen_step_cache=None
    def clear_source_cache(self):self._frozen_step_cache=None
    def train(self,mode=True):
        super().train(mode);self.clear_source_cache();return self
    def groups(self):return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='private_PERM_adaptive_mass_head')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(readout='16adaptive intervals in SOURCE conditionalCDF coordinate',
                    new_component_logweight_correction=False,own_new_locations_only=False,
                    SOURCE_means_scales_weights_all_frozen=True,widths_and_masses_each_sum_one=True,
                    interval_minimum=.001,mass_minimum=.001,logits_bound=1.,
                    zero_output_uniform_width_and_mass=True,same_initial_predictive_function='algebraic identity; verify actual score',
                    conditionalRPS1=True,utility_weight=.1,SAM_rho=.05,
                    adaptive_target_space_bins=False,not_exact_AdaCat_reproduction=True,
                    source_features_cached_only_within_same_SAM_step=True,cache_not_persisted=True,
                    training_NLL='SOURCE Gaussian standardizedlog10 density plus exact transport Jacobian',
                    no_trained_v174_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if self.training and self._frozen_step_cache is not None:
            gate,raw,features=self._frozen_step_cache
        else:
            self.local_captured.clear()
            with torch.no_grad():
                gate,raw=CompactReadout.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
                features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1).detach()
                self.local_captured.clear()
            if self.training:self._frozen_step_cache=(gate.detach(),raw.detach(),features)
        return gate,(raw,self.context_head(features,valid))

def make_model(stage,master,metadata):return Network(master,stage)

def extra_score(g,r,c,y,d,edges,centers,median,scale):
    cdf,_=transport(source_cdf(r,edges),c)
    rps=(cdf-(y[:,None]<=edges[None,:]).float()).square().mean(-1).masked_fill(d,0)
    mass=torch.cat((torch.zeros_like(cdf[:,:1]),cdf,torch.ones_like(cdf[:,:1])),-1).diff(dim=-1).clamp_min(0)
    mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
    target_log=torch.where(d,torch.full_like(y,-2.),y*scale+median)
    reward=(1-(centers.clamp_min(1e-12).log10()[None,:]-target_log[:,None]).abs()).clamp_min(0)
    atom=(1-(-2.-target_log).abs()).clamp_min(0);p=g.sigmoid()
    return rps+.1*(1-p*atom-(1-p)*(mass*reward).sum(-1))

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    gate,(raw,correction)=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float();c=correction[mask].float()
    d=default[mask][:,1];y=truth[mask][:,1].float().masked_fill(d,0);weight=quality[mask].detach().float()
    sd=F.softplus(r[:,2])+.025
    logpdf=torch.logsumexp(r[:,0].log_softmax(-1)-sd.log()-.5*((y[:,None]-r[:,1])/sd).square()-.5*math.log(2*math.pi),-1)
    _,logjac=transport(source_cdf(r,y[:,None]),c)
    nll=-(logpdf+logjac[:,0]).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');parts=[]
    for start in range(0,len(g),1024):
        take=slice(start,start+1024)
        parts.append(checkpoint(extra_score,g[take],r[take],c[take],y[take],d[take],model.rps_edges,
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
                gate,(raw,correction)=model(d['x'][ix],relative,valid,absolute_depth=d['absolute'][ix],well_inputs=d['well_bank'][d['row_well'][anchors]])
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=gate[:,lo:hi][take][:,0].float().sigmoid();r=raw[:,lo:hi][take][:,0].float();c=correction[:,lo:hi][take].float()
            cdf,_=transport(source_cdf(r,self.edges[1:-1]),c)
            mass=torch.cat((torch.zeros_like(cdf[:,:1]),cdf,torch.ones_like(cdf[:,:1])),-1).diff(dim=-1).clamp_min(0)
            mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
            joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,1]=result
        return result,float(base.metric(y,p)['PERM'])
