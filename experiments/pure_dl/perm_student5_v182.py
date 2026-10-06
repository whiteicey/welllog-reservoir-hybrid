"""Log-space moment-matched Student-t5 PERM likelihood/CDF/utility readout."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from perm_conditional_cdf_v174 import ROOT,SOURCE,PARENT,STAGES,Network as Parent,load_data,Evaluator as PreviousEvaluator
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/perm_student5_v182_20261001'
OUT=ROOT/'outputs/pure_dl_perm_student5_v182_fold5'
class Network(Parent):
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(family_mode='StudentT5',same_initial_predictive_function=False,log_component_means_variances_matched=True,
                    t_scale_from_standard_sd=math.sqrt(3/5),finite_standardized_log_moments=True,
                    physical_PERM_moments_not_asserted=True,training_and_decoder_CDF_same=True,no_trained_v174_weights_loaded=True)
        return info
def mixed_cdf(r,edges):
    sd=F.softplus(r[:,2])+.025;scale=sd*math.sqrt(3/5)
    u=torch.atan((edges[None,:,None]-r[:,1,None,:])/(scale[:,None,:]*math.sqrt(5)))
    cdf=(.5+u/math.pi+2*torch.sin(2*u)/(3*math.pi)+torch.sin(4*u)/(12*math.pi)).clamp(0,1)
    return (cdf*r[:,0].softmax(-1)[:,None,:]).sum(-1)
def ranked_utility(g,r,y,d,edges,centers,median,scale):
    conditional=mixed_cdf(r,edges)
    rps=(conditional-(y[:,None]<=edges[None,:]).float()).square().mean(-1).masked_fill(d,0)
    full=torch.cat((torch.zeros_like(conditional[:,:1]),conditional,torch.ones_like(conditional[:,:1])),-1)
    mass=full.diff(dim=-1).clamp_min(0);mass=mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)
    target=torch.where(d,torch.full_like(y,-2.),y*scale+median)
    utility=(1-(centers.clamp_min(1e-12).log10()[None,:]-target[:,None]).abs()).clamp_min(0)
    atom=(1-(-2.-target).abs()).clamp_min(0);gate=g.sigmoid()
    return rps+.1*(1-gate*atom-(1-gate)*(mass*utility).sum(-1))
def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==1
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float();d=default[mask][:,1]
    y=truth[mask][:,1].float();q=quality[mask].detach().float();scale=(F.softplus(r[:,2])+.025)*math.sqrt(3/5)
    safe=y.masked_fill(d,0)
    logp=math.lgamma(3)-math.lgamma(2.5)-.5*math.log(5*math.pi)-scale.log()-3*torch.log1p(((safe[:,None]-r[:,1])/scale).square()/5)
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');extra=[]
    for start in range(0,len(g),1024):
        take=slice(start,start+1024)
        extra.append(checkpoint(ranked_utility,g[take],r[take],y[take],d[take],model.rps_edges,model.physical_centers,model.target_median,model.target_scale,use_reentrant=False,preserve_rng_state=False))
    add=torch.cat(extra)
    return (q*(bce+nll+add)).mean()/3,(q*bce).mean(),(q*(nll+add)).mean()
def make_model(stage,master,metadata):return Network(master,stage)

class Evaluator(PreviousEvaluator):
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
            cdf=mixed_cdf(r,self.edges);cp=cdf.diff(dim=-1).clamp_min(0);cp=cp/cp.sum(-1,keepdim=True).clamp_min(1e-12)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        # Calculate this target score without treating another channel's labels
        # as its loss or using a fabricated full-fold TOTAL for stage selection.
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,self.c]=result
        score=base.metric(y,p)[base.TS[self.c]]
        return result,float(score)

