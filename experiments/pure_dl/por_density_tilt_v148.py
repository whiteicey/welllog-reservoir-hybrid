"""Proper conditional residual log-density on unchanged original support intervals."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from por_joint_density_width_v140 import ROOT,SOURCE,PARENT,Network as ParentNetwork,ContextHead as ParentHead,load_data
from compact_por_mixed_family_v113 import choose,channel_objective as base_loss,Evaluator as ParentEvaluator
from compact_por_cauchy_v112 import IQR_SCALE
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/por_density_tilt_v148_20261001'
OUT=ROOT/'outputs/pure_dl_por_density_tilt_v148_fold5'
STAGES={'tilt':('POR',256,False,0.,3e-4)}

def base_probability(weight,location,sd,edges):
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges))
    delta=safe[None,:,None]-location[:,None,:];sd=sd[:,None,:]
    phi=torch.atan(delta/sd)
    t3=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
    cauchy=.5+torch.atan(delta/(sd*IQR_SCALE))/math.pi
    cdf=choose('t3_cauchy',t3,cauchy)
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf).clamp(0,1)
    mass=(cdf.diff(dim=1)*weight[:,None,:]).sum(-1).clamp_min(0)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class Network(ParentNetwork):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.context_head.output=nn.Conv1d(256,288,1)
        nn.init.zeros_(self.context_head.output.weight);nn.init.zeros_(self.context_head.output.bias)
        self.register_buffer('tilt_edges',torch.as_tensor(master['density_edges'][0],dtype=torch.float32))
        assert len(self.tilt_edges)==257

    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(residual_log_density=True,residual_limit=.25,unchanged_density_intervals=256,
            proper_piecewise_density_normalization=True,scale_parameters_frozen=True,
            training_label_bin_convention='right=True',extra_output_parameters=65792,
            no_trained_v140_head_loaded=True)
        return info

    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        # Call the same frozen v113 predictor; bypass the parent's32-output correction.
        from compact_por_mixed_family_v113 import Network as FrozenPredictor
        with torch.no_grad():gate,raw=FrozenPredictor.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid)
        own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
        locations=own[:,:,1]+reference*torch.tanh(u[...,:16]/reference)
        weights=own[:,:,0]+torch.tanh(u[...,16:32])
        corrected=torch.stack((weights,locations,own[:,:,2]),2).unsqueeze(2)
        tilt=.25*torch.tanh(u[...,32:]/.25)
        return gate,corrected,tilt

def make_model(stage,master,metadata):return Network(master,stage)

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    assert target_index==0
    gate,raw,tilt=output
    original=base_loss((gate,raw),truth,default,mask,quality,0,'t3_cauchy')
    use=mask&~default[...,0];r=raw[use][:,0].float();t=tilt[use].float()
    if not bool(use.any()):return original
    cp=base_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,model.tilt_edges)
    normalizer=(cp*t.exp()).sum(-1).clamp_min(1e-12)
    bin_index=torch.bucketize(truth[...,0][use].float().contiguous(),model.tilt_edges[1:-1],right=True)
    correction=normalizer.log()-t.gather(1,bin_index[:,None]).squeeze(1)
    q=quality[use].detach().float()
    # p_new(y)=p_base(y)*exp(tilt[bin(y)])/Z; exact likelihood correction.
    extra=(q*correction).sum()/mask.sum().clamp_min(1)/3
    return original[0]+extra,original[1],original[2]+3*extra

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length,'t3_cauchy')

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
            cp=base_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            cp=cp*t.exp();cp=cp/cp.sum(-1,keepdim=True).clamp_min(1e-12)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,0]=result
        return result,float(base.metric(y,p)['POR'])
