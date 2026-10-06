"""Own SW Student-t3 likelihood and utility on unchanged random local backbone."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from compact_frequency_v104 import ROOT,SOURCE,Network as Parent,load_data
from sw_head_utility_v80r1 import DecisionObjective as BaseObjective
from train_stage_fullcv_v97 import Evaluator as BaseEvaluator
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/compact_sw_student_v139_20261001'
OUT=ROOT/'outputs/pure_dl_compact_sw_student_v139_fold5'
STAGES={'student3':('SW',1024,True,1.,3e-4)}

def conditional_probability(weight,mean,std,edges):
    # Student-t df3 scale=std/sqrt(3); std retains its finite SD meaning.
    safe=torch.where(torch.isfinite(edges),edges,torch.zeros_like(edges))
    phi=torch.atan((safe[None,:,None]-mean[:,None,:])/std[:,None,:])
    cdf=.5+(phi+torch.sin(phi)*torch.cos(phi))/math.pi
    cdf=torch.where(torch.isneginf(edges)[None,:,None],torch.zeros_like(cdf),cdf)
    cdf=torch.where(torch.isposinf(edges)[None,:,None],torch.ones_like(cdf),cdf)
    mass=(cdf.diff(dim=1).clamp_min(0)*weight[:,None,:]).sum(-1)
    return mass/mass.sum(-1,keepdim=True).clamp_min(1e-12)

class Network(Parent):
    def __init__(self,metadata,mode):super().__init__(metadata,'local')
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(SW_family='Student-t3',component_sd_is_finite=True,student_scale='sd/sqrt(3)',
            architecture_and_parameter_count_same_as_v104=True,all_SW_weights_random=True,
            no_old_SW_weights_loaded=True,initial_raw_same_but_predictive_family_different=True,
            own_SW_likelihood_utility_and_inference_CDF_synchronized=True,original_actions_and_density_edges=True)
        return info

def make_model(stage,master,metadata):return Network(metadata,stage)

class DecisionObjective(BaseObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'joint');self.alpha=1.
    def forward(self,output,labels,default,mask,quality):
        gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        mu=r[:,1];std=F.softplus(r[:,2])+.025;scale=std/math.sqrt(3)
        y=z.masked_fill(d,0)
        logp=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((y[:,None]-mu)/scale).square()/3)
        nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
        bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
        likelihood=(q*(bce+nll)).mean()/3
        cp=conditional_probability(r[:,0].softmax(-1),mu,std,self.edges)
        pg=g.sigmoid();joint=torch.cat((pg[:,None],(1-pg[:,None])*cp),-1)
        policy=((joint@self.utilities)/self.temperature).softmax(-1)
        physical_y=torch.exp(z*self.target_scale+self.target_median)
        physical_y=torch.where(d,torch.full_like(physical_y,99.9),physical_y)
        true_utility=(1-(self.actions[None]-physical_y[:,None]).abs()/(.05*(physical_y.abs()[:,None]+1e-5))).clamp_min(0)
        decision=(q*(1-(policy*true_utility).sum(-1))).mean()/3
        return likelihood+decision,likelihood,decision

class Evaluator(BaseEvaluator):
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
            cp=conditional_probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        # Calculate this target score without treating another channel's labels
        # as its loss or using a fabricated full-fold TOTAL for stage selection.
        y=d['truth'][d['validation']].copy();p=y.copy();p[:,self.c]=result
        score=base.metric(y,p)[base.TS[self.c]]
        return result,float(score)

