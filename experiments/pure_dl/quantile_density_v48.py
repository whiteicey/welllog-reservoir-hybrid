"""Conditional monotone CDFs: linear versus rational-quadratic interpolation.

RQS CDF/derivative follow Durkan et al. 2019 Appendix A equations19/22.
https://arxiv.org/html/1906.04032
Our fixed CDF heights, harmonic secants and exponential tails are explicit
adaptations; this is not a reproduction of their coupling-flow benchmark.
"""
import hashlib,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from well_conditioned_v46 import ROOT,Network as Parent,load_data,modulate
from train_scaling_v21 import Evaluator as OriginalEvaluator
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/quantile_density_v48_20260929'
PREVIOUS=ROOT/'results/deep_learning/well_conditioned_v46_20260929'
PARENT=ROOT/'outputs/pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt'
MODES={m:dict(target='SW',c=2,length=1024) for m in ('linear','rqs')}
K=32;TAIL=.005;DP=(1-2*TAIL)/K

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

def knots(raw):
    span=F.softplus(raw[:,1:2])+.1
    widths=.001+span*raw[:,2:].softmax(-1)
    first=raw[:,:1]-.5*widths.sum(-1,keepdim=True)
    return torch.cat((first,first+widths.cumsum(-1)),-1)

def cdf_logpdf(raw,values,mode):
    """raw N34; values NM; evaluates full-real-support CDF and exact density."""
    q=knots(raw.float());values=values.float();width=q[:,1:]-q[:,:-1];slope=DP/width
    derivative=torch.cat((slope[:,:1],2*slope[:,:-1]*slope[:,1:]/(slope[:,:-1]+slope[:,1:]),slope[:,-1:]),-1)
    idx=(torch.searchsorted(q.contiguous(),values.contiguous(),right=True)-1).clamp(0,K-1)
    q0=q.gather(1,idx);w=width.gather(1,idx);s=slope.gather(1,idx)
    t=((values-q0)/w).clamp(0,1);p0=TAIL+idx.float()*DP
    if mode=='linear':cdf=p0+DP*t;logpdf=s.log()
    else:
        left=derivative.gather(1,idx);right=derivative.gather(1,idx+1);mix=t*(1-t)
        den=s+(left+right-2*s)*mix
        cdf=p0+DP*(s*t.square()+left*mix)/den
        logpdf=2*s.log()+(right*t.square()+2*s*mix+left*(1-t).square()).log()-2*den.log()
    # Boundary slopes match both interpolants; tails have mass exactly .005.
    left_scale=TAIL/derivative[:,:1];right_scale=TAIL/derivative[:,-1:]
    a=((values-q[:,:1])/left_scale).clamp_max(0)
    b=(-(values-q[:,-1:])/right_scale).clamp_max(0)
    left_cdf=TAIL*a.exp();right_cdf=1-TAIL*b.exp()
    cdf=torch.where(values<q[:,:1],left_cdf,torch.where(values>q[:,-1:],right_cdf,cdf))
    logpdf=torch.where(values<q[:,:1],math.log(TAIL)-left_scale.log()+a,
        torch.where(values>q[:,-1:],math.log(TAIL)-right_scale.log()+b,logpdf))
    return cdf.clamp(0,1),logpdf

def channel_objective(output,truth,default,mask,quality,target_index,mode):
    gate,raw=output;d=default[mask][:,target_index];g=gate[mask][:,0].float()
    r=raw[mask].float();y=truth[mask][:,target_index].masked_fill(d,0)
    _,logpdf=cdf_logpdf(r,y[:,None],mode)
    nll=-logpdf[:,0].masked_fill(d,0);bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    weight=quality[mask].detach().float()
    return (weight*(bce+nll)).mean()/3,(weight*bce).mean(),(weight*nll).mean()

class Network(Parent):
    def __init__(self,mode,metadata):
        super().__init__('well',metadata)
        parent=torch.load(PARENT,map_location='cpu',weights_only=False);self.load_state_dict(parent['state_dict'],strict=True)
        self.mode=mode
        self.gate=nn.Linear(128,1)
        with torch.no_grad():self.gate.weight.copy_(self.output.weight[:1]);self.gate.bias.copy_(self.output.bias[:1])
        self.output=nn.Linear(128,K+2);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.parent_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            best_epoch=parent['best_epoch'],target_score=parent['target_score'],optimizer_reset=True,
            initialization='v46 body/FiLM/gate; new conditional quantile density, training-only empirical prior; not function-preserving')

    @torch.no_grad()
    def initialize_priors(self,data):
        ids=data['train_raw'];selected=ids[~data['default'][ids,2]]
        prior=torch.quantile(data['labels'][selected,2].float(),torch.linspace(TAIL,1-TAIL,K+1,device='cuda'))
        gap=prior.diff().clamp_min(.0011);span=(gap-.001).sum().clamp_min(.101)
        self.output.bias[0].copy_((prior[0]+prior[-1])/2)
        self.output.bias[1].copy_(torch.log(torch.expm1(span-.1)))
        self.output.bias[2:].copy_((gap-.001).log())
        return self.parent_info|dict(prior_quantiles=prior.cpu().tolist(),knots=K+1,tail_mass=TAIL,
            prior_training_nondefault_rows=len(selected),prior_fit='original64 training labels only',minimum_interval=.001)

    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Input-only current-well bank required')
        b,l,_=x.shape;c=self.body;modulation=self.condition(well_inputs)
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2);loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),modulation[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,modulation[5],valid)
        for block in c.detail:h=block(h,valid)
        h=self.norm(h);output=(self.gate(h).float(),self.output(h).float())
        return (output,h.new_zeros(()),None) if return_aux else output

class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def __call__(self,model):
        model.eval();d=self.data;c=self.c;length=model.length;lo=length//2-64;hi=lo+128
        result=np.full(len(d['validation']),np.nan,np.float64);coverage=np.zeros(len(result),np.int8)
        for begin in range(0,len(self.centers),16):
            centers=torch.tensor(self.centers[begin:begin+16],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+16],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],length=length,anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):go,raw=model(d['x'][ix],relative,valid,well_inputs=d['well_bank'][d['row_well'][anchors]])
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()];use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=self.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda')
            g=go[:,lo:hi][take][:,0].sigmoid();r=raw[:,lo:hi][take]
            cdf,_=cdf_logpdf(r,self.edges[None,:].expand(len(r),-1),model.mode)
            cp=(cdf[:,1:]-cdf[:,:-1]).clamp_min(0);cp=cp/cp.sum(-1,keepdim=True).clamp_min(1e-12)
            joint=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
            result[dest]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all() and np.isfinite(result).all()
        combined=self.reference.copy();combined[:,c]=result;scores=base.metric(d['truth'][d['validation']],combined)
        return result,float(scores['SW']),scores
