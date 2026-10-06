"""Frozen champion PERM with bounded polynomial CDF calibration heads."""
import json,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_split_route_v29 import Network as Parent
from champion_all_heads_v26 import ROOT,load_data,Evaluator as OriginalEvaluator
from por_t3_patch_loss import probability
from multiscale_sequence import window_indices
import train_invariant as base
STUDY=ROOT/'results/deep_learning/perm_cdfwarp_v70_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_curvature_v69_20260929'
MODES=('affine','nonlinear')
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class Warp(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.input=nn.Conv1d(319,32,1);self.output=nn.Conv1d(32,2,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,hidden,side,valid):
        z=torch.cat((*hidden,side.transpose(1,2).to(hidden[0].dtype)),1).masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            h=self.input(z)
            if self.mode=='nonlinear':h=F.gelu(h)
            uv=.25*torch.tanh(self.output(h).transpose(1,2).float())
        return uv.masked_fill(~valid[...,None],0)
class Network(nn.Module):
    def __init__(self,target,master,mode):
        super().__init__()
        if target!='PERM' or mode not in MODES:raise ValueError((target,mode))
        self.target=target;self.c=1;self.mode=mode
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        self.choice=next(x for x in recipe['choices'] if x['target']==target)
        assert self.choice['model_module']=='champion_split_route_v29' and self.choice['mode']=='coupled'
        self.current=Parent('PERM',master,modes=('coupled',))
        saved=torch.load(ROOT/self.choice['checkpoint'],map_location='cpu',weights_only=False)
        self.current.heads['coupled'].load_state_dict(saved['delta_state'],strict=True);self.current.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261170);self.warp=Warp(mode)
        self.trainable_names={n for n,p in self.named_parameters() if p.requires_grad}
        self.manifest=dict(mode=mode,parameters=sum(p.numel() for p in self.warp.parameters()),bound=.25,
            transform='F+u*F*(1-F)+v*F*(1-F)*(1-2F)',derivative_lower_bound=.5,
            frozen='entire v29 PERM coupled predictor',initialization='zero warp output, same tensors',hidden=319,rank=32)
    def train(self,mode=True):
        super().train(False);self.training=mode;return self
    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        if factors is not None and tuple(factors)!=(self.mode,):raise ValueError(factors)
        with torch.no_grad():
            hidden,outputs,contexts,_=self.current.extract(x,relative,valid,absolute_depth)
            gate,raw=self.current.heads['coupled'](hidden,outputs,contexts,x[...,206:],valid)
        uv=self.warp(hidden,x[...,206:],valid)
        return {self.mode:(gate,raw,uv)}
    def trainable_state(self):return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}
    def load_trainable_state(self,state):
        assert set(state)==self.trainable_names
        params=dict(self.named_parameters())
        with torch.no_grad():
            for n,v in state.items():params[n].copy_(v.to(params[n].device))
    def frozen_state(self):return {n:v for n,v in self.state_dict().items() if n not in self.trainable_names}
    def close(self):self.current.close()

def channel_objective(output,truth,default,mask,quality,target_index):
    assert target_index==1
    gate,raw,uv=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float();u,v=uv[mask].float().unbind(-1)
    d=default[mask][:,1];y=truth[mask][:,1].float().masked_fill(d,0)
    mu=r[:,1];sd=F.softplus(r[:,2])+.025;w=r[:,0].softmax(-1);lw=r[:,0].log_softmax(-1)
    z=(y[:,None]-mu)/sd
    logpdf=(lw-.5*z.square()-sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    cdf=(w*(.5*(1+torch.erf(z/math.sqrt(2))))).sum(-1).clamp(0,1)
    derivative=1+u*(1-2*cdf)+v*(1-6*cdf+6*cdf.square())
    nll=-(logpdf+derivative.log()).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');q=quality[mask].detach().float()
    return (q*(bce+nll)).mean()/3,(q*bce).mean(),(q*nll).mean()

def warp_probability(raw,uv,edges):
    r=raw.float();p=probability(1,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,edges)
    cdf=torch.cat((torch.zeros_like(p[:,:1]),p.cumsum(-1)),-1).clamp(0,1);cdf[:,-1]=1
    u,v=uv.float().unbind(-1)
    delta=cdf*(1-cdf)*(u[:,None]+v[:,None]*(1-2*cdf))
    # Residual arithmetic keeps the zero head bitwise identical to original p.
    # Endpoints make correction mass telescope to zero; roundoff only is clamped.
    return (p+delta.diff(dim=-1)).clamp_min(0)
class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def many(self,model,factors):
        model.eval();d=self.data;preds={m:np.full(len(d['validation']),np.nan,np.float64) for m in factors}
        coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            output=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],factors=factors)
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()];use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
            for mode,(gate,raw,uv) in output.items():
                g=gate[:,64:192][take][:,0].float().sigmoid();r=raw[:,64:192][take][:,0]
                cp=warp_probability(r,uv[:,64:192][take],self.edges)
                mass=torch.cat((g[:,None],(1-g[:,None])*cp),-1)
                preds[mode][loc]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all();result={}
        for mode,pred in preds.items():
            assert np.isfinite(pred).all();combined=self.reference.copy();combined[:,1]=pred;s=base.metric(d['truth'][d['validation']],combined)
            result[mode]=(pred,float(s['PERM']),float(s['TOTAL']))
        return result
