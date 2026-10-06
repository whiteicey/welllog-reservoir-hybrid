"""Matched bounded PERM density readouts: Gaussian versus fixed-df Student-t3.

Both start with identical component means and standard deviations. The t3
initial function is a distribution intervention, not a parent replay claim.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from quality_readout_v37 import Network as ParentBuilder,ROOT,PREVIOUS,load_data,reference_prediction
from train_scaling_v21 import Evaluator as ParentEvaluator
from train_invariant import conditional_probability
from por_student3_sequence import por_probability
from multiscale_sequence import window_indices
import train_invariant as base

STUDY=ROOT/'results/deep_learning/perm_family_v41_20260929'
MODES=('gaussian','student3')
LEARNING_RATE=3e-4

class DensityHead(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261141)
            self.input=nn.Conv1d(319,32,1);self.output=nn.Conv1d(32,48,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,hidden,parent,context,side,valid):
        x=torch.cat((hidden[0].float(),hidden[1].float(),side.transpose(1,2).float()),1).masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            delta=self.output(F.gelu(self.input(x))).transpose(1,2).float()
        delta=delta.masked_fill(~valid[...,None],0).reshape(*valid.shape,3,16)
        gate,raw=parent;r=raw[:,:,0].float();w,mu,s=r.unbind(-2)
        dw,dm,ds=delta.unbind(-2);sd=F.softplus(s.detach())+.025
        updated=torch.stack((w+.25*torch.tanh(dw/.25),mu+.25*sd*torch.tanh(dm/.25),s+.25*torch.tanh(ds/.25)),-2)
        return gate,updated.unsqueeze(2)

class Network(ParentBuilder):
    def __init__(self,target,master,modes=MODES):
        if target!='PERM':raise ValueError('PERM only')
        super().__init__(target,master,modes=())
        self.heads=nn.ModuleDict({m:DensityHead() for m in modes})
        self.metadata=dict(parent=self.choice,parameters_per_mode=11824,gate_fixed=True,
            component_count=16,location_bound_in_parent_sd=.25,weight_and_rawscale_bound=.25,
            family_difference='Gaussian versus df3 Student; Student scale=standard_deviation/sqrt(3)',
            transformed_target='original standardized log10 PERM; no new target transform')
        self.train(False)

def channel_objective(output,truth,default,mask,quality,target_index,mode):
    if target_index!=1 or mode not in MODES:raise ValueError((target_index,mode))
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    d=default[mask][:,1];y=truth[mask][:,1].masked_fill(d,0)
    sd=F.softplus(r[:,2])+.025;mu=r[:,1]
    if mode=='gaussian':logp=-.5*((y[:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    else:
        scale=sd/math.sqrt(3)
        logp=math.lgamma(2)-math.lgamma(1.5)-.5*math.log(3*math.pi)-scale.log()-2*torch.log1p(((y[:,None]-mu)/scale).square()/3)
    nll=-(r[:,0].log_softmax(-1)+logp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');w=quality[mask].detach().float()
    return (w*(bce+nll)).mean()/3,(w*bce).mean(),(w*nll).mean()

class Evaluator(ParentEvaluator):
    @torch.no_grad()
    def many(self,model,factors):
        assert self.c==1;model.eval();d=self.data;factors=tuple(factors)
        predictions={m:np.full(len(d['validation']),np.nan,np.float64) for m in factors};coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),32):
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda');anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            relative=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            outputs=model(d['x'][ix],relative,valid,absolute_depth=d['absolute'][ix],factors=factors)
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()];use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
            for m,(go,raw) in outputs.items():
                g=go[:,64:192][take][:,0].float().sigmoid();r=raw[:,64:192][take][:,0].float()
                probability=conditional_probability if m=='gaussian' else por_probability
                mass=probability(r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
                joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
                predictions[m][loc]=self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all()
        result={}
        for m,p in predictions.items():
            assert np.isfinite(p).all();combined=self.reference.copy();combined[:,1]=p
            scores=base.metric(d['truth'][d['validation']],combined);result[m]=(p,float(scores['PERM']),float(scores['TOTAL']))
        return result

@torch.no_grad()
def factor_scope(model,data,mode):
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper']);rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,parent,context,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    g,r=model.heads[mode](h,parent,context,data['x'][ix,...,206:],valid)
    assert torch.equal(g,parent[0]);delta=r[:,:,0]-parent[1][:,:,0]
    sd=F.softplus(parent[1][:,:,0,2].float())+.025
    assert float(delta[:,:,0].abs().max())<=.250001 and float(delta[:,:,2].abs().max())<=.250001
    assert float((delta[:,:,1]/sd).abs().max())<=.250001
    return dict(gate_logits_exact=True,weight_delta=float(delta[:,:,0].abs().max()),location_delta_in_parent_sd=float((delta[:,:,1]/sd).abs().max()),rawscale_delta=float(delta[:,:,2].abs().max()),
                family=mode,student_std_convention='scale=sd/sqrt(3)',scope='two training windows; not validation fitting')
