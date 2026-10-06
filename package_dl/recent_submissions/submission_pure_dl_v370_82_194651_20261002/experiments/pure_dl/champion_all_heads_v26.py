"""Independent factor corrections on each target's frozen retained model.

One target process shares only immutable source inference between alternatives.
No trainable parameter, optimizer or stopping state is shared among candidates.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_shape_v25 import load_data as load_amplitude_data, make_model as por_builder
from champion_scaling_v21 import HEADS
from independent_channels_v8 import Network as PrivateNetwork, TARGETS
from paired_detached_router_sequence import routed_merge
from train_scaling_v21 import Evaluator as OriginalEvaluator
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/all_heads_v26_20260928'
PREVIOUS=ROOT/'results/deep_learning/champion_shape_v25_20260928'
FACTORS={'POR':('gate','weight','scale'),
         'PERM':('gate','weight','location','scale'),
         'SW':('gate','weight','location','scale')}


def load_data():
    return load_amplitude_data('amplitude')


def reference_prediction():
    return np.load(PREVIOUS/'assembled_amplitude_pred.npy')


def build_source(target,master):
    if target=='POR':
        model=por_builder('amplitude',master)
        path=ROOT/'outputs/pure_dl_champion_shape_v25_amplitude_fold5/model_best.pt'
    else:
        previous=torch.load(HEADS,map_location='cpu',weights_only=False)
        model=PrivateNetwork(master,previous,target,'dual')
        path=ROOT/'outputs'/f'pure_dl_independent_v8_{target}_dual80_fold5/model_best.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True)
    model.requires_grad_(False);model.eval()
    return model,str(path.relative_to(ROOT))


class ExpertCorrection(nn.Module):
    def __init__(self,outputs):
        super().__init__()
        self.input=nn.Conv1d(191,32,1)
        self.output=nn.Conv1d(32,outputs,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,h,side,valid):
        z=torch.cat((h,side.transpose(1,2).to(h.dtype)),1)
        z=z.masked_fill(~valid[:,None],0)
        return self.output(F.gelu(self.input(z))).transpose(1,2).float().masked_fill(~valid[...,None],0)


class FactorHead(nn.Module):
    def __init__(self,factor):
        super().__init__();self.factor=factor
        self.experts=nn.ModuleList()
        # Equal shared input projection initialization across factors of a target.
        for e in range(2):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261102+e)
                self.experts.append(ExpertCorrection(1 if factor=='gate' else 8))

    def corrected_members(self,hidden,outputs,side,valid):
        result=[]
        for head,h,(g,r) in zip(self.experts,hidden,outputs):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proposal=head(h,side,valid)
            if self.factor=='gate':
                result.append((g.float()+.25*torch.tanh(proposal/.25),r));continue
            field={'weight':0,'location':1,'scale':2}[self.factor]
            bound=.25*(F.softplus(r[:,:,0,2,:].detach().float())+.025) if field==1 else .25
            delta=bound*torch.tanh(proposal/bound)
            slices=[r[...,k,:] for k in range(3)]
            slices[field]=slices[field]+delta.unsqueeze(2)
            result.append((g,torch.stack(slices,dim=-2)))
        return result

    def forward(self,hidden,outputs,logits,side,valid):
        return routed_merge(self.corrected_members(hidden,outputs,side,valid),logits)


class Network(nn.Module):
    def __init__(self,target,master,factors=None):
        super().__init__()
        if target not in FACTORS:raise ValueError(target)
        self.target=target;self.c=TARGETS.index(target)
        factors=tuple(FACTORS[target] if factors is None else factors)
        if not factors or any(f not in FACTORS[target] for f in factors):raise ValueError(factors)
        self.source,self.source_path=build_source(target,master)
        self.heads=nn.ModuleDict({f:FactorHead(f) for f in factors})
        self.captured={};self.handles=[]
        for e,member in enumerate(self.source.members):
            def capture(_module,args,output,e=e):
                self.captured['h'+str(e)]=args[0].detach()
                self.captured['o'+str(e)]=tuple(v.detach() for v in output)
            self.handles.append(member.main_head.register_forward_hook(capture))
        self.handles.append(self.source.router[2].register_forward_hook(
            lambda _m,_a,v:self.captured.update(logits=v.detach())))

    def train(self,mode=True):
        super().train(False);self.training=mode;self.heads.train(mode);return self

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        self.captured.clear();self.source.eval()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            original=self.source(x if self.c==0 else x[...,:206].contiguous(),relative,valid,absolute_depth=absolute)
        h=[self.captured['h'+str(e)] for e in range(2)]
        o=[self.captured['o'+str(e)] for e in range(2)]
        logits=self.captured['logits'].reshape(*x.shape[:2],3,2)[:,:,self.c:self.c+1]
        self.captured.clear()
        return h,o,logits,original

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        chosen=tuple(self.heads) if factors is None else tuple(factors)
        h,o,logits,_=self.extract(x,relative,valid,absolute_depth)
        return {f:self.heads[f](h,o,logits,x[...,206:],valid) for f in chosen}

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[];self.captured.clear()


class Evaluator(OriginalEvaluator):
    @torch.no_grad()
    def many(self,model,factors):
        model.eval();d=self.data;c=self.c;factors=tuple(factors)
        predictions={f:np.full(len(d['validation']),np.nan,np.float64) for f in factors}
        coverage=np.zeros(len(d['validation']),np.int8)
        for begin in range(0,len(self.centers),32):
            from multiscale_sequence import window_indices
            centers=torch.tensor(self.centers[begin:begin+32],device='cuda')
            anchors=torch.tensor(self.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(centers,d['lower'],d['upper'],anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            output=model(d['x'][ix],rel,valid,absolute_depth=d['absolute'][ix],factors=factors)
            rows=d['raw_to_valid'][ix[:,64:192].cpu().numpy()]
            use=valid[:,64:192].cpu().numpy()&(rows>=0);loc=self.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda')
            for factor,(go,raw) in output.items():
                from por_t3_patch_loss import probability
                g=go[:,64:192][take][:,0].float().sigmoid()
                r=raw[:,64:192][take][:,0].float()
                p=probability(c,r[:,0].softmax(-1),r[:,1],F.softplus(r[:,2])+.025,self.edges)
                mass=torch.cat((g[:,None],(1-g[:,None])*p),-1)
                predictions[factor][loc]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage==1).all()
        result={}
        for f,pred in predictions.items():
            assert np.isfinite(pred).all()
            combined=self.reference.copy();combined[:,c]=pred
            scores=base.metric(d['truth'][d['validation']],combined)
            result[f]=(pred,float(scores[TARGETS[c]]),float(scores['TOTAL']))
        return result
