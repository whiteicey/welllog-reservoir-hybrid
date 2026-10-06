"""Same-bound POR route optimization from the retained v30 function.

Only existing route parameters learn. Both alternatives carry an identical
frozen anchor copy; the anchored alternative uses it in the parameterization.
"""
from pathlib import Path
import copy,json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_coadapt_v30 import Network as CurrentLoader
from champion_all_heads_v26 import Evaluator,load_data,TARGETS
from train_scaling_v21 import channel_objective

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/mechanism_checks_v31_20260929'
PREVIOUS=ROOT/'results/deep_learning/champion_coadapt_v30_20260929'
MODES=('tanh','anchored')
LEARNING_RATE=3e-5


def reference_prediction():
    return np.load(PREVIOUS/'assembled_pred.npy')


class ReparameterizedBranch(nn.Module):
    def __init__(self,original,bound,mode):
        super().__init__()
        self.input=copy.deepcopy(original.input)
        self.output=copy.deepcopy(original.output)
        self.anchor_input=copy.deepcopy(original.input)
        self.anchor_output=copy.deepcopy(original.output)
        self.anchor_input.requires_grad_(False);self.anchor_output.requires_grad_(False)
        self.input.requires_grad_(True);self.output.requires_grad_(True)
        self.bound=float(bound);self.mode=mode

    def forward(self,x,valid):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            u=self.output(F.gelu(self.input(x))).squeeze(1).float()
            if self.mode=='anchored':
                with torch.no_grad():
                    u0=self.anchor_output(F.gelu(self.anchor_input(x))).squeeze(1).float()
                # Parenthesize the zero residual: preserve d0 exactly initially.
                d0=self.bound*torch.tanh(u0/self.bound)
                d=(d0+(u-u0)).clamp(-self.bound,self.bound)
            else:
                d=self.bound*torch.tanh(u/self.bound)
        return d.masked_fill(~valid,0)


class ReparameterizedRoute(nn.Module):
    def __init__(self,original,mode):
        super().__init__();self.unit=ReparameterizedBranch(original,.25,mode)

    def correction(self,hidden,side,valid):
        x=torch.cat((hidden[0],hidden[1],side.transpose(1,2).to(hidden[0].dtype)),1)
        return self.unit(x.masked_fill(~valid[:,None],0),valid)


class Network(CurrentLoader):
    def __init__(self,target,master,modes=MODES):
        if target!='POR':raise ValueError('v31 reparameterization is POR only')
        super().__init__(target,master,modes=('coadapt',))
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        self.choice=next(c for c in recipe['choices'] if c['target']==target)
        self.parent_path=self.choice['checkpoint']
        saved=torch.load(ROOT/self.parent_path,map_location='cpu',weights_only=False)
        original=self.heads['coadapt']
        original.load_state_dict(saved['delta_state'],strict=True)
        original.requires_grad_(False)
        heads={}
        for mode in modes:
            bundle=copy.deepcopy(original);bundle.mode='route_only'
            bundle.route28=ReparameterizedRoute(original.route28,mode)
            bundle.route29.branches=nn.ModuleList([
                ReparameterizedBranch(branch,.125,mode) for branch in original.route29.branches])
            heads[mode]=bundle
        self.heads=nn.ModuleDict(heads);self.factor='same_bound_route_parameterization'
        self.train(False)
        for head in self.heads.values():
            assert sum(p.numel() for p in head.parameters() if p.requires_grad)==30819

    def trainable_groups(self,mode):
        return [dict(params=[p for p in self.heads[mode].parameters() if p.requires_grad],lr=LEARNING_RATE,name='existing_routes')]


@torch.no_grad()
def factor_scope(model,data,mode):
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,o,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix]);side=data['x'][ix,...,206:]
    head=model.heads[mode]
    x=torch.cat((h[0],h[1],side.transpose(1,2).to(h[0].dtype)),1).masked_fill(~valid[:,None],0)
    units=[head.route28.unit,*head.route29.branches];scope={}
    for i,unit in enumerate(units):
        d=unit(x,valid)
        assert float(d.abs().max())<=unit.bound+1e-7
        scope[str(i)]=dict(bound=unit.bound,max_abs=float(d[valid].abs().max()),
            fraction_near_bound=float((d[valid].abs()>.95*unit.bound).float().mean()))
    return dict(mode=mode,route_bounds=scope,scope='two fixed training windows; component outputs remain frozen by parameter checks')


def mechanism_probe(model,data,mode):
    """Check the actual parameterization derivative at the initial trained weights."""
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,_,_,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix]);side=data['x'][ix,...,206:]
    x=torch.cat((h[0],h[1],side.transpose(1,2).to(h[0].dtype)),1).masked_fill(~valid[:,None],0)
    result={}
    for name,unit in [('v28',model.heads[mode].route28.unit)]+[(f'v29_{j}',b) for j,b in enumerate(model.heads[mode].route29.branches)]:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            u0=unit.anchor_output(F.gelu(unit.anchor_input(x))).squeeze(1).float()
        u=u0.detach().clone().requires_grad_(True)
        b=unit.bound;d0=b*torch.tanh(u0/b)
        d=(d0+(u-u0)).clamp(-b,b) if mode=='anchored' else b*torch.tanh(u/b)
        derivative=torch.autograd.grad(d[valid].sum(),u)[0][valid]
        result[name]=dict(derivative_mean=float(derivative.mean()),derivative_min=float(derivative.min()),
            fraction_below_002=float((derivative<.02).float().mean()),initial_bound=b)
    return result
