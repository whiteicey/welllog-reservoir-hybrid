"""Equal-parameter coupled vs separated route increments over retained v28.

The v28 atom and conditional routing bases are frozen and preserved exactly.
Both candidates own two active, independently initialized nonlinear branches.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_posterior_v27 import Network as FrozenV26
from champion_route_v28 import RoutingHead as RetainedRoute
from champion_all_heads_v26 import Evaluator,load_data,TARGETS
from paired_detached_router_sequence import routed_merge

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'results/deep_learning/champion_route_v28_20260929'
STUDY=ROOT/'results/deep_learning/champion_split_route_v29_20260929'
MODES=('coupled','split')
BOUND=.125


def reference_prediction():
    return np.load(PREVIOUS/'assembled_pred.npy')


def shift(logits,delta):
    return logits.float()+torch.stack((.5*delta,-.5*delta),-1).unsqueeze(2)


class Branch(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.input=nn.Conv1d(319,32,1)
            self.output=nn.Conv1d(32,1,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,x,valid):
        u=self.output(F.gelu(self.input(x))).squeeze(1).float()
        return (BOUND*torch.tanh(u/BOUND)).masked_fill(~valid,0)


class SplitRouteHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode
        self.branches=nn.ModuleList([Branch(20261105+i) for i in range(2)])

    def corrections(self,hidden,side,valid):
        x=torch.cat((hidden[0],hidden[1],side.transpose(1,2).to(hidden[0].dtype)),1)
        x=x.masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a,b=[branch(x,valid) for branch in self.branches]
        if self.mode=='coupled':
            common=(a+b)*.5
            return common,common
        return a,b

    def forward(self,hidden,outputs,contexts,side,valid):
        atom_logits,density_logits=contexts
        da,dc=self.corrections(hidden,side,valid)
        gate,_=routed_merge(outputs,shift(atom_logits,da))
        _,raw=routed_merge(outputs,shift(density_logits,dc))
        return gate,raw


class Network(FrozenV26):
    def __init__(self,target,master,modes=MODES):
        super().__init__(target,master,modes=())
        recipe=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))
        item=next(x for x in recipe['choices'] if x['target']==target)
        self.parent_mode=item['mode'];self.route_path=item['checkpoint']
        cp=torch.load(ROOT/self.route_path,map_location='cpu',weights_only=False)
        if cp['target']!=target or cp['mode']!=self.parent_mode:raise ValueError('Wrong v28 route checkpoint')
        self.retained_route=RetainedRoute(self.parent_mode)
        self.retained_route.load_state_dict(cp['delta_state'],strict=True)
        self.retained_route.requires_grad_(False);self.retained_route.eval()
        self.heads=nn.ModuleDict({m:SplitRouteHead(m) for m in modes})
        self.factor='atom_conditional_routing';self.train(False)

    @torch.no_grad()
    def extract(self,x,relative,valid,absolute):
        h,o,logits,_=super().extract(x,relative,valid,absolute)
        delta=self.retained_route.correction(h,x[...,206:],valid)
        density_logits=shift(logits,delta)
        atom_logits=density_logits if self.parent_mode=='joint' else logits
        return h,o,(atom_logits,density_logits),None


@torch.no_grad()
def routing_scope(model,data,mode):
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,o,contexts,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    ga,_=routed_merge(o,contexts[0]);_,raw=routed_merge(o,contexts[1])
    after=model.heads[mode](h,o,contexts,data['x'][ix,...,206:],valid)
    da,dc=model.heads[mode].corrections(h,data['x'][ix,...,206:],valid)
    assert torch.equal(raw[...,1:,:],after[1][...,1:,:])
    assert max(float(da.abs().max()),float(dc.abs().max()))<=BOUND+1e-6
    if mode=='coupled':assert torch.equal(da,dc)
    old_share=raw[...,0,:].softmax(-1)[...,:8].sum(-1)
    new_share=after[1][...,0,:].softmax(-1)[...,:8].sum(-1)
    return dict(mode=mode,means_and_scales_exact=True,
        atom_increment_max=float(da[valid].abs().max()),density_increment_max=float(dc[valid].abs().max()),
        increment_difference_mean=float((da-dc)[valid].abs().mean()),
        default_probability_max_change=float((ga.sigmoid()-after[0].sigmoid())[valid].abs().max()),
        conditional_share_mean_change=float((new_share-old_share)[valid].mean()),
        scope='two training windows; increments relative to selected v28 paths')


def gradient_isolation(model,data,mode):
    """Small training-input check. No parameter updates, labels or validation selection."""
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,o,contexts,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    head=model.heads[mode]
    gate,raw=head(h,o,contexts,data['x'][ix,...,206:],valid)
    params=[list(branch.parameters()) for branch in head.branches]
    flat=params[0]+params[1]
    gs=torch.autograd.grad(gate[valid].sigmoid().sum(),flat,retain_graph=True,allow_unused=True)
    ds=torch.autograd.grad(raw[...,0,:][valid].square().sum(),flat,allow_unused=True)
    cut=len(params[0])
    def norm(grads):
        vals=[g.detach().float().square().sum() for g in grads if g is not None]
        return float(torch.stack(vals).sum().sqrt()) if vals else 0.
    out=dict(atom_to_branch0=norm(gs[:cut]),atom_to_branch1=norm(gs[cut:]),
             density_to_branch0=norm(ds[:cut]),density_to_branch1=norm(ds[cut:]))
    if mode=='split':
        assert out['atom_to_branch1']==0. and out['density_to_branch0']==0.
    return out
