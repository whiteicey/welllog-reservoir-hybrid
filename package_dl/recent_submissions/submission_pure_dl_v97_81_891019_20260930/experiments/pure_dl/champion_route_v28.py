"""Bounded internal expert-routing corrections on the retained v26 predictor.

joint: modifies atom probability and conditional expert shares together.
conditional: preserves the old merged default logit exactly, only modifies the
conditional expert shares. Neither arm modifies an expert's component outputs.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from champion_posterior_v27 import Network as FrozenRecipeLoader
from champion_all_heads_v26 import Evaluator,load_data,TARGETS
from paired_detached_router_sequence import routed_merge

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/champion_route_v28_20260929'
PREVIOUS=ROOT/'results/deep_learning/all_heads_v26_20260928'
MODES=('joint','conditional')


def reference_prediction():
    return np.load(PREVIOUS/'assembled_pred.npy')


class RoutingHead(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:raise ValueError(mode)
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261104)
            self.input=nn.Conv1d(128*2+63,32,1)
            self.output=nn.Conv1d(32,1,1)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def correction(self,hidden,side,valid):
        z=torch.cat((hidden[0],hidden[1],side.transpose(1,2).to(hidden[0].dtype)),1)
        z=z.masked_fill(~valid[:,None],0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            proposal=self.output(F.gelu(self.input(z))).squeeze(1).float()
        return (.25*torch.tanh(proposal/.25)).masked_fill(~valid,0)

    def forward(self,hidden,outputs,logits,side,valid):
        delta=self.correction(hidden,side,valid)
        # Contrast correction has zero common-mode: log(a1/a2) changes by delta.
        shift=torch.stack((.5*delta,-.5*delta),-1).unsqueeze(2)
        gate,raw=routed_merge(outputs,logits.float()+shift)
        if self.mode=='conditional':
            original_gate,_=routed_merge(outputs,logits)
            gate=original_gate
        return gate,raw


class Network(FrozenRecipeLoader):
    def __init__(self,target,master,modes=MODES):
        # Reuse only v27's exact v26 source-loading/extraction interface. No
        # trained v27 correction is loaded or executed.
        super().__init__(target,master,modes=())
        self.factor='expert_routing'
        self.heads=nn.ModuleDict({m:RoutingHead(m) for m in modes})
        self.train(False)


@torch.no_grad()
def routing_scope(model,data,mode):
    from multiscale_sequence import window_indices
    anchors=data['train_raw'][torch.tensor([0,len(data['train_raw'])//2],device='cuda')]
    ix,valid=window_indices(anchors,data['lower'],data['upper'])
    rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    h,o,logits,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
    before=routed_merge(o,logits)
    after=model.heads[mode](h,o,logits,data['x'][ix,...,206:],valid)
    assert torch.equal(before[1][...,1:,:],after[1][...,1:,:])
    delta=model.heads[mode].correction(h,data['x'][ix,...,206:],valid)
    assert float(delta.abs().max())<=.250001
    if mode=='conditional':assert torch.equal(before[0],after[0])
    old_share=before[1][...,0,:].softmax(-1)[...,:8].sum(-1)
    new_share=after[1][...,0,:].softmax(-1)[...,:8].sum(-1)
    return dict(mode=mode,all_component_means_and_scales_exact=True,
        merged_default_logit_exact=bool(torch.equal(before[0],after[0])),
        routing_log_odds_max_abs=float(delta.abs()[valid].max()),
        expert1_conditional_share_mean_change=float((new_share-old_share)[valid].mean()),
        expert1_conditional_share_max_abs_change=float((new_share-old_share).abs()[valid].max()),
        merged_default_probability_max_abs_change=float((after[0].sigmoid()-before[0].sigmoid()).abs()[valid].max()),
        scope='two fixed training windows, diagnostic only; fixed default probability does not force the final default action')
