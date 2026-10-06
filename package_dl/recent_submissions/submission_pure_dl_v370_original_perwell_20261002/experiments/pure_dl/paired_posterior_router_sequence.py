"""Detached main-posterior descriptors augment the calibrated68 input router."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paired_calibrated_view_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge


def posterior_descriptors(outputs):
    descriptors=[]
    for gate,raw in outputs:
        # Detach BOTH routes into router input. Expert NLL remains independent.
        gate=gate.detach().float();raw=raw.detach().float()
        weight=raw[...,0,:].softmax(-1)
        mu=raw[...,1,:]
        sd=F.softplus(raw[...,2,:])+.025
        mean=(weight*mu).sum(-1)
        # POR t3 scale=sd/sqrt3, hence its actual component variance is sd^2;
        # the same expression therefore applies to all three target families.
        variance=(weight*(sd.square()+(mu-mean[...,None]).square())).sum(-1)
        entropy=-(weight*weight.clamp_min(1e-30).log()).sum(-1)/math.log(8.)
        values=torch.stack([gate.sigmoid(),entropy.clamp(0,1),mean.clamp(-8,8),
                            torch.log1p(variance.clamp_min(0).sqrt()).clamp(max=8)],-1)
        descriptors.append(values.flatten(-2))
    return torch.cat(descriptors,-1)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.register_buffer('router_indices',torch.tensor(retain_indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.router=nn.Sequential(nn.Linear(92,32),nn.GELU(),nn.Linear(32,6))
            nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)

    def route_logits(self, features, valid, main_outputs):
        original=features.index_select(-1,self.router_indices).float()
        descriptors=posterior_descriptors(main_outputs)
        x=torch.cat([original,descriptors],-1).masked_fill(~valid[...,None],0)
        return self.router(x).reshape(*features.shape[:2],3,2)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members=[m(features,relative_depth,valid,absolute_depth=absolute_depth) for m in self.members]
        logits=self.route_logits(features,valid,members)
        return members,logits.float().softmax(-1),routed_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members=[m(features,relative_depth,valid,auxiliary,absolute_depth) for m in self.members]
        main=[output[0] for output in members] if auxiliary else members
        logits=self.route_logits(features,valid,main)
        if auxiliary:
            detached=[tuple(value.detach() for value in output) for output in main]
            return members,routed_merge(detached,logits)
        return routed_merge(main,logits)
