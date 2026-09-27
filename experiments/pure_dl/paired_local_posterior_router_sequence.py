"""Zero-init depth-local context in the posterior-descriptor router's hidden layer."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from paired_calibrated_view_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge
from paired_posterior_router_sequence import posterior_descriptors


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        # Same construction order as the posterior router: experts, then router.
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.register_buffer('router_indices',torch.tensor(retain_indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.router=nn.Sequential(nn.Linear(92,32),nn.GELU(),nn.Linear(32,6))
            nn.init.zeros_(self.router[-1].weight);nn.init.zeros_(self.router[-1].bias)
            context=nn.Conv1d(32,32,9,padding=4,groups=32,bias=False)
            nn.init.zeros_(context.weight)
        # Child of router so the separate router optimizer includes this parameter.
        self.router.add_module('local_context',context)

    def route_logits(self, features, valid, main_outputs):
        original=features.index_select(-1,self.router_indices).float()
        descriptors=posterior_descriptors(main_outputs)
        x=torch.cat([original,descriptors],-1).masked_fill(~valid[...,None],0)
        hidden=self.router[1](self.router[0](x)).masked_fill(~valid[...,None],0)
        correction=self.router.local_context(hidden.transpose(1,2)).transpose(1,2)
        correction=correction.masked_fill(~valid[...,None],0)
        hidden=(hidden+correction).masked_fill(~valid[...,None],0)
        return self.router[2](hidden).reshape(*features.shape[:2],3,2)

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
