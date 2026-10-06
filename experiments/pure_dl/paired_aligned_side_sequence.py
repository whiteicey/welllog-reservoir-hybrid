"""Proxy side for full206 expert, spike side for calibrated68 expert; raw router view."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge
from paired_spike_side_sequence import spike_scores, CURVES
from paired_proxy_relative_sequence import proxy_relative_features, PROXIES


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        assert features==206
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.spike_adapter=nn.Linear(9,206)
            self.proxy_adapter=nn.Linear(12,206)
            for layer in (self.spike_adapter,self.proxy_adapter):
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def input_views(self, features, valid):
        assert features.shape[-1]==227
        original=features[...,:206].to(torch.float16 if features.device.type=='cuda' else features.dtype)
        spike=features[...,206:215].masked_fill(~valid[...,None],0)
        proxy=features[...,215:].float().masked_fill(~valid[...,None],0)
        correction0=self.proxy_adapter(proxy).masked_fill(~valid[...,None],0)
        correction1=self.spike_adapter(spike).masked_fill(~valid[...,None],0)
        views=((original+correction0).to(original.dtype),(original+correction1).to(original.dtype))
        return original,views

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        original,views=self.input_views(features,valid)
        members=[member(view,relative_depth,valid,absolute_depth=absolute_depth)
                 for member,view in zip(self.members,views)]
        logits=self.route_logits(original,valid,members)
        return members,logits.float().softmax(-1),routed_merge(members,logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        original,views=self.input_views(features,valid)
        members=[member(view,relative_depth,valid,auxiliary,absolute_depth)
                 for member,view in zip(self.members,views)]
        main=[output[0] for output in members] if auxiliary else members
        logits=self.route_logits(original,valid,main)
        if auxiliary:
            detached=[tuple(value.detach() for value in output) for output in main]
            return members,routed_merge(detached,logits)
        return routed_merge(main,logits)
