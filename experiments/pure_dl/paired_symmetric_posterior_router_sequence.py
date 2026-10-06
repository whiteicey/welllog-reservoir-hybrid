"""Shared quality scorer with expert-permutation-equivariant posterior routing."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, posterior_descriptors, retained_indices, calibration_indices, VIEW_NAMES


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        old=self.router[2]
        with torch.random.fork_rng(devices=[]):
            final=nn.Linear(32,3,device=old.weight.device,dtype=old.weight.dtype)
            nn.init.zeros_(final.weight);nn.init.zeros_(final.bias)
        self.router[2]=final

    def route_logits(self, features, valid, main_outputs):
        original=features.index_select(-1,self.router_indices).float()
        descriptor=posterior_descriptors(main_outputs)
        first,second=descriptor.split(12,dim=-1)
        views=torch.stack([torch.cat([original,first,second],-1),
                           torch.cat([original,second,first],-1)],dim=-2)
        views=views.masked_fill(~valid[...,None,None],0)
        # The same scorer is evaluated twice; expert axis moves after targets.
        return self.router(views).transpose(-1,-2)
