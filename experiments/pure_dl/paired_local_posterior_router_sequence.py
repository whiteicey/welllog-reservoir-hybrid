"""Combine detached posterior descriptors with depth-local router context."""
import torch
from torch import nn
from paired_posterior_router_sequence import (
    Network as Baseline, posterior_descriptors, retained_indices,
    calibration_indices, VIEW_NAMES,
)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices,
                 numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices,
                         conditioning_indices=conditioning_indices)
        # Preserve the posterior-router control initialization and RNG stream.
        with torch.random.fork_rng(devices=[]):
            context = nn.Conv1d(32, 32, 9, padding=4, groups=32, bias=False)
            nn.init.zeros_(context.weight)
        self.router.add_module('local_context', context)

    def route_logits(self, features, valid, main_outputs):
        original = features.index_select(-1, self.router_indices).float()
        descriptors = posterior_descriptors(main_outputs)
        x = torch.cat([original, descriptors], -1).masked_fill(~valid[..., None], 0)
        hidden = self.router[1](self.router[0](x)).masked_fill(~valid[..., None], 0)
        correction = self.router.local_context(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = (hidden + correction).masked_fill(~valid[..., None], 0)
        return self.router[2](hidden).reshape(*features.shape[:2], 3, 2)
