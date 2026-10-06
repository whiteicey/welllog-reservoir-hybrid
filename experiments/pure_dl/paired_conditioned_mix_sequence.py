"""Well-conditioned rank-eight numeric mixing in the calibrated expert only."""
import torch
from torch import nn
from paired_posterior_router_sequence import (
    Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES)


class ConditionedMixMember(nn.Module):
    def __init__(self, original, numeric_indices, conditioning_indices):
        super().__init__()
        self.original = original
        self.register_buffer('numeric_indices', torch.tensor(numeric_indices))
        self.register_buffer('conditioning_indices', torch.tensor(conditioning_indices))
        with torch.random.fork_rng(devices=[]):
            self.encode = nn.Linear(len(numeric_indices), 8)
            self.condition = nn.Linear(len(conditioning_indices), 8)
            self.decode = nn.Linear(8, len(numeric_indices))
            nn.init.zeros_(self.decode.weight)
            nn.init.zeros_(self.decode.bias)

    def forward(self, features, relative_depth, valid, auxiliary=False,
                absolute_depth=None):
        clean = features.masked_fill(~valid[..., None], 0)
        numeric = clean.index_select(-1, self.numeric_indices)
        context = clean.index_select(-1, self.conditioning_indices)
        cross = self.encode(numeric).tanh() * self.condition(context).tanh()
        replacement = (numeric + self.decode(cross)).masked_fill(
            ~valid[..., None], 0).to(clean.dtype)
        updated = clean.index_copy(-1, self.numeric_indices, replacement)
        return self.original(updated, relative_depth, valid, auxiliary,
                             absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices,
                 numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices,
                         conditioning_indices=conditioning_indices)
        self.members[1] = ConditionedMixMember(
            self.members[1], numeric_indices, conditioning_indices)
