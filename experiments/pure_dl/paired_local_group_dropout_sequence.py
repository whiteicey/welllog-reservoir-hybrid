"""Drop local engineered feature groups during full-expert training only."""
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_pruned_local_view_sequence import CURVES, SUFFIXES


def local_group_indices(whitelist):
    assert len(whitelist) == len(set(whitelist)) == 206
    groups = []
    for curve in CURVES:
        names = [curve + '_' + suffix for suffix in SUFFIXES]
        assert all(whitelist.count(name) == 1 for name in names)
        groups.append([whitelist.index(name) for name in names])
    assert len(groups) == 9 and all(len(group) == 10 for group in groups)
    return groups


class LocalGroupDropoutMember(nn.Module):
    def __init__(self, member, groups, probability=0.12):
        super().__init__()
        assert len(groups) == 9 and all(len(group) == 10 for group in groups)
        assert 0 < probability < 1
        self.base = member
        self.probability = float(probability)
        self.register_buffer('local_indices', torch.tensor(groups, dtype=torch.long).flatten())
        self._generator = None

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        if self.training:
            if self._generator is None:
                self._generator = torch.Generator(device=features.device)
                self._generator.manual_seed(20260928)
            keep = torch.rand((features.shape[0], 9), device=features.device,
                              generator=self._generator) >= self.probability
            factor = keep.repeat_interleave(10, dim=-1).unsqueeze(1).to(features.dtype)
            factor = factor / (1 - self.probability)
            selected = features.index_select(-1, self.local_indices)
            changed = selected * factor
            features = features.index_copy(-1, self.local_indices, changed)
        return self.base(features, relative_depth, valid, auxiliary, absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices, group_indices):
        assert features == 206
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[0] = LocalGroupDropoutMember(self.members[0], group_indices)
