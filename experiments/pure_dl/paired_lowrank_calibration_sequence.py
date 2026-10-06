"""Context-controlled rank-eight cross-feature calibration of the second expert."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import WellViewMember


class LowRankCalibrationMember(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        with torch.random.fork_rng(devices=[]):
            self.project = nn.Linear(31, 8, bias=False)
            self.condition = nn.Linear(27, 8)
            self.restore = nn.Linear(8, 31, bias=False)
            nn.init.zeros_(self.restore.weight)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        member = self.original
        clean = features.masked_fill(~valid[..., None], 0)
        context = clean.index_select(-1, member.conditioning_indices)
        numeric = clean.index_select(-1, member.numeric_indices)
        gamma, beta = member.calibration(context).chunk(2, -1)
        # Bounded factors and zero output preserve the original initial function.
        cross = self.restore(self.project(numeric).tanh() * self.condition(context).tanh())
        replacement = numeric * (1 + gamma.tanh()) + beta.tanh() + cross.tanh()
        updated = clean.index_copy(-1, member.numeric_indices,
                                   replacement.masked_fill(~valid[..., None], 0).to(clean.dtype))
        return WellViewMember.forward(member, updated, relative_depth, valid, auxiliary, absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[1] = LowRankCalibrationMember(self.members[1])
