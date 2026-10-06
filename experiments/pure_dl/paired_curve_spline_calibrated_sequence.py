"""Learn a tiny per-curve piecewise correction inside the calibrated expert."""
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES


class CurveSplineMember(nn.Module):
    def __init__(self, member, raw_indices):
        super().__init__()
        assert len(raw_indices) == len(set(raw_indices)) == 9
        self.base = member
        self.register_buffer('raw_indices', torch.tensor(raw_indices, dtype=torch.long))
        self.register_buffer('knots', torch.arange(-4, 5, dtype=torch.float32))
        self.spline_weight = nn.Parameter(torch.zeros(9, 9))

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        raw = features.index_select(-1, self.raw_indices).float()
        basis = (1 - (raw[..., None] - self.knots).abs()).clamp_min(0)
        correction = (basis * self.spline_weight).sum(-1).tanh() * .5
        updated = features.index_copy(-1, self.raw_indices,
                                      (raw + correction).to(features.dtype))
        return self.base(updated, relative_depth, valid, auxiliary, absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices, raw_indices):
        assert features == 206
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[1] = CurveSplineMember(self.members[1], raw_indices)
