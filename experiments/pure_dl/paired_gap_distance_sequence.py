"""Input-only whole-well gap geometry for the calibrated expert."""
import numpy as np
import torch
from torch import nn
from paired_well_view_sequence import RAW_CURVES
from paired_posterior_router_sequence import (
    Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES)

GAP_NAMES = tuple(c + '_' + side + '_VALID_DISTANCE'
                  for c in RAW_CURVES for side in ('PREVIOUS', 'NEXT'))


def gap_features(frame):
    depth = frame['DEPTH'].to_numpy(np.float64)
    n = len(depth)
    pos = np.arange(n)
    result = []
    for curve in RAW_CURVES:
        observed = np.isfinite(frame[curve].to_numpy(np.float64))
        previous = np.maximum.accumulate(np.where(observed, pos, -1))
        following = np.minimum.accumulate(np.where(observed, pos, n)[::-1])[::-1]
        for index, available in [(previous, previous >= 0), (following, following < n)]:
            distance = np.where(available, np.abs(depth - depth[np.clip(index, 0, n-1)]), 100.)
            result.append(np.log1p(np.minimum(distance, 100.)) / np.log(101.))
    return np.stack(result, axis=1).astype(np.float32)


class OriginalInputs(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, relative_depth, valid, auxiliary=False, absolute_depth=None):
        return self.original(x[..., :206], relative_depth, valid, auxiliary, absolute_depth)


class GapCalibratedInputs(nn.Module):
    def __init__(self, original, numeric_indices):
        super().__init__()
        self.original = original
        self.register_buffer('numeric_indices', torch.tensor(numeric_indices))
        with torch.random.fork_rng(devices=[]):
            self.gap_calibration = nn.Sequential(nn.Linear(18,32), nn.GELU(), nn.Linear(32,62))
            nn.init.zeros_(self.gap_calibration[-1].weight)
            nn.init.zeros_(self.gap_calibration[-1].bias)

    def forward(self, x, relative_depth, valid, auxiliary=False, absolute_depth=None):
        clean = x.masked_fill(~valid[...,None], 0)
        gamma, beta = self.gap_calibration(clean[...,206:]).chunk(2,-1)
        original = clean[...,:206]
        numeric = original.index_select(-1,self.numeric_indices)
        replacement = (numeric*(1+gamma.tanh())+beta.tanh()).to(original.dtype)
        original = original.index_copy(-1,self.numeric_indices,replacement)
        return self.original(original,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices,
                 numeric_indices, conditioning_indices):
        super().__init__(206,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0] = OriginalInputs(self.members[0])
        self.members[1] = GapCalibratedInputs(self.members[1],numeric_indices)
