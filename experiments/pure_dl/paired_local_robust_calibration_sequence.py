"""Input-only local robust coordinates for the calibrated expert."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_calibrated_view_sequence import CalibratedMember
from paired_well_view_sequence import WellViewMember, RAW_CURVES
from paired_detached_router_sequence import routed_merge


SCALES = (101, 401)


def local_robust_features(frame):
    columns = []
    for name in RAW_CURVES:
        original = frame[name].to_numpy(np.float64)
        series = pd.Series(np.where(np.isfinite(original), original, np.nan))
        if name in ('RXO', 'RT'):
            series = np.log1p(series.clip(lower=0))
        valid = series.dropna()
        if valid.empty:
            columns.extend([np.zeros(len(frame), np.float32) for _ in SCALES])
            continue
        well_q25, well_median, well_q75 = valid.quantile([.25, .5, .75]).to_numpy()
        floor = max(.1 * (well_q75 - well_q25), 1e-6)
        for width in SCALES:
            rolling = series.rolling(width, center=True, min_periods=3)
            q25 = rolling.quantile(.25).fillna(well_q25).to_numpy()
            q50 = rolling.median().fillna(well_median).to_numpy()
            q75 = rolling.quantile(.75).fillna(well_q75).to_numpy()
            z = (series.to_numpy() - q50) / np.maximum(q75 - q25, floor)
            columns.append((np.clip(np.nan_to_num(z, nan=0., posinf=0., neginf=0.), -4, 4) / 4).astype(np.float32))
    result = np.column_stack(columns)
    assert result.shape == (len(frame), 18) and np.isfinite(result).all()
    return result


class LocalCalibratedMember(CalibratedMember):
    def __init__(self, original):
        nn.Module.__init__(self)
        for name, child in original.named_children():
            self.add_module(name, child)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in original.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        with torch.random.fork_rng(devices=[]):
            self.local_calibration = nn.Linear(18, 62)
            nn.init.zeros_(self.local_calibration.weight)
            nn.init.zeros_(self.local_calibration.bias)

    def forward(self, features, side, relative_depth, valid, auxiliary=False, absolute_depth=None):
        clean = features.masked_fill(~valid[..., None], 0)
        context = clean.index_select(-1, self.conditioning_indices)
        delta = self.calibration(context) + self.local_calibration(side.float().masked_fill(~valid[..., None], 0))
        gamma, beta = delta.chunk(2, -1)
        numeric = clean.index_select(-1, self.numeric_indices)
        replacement = (numeric * (1 + gamma.tanh()) + beta.tanh()).masked_fill(~valid[..., None], 0).to(clean.dtype)
        updated = clean.index_copy(-1, self.numeric_indices, replacement)
        return WellViewMember.forward(self, updated, relative_depth, valid, auxiliary, absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[1] = LocalCalibratedMember(self.members[1])

    def _members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        assert features.shape[-1] == 224
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        side = features[..., 206:]
        return [self.members[0](original, relative_depth, valid, auxiliary, absolute_depth),
                self.members[1](original, side, relative_depth, valid, auxiliary, absolute_depth)], original

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        members, original = self._members(features, relative_depth, valid, False, absolute_depth)
        logits = self.route_logits(original, valid, members)
        return members, logits.float().softmax(-1), routed_merge(members, logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        members, original = self._members(features, relative_depth, valid, auxiliary, absolute_depth)
        main = [output[0] for output in members] if auxiliary else members
        logits = self.route_logits(original, valid, main)
        if auxiliary:
            detached = [tuple(value.detach() for value in output) for output in main]
            return members, routed_merge(detached, logits)
        return routed_merge(main, logits)
