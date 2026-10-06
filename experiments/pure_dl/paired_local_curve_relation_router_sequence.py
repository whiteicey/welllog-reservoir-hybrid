"""Input-only local cross-curve correlations inform the posterior router."""
import numpy as np
import pandas as pd
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_detached_router_sequence import routed_merge


SCALES = (101, 401)
PAIRS = (('GR', 'RT'), ('CNL', 'DEN'), ('SP', 'GR'), ('RXO', 'RT'))


def local_curve_relations(frame):
    curves = {}
    for name in {name for pair in PAIRS for name in pair}:
        raw = frame[name].to_numpy(np.float64)
        series = pd.Series(np.where(np.isfinite(raw), raw, np.nan))
        if name in ('RT', 'RXO'):
            series = np.log1p(series.clip(lower=0))
        values = series.dropna()
        if values.empty:
            curves[name] = series
        else:
            lo, hi = values.quantile([.01, .99]).to_numpy()
            curves[name] = series.clip(lower=lo, upper=hi)
    columns = []
    for left, right in PAIRS:
        for width in SCALES:
            corr = curves[left].rolling(width, center=True, min_periods=max(8, width//4)).corr(curves[right])
            columns.append(np.clip(np.nan_to_num(corr.to_numpy(), nan=0., posinf=0., neginf=0.), -1, 1).astype(np.float32))
    result = np.column_stack(columns)
    assert result.shape == (len(frame), 8) and np.isfinite(result).all()
    return result


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.relation_router = nn.Linear(8, 6)
            nn.init.zeros_(self.relation_router.weight)
            nn.init.zeros_(self.relation_router.bias)

    def route_logits(self, features, valid, main_outputs):
        assert features.shape[-1] == 214
        base = super().route_logits(features[..., :206], valid, main_outputs)
        side = features[..., 206:].float().masked_fill(~valid[..., None], 0)
        correction = self.relation_router(side).reshape(*features.shape[:2], 3, 2)
        return base + correction.masked_fill(~valid[..., None, None], 0)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        assert features.shape[-1] == 214
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        members = [m(original, relative_depth, valid, absolute_depth=absolute_depth) for m in self.members]
        logits = self.route_logits(features, valid, members)
        return members, logits.float().softmax(-1), routed_merge(members, logits)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        assert features.shape[-1] == 214
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        members = [m(original, relative_depth, valid, auxiliary, absolute_depth) for m in self.members]
        main = [output[0] for output in members] if auxiliary else members
        logits = self.route_logits(features, valid, main)
        if auxiliary:
            detached = [tuple(value.detach() for value in output) for output in main]
            return members, routed_merge(detached, logits)
        return routed_merge(main, logits)
