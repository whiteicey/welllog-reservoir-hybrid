"""Input-only whole-well summaries condition each expert's SW default gate."""
import numpy as np
import torch
from torch import nn

from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES
from paired_detached_router_sequence import routed_merge


def well_context_features(frame):
    values = []
    for name in RAW_CURVES:
        raw = frame[name].to_numpy(np.float64)
        finite = np.isfinite(raw)
        curve = raw[finite]
        if name in ('RXO', 'RT'):
            curve = np.log1p(np.maximum(curve, 0))
        q = np.quantile(curve, [.1, .5, .9]) if len(curve) else [np.nan]*3
        values.extend([float(q[0]), float(q[1]), float(q[2]), float(1-finite.mean())])
    result = np.asarray(values, np.float32)
    assert result.shape == (36,)
    return result


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        with torch.random.fork_rng(devices=[]):
            self.sw_well_bias = nn.Sequential(nn.Linear(36, 16), nn.GELU(), nn.Linear(16, 2))
            nn.init.zeros_(self.sw_well_bias[-1].weight)
            nn.init.zeros_(self.sw_well_bias[-1].bias)

    def _members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        assert features.shape[-1] == 242
        original = features[..., :206].to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        context = features[..., 206:].float().masked_fill(~valid[..., None], 0)
        bias = self.sw_well_bias(context).masked_fill(~valid[..., None], 0)
        result = []
        for index, member in enumerate(self.members):
            output = member(original, relative_depth, valid, auxiliary, absolute_depth)
            main = output[0] if auxiliary else output
            gate, raw = main
            sw_gate = (gate[..., 2] + bias[..., index]).to(gate.dtype)
            corrected = (torch.cat([gate[..., :2], sw_gate[..., None]], -1), raw)
            result.append((corrected, output[1], output[2]) if auxiliary else corrected)
        return result, original

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
