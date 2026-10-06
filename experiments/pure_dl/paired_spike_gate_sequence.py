"""Curve-specific reliability modulation; only named standardized raw9 columns change."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_spike_side_sequence import spike_scores, CURVES


def raw_indices_from_schema(whitelist):
    assert len(whitelist)==len(set(whitelist))==206
    assert all(whitelist.count(name)==1 for name in CURVES)
    return [whitelist.index(name) for name in CURVES]


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices, raw_curve_indices):
        assert features==206 and len(set(raw_curve_indices))==9
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.register_buffer('raw_curve_indices',torch.tensor(raw_curve_indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.reliability_gate=nn.Linear(9,9)
            nn.init.zeros_(self.reliability_gate.weight);nn.init.zeros_(self.reliability_gate.bias)

    def corrected(self, features, valid):
        assert features.shape[-1]==215
        original=features[...,:206]
        side=features[...,206:].masked_fill(~valid[...,None],0)
        gate=self.reliability_gate(side).float().tanh()
        raw=original.index_select(-1,self.raw_curve_indices).float()
        replacement=(raw*(1-gate*side.float())).masked_fill(~valid[...,None],0).to(original.dtype)
        return original.index_copy(-1,self.raw_curve_indices,replacement)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        return super().forward(self.corrected(features,valid),relative_depth,valid,auxiliary,absolute_depth)

    def prediction_parts(self, features, relative_depth, valid, absolute_depth=None):
        return super().prediction_parts(self.corrected(features,valid),relative_depth,valid,absolute_depth)
