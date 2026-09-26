"""Zero-initialized per-raw-curve temporal residual on the paired41 view."""
import torch
from torch import nn
from paired_backbone_sequence import Network as Baseline, objective, merge_predictions
from paired_point_view_sequence import PointViewMember, retained_indices, RAW_CURVES, VIEW_NAMES


def raw_indices(whitelist):
    assert len(whitelist) == len(set(whitelist)) == 206
    assert all(whitelist.count(name) == 1 for name in RAW_CURVES)
    indices = [whitelist.index(name) for name in RAW_CURVES]
    assert [whitelist[i] for i in indices] == list(RAW_CURVES)
    return indices


class CurveFilterMember(PointViewMember):
    def __init__(self, original, indices, features, raw_curve_indices):
        super().__init__(original, indices, features)
        assert len(raw_curve_indices) == len(set(raw_curve_indices)) == 9
        assert set(raw_curve_indices).issubset(indices)
        self.register_buffer('raw_curve_indices',torch.tensor(raw_curve_indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.curve_filter = nn.Conv1d(9,9,9,padding=4,groups=9,bias=False)
            nn.init.zeros_(self.curve_filter.weight)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        clean = features.masked_fill(~valid[...,None],0)
        raw = clean.index_select(-1,self.raw_curve_indices)
        correction = self.curve_filter(raw.transpose(1,2)).transpose(1,2)
        correction = correction.masked_fill(~valid[...,None],0)
        replacement = (raw+correction).to(clean.dtype)
        updated = clean.index_copy(-1,self.raw_curve_indices,replacement)
        return super().forward(updated,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, raw_curve_indices):
        super().__init__(features,initial_means)
        self.members[1] = CurveFilterMember(self.members[1],retain_indices,features,raw_curve_indices)
