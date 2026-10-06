"""Per-point well-relative input calibration on paired68 view."""
import torch
from torch import nn
from paired_backbone_sequence import Network as Baseline, objective, merge_predictions
from paired_well_view_sequence import WellViewMember, retained_indices, VIEW_NAMES, RAW_CURVES
from paired_point_view_sequence import VIEW_NAMES as POINT_NAMES


def calibration_indices(whitelist):
    retained = retained_indices(whitelist)
    conditioning_names = [c+'_'+s for c in RAW_CURVES for s in ('WELL_RANK','WELL_ZIQR','WELL_QNORM')]
    numeric_names = [n for n in POINT_NAMES if not n.endswith('_MISSING') and n!='DEPTH_REL']
    assert len(numeric_names)==len(set(numeric_names))==31
    assert len(conditioning_names)==len(set(conditioning_names))==27
    assert all(whitelist.count(n)==1 for n in numeric_names+conditioning_names)
    numeric = [whitelist.index(n) for n in numeric_names]
    conditioning = [whitelist.index(n) for n in conditioning_names]
    assert set(numeric).isdisjoint(conditioning)
    assert set(numeric+conditioning).issubset(retained)
    return numeric,conditioning


class CalibratedMember(WellViewMember):
    def __init__(self, original, retained, features, numeric, conditioning):
        super().__init__(original,retained,features)
        assert len(set(numeric))==31 and len(set(conditioning))==27
        assert set(numeric).isdisjoint(conditioning)
        assert set(numeric+conditioning).issubset(retained)
        self.register_buffer('numeric_indices',torch.tensor(numeric,dtype=torch.long))
        self.register_buffer('conditioning_indices',torch.tensor(conditioning,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            self.calibration=nn.Sequential(nn.Linear(27,32),nn.GELU(),nn.Linear(32,62))
            nn.init.zeros_(self.calibration[-1].weight)
            nn.init.zeros_(self.calibration[-1].bias)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        clean=features.masked_fill(~valid[...,None],0)
        context=clean.index_select(-1,self.conditioning_indices)
        gamma,beta=self.calibration(context).chunk(2,-1)
        numeric=clean.index_select(-1,self.numeric_indices)
        replacement=(numeric*(1+gamma.tanh())+beta.tanh()).masked_fill(~valid[...,None],0).to(clean.dtype)
        updated=clean.index_copy(-1,self.numeric_indices,replacement)
        return super().forward(updated,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features,initial_means)
        self.members[1]=CalibratedMember(self.members[1],retain_indices,features,numeric_indices,conditioning_indices)
