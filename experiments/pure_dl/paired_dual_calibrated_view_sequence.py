"""Calibrate the same31 numeric columns independently in both feature-view members."""
import torch
from torch import nn
from global_cosine_sequence import Network as Member
from paired_calibrated_view_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES


class FullCalibratedMember(Member):
    def __init__(self, original, features, numeric, conditioning):
        nn.Module.__init__(self)
        for name, child in original.named_children():
            self.add_module(name,child)
        for name,parameter in original.named_parameters(recurse=False):
            self.register_parameter(name,parameter)
        for name,buffer in original.named_buffers(recurse=False):
            self.register_buffer(name,buffer)
        assert len(set(numeric))==31 and len(set(conditioning))==27
        assert set(numeric).isdisjoint(conditioning)
        assert all(0<=i<features for i in numeric+conditioning)
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
        # No feature-view masking here: every other175 original columns remain.
        return super().forward(updated,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        # Finish BOTH baseline backbones and the existing member1 adapter first.
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0]=FullCalibratedMember(self.members[0],features,numeric_indices,conditioning_indices)
