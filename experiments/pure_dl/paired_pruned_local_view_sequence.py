"""Only expert0 omits fixed residual/range/slope engineered inputs."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from global_cosine_sequence import Network as Member

CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
SUFFIXES=tuple('RESID_MEAN_'+str(w) for w in (9,51,201))+tuple('RESID_MEDIAN_'+str(w) for w in (9,51,201))+tuple('ROLL_RANGE_'+str(w) for w in (9,51))+tuple('SLOPE_'+str(w) for w in (5,20))


def removed_indices(whitelist):
    names=[curve+'_'+suffix for curve in CURVES for suffix in SUFFIXES]
    assert len(whitelist)==len(set(whitelist))==206
    assert len(names)==len(set(names))==90
    assert all(whitelist.count(name)==1 for name in names)
    return sorted(whitelist.index(name) for name in names)


class PrunedMember(Member):
    def __init__(self, original, removed, features):
        nn.Module.__init__(self)
        for name,child in original.named_children():self.add_module(name,child)
        for name,parameter in original.named_parameters(recurse=False):self.register_parameter(name,parameter)
        for name,buffer in original.named_buffers(recurse=False):self.register_buffer(name,buffer)
        assert len(removed)==len(set(removed))==90 and all(0<=i<features for i in removed)
        mask=torch.zeros(features,dtype=torch.bool);mask[removed]=True
        self.register_buffer('removed_feature_mask',mask)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        view=features.masked_fill(self.removed_feature_mask[None,None,:],0)
        return super().forward(view,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices, prune_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0]=PrunedMember(self.members[0],prune_indices,features)
