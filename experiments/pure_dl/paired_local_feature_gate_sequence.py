"""Learn point-specific use of each curve's local statistics in the full expert."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES
from paired_well_view_sequence import RAW_CURVES

LOCAL_SUFFIXES = ('RESID_MEAN_9','RESID_MEDIAN_9','ROLL_RANGE_9',
                  'RESID_MEAN_51','RESID_MEDIAN_51','ROLL_RANGE_51',
                  'RESID_MEAN_201','RESID_MEDIAN_201','SLOPE_5','SLOPE_20')


def feature_gate_indices(whitelist):
    assert len(whitelist) == len(set(whitelist)) == 206
    local, context = [], []
    for curve in RAW_CURVES:
        local_names = [curve + '_' + suffix for suffix in LOCAL_SUFFIXES]
        context_names = [curve, curve+'_MISSING', curve+'_WELL_RANK',
                         curve+'_WELL_ZIQR', curve+'_WELL_QNORM'] + local_names
        assert all(whitelist.count(n) == 1 for n in context_names)
        local.append([whitelist.index(n) for n in local_names])
        context.append([whitelist.index(n) for n in context_names])
    return local, context


class LocalFeatureGateMember(nn.Module):
    def __init__(self, original, local_indices, gate_indices):
        super().__init__()
        self.original = original
        self.register_buffer('local_indices',torch.tensor(local_indices).flatten())
        self.register_buffer('gate_indices',torch.tensor(gate_indices).flatten())
        with torch.random.fork_rng(devices=[]):
            self.curve_embedding = nn.Parameter(torch.zeros(9,4))
            nn.init.normal_(self.curve_embedding,std=.02)
            self.selector = nn.Sequential(nn.Linear(19,16),nn.GELU(),nn.Linear(16,1))
            nn.init.zeros_(self.selector[-1].weight)
            nn.init.zeros_(self.selector[-1].bias)

    def forward(self,features,relative_depth,valid,auxiliary=False,absolute_depth=None):
        clean=features.masked_fill(~valid[...,None],0)
        b,l,_=clean.shape
        context=clean.index_select(-1,self.gate_indices).reshape(b,l,9,15)
        embedding=self.curve_embedding.to(context.dtype)[None,None].expand(b,l,-1,-1)
        gain=1+self.selector(torch.cat([context,embedding],-1)).tanh()
        local=clean.index_select(-1,self.local_indices).reshape(b,l,9,10)
        replacement=(local*gain).flatten(-2).to(clean.dtype)
        updated=clean.index_copy(-1,self.local_indices,replacement)
        return self.original(updated,relative_depth,valid,auxiliary,absolute_depth)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,
                 conditioning_indices,local_indices,gate_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,
                         numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0]=LocalFeatureGateMember(self.members[0],local_indices,gate_indices)
