"""Append directional side features through a zero-init full-expert stem branch."""
import torch
from torch import nn
from directional_depth_features import directional_features, FEATURE_NAMES
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES

class DirectionalStem(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
        with torch.random.fork_rng(devices=[]):
            self.side=nn.Conv1d(24,64,1,bias=False)
            nn.init.zeros_(self.side.weight)
    def forward(self,x):
        # Original forward concatenates230 features, relative depth,valid,absolute.
        assert x.shape[1]==233
        old_input=torch.cat([x[:,:206,:],x[:,-3:,:]],1)
        return self.original(old_input)+self.side(x[:,206:230,:])*(x[:,-2:-1,:]>0)

class OriginalInputs(nn.Module):
    def __init__(self,original):
        super().__init__();self.original=original
    def forward(self,x,relative_depth,valid,auxiliary=False,absolute_depth=None):
        return self.original(x[...,:206],relative_depth,valid,auxiliary,absolute_depth)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,conditioning_indices):
        super().__init__(206,initial_means,retain_indices=retain_indices,numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        self.members[0].stem=DirectionalStem(self.members[0].stem)
        self.members[1]=OriginalInputs(self.members[1])
