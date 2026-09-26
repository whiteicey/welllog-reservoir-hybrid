"""Rank-ordered mixture locations, with weights/scales tied to rank slots."""
import torch
from torch import nn
from patch_hierarchy_sequence import Network as Baseline

class OrderedHead(nn.Module):
    def __init__(self,old):
        super().__init__();self.gate=old.gate;self.density=old.density
    def forward(self,x):
        gate=self.gate(x).transpose(1,2)
        raw=self.density(x).transpose(1,2).reshape(len(x),-1,3,3,8)
        ordered=raw[:,:,:,1,:].sort(dim=-1).values
        return gate,torch.stack([raw[:,:,:,0,:],ordered,raw[:,:,:,2,:]],dim=3)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        for name in ['main_head','aux_mid','aux_coarse']:setattr(self,name,OrderedHead(getattr(self,name)))
