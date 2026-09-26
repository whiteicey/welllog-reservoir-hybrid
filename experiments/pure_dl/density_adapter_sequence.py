"""Separate late continuous-density representation from the default-state gate."""
import torch
from torch import nn
from torch.nn import functional as F
from patch_hierarchy_sequence import Network as Baseline

class DensityOnlyHead(nn.Module):
    def __init__(self,old):
        super().__init__();self.gate=old.gate;self.density=old.density
        width=self.density.in_channels
        self.norm=nn.LayerNorm(width)
        self.input=nn.Conv1d(width,32,1)
        self.context=nn.Conv1d(32,32,7,padding=3,groups=32)
        self.output=nn.Conv1d(32,width,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x):
        z=self.norm(x.transpose(1,2)).transpose(1,2)
        correction=self.output(F.gelu(self.context(F.gelu(self.input(z)))))
        gate=self.gate(x).transpose(1,2)
        raw=self.density(x+correction).transpose(1,2).reshape(len(x),-1,3,3,8)
        return gate,raw

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.main_head=DensityOnlyHead(self.main_head)
