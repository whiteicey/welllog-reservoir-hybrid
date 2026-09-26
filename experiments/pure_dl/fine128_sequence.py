"""Widen the highest-resolution encoder/decoder path from64 to128 channels."""
import torch
from torch import nn
from torch.nn import functional as F
from global_cosine_sequence import Network as Baseline
from relative_only_sequence import RelativeOnlyStem
from patch_hierarchy_sequence import PatchHierarchy
from depthwise_norm_sequence import DepthNorm
from blur_down_sequence import BlurDown


class FineBlock(nn.Module):
    def __init__(self):
        super().__init__();channels=128
        self.norm=DepthNorm(channels,eps=1e-5)
        self.depthwise=nn.Conv1d(channels,channels,7,padding=3,groups=channels)
        self.expand=nn.Conv1d(channels,channels,1)
        self.project=nn.Conv1d(channels,channels,1)
        self.dropout=nn.Dropout(0.)
        self.gain=nn.Parameter(torch.full((1,channels,1),.1))
    def forward(self,x):
        return x+self.gain*self.project(self.dropout(.5*F.gelu(self.expand(self.depthwise(self.norm(x))))))


class FineHierarchy(PatchHierarchy):
    def __init__(self,old):
        nn.Module.__init__(self);self.patch_size=16
        self.norm=nn.LayerNorm(128)
        self.input=nn.Linear(128,32,bias=False)
        self.local=old.local;self.global_patches=old.global_patches
        self.output=nn.Linear(32,128,bias=False)
        nn.init.zeros_(self.output.weight)


class FineStem(RelativeOnlyStem):
    def __init__(self,features,old_hierarchy):
        super().__init__(features+2,128,1)
        self.hierarchy=FineHierarchy(old_hierarchy)
    def forward(self,x):
        stem=super().forward(x)
        return stem+self.hierarchy(stem,x[:,-2,:]>0)


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):
            self.stem=FineStem(features,self.stem.hierarchy)
            self.enc0=FineBlock()
            self.down1=BlurDown(nn.Conv1d(128,96,4,stride=2,padding=1))
            self.merge0=nn.Conv1d(96+128,128,1)
            self.dec0=FineBlock()
            self.combine[0]=nn.Conv1d(128,128,1)
