"""Content-conditioned tied-kernel dilation residual in full expert bottleneck."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES

class AdaptiveDilation(nn.Module):
    def __init__(self,original):
        super().__init__(); self.original=original
        c=original.in_channels
        assert c%8==0 and original.groups==c
        self.channels=c
        with torch.random.fork_rng(devices=[]):
            self.route=nn.Conv1d(c,8,1)
            nn.init.zeros_(self.route.weight); nn.init.zeros_(self.route.bias)
            self.strength=nn.Parameter(torch.zeros(()))
    def forward(self,x):
        base=self.original(x)
        wide=F.conv1d(x,self.original.weight,self.original.bias,
                      stride=self.original.stride,padding=2*self.original.padding[0],
                      dilation=2*self.original.dilation[0],groups=self.original.groups)
        route=self.route(x).sigmoid().repeat_interleave(self.channels//8,dim=1)
        return base+self.strength.tanh()*route*(wide-base)

class Network(Baseline):
    def __init__(self,features=206,initial_means=None,*,retain_indices,numeric_indices,conditioning_indices):
        super().__init__(features,initial_means,retain_indices=retain_indices,numeric_indices=numeric_indices,conditioning_indices=conditioning_indices)
        for block in self.members[0].bottleneck:
            block.depthwise=AdaptiveDilation(block.depthwise)
