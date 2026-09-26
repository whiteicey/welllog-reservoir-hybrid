"""Per-depth channel LayerNorm replaces nine spatial GroupNorm modules."""
import torch
from torch import nn
from .multiscale_sequence import Network as Baseline, GatedBlock

class DepthNorm(nn.LayerNorm):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class Network(Baseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        for module in self.modules():
            if isinstance(module, GatedBlock):
                original = module.norm
                module.norm = DepthNorm(original.num_channels, eps=original.eps)
                module.norm.load_state_dict(original.state_dict())
