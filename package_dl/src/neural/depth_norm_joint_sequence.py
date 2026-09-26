"""Combine physical depth input with channel-only per-depth normalization."""
import torch
from .absolute_depth_sequence import Network as DepthBaseline
from .multiscale_sequence import GatedBlock
from .depthwise_norm_sequence import DepthNorm

class Network(DepthBaseline):

    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        for module in self.modules():
            if isinstance(module, GatedBlock):
                old = module.norm
                module.norm = DepthNorm(old.num_channels, eps=old.eps)
                module.norm.load_state_dict(old.state_dict())
