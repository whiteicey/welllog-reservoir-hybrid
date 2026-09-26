"""Kernel3 depthwise residual convolutions on the frozen global-cosine backbone."""
import torch
from torch import nn
from global_cosine_sequence import Network as Baseline
from blur_ungated_sequence import UngatedBlock
from multiscale_sequence import GatedBlock


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        count = 0
        # One fork retains sequential random draws between the nine new layers
        # while preserving the outer fixed-seed sampling RNG state.
        with torch.random.fork_rng(devices=[]):
            for block in self.modules():
                if isinstance(block, (UngatedBlock, GatedBlock)):
                    old = block.depthwise
                    dilation = old.dilation[0]
                    block.depthwise = nn.Conv1d(
                        old.in_channels, old.out_channels, 3,
                        stride=old.stride, padding=dilation, dilation=dilation,
                        groups=old.groups, bias=old.bias is not None,
                        padding_mode=old.padding_mode,
                        device=old.weight.device, dtype=old.weight.dtype)
                    count += 1
        assert count == 9
