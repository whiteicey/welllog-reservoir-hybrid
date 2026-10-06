"""Keep channel common-mode information in full-feature expert normalization."""
import torch
from torch import nn
from depthwise_norm_sequence import DepthNorm
from paired_posterior_router_sequence import (
    Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES)


class AffineDepthRMS(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.eps = original.eps
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())

    def forward(self, x):
        # Normalize channels independently at each depth; no spatial statistics.
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(1, keepdim=True) + self.eps)
        return (z * self.weight[None, :, None]
                + self.bias[None, :, None]).to(x.dtype)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices,
                 numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices,
                         conditioning_indices=conditioning_indices)
        count = 0
        for module in list(self.members[0].modules()):
            for name, child in list(module.named_children()):
                if isinstance(child, DepthNorm):
                    setattr(module, name, AffineDepthRMS(child))
                    count += 1
        assert count == 9, count
