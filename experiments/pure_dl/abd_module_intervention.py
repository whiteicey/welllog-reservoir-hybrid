"""Explicit architecture intervention for the A+B+D module study."""
import torch
from torch import nn


class ZeroAttention(nn.Module):
    def forward(self, features, valid):
        return torch.zeros_like(features)


def disable_full_global_attention(model):
    """Remove only A's full206 branch global context; preserve local attention."""
    hierarchy=model.members[0].stem.hierarchy
    if isinstance(hierarchy.global_patches,ZeroAttention):
        raise ValueError('Global attention already removed')
    hierarchy.global_patches=ZeroAttention()
    return model
