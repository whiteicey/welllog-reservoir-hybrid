"""Add three spatial-variation descriptors without altering initial predictions."""
import torch
from torch import nn
from frozen_numeric_patch import NumericPatch as Original, apply_patch_values


class NumericPatch(Original):
    def __init__(self):
        super().__init__()
        old=self.net[0]
        with torch.random.fork_rng(devices=[]):
            layer=nn.Linear(215,32)
        with torch.no_grad():
            layer.weight.zero_();layer.weight[:,:212].copy_(old.weight);layer.bias.copy_(old.bias)
        self.net[0]=layer
