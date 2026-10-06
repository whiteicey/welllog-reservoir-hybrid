"""Small zero-initialized correction; original default decisions remain fixed."""
import torch
from torch import nn


class NumericPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(212, 32), nn.GELU(), nn.Linear(32, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('limits', torch.tensor([.04, .03, .025]))

    def forward(self, inputs):
        return self.net(inputs).tanh() * self.limits


def apply_patch_values(baseline, correction, defaults):
    # Relative POR/SW corrections, log10 PERM correction. No state switches.
    result = torch.stack((baseline[:, 0]*(1+correction[:, 0]),
                          baseline[:, 1]*torch.pow(10., correction[:, 1]),
                          baseline[:, 2]*(1+correction[:, 2])), -1)
    return torch.where(defaults, baseline, result)
