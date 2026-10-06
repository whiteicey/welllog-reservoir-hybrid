"""Low-capacity bounded linear calibration on frozen block summaries."""
import torch
from torch import nn
from frozen_numeric_patch import apply_patch_values


class NumericPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Linear(212,3)
        nn.init.zeros_(self.net.weight);nn.init.zeros_(self.net.bias)
        self.register_buffer('limits',torch.tensor([.04,.03,.025]))

    def forward(self,inputs):return self.net(inputs).tanh()*self.limits
