"""Separate target branches with matched shared-patch initialization."""
import copy
import torch
from torch import nn
from frozen_profile_patch import NumericPatch as SharedPatch, profile_descriptors, apply_patch_values


class NumericPatch(nn.Module):
    def __init__(self):
        super().__init__()
        original=SharedPatch()
        self.branches=nn.ModuleList()
        with torch.random.fork_rng(devices=[]):
            for j in range(3):
                output=nn.Linear(32,2)
                with torch.no_grad():
                    output.weight.copy_(original.net[-1].weight[j*2:j*2+2])
                    output.bias.copy_(original.net[-1].bias[j*2:j*2+2])
                self.branches.append(nn.Sequential(copy.deepcopy(original.net[0]),nn.GELU(),output))
        self.register_buffer('limits',original.limits.clone())

    def forward(self,inputs):
        coefficient=torch.stack([branch(inputs).tanh() for branch in self.branches],1)
        evidence=inputs[:,-6:].reshape(-1,3,2)
        return (coefficient*evidence).mean(-1)*self.limits
