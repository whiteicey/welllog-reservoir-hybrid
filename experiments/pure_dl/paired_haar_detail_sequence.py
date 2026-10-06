"""Preserve signed fine-scale details alongside the first blurred downsample."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES


class DetailDown(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        conv = original.conv
        with torch.random.fork_rng(devices=[]):
            self.detail = nn.Conv1d(conv.in_channels, conv.out_channels, 1, bias=False)
            nn.init.zeros_(self.detail.weight)

    def forward(self, x):
        # Fixed even/odd pairs within each well window; no target-derived signal.
        detail = (x[..., 0::2] - x[..., 1::2]) * 0.5
        return self.original(x) + self.detail(detail)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[0].down1 = DetailDown(self.members[0].down1)
