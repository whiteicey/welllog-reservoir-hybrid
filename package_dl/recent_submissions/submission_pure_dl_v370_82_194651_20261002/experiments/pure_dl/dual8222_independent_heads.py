"""Independent target readouts on the retained 80.221861 local-router dual model.

The two candidate residual structures have exactly the same parameter count.
Both preserve original gate/density rows and start at the original function.
No external ABD model, original-output residual vote, or trainable router is used.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from frozen_channel_heads import ChannelHead
from paired_detached_router_sequence import routed_merge


class ResidualReadout(ChannelHead):
    """Copied linear rows plus a zero-output nonlinear correction in head space."""

    def __init__(self, original: nn.Module, channel: int, architecture: str):
        super().__init__(original, channel)
        if architecture not in {"pointwise", "local5"}:
            raise ValueError(f"Unknown head architecture: {architecture}")
        self.architecture = architecture
        width = 80 if architecture == "pointwise" else 77
        # 154*80+25 == (154+6)*77+25 == 12345 residual parameters.
        self.reduce = nn.Conv1d(128, width, 1)
        self.context = (
            nn.Conv1d(width, width, 5, padding=2, groups=width, bias=True)
            if architecture == "local5" else None
        )
        if self.context is not None:
            nn.init.zeros_(self.context.weight)
            nn.init.zeros_(self.context.bias)
        self.expand = nn.Conv1d(width, 25, 1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)
        count = sum(p.numel() for p in self.parameters())
        assert count == 15570, count  # 3225 copied + 12345 residual.

    def residual(self, feature: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """feature: B,128,L; valid: B,L; result: B,25,L."""
        hidden = F.gelu(self.reduce(feature))
        hidden = hidden.masked_fill(~valid[:, None], 0)
        if self.context is not None:
            hidden = hidden + self.context(hidden).masked_fill(~valid[:, None], 0)
        return self.expand(hidden).masked_fill(~valid[:, None], 0)


class Network(nn.Module):
    """Three independent head groups, each with one readout for each expert."""

    def __init__(self, original: nn.Module, architecture: str):
        super().__init__()
        if len(original.members) != 2:
            raise ValueError("This study requires exactly the original two experts")
        self.backbone = original.eval().requires_grad_(False)
        # Fork construction RNG so head widths cannot change training-window RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.heads = nn.ModuleList([
                nn.ModuleList([
                    ResidualReadout(member.main_head, channel, architecture)
                    for member in original.members
                ]) for channel in range(3)
            ])
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = [
            member.main_head.register_forward_pre_hook(self._capture(index))
            for index, member in enumerate(original.members)
        ]
        self.architecture = architecture
        assert sum(p.numel() for p in self.heads.parameters()) == 93420

    def _capture(self, index: int):
        def hook(_module, inputs):
            self.captured[index] = inputs[0].detach()
        return hook

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        if auxiliary:
            raise ValueError("Frozen-backbone readout study has no restarted auxiliary loss")
        self.captured.clear()
        with torch.no_grad():
            for member in self.backbone.members:
                # Source heads execute only to capture their inputs; outputs are discarded.
                member(features, relative_depth, valid, absolute_depth=absolute_depth)
            # This source has an INPUT-only local router, independent of new output heads.
            logits = self.backbone.route_logits(features, valid)
        outputs = []
        for branch in range(2):
            h = self.captured[branch]
            target_heads = [self.heads[c][branch] for c in range(3)]
            # Keep original convolution shapes/order for source-function initialization.
            gate = F.conv1d(h, torch.cat([m.gate_weight for m in target_heads]),
                           torch.cat([m.gate_bias for m in target_heads]))
            density = F.conv1d(h, torch.cat([m.density_weight for m in target_heads]),
                              torch.cat([m.density_bias for m in target_heads]))
            corrections = [m.residual(h, valid) for m in target_heads]
            gate = gate + torch.cat([z[:, :1] for z in corrections], 1)
            density = density + torch.cat([z[:, 1:] for z in corrections], 1)
            gate = gate.transpose(1, 2)
            raw = density.transpose(1, 2).reshape(len(h), -1, 3, 3, 8)
            outputs.append((gate, raw))
        self.captured.clear()
        # Only the two INTERNAL experts are merged. No B43/D52 or output-stage patch.
        return routed_merge(outputs, logits)
