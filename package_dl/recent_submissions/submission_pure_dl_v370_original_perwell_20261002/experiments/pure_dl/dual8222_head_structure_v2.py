"""Round two: linear control and target-specific cross-expert hidden readout.

All variants retain the original source function at initialization. The cross80
arm combines only the two INTERNAL expert features, not external model votes.
Historical source/model files are immutable.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from frozen_channel_heads import ChannelHead
from paired_detached_router_sequence import routed_merge


class TargetReadout(nn.Module):
    def __init__(self, members: nn.ModuleList, channel: int, architecture: str):
        super().__init__()
        self.branches = nn.ModuleList([ChannelHead(member.main_head, channel) for member in members])
        self.architecture = architecture
        if architecture == 'cross80':
            # A shared target-specific residual sees both unnormalized 128-D features.
            # 256->80->50 = 24,610 parameters versus 24,690 for two separate128->80->25.
            self.reduce = nn.Conv1d(256, 80, 1)
            self.expand = nn.Conv1d(80, 50, 1)
            nn.init.zeros_(self.expand.weight)
            nn.init.zeros_(self.expand.bias)
        elif architecture != 'linear':
            raise ValueError(f'Unknown architecture {architecture}')

    def corrections(self, features: torch.Tensor, valid: torch.Tensor):
        if self.architecture == 'linear':
            return None
        hidden = F.gelu(self.reduce(features)).masked_fill(~valid[:, None], 0)
        return self.expand(hidden).masked_fill(~valid[:, None], 0)


class Network(nn.Module):
    @staticmethod
    def parameter_count(architecture: str) -> int:
        return {'linear':19350, 'cross80':93180}[architecture]

    def __init__(self, original: nn.Module, architecture: str):
        super().__init__()
        if len(original.members) != 2:
            raise ValueError('Exactly two source experts are required')
        self.backbone = original.eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.heads = nn.ModuleList([
                TargetReadout(original.members, c, architecture) for c in range(3)
            ])
        self.architecture = architecture
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = [
            member.main_head.register_forward_pre_hook(self._capture(i))
            for i, member in enumerate(original.members)
        ]
        assert sum(p.numel() for p in self.heads.parameters()) == self.parameter_count(architecture)

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
            raise ValueError('Readout-only study has no restarted auxiliary objective')
        self.captured.clear()
        with torch.no_grad():
            for member in self.backbone.members:
                member(features, relative_depth, valid, absolute_depth=absolute_depth)
            logits = self.backbone.route_logits(features, valid)
        # Tensor shapes: each hidden B,128,L; concat B,256,L; corrections B,50,L.
        combined = torch.cat([self.captured[0], self.captured[1]], dim=1)
        corrections = [head.corrections(combined, valid) for head in self.heads]
        outputs = []
        for e in range(2):
            h = self.captured[e]
            heads = [group.branches[e] for group in self.heads]
            gate = F.conv1d(h, torch.cat([head.gate_weight for head in heads]),
                           torch.cat([head.gate_bias for head in heads]))
            density = F.conv1d(h, torch.cat([head.density_weight for head in heads]),
                              torch.cat([head.density_bias for head in heads]))
            if self.architecture != 'linear':
                # Preserve each expert's own gate and eight density components.
                # This changes their readouts, not the fixed input-only router.
                per_target = [value[:, 25*e:25*(e+1)] for value in corrections]
                gate = gate + torch.cat([value[:, :1] for value in per_target], dim=1)
                density = density + torch.cat([value[:, 1:] for value in per_target], dim=1)
            outputs.append((gate.transpose(1,2), density.transpose(1,2).reshape(len(h),-1,3,3,8)))
        self.captured.clear()
        return routed_merge(outputs, logits)
