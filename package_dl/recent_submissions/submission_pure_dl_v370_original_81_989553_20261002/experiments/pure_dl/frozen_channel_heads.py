"""Independent channel heads on frozen B features and original frozen routes."""
import torch
from torch import nn
from torch.nn import functional as F
from paired_detached_router_sequence import routed_merge


class ChannelHead(nn.Module):
    def __init__(self, original, channel):
        super().__init__()
        # Copy original output rows without consuming RNG or changing capacity.
        j = channel
        self.gate_weight = nn.Parameter(original.gate.weight[j:j+1].detach().clone())
        self.gate_bias = nn.Parameter(original.gate.bias[j:j+1].detach().clone())
        self.density_weight = nn.Parameter(original.density.weight[24*j:24*(j+1)].detach().clone())
        self.density_bias = nn.Parameter(original.density.bias[24*j:24*(j+1)].detach().clone())


class FrozenBackboneHeads(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.backbone = original.eval().requires_grad_(False)
        self.heads = nn.ModuleList([nn.ModuleList([ChannelHead(member.main_head, j)
                                                  for member in original.members]) for j in range(3)])
        self.captured = {}
        self.handles = [member.main_head.register_forward_pre_hook(self.capture(i))
                        for i, member in enumerate(self.backbone.members)]

    def capture(self, index):
        def hook(module, inputs):
            self.captured[index] = inputs[0].detach()
        return hook

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        assert not auxiliary
        self.captured.clear()
        with torch.no_grad():
            original_outputs = [member(features, relative_depth, valid, absolute_depth=absolute_depth)
                                for member in self.backbone.members]
            # Route input remains the ORIGINAL frozen heads, never new trainable heads.
            logits = self.backbone.route_logits(features, valid, original_outputs)
        outputs = []
        for branch in range(2):
            h = self.captured[branch]
            heads = [self.heads[j][branch] for j in range(3)]
            # Same convolution output shapes as original model: exact function at initialization.
            gate = F.conv1d(h, torch.cat([m.gate_weight for m in heads]),
                           torch.cat([m.gate_bias for m in heads])).transpose(1, 2)
            raw = F.conv1d(h, torch.cat([m.density_weight for m in heads]),
                          torch.cat([m.density_bias for m in heads])).transpose(1, 2)
            outputs.append((gate, raw.reshape(len(h), -1, 3, 3, 8)))
        self.captured.clear()
        return routed_merge(outputs, logits)
