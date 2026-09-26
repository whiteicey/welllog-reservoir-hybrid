"""Dense well-depth U-Net with a pointwise MLP and optional deep supervision.

No learned numerical encoding, label retrieval, or ensemble-member branches.
Every input window belongs to exactly one well.
"""
import json
import math
import torch
from torch import nn
from torch.nn import functional as F

class GatedBlock(nn.Module):

    def __init__(self, channels, dilation=1):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation, groups=channels)
        self.expand = nn.Conv1d(channels, channels * 2, 1)
        self.project = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(0.1)
        self.gain = nn.Parameter(torch.full((1, channels, 1), 0.1))

    def forward(self, x):
        (value, gate) = self.expand(self.depthwise(self.norm(x))).chunk(2, dim=1)
        return x + self.gain * self.project(self.dropout(F.gelu(value) * gate.sigmoid()))

class DensityHead(nn.Module):

    def __init__(self, width, initial_means=None):
        super().__init__()
        self.gate = nn.Conv1d(width, 3, 1)
        self.density = nn.Conv1d(width, 72, 1)
        nn.init.normal_(self.density.weight, 0, 0.005)
        nn.init.zeros_(self.density.bias)
        with torch.no_grad():
            bias = self.density.bias.view(3, 3, 8)
            if initial_means is not None:
                bias[:, 1].copy_(torch.as_tensor(initial_means))
            bias[:, 2].fill_(math.log(math.expm1(0.5 - 0.025)))

    def forward(self, x):
        return (self.gate(x).transpose(1, 2), self.density(x).transpose(1, 2).reshape(len(x), -1, 3, 3, 8))

class Network(nn.Module):

    def __init__(self, features=206, initial_means=None):
        super().__init__()
        self.stem = nn.Conv1d(features + 2, 64, 1)
        self.enc0 = GatedBlock(64)
        self.down1 = nn.Conv1d(64, 96, 4, stride=2, padding=1)
        self.enc1 = GatedBlock(96)
        self.down2 = nn.Conv1d(96, 128, 4, stride=2, padding=1)
        self.enc2 = GatedBlock(128, dilation=2)
        self.down3 = nn.Conv1d(128, 192, 4, stride=2, padding=1)
        self.bottleneck = nn.Sequential(GatedBlock(192, 1), GatedBlock(192, 2), GatedBlock(192, 4))
        self.merge2 = nn.Conv1d(192 + 128, 128, 1)
        self.dec2 = GatedBlock(128)
        self.merge1 = nn.Conv1d(128 + 96, 96, 1)
        self.dec1 = GatedBlock(96)
        self.merge0 = nn.Conv1d(96 + 64, 64, 1)
        self.dec0 = GatedBlock(64)
        self.point = nn.Sequential(nn.Conv1d(features, 128, 1), nn.GELU(), nn.Conv1d(128, 64, 1))
        self.combine = nn.Sequential(nn.Conv1d(128, 128, 1), nn.GELU(), nn.Dropout(0.1))
        self.main_head = DensityHead(128, initial_means)
        self.aux_mid = DensityHead(96, initial_means)
        self.aux_coarse = DensityHead(128, initial_means)

    def forward(self, features, relative_depth, valid, auxiliary=False):
        mask = valid[:, None].to(features.dtype)
        point = features.transpose(1, 2) * mask
        x = torch.cat([point, relative_depth[:, None] * mask, mask], dim=1)
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        h = self.bottleneck(self.down3(e2))
        h = self.dec2(self.merge2(torch.cat([F.interpolate(h, size=e2.shape[-1], mode='linear', align_corners=False), e2], 1)))
        coarse = h
        h = self.dec1(self.merge1(torch.cat([F.interpolate(h, size=e1.shape[-1], mode='linear', align_corners=False), e1], 1)))
        mid = h
        h = self.dec0(self.merge0(torch.cat([F.interpolate(h, size=e0.shape[-1], mode='linear', align_corners=False), e0], 1)))
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        return (main, self.aux_mid(mid), self.aux_coarse(coarse))

def window_indices(centers, lower, upper, length=256, anchors=None):
    assert length % 8 == 0
    anchors = centers if anchors is None else anchors
    index = centers[:, None] + torch.arange(-length // 2, length // 2, device=centers.device)
    valid = (index >= lower[anchors, None]) & (index < upper[anchors, None])
    safe = torch.maximum(lower[anchors, None], torch.minimum(index, upper[anchors, None] - 1))
    return (safe, valid)
