"""WellNet：整井一维扩张卷积网络（门控头 + 分箱分布头）。"""
import numpy as np
import torch
import torch.nn as nn

from .config import (NN_CHANNELS, NN_DILATIONS, NN_DROPOUT, N_BINS, TARGETS)


class SE(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r), nn.GELU(),
            nn.Linear(ch // r, ch), nn.Sigmoid())

    def forward(self, x):           # (B, C, L)
        s = x.mean(dim=-1)
        return x * self.fc(s).unsqueeze(-1)


class ResBlock(nn.Module):
    def __init__(self, ch, dilation, k=5, dropout=NN_DROPOUT):
        super().__init__()
        pad = dilation * (k // 2)
        self.body = nn.Sequential(
            nn.GroupNorm(8, ch), nn.GELU(),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation),
            nn.GroupNorm(8, ch), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation))
        self.se = SE(ch)

    def forward(self, x):
        return x + self.se(self.body(x))


class WellNet(nn.Module):
    """输出 dict: 每目标 (gate logit (B,L), bin logits (B,K,L))。"""

    def __init__(self, n_in, ch=NN_CHANNELS):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 7, padding=3)
        self.blocks = nn.Sequential(
            *[ResBlock(ch, d) for d in NN_DILATIONS])
        self.ctx = nn.Sequential(
            nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, ch))
        self.mix = nn.Sequential(nn.Conv1d(ch * 2, ch, 1), nn.GELU())
        self.heads = nn.ModuleDict(
            {t: nn.Conv1d(ch, 1 + N_BINS[t], 1) for t in TARGETS})

    def forward(self, x):           # (B, C, L)
        h = self.blocks(self.inp(x))
        g = self.ctx(h.mean(dim=-1)).unsqueeze(-1).expand_as(h)
        h = self.mix(torch.cat([h, g], dim=1))
        out = {}
        for t in TARGETS:
            o = self.heads[t](h)
            out[t] = (o[:, 0], o[:, 1:])
        return out
