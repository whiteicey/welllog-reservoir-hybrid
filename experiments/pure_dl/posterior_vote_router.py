"""Small trainable vote router over frozen model posterior summaries."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def posterior_summary(gate_logits, raw):
    """Return [point, target, 4] detached-compatible model confidence features."""
    gate = gate_logits.float().sigmoid()
    r = raw.float()
    weights = r[:, :, 0].softmax(-1)
    mu = r[:, :, 1]
    sd = F.softplus(r[:, :, 2])+.025
    mean = (weights*mu).sum(-1)
    variance = (weights*(sd.square()+(mu-mean[..., None]).square())).sum(-1)
    entropy = -(weights*weights.clamp_min(1e-30).log()).sum(-1)/math.log(16)
    return torch.stack([gate, mean.clamp(-8, 8),
                        torch.log1p(variance.clamp_min(0).sqrt()).clamp(max=8),
                        entropy.clamp(0, 1)], dim=-1)


class VoteRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(12), nn.Linear(12, 16), nn.GELU(), nn.Linear(16, 3))
            for _ in range(3)
        ])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, summaries):
        assert summaries.shape[-2:] == (3, 12)
        logits = torch.stack([head(summaries[:, j]) for j, head in enumerate(self.heads)], dim=1)
        return logits.softmax(-1)
