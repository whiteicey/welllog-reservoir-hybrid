"""Symmetric, differentiable agreement of overlapping-window distributions."""
import math
import torch
from torch.nn import functional as F


def cdf_signature(output):
    gate, raw = output
    r = raw.float()
    w = r[..., 0, :].softmax(-1)
    mu = r[..., 1, :]
    sd = F.softplus(r[..., 2, :]) + .025
    thresholds = torch.linspace(-3, 3, 17, device=r.device)
    z = (thresholds - mu[..., None]) / sd[..., None]
    normal = torch.special.ndtr(z)
    phi = torch.atan(z[..., 0, :, :])
    por = .5 + (phi + phi.sin() * phi.cos()) / math.pi
    cdf = torch.cat((por.unsqueeze(-3), normal[..., 1:, :, :]), -3)
    return gate.float().sigmoid(), (w[..., None] * cdf).sum(-2)


def agreement(first, second, mask, point_weight):
    a, ca = cdf_signature(first)
    b, cb = cdf_signature(second)
    # Both sides receive gradients. Conditional agreement matters where either
    # view assigns nondefault mass; default probability itself is also matched.
    point = ((a-b).square() + (1-(a+b)/2) * (ca-cb).square().mean(-1)).mean(-1)
    weighted = point * point_weight.detach().float()
    return weighted.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)
