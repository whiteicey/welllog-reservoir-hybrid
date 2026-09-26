"""Composite likelihood and expected official triangular utility (training labels only)."""
import math
import numpy as np
import torch
from torch.nn import functional as F

def expected_utility(raw, target, med, scale):
    # Layer-cake identity E[(1-error/radius)+] = integral_0^1 P(error < t*radius) dt.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    t = torch.as_tensor((nodes+1)/2, device=raw.device, dtype=raw.dtype)
    qw = torch.as_tensor(weights/2, device=raw.device, dtype=raw.dtype)
    med = torch.as_tensor(med, device=raw.device, dtype=raw.dtype)
    scale = torch.as_tensor(scale, device=raw.device, dtype=raw.dtype)
    transformed = target*scale+med
    por = torch.expm1(transformed[:, 0]).clamp_min(0)
    sw = transformed[:, 2].exp()
    por_radius = .08*(por.abs()+1e-5)
    sw_radius = .05*(sw.abs()+1e-5)
    lo = torch.stack([torch.log1p((por[:, None]-por_radius[:, None]*t).clamp_min(0)),
                      transformed[:, 1, None]-t,
                      (sw[:, None]-sw_radius[:, None]*t).clamp_min(1e-8).log()], 1)
    hi = torch.stack([torch.log1p(por[:, None]+por_radius[:, None]*t),
                      transformed[:, 1, None]+t,
                      (sw[:, None]+sw_radius[:, None]*t).log()], 1)
    lo = (lo-med[None, :, None])/scale[None, :, None]
    hi = (hi-med[None, :, None])/scale[None, :, None]
    mean = raw[:, :, 1, :, None]
    sd = (F.softplus(raw[:, :, 2])+.025)[..., None]
    # CDF difference; reflected erfc on the positive side avoids 1-1 cancellation.
    a, b = (lo[:, :, None]-mean)/sd, (hi[:, :, None]-mean)/sd
    mass = torch.where(a > 0, .5*(torch.erfc(a/math.sqrt(2))-torch.erfc(b/math.sqrt(2))),
                       .5*(torch.erfc(-b/math.sqrt(2))-torch.erfc(-a/math.sqrt(2))))
    return ((mass*qw).sum(-1)*raw[:, :, 0].softmax(-1)).sum(-1).clamp(0, 1)

def head_loss(output, target, default, mask, weights, med, scale):
    gate, raw = output
    if gate.shape[1] != target.shape[1]:
        n = target.shape[1]//gate.shape[1]
        gate, raw = gate.repeat_interleave(n, 1), raw.repeat_interleave(n, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    sd = F.softplus(r[:, :, 2])+.025
    nll = -(r[:, :, 0].log_softmax(-1)-.5*((y[:, :, None]-r[:, :, 1])/sd).square()
            -sd.log()-.5*math.log(2*math.pi)).logsumexp(-1)
    alignment = -expected_utility(r, y, med, scale).clamp_min(1e-6).log()
    per_point = (F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
                 +(nll+.25*alignment*r.new_tensor([0.,1.,1.])).masked_fill(d, 0)).mean(-1)
    return (weights[mask].detach().float()*per_point).mean()

def objective(outputs, target, default, mask, deep_supervision=False, weights=None, *, med, scale):
    assert weights is not None and weights.shape == mask.shape
    if not deep_supervision:
        return head_loss(outputs, target, default, mask, weights, med, scale)
    return sum(a*head_loss(o, target, default, mask, weights, med, scale)
               for a, o in zip([1, .2, .1], outputs))
