"""Apply persistent point weights to the completed prediction loss before reduction."""
import math
import torch
from torch.nn import functional as F
from relative_only_sequence import Network


def head_loss(output, target, default, mask, weights):
    gate, raw = output
    if gate.shape[1] != target.shape[1]:
        n = target.shape[1] // gate.shape[1]
        gate = gate.repeat_interleave(n, 1)
        raw = raw.repeat_interleave(n, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    sd = F.softplus(r[:, :, 2]) + .025
    nll = -(r[:, :, 0].log_softmax(-1) - .5*((y[:, :, None]-r[:, :, 1])/sd).square()
            - sd.log() - .5*math.log(2*math.pi)).logsumexp(-1)
    per_point = (F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
                 + nll.masked_fill(d, 0)).mean(-1)
    return (weights[mask].detach().float()*per_point).mean()


def objective(outputs, target, default, mask, deep_supervision=False, weights=None):
    assert weights is not None and weights.shape == mask.shape
    if not deep_supervision:
        return head_loss(outputs, target, default, mask, weights)
    return sum(a*head_loss(o, target, default, mask, weights)
               for a, o in zip([1, .2, .1], outputs))
