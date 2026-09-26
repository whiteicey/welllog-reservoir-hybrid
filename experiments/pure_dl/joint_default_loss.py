"""Joint current-row categorical gate CE/3 plus unchanged conditional mixture NLL."""
import torch
from torch.nn import functional as F
from por_t3_patch_loss import components, probability


def head_loss(output, target, default, mask, weights):
    gate, raw = output
    assert gate.shape[-1] == 8
    if gate.shape[1] != target.shape[1]:
        repeat = target.shape[1] // gate.shape[1]
        assert repeat * gate.shape[1] == target.shape[1]
        gate, raw = gate.repeat_interleave(repeat, 1), raw.repeat_interleave(repeat, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    nll = -(r[:, :, 0].log_softmax(-1) + components(r, y)).logsumexp(-1)
    state = (d.long() * torch.tensor([1, 2, 4], device=d.device)).sum(-1)
    point = F.cross_entropy(g, state, reduction='none') / 3 + nll.masked_fill(d, 0).mean(-1)
    return (weights[mask].detach().float() * point).mean()


def objective(outputs, target, default, mask, deep_supervision=False, weights=None, *, epoch):
    assert weights is not None and weights.shape == mask.shape
    if not deep_supervision:
        return head_loss(outputs, target, default, mask, weights)
    main, mid, coarse = outputs
    factor = max(0., 1. - epoch / 40.)
    value = head_loss(main, target, default, mask, weights)
    if factor:
        value += factor * (.2 * head_loss(mid, target, default, mask, weights) +
                           .1 * head_loss(coarse, target, default, mask, weights))
    return value
