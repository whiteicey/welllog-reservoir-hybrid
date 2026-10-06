"""Train-prior SW hurdle gate balance; original density NLL and point Q."""
import torch
from torch.nn import functional as F

from por_t3_patch_loss import components


def head_loss(output, target, default, mask, weights, *, sw_gate_weights):
    gate, raw = output
    if gate.shape[1] != target.shape[1]:
        repeat = target.shape[1] // gate.shape[1]
        gate, raw = gate.repeat_interleave(repeat, 1), raw.repeat_interleave(repeat, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    nll = -(r[:, :, 0].log_softmax(-1) + components(r, y)).logsumexp(-1)
    gate_bce = F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
    default_weight, nondefault_weight = sw_gate_weights
    sw_bce = torch.where(d[:, 2:3], gate_bce[:, 2:3] * default_weight,
                         gate_bce[:, 2:3] * nondefault_weight)
    gate_bce = torch.cat([gate_bce[:, :2], sw_bce], -1)
    point = (gate_bce + nll.masked_fill(d, 0)).mean(-1)
    return (weights[mask].detach().float() * point).mean()


def member_objective(outputs, target, default, mask, deep_supervision, weights, *, epoch, sw_gate_weights):
    if not deep_supervision:
        return head_loss(outputs, target, default, mask, weights, sw_gate_weights=sw_gate_weights)
    main, mid, coarse = outputs
    factor = max(0., 1. - epoch/40.)
    result = head_loss(main, target, default, mask, weights, sw_gate_weights=sw_gate_weights)
    if factor:
        result = result + factor * (
            .2 * head_loss(mid, target, default, mask, weights, sw_gate_weights=sw_gate_weights)
            + .1 * head_loss(coarse, target, default, mask, weights, sw_gate_weights=sw_gate_weights)
        )
    return result


def paired_objective(outputs, target, default, mask, deep_supervision, weights, *, epoch, sw_gate_weights):
    assert len(outputs) == 2 and deep_supervision
    return sum(member_objective(output, target, default, mask, deep_supervision, weights,
                                epoch=epoch, sw_gate_weights=sw_gate_weights)
               for output in outputs) / 2
