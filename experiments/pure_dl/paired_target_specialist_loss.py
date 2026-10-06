"""Complementary target priorities across two experts; total average stays one."""
import torch
from torch.nn import functional as F

from por_t3_patch_loss import components


TARGET_WEIGHTS = ((1.25, 1.25, .5), (.75, .75, 1.5))


def head_loss(output, target, default, mask, point_weights, channel_weights):
    gate, raw = output
    if gate.shape[1] != target.shape[1]:
        repeat = target.shape[1] // gate.shape[1]
        gate, raw = gate.repeat_interleave(repeat, 1), raw.repeat_interleave(repeat, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    nll = -(r[:, :, 0].log_softmax(-1) + components(r, y)).logsumexp(-1)
    channels = F.binary_cross_entropy_with_logits(g, d.float(), reduction='none') + nll.masked_fill(d, 0)
    factors = torch.as_tensor(channel_weights, device=g.device, dtype=channels.dtype)
    point = (channels * factors[None]).mean(-1)
    return (point_weights[mask].detach().float() * point).mean()


def member_objective(outputs, target, default, mask, point_weights, channel_weights, *, epoch):
    main, mid, coarse = outputs
    factor = max(0., 1. - epoch/40.)
    result = head_loss(main, target, default, mask, point_weights, channel_weights)
    if factor:
        result = result + factor * (
            .2 * head_loss(mid, target, default, mask, point_weights, channel_weights)
            + .1 * head_loss(coarse, target, default, mask, point_weights, channel_weights)
        )
    return result


def paired_objective(outputs, target, default, mask, deep_supervision, weights, *, epoch):
    assert deep_supervision and len(outputs) == 2
    return sum(member_objective(output, target, default, mask, weights, TARGET_WEIGHTS[index], epoch=epoch)
               for index, output in enumerate(outputs)) / 2
