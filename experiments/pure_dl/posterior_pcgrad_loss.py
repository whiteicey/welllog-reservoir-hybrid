"""Target-wise likelihoods and PCGrad; no data loading or job launching."""
import torch
from torch.nn import functional as F
from por_t3_patch_loss import components


def channel_head(output, target, default, mask, weights):
    gate, raw = output
    if gate.shape[1] != target.shape[1]:
        repeat = target.shape[1] // gate.shape[1]
        gate = gate.repeat_interleave(repeat, 1)
        raw = raw.repeat_interleave(repeat, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    nll = -(r[:, :, 0].log_softmax(-1) + components(r, y)).logsumexp(-1)
    point = F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
    point = point + nll.masked_fill(d, 0)
    return (weights[mask].detach().float()[:, None] * point).mean(0)


def channel_objective(outputs, target, default, mask, weights, epoch):
    total = 0
    factor = max(0., 1. - epoch / 40.)
    for main, mid, coarse in outputs:
        loss = channel_head(main, target, default, mask, weights)
        if factor:
            loss = loss + factor * (
                .2 * channel_head(mid, target, default, mask, weights)
                + .1 * channel_head(coarse, target, default, mask, weights))
        total = total + loss / len(outputs)
    return total


def projected_backward(losses, parameters, generator):
    """Project full expert parameter vectors; keep unused parameters grad=None.

    Includes expert heads, not just shared trunk. Average task gradients to
    preserve the original loss scale when there are no conflicts. Router loss
    is backpropagated separately by the caller. Generator is CPU-only and
    independent of the training-window RNG.
    """
    gradients = []
    used = [False] * len(parameters)
    for i in range(3):
        values = torch.autograd.grad(
            losses[i], parameters, retain_graph=i < 2, allow_unused=True)
        used = [u or v is not None for u, v in zip(used, values)]
        gradients.append(torch.cat([
            (torch.zeros_like(p) if v is None else v).reshape(-1)
            for p, v in zip(parameters, values)]))
    original = torch.stack(gradients)
    projected = original.clone()
    conflicts = original.new_zeros(())
    for i in range(3):
        for j in torch.randperm(3, generator=generator).tolist():
            if i == j:
                continue
            dot = torch.dot(projected[i], original[j])
            conflicts += (dot < 0).float()
            projected[i] -= (dot.clamp(max=0)
                             / original[j].square().sum().clamp_min(1e-20)
                             * original[j])
    merged = projected.mean(0)
    offset = 0
    for p, is_used in zip(parameters, used):
        size = p.numel()
        p.grad = merged[offset:offset + size].view_as(p) if is_used else None
        offset += size
    return conflicts.detach() / 6
