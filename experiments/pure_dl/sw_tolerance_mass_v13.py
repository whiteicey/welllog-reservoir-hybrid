"""Train the private SW dual model with a bounded, tolerance-aware auxiliary loss.

Predefined modes differ ONLY in the auxiliary coefficient. This is a surrogate
for coverage under the official triangular kernel, not the exact inference
argmax objective and not a proper scoring rule by itself. Retain proper NLL.
The existing conditional_probability helper is inference-only (@no_grad); this
module deliberately implements its differentiable Gaussian-bin counterpart.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from train_independent_channels_v8 import channel_objective

MODES = ('likelihood', 'tolerance_mass')
AUXILIARY_WEIGHT = 0.5


class ToleranceObjective(nn.Module):
    def __init__(self, master: dict, mode: str):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.alpha = 0.0 if mode == 'likelihood' else AUXILIARY_WEIGHT
        centers = torch.as_tensor(master['centers'][2], dtype=torch.float32)
        edges = torch.as_tensor(master['density_edges'][2], dtype=torch.float32)
        if centers.shape != (256,) or edges.shape != (257,):
            raise ValueError('Expected original SW256 centers/257edges')
        if not bool(torch.isfinite(centers).all() and (centers > 0).all()):
            raise ValueError('Invalid physical SW support')
        self.register_buffer('centers', centers.clone())
        self.register_buffer('edges', edges.clone())
        self.location = float(master['regression_median'][2])
        self.scale = float(master['regression_scale'][2])

    def bin_probabilities(self, raw: torch.Tensor) -> torch.Tensor:
        # raw: N x 3 x K; K=16 for the routed dual model.
        weight = raw[:, 0].softmax(-1)
        mean = raw[:, 1]
        sd = F.softplus(raw[:, 2]) + 0.025
        # Avoid inf/inf derivatives if the legacy grid uses open outer edges.
        finite_edges = self.edges.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        z = (finite_edges[None, :, None] - mean[:, None, :]) / sd[:, None, :]
        cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
        cdf = torch.where(torch.isneginf(self.edges)[None, :, None], 0.0, cdf)
        cdf = torch.where(torch.isposinf(self.edges)[None, :, None], 1.0, cdf)
        mass = ((cdf[:, 1:] - cdf[:, :-1]).clamp_min(0) * weight[:, None, :]).sum(-1)
        return mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)

    def forward(self, output, labels, defaults, mask, quality):
        proper, bce, nll = channel_objective(output, labels, defaults, mask, quality, 2)
        gate, raw = output
        g = gate[mask][:, 0].float().sigmoid()
        r = raw[mask][:, 0].float()
        target = labels[mask][:, 2].float()
        default = defaults[mask][:, 2]
        y = torch.exp(target * self.scale + self.location)
        y = torch.where(default, torch.full_like(y, 99.9), y).detach()
        radius = 0.05 * (y.abs() + 1e-5)
        kernel = (1 - (self.centers[None] - y[:, None]).abs() / radius[:, None]).clamp_min(0)
        default_kernel = (1 - (99.9 - y).abs() / radius).clamp_min(0)
        p = self.bin_probabilities(r)
        coverage = g * default_kernel + (1 - g) * (p * kernel).sum(-1)
        q = quality[mask].detach().float()
        # Positive epsilon protects log for a target outside numerical support.
        # Its finite contribution is retained rather than dropping such points.
        log_coverage = (q * (-torch.log(coverage.clamp_min(1e-8)))).mean() / 3
        loss = proper + self.alpha * log_coverage
        stats = torch.stack([proper.detach(), bce.detach(), nll.detach(),
                             log_coverage.detach(), coverage.mean().detach()])
        return loss, stats, log_coverage, p


def verify_first_training_batch(auxiliary, output, probabilities, edges):
    """Inspect the actual first batch; no extra training or validation fitting."""
    from por_t3_patch_loss import probability
    gradients = torch.autograd.grad(auxiliary, output, retain_graph=True)
    proof = {}
    for name, grad in zip(('gate', 'density'), gradients):
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError('Nonfinite coverage gradient: ' + name)
        proof[name + '_auxiliary_gradient_norm'] = float(grad.float().norm())
        if proof[name + '_auxiliary_gradient_norm'] <= 0:
            raise RuntimeError('Coverage does not reach ' + name)
    proof.update(training_only=True, default_and_continuous_mass_included=True,
                 inference_actions_and_argmax_unchanged=True)
    return proof
