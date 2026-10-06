"""Joint SW decision surrogate over the unchanged official candidate actions.

The predictor remains the v11 shared two-TCN model. Only the training objective
changes: default mass and conditional density jointly determine action utility.
No validation labels, selected actions, or other-target predictions enter loss.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from independent_sw_separation_v11 import Network

MODES = ('likelihood', 'joint_utility')
TEMPERATURE = 0.05
UTILITY_WEIGHT = 0.5


class DecisionObjective(nn.Module):
    def __init__(self, actions, utilities, target_median, target_scale, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        a = torch.as_tensor(actions, dtype=torch.float32).clone()
        u = torch.as_tensor(utilities, dtype=torch.float32).detach().clone()
        if a.ndim != 1 or u.shape != (257, len(a)):
            raise ValueError('Expected A actions and257xA default-plus-density utility matrix')
        if not bool(torch.isfinite(a).all() and torch.isfinite(u).all()):
            raise ValueError('Nonfinite actions/utilities')
        self.register_buffer('actions', a)
        self.register_buffer('utilities', u)
        self.target_median = float(target_median)
        self.target_scale = float(target_scale)
        self.alpha = 0.0 if mode == 'likelihood' else UTILITY_WEIGHT
        self.temperature = TEMPERATURE

    def forward(self, model, output, labels, default, mask, quality):
        gate, logits = output
        gl = gate[mask][:, 0].float()
        lp = logits[mask].float().log_softmax(-1)
        d = default[mask][:, 2]
        z = labels[mask][:, 2].float()
        q = quality[mask].detach().float()
        lo, hi, frac = model.soft_targets(z.masked_fill(d, 0))
        ce = -((1-frac)*lp.gather(1, lo[:, None]).squeeze(1)
               + frac*lp.gather(1, hi[:, None]).squeeze(1))
        bce = F.binary_cross_entropy_with_logits(gl, d.float(), reduction='none')
        numeric = (~d).float()
        likelihood = (q*(bce+numeric*ce)).mean()/3
        g = gl.sigmoid()
        p = lp.exp()
        joint = torch.cat([g[:, None], (1-g[:, None])*p], dim=-1)
        # Same257support points and all candidate actions as inference. This
        # soft action selection is used in training only; inference keeps argmax.
        predicted_utility = joint @ self.utilities
        action_probability = (predicted_utility/self.temperature).softmax(-1)
        y = torch.exp(z*self.target_scale+self.target_median)
        y = torch.where(d, torch.full_like(y, 99.9), y)
        true_utility = (1-(self.actions[None]-y[:, None]).abs()/
                        (.05*(y.abs()[:, None]+1e-5))).clamp_min(0)
        expected_reward = (action_probability*true_utility).sum(-1)
        decision_loss = (q*(1-expected_reward)).mean()/3
        loss = likelihood+self.alpha*decision_loss
        # Computing the same surrogate in both arms makes the diagnostics and
        # arithmetic comparable. Alpha0 contributes exactly zero gradient.
        stats = torch.stack([likelihood.detach(), (q*bce).mean().detach(),
             ((q*numeric*ce).sum()/(q*numeric).sum().clamp_min(1)).detach(),
             decision_loss.detach(), (q*expected_reward).sum().detach()/q.sum().clamp_min(1)])
        return loss, stats, decision_loss


def verify_loss_paths(decision_loss, output):
    """Inspect the actual first training batch, without an optimizer update."""
    grads = torch.autograd.grad(decision_loss, output, retain_graph=True, allow_unused=False)
    names = ('gate_logit', 'density_logits')
    proof = {}
    for name, grad in zip(names, grads):
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError('Nonfinite joint objective gradient')
        value = float(grad.detach().float().norm())
        if value <= 0:
            raise RuntimeError('Joint objective does not reach '+name)
        proof[name+'_gradient_norm'] = value
    return dict(proof, used_training_labels_only=True,
                mask='original training labelled center128points',
                includes_default_and_continuous_actions=True,
                inference_argmax_unchanged=True)
