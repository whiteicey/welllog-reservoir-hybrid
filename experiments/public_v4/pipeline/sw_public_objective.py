"""Differentiable SW utility objective with public SW=1 semantics and identity target scale."""
import numpy as np
import torch
from torch import nn
from train_scaling_v21 import channel_objective
from sw_head_utility_v80r1 import differentiable_gaussian_probability


class DecisionObjective(nn.Module):
    def __init__(self, evaluator, metadata, mode):
        super().__init__()
        self.alpha = 1.0 if mode == 'a100' else 0.25
        self.temperature = .05
        self.register_buffer('actions', torch.as_tensor(evaluator.actions, dtype=torch.float32))
        self.register_buffer('utilities', evaluator.utilities.detach().clone())
        self.register_buffer('edges', evaluator.edges.detach().clone())
        self.target_median = metadata['sw_target_median']
        self.target_scale = metadata['sw_target_scale']

    def forward(self, output, labels, default, mask, quality):
        likelihood, _, _ = channel_objective(output, labels, default, mask, quality, 1)
        gate, raw = output
        g = gate[mask][:, 0].float().sigmoid()
        r = raw[mask][:, 0].float()
        d = default[mask][:, 1]
        z = labels[mask][:, 1].float()
        q = quality[mask].detach().float()
        conditional = differentiable_gaussian_probability(
            r[:, 0].softmax(-1), r[:, 1], torch.nn.functional.softplus(r[:, 2]) + .025, self.edges)
        joint = torch.cat((g[:, None], (1 - g[:, None]) * conditional), -1)
        predicted_utility = joint @ self.utilities
        action_probability = (predicted_utility / self.temperature).softmax(-1)
        physical_y = z * self.target_scale + self.target_median
        physical_y = torch.where(d, torch.full_like(physical_y, 1.0), physical_y)
        true_utility = (1 - (self.actions[None] - physical_y[:, None]).abs() /
                        (.05 * (physical_y.abs()[:, None] + 1e-7))).clamp_min(0)
        reward = (action_probability * true_utility).sum(-1)
        decision = (q * (1 - reward)).mean() / 3
        return likelihood + self.alpha * decision, likelihood, decision
