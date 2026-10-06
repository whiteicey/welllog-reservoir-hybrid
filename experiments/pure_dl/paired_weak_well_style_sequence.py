"""Training-only weak cross-well hidden-statistics augmentation."""
import torch
from torch import nn
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES


class WeakWellStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.valid = None
        self.domains = None
        self.generator = None

    def forward(self, h):
        if not self.training:
            return h
        assert self.valid is not None and self.domains is not None
        assert self.valid.shape == (h.shape[0], h.shape[-1])
        if self.generator is None:
            self.generator = torch.Generator(device=h.device).manual_seed(20261117)
        mask = self.valid[:, None].float()
        x = h.float()
        count = mask.sum(-1, keepdim=True).clamp_min(1)
        mu = ((x * mask).sum(-1, keepdim=True) / count).detach()
        sd = (((x - mu).square() * mask).sum(-1, keepdim=True) / count + 1e-5).sqrt().detach()
        different = self.domains[:, None] != self.domains[None, :]
        donor_scores = torch.rand((len(h), len(h)), device=h.device, generator=self.generator)
        donor = donor_scores.masked_fill(~different, -1).argmax(-1)
        strength = .2 * torch.rand((len(h), 1, 1), device=h.device, generator=self.generator)
        use = torch.rand((len(h), 1, 1), device=h.device, generator=self.generator) < .5
        strength = strength * use * different.any(-1)[:, None, None]
        mixed_mu = mu + strength * (mu[donor] - mu)
        mixed_sd = sd + strength * (sd[donor] - sd)
        mixed = (x - mu) / sd * mixed_sd + mixed_mu
        # Keep padding and non-selected examples exactly unchanged.
        active = self.valid[:, None] & (strength > 0)
        return torch.where(active, mixed, x).to(h.dtype)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        self.members[0].weak_style = WeakWellStyle()
        self.members[0].stem.register_forward_hook(self._style_hook)
        self.training_domains = None

    def _style_hook(self, module, inputs, output):
        return self.members[0].weak_style(output)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        style = self.members[0].weak_style
        style.valid = valid
        style.domains = self.training_domains
        try:
            return super().forward(features, relative_depth, valid, auxiliary, absolute_depth)
        finally:
            style.valid = None
            style.domains = None
