"""Private SW model: frozen private dual prior plus an input-driven dilated TCN.

The new backbone reads 206 curve features and relative depth directly. It does
not read POR/PERM outputs or hidden states. The zero-output density correction
is trained independently; only its supervised objective differs between arms.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from independent_channels_v8 import Network as PrivatePrior
from por_t3_patch_loss import probability


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2*dilation,
                                   dilation=dilation, groups=width)
        self.gate = nn.Conv1d(width, 2*width, 1)
        self.project = nn.Conv1d(width, width, 1)

    def forward(self, x, valid):
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)
        z = z.masked_fill(~valid[:, None], 0)
        z = self.depthwise(z).masked_fill(~valid[:, None], 0)
        z = self.project(F.glu(self.gate(z), dim=1))
        return (x+z).masked_fill(~valid[:, None], 0)


class DensityTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(207, 64, 1)
        self.blocks = nn.ModuleList([TemporalBlock(64, d) for d in (1, 2, 4, 8)])
        self.norm = nn.LayerNorm(64)
        self.output = nn.Conv1d(64, 256, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, relative, valid):
        inp = torch.cat([x.float(), relative[..., None].float()], -1)
        inp = inp.masked_fill(~valid[..., None], 0)
        h = F.gelu(self.stem(inp.transpose(1, 2))).masked_fill(~valid[:, None], 0)
        for block in self.blocks:
            h = block(h, valid)
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        raw = self.output(h).transpose(1, 2).float()
        # Fixed protection, not a tuned hyperparameter: at most exp(+/-2)
        # unnormalised odds adjustment at any density support point.
        return (2*torch.tanh(raw/2)).masked_fill(~valid[..., None], 0)


class Network(nn.Module):
    def __init__(self, master: dict, previous: dict, prior_state: dict):
        super().__init__()
        self.prior = PrivatePrior(master, previous, 'SW', 'dual')
        self.prior.load_state_dict(prior_state, strict=True)
        self.prior.requires_grad_(False).eval()
        self.correction = DensityTCN()
        self.register_buffer('edges', torch.as_tensor(master['density_edges'][2], dtype=torch.float32))
        self.register_buffer('centers', torch.as_tensor(master['centers'][2], dtype=torch.float32))
        center_z = ((self.centers.log()-float(master['regression_median'][2])) /
                    float(master['regression_scale'][2]))
        self.register_buffer('centers_z', center_z)
        self.log_target_median = float(master['regression_median'][2])
        self.log_target_scale = float(master['regression_scale'][2])
        assert len(self.centers) == 256 and bool((center_z[1:] > center_z[:-1]).all())

    def train(self, mode=True):
        super().train(mode)
        self.prior.eval()
        return self

    def forward(self, x, relative, valid, absolute_depth=None):
        with torch.no_grad():
            gate, raw = self.prior(x, relative, valid, absolute_depth=absolute_depth)
        return gate, raw, self.correction(x, relative, valid)

    def distributions(self, output, mask):
        gate, raw, delta = output
        g = gate[mask][:, 0].float().sigmoid()
        r = raw[mask][:, 0].float()
        p0 = probability(2, r[:, 0].softmax(-1), r[:, 1],
                         F.softplus(r[:, 2])+.025, self.edges).detach()
        # A tiny fixed floor avoids impossible density bins during fitting.
        # Initial decode is compared against the saved old model, not assumed.
        p0 = p0.clamp_min(1e-12)
        p0 = p0 / p0.sum(-1, keepdim=True)
        logp = (p0.log()+delta[mask].float()).log_softmax(-1)
        return g, logp.exp(), logp, p0


def objective(model: Network, output, labels, default, mask, quality, utility_weight: float):
    """CE+0.01 prior KL; optionally +0.1 negative log conditional utility.

    Only SW labels from the outer training wells are read. Utility is a
    differentiable training surrogate, not a claim of exact argmax optimisation.
    """
    g, p, logp, p0 = model.distributions(output, mask)
    d = default[mask][:, 2]
    z = labels[mask][:, 2].float()
    upper = torch.searchsorted(model.centers_z, z.contiguous()).clamp(1, 255)
    lower = upper-1
    fraction = ((z-model.centers_z[lower]) /
                (model.centers_z[upper]-model.centers_z[lower])).clamp(0, 1)
    ce = -((1-fraction)*logp.gather(1, lower[:, None]).squeeze(1) +
           fraction*logp.gather(1, upper[:, None]).squeeze(1))
    y = torch.exp(z*model.log_target_scale+model.log_target_median)
    kernel = (1-(model.centers[None]-y[:, None]).abs() /
              (.05*(y.abs()[:, None]+1e-5))).clamp_min(0)
    utility = (p*kernel).sum(-1)
    score_loss = -utility.clamp_min(1e-8).log()
    kl = (p0*(p0.log()-logp)).sum(-1).clamp_min(0)
    w = quality[mask].detach().float()
    numeric = (~d).float()
    loss = (w*(numeric*(ce+utility_weight*score_loss)+.01*kl)).mean()/3
    # Diagnostic means on actual non-default training points, never validation.
    denom = (w*numeric).sum().clamp_min(1)
    metrics = {'ce': float((w*numeric*ce).sum().detach()/denom),
               'conditional_utility': float((w*numeric*utility).sum().detach()/denom),
               'prior_kl': float((w*kl).mean().detach()),
               'supervised_numeric_count': int(numeric.sum().detach())}
    return loss, metrics
