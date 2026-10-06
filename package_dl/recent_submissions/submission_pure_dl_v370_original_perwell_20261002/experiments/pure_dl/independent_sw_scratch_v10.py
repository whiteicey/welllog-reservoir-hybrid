"""Two standalone SW models: dilated gated TCN versus bidirectional GRU.

No pretrained network, teacher logits, previous predictions or other-target
features enter either model. The old training-fold preprocessing/grid is reused.
The two arms share the loss and readout shape, not parameter storage.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

ARCHITECTURES = ('tcn', 'bigru')


class TemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, 5, padding=2*dilation,
                                  dilation=dilation, groups=width)
        self.gate = nn.Conv1d(width, 2*width, 1)
        self.project = nn.Conv1d(width, width, 1)

    def forward(self, h, valid):
        z = self.norm(h).masked_fill(~valid[..., None], 0).transpose(1, 2)
        z = self.depthwise(z).masked_fill(~valid[:, None], 0)
        z = self.project(F.glu(self.gate(z), dim=1)).transpose(1, 2)
        return (h+z).masked_fill(~valid[..., None], 0)


class RecurrentBlock(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.project = nn.Linear(2*width, width)

    def forward(self, h, valid):
        # Source windows have one contiguous valid segment. Move that segment
        # to the left, pack it, then restore original indices. Padding never
        # advances either recurrent direction. Empty segments return zero.
        batch, length, width = h.shape
        start = valid.long().argmax(1)
        lengths = valid.long().sum(1)
        steps = torch.arange(length, device=h.device)[None]
        gather = (start[:, None]+steps).clamp_max(length-1)
        shifted = h.gather(1, gather[..., None].expand(batch, length, width))
        shifted = shifted.masked_fill(~(steps < lengths[:, None])[..., None], 0)
        # Use float32 for packed cuDNN GRU; convolutions/readouts may use AMP.
        with torch.autocast(device_type=h.device.type, enabled=False):
            packed = pack_padded_sequence(shifted.float(), lengths.clamp_min(1).cpu(),
                                         batch_first=True, enforce_sorted=False)
            packed_output, _ = self.gru(packed)
            y, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=length)
            inverse = (steps-start[:, None]).clamp(0, length-1)
            y = y.gather(1, inverse[..., None].expand(batch, length, 2*width))
        return (h+self.project(y)).masked_fill(~valid[..., None], 0)


class Network(nn.Module):
    def __init__(self, architecture: str, centers, target_median: float, target_scale: float):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.architecture = architecture
        # Shared-shaped modules get the same initial values in either arm.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010)
            self.stem = nn.Linear(207, 64)
        self.blocks = (nn.ModuleList([TemporalBlock(64, d) for d in (1, 2, 4, 8)])
                       if architecture == 'tcn' else nn.ModuleList([RecurrentBlock(64)]))
        self.norm = nn.LayerNorm(64)
        self.output = nn.Linear(64, 257)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        c = torch.as_tensor(centers, dtype=torch.float32).clone()
        if c.shape != (256,) or not bool((c[1:] > c[:-1]).all()) or not bool((c > 0).all()):
            raise ValueError('Expected 256 increasing positive SW support points')
        self.register_buffer('centers', c)
        self.register_buffer('centers_z', (c.log()-float(target_median))/float(target_scale))
        self.target_median, self.target_scale = float(target_median), float(target_scale)

    def forward(self, x, relative, valid, absolute_depth=None):
        inp = torch.cat([x.float(), relative[..., None].float()], dim=-1)
        inp = inp.masked_fill(~valid[..., None], 0)
        h = F.gelu(self.stem(inp)).masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            h = block(h, valid)
        output = self.output(self.norm(h)).float().masked_fill(~valid[..., None], 0)
        return output[..., :1], output[..., 1:]

    def distributions(self, output, mask):
        gate, logits = output
        g = gate[mask][:, 0].float().sigmoid()
        logp = logits[mask].float().log_softmax(-1)
        return g, logp.exp(), logp, None

    def soft_targets(self, z):
        upper = torch.searchsorted(self.centers_z, z.contiguous()).clamp(1, 255)
        lower = upper-1
        frac = ((z-self.centers_z[lower]) /
                (self.centers_z[upper]-self.centers_z[lower])).clamp(0, 1)
        return lower, upper, frac

    @torch.no_grad()
    def initialize_training_priors(self, data):
        ids = data['train_raw']
        default = data['default'][ids, 2]
        q = data['quality'][ids].float()
        rate = (q*default.float()).sum()/q.sum()
        z = data['labels'][ids, 2][~default].float()
        weight = q[~default]
        lower, upper, frac = self.soft_targets(z)
        counts = torch.ones(256, device=z.device)  # fixed one-count smoothing
        counts.scatter_add_(0, lower, weight*(1-frac))
        counts.scatter_add_(0, upper, weight*frac)
        probabilities = counts/counts.sum()
        self.output.bias[0].copy_(torch.logit(rate.clamp(1e-5, 1-1e-5)))
        self.output.bias[1:].copy_(probabilities.log())
        return {'training_default_rate': float(rate), 'training_nondefault_rows': int((~default).sum()),
                'histogram_pseudocount_per_bin': 1.0, 'fit_population': 'outer64trainingwells_only',
                'initialization': 'random backbone;zero final weights;training-only label prior biases'}


def objective(model, output, labels, default, mask, quality):
    gate, logits = output
    g = gate[mask][:, 0].float()
    d = default[mask][:, 2]
    z = labels[mask][:, 2].float().masked_fill(d, 0)
    logp = logits[mask].float().log_softmax(-1)
    lower, upper, frac = model.soft_targets(z)
    ce = -((1-frac)*logp.gather(1, lower[:, None]).squeeze(1)
           + frac*logp.gather(1, upper[:, None]).squeeze(1))
    bce = F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
    w = quality[mask].detach().float()
    numeric = (~d).float()
    loss = (w*(bce+numeric*ce)).mean()/3
    return loss, (w*bce).mean().detach(), (w*numeric*ce).sum().detach()/(w*numeric).sum().clamp_min(1)
