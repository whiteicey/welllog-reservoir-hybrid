"""Exact-parameter SW shared-representation versus gate/density separation.

Both arms contain two differently initialized width48 TCN encoders and identical
heads. Shared: both heads read (E0+E1)/2. Split: gate reads E0, density reads E1.
Every parameter is used; no padded capacity or label-dependent inference routing.
Only the input-to-head connectivity differs between arms. No other-target model
or pretrained network is used; training-side preprocessing/grid/Q are reused.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from independent_sw_scratch_v10 import TemporalBlock, Network as GridMethods

ARCHITECTURES = ('shared', 'split')
WIDTH = 48


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(207, WIDTH)
        self.blocks = nn.ModuleList([TemporalBlock(WIDTH, d) for d in (1, 2, 4, 8)])
        self.norm = nn.LayerNorm(WIDTH)

    def forward(self, inp, valid):
        h = F.gelu(self.stem(inp)).masked_fill(~valid[..., None], 0)
        for block in self.blocks:
            h = block(h, valid)
        return self.norm(h).masked_fill(~valid[..., None], 0)


class Network(GridMethods):
    def __init__(self, architecture, centers, target_median, target_scale):
        nn.Module.__init__(self)
        if architecture not in ARCHITECTURES:
            raise ValueError(architecture)
        self.architecture = architecture
        self.encoders = nn.ModuleList()
        # Two distinct banks, same exact initialization across experimental arms.
        for seed in (20261010, 20261011):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.encoders.append(Encoder())
        self.output = nn.ModuleDict({'gate': nn.Linear(WIDTH, 1),
                                     'density': nn.Linear(WIDTH, 256)})
        for head in self.output.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        c = torch.as_tensor(centers, dtype=torch.float32).clone()
        if c.shape != (256,) or not bool((c[1:] > c[:-1]).all()) or not bool((c > 0).all()):
            raise ValueError('Expected256positiveincreasingSWsupportpoints')
        if not float(target_scale) > 0:
            raise ValueError('Positive target scale required')
        self.register_buffer('centers', c)
        self.register_buffer('centers_z', (c.log()-float(target_median))/float(target_scale))
        self.target_median = float(target_median)
        self.target_scale = float(target_scale)

    def forward(self, x, relative, valid, absolute_depth=None):
        inp = torch.cat([x.float(), relative[..., None].float()], -1)
        inp = inp.masked_fill(~valid[..., None], 0)
        h0, h1 = [encoder(inp, valid) for encoder in self.encoders]
        if self.architecture == 'shared':
            gate_h = density_h = (h0+h1)*.5
        else:
            gate_h, density_h = h0, h1
        gate = self.output['gate'](gate_h).float().masked_fill(~valid[..., None], 0)
        density = self.output['density'](density_h).float().masked_fill(~valid[..., None], 0)
        return gate, density

    @torch.no_grad()
    def initialize_training_priors(self, data):
        ids = data['train_raw']
        default = data['default'][ids, 2]
        q = data['quality'][ids].float()
        rate = (q*default.float()).sum()/q.sum()
        lower, upper, frac = self.soft_targets(data['labels'][ids, 2][~default].float())
        counts = torch.ones(256, device=ids.device)
        counts.scatter_add_(0, lower, q[~default]*(1-frac))
        counts.scatter_add_(0, upper, q[~default]*frac)
        self.output['gate'].bias[0].copy_(torch.logit(rate.clamp(1e-5, 1-1e-5)))
        self.output['density'].bias.copy_((counts/counts.sum()).log())
        digest = hashlib.sha256()
        for name, value in self.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return {'training_default_rate': float(rate), 'training_nondefault_rows': int((~default).sum()),
                'histogram_pseudocount_per_bin': 1., 'fit_population': 'outer64trainingwells_only',
                'initialization': 'two random different TCNbanks,zero headweights,training-only labelbiases',
                'initial_state_sha256': digest.hexdigest()}

    def clipping_groups(self):
        return [list(self.encoders[0].parameters())+list(self.output['gate'].parameters()),
                list(self.encoders[1].parameters())+list(self.output['density'].parameters())]


def objective_terms(model, output, labels, default, mask, quality):
    gate, logits = output
    d = default[mask][:, 2]
    z = labels[mask][:, 2].float().masked_fill(d, 0)
    logp = logits[mask].float().log_softmax(-1)
    lo, hi, frac = model.soft_targets(z)
    ce = -((1-frac)*logp.gather(1, lo[:, None]).squeeze(1)
           + frac*logp.gather(1, hi[:, None]).squeeze(1))
    bce = F.binary_cross_entropy_with_logits(gate[mask][:, 0].float(), d.float(), reduction='none')
    w = quality[mask].detach().float()
    numeric = (~d).float()
    gate_loss = (w*bce).mean()/3
    density_loss = (w*numeric*ce).mean()/3
    return gate_loss, density_loss, (w*numeric*ce).sum()/(w*numeric).sum().clamp_min(1)


def objective(model, output, labels, default, mask, quality):
    g, d, conditional_ce = objective_terms(model, output, labels, default, mask, quality)
    return g+d, (3*g).detach(), conditional_ce.detach()


def training_gradient_diagnostic(model, data, batches=8):
    """Fixed training-only batches, no update/RNG disturbance; after best restore."""
    from multiscale_sequence import window_indices
    model.eval()
    generator = torch.Generator(device='cuda').manual_seed(20261111)
    params = [p for encoder in model.encoders for p in encoder.parameters()]
    n0 = len(list(model.encoders[0].parameters()))
    core = torch.zeros((1,256), device='cuda', dtype=torch.bool)
    core[:,64:192] = True
    records = []
    for _ in range(batches):
        anchor = data['train_raw'][torch.randint(len(data['train_raw']), (16,), device='cuda', generator=generator)]
        ix, valid = window_indices(anchor, data['lower'], data['upper'])
        relative = torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
        mask = valid & core & data['labeled'][ix]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(data['x'][ix], relative, valid)
        g, d, _ = objective_terms(model, out, data['labels'][ix], data['default'][ix], mask, data['quality'][ix])
        gg = torch.autograd.grad(g, params, retain_graph=True, allow_unused=True)
        gd = torch.autograd.grad(d, params, allow_unused=True)
        def norm(grads):
            values = [v.float().square().sum() for v in grads if v is not None]
            return float(torch.stack(values).sum().sqrt()) if values else 0.
        ng, nd = norm(gg), norm(gd)
        terms = [(a.float()*b.float()).sum() for a,b in zip(gg,gd) if a is not None and b is not None]
        dot = float(torch.stack(terms).sum()) if terms else 0.
        cosine = dot/(ng*nd) if ng*nd > 0 else None
        cross_gate = norm(gg[n0:]); cross_density = norm(gd[:n0])
        if model.architecture == 'split':
            assert cross_gate == 0. and cross_density == 0.
        records.append({'gate_gradient_norm': ng, 'density_gradient_norm': nd, 'cosine': cosine,
                        'gate_to_density_encoder_norm': cross_gate,
                        'density_to_gate_encoder_norm': cross_density})
    valid_cosines = [x['cosine'] for x in records if x['cosine'] is not None]
    return {'population': 'fixed128windows_from_outer64trainingwells_only', 'parameter_updates': 0,
            'records': records, 'negative_cosine_batches': sum(x < 0 for x in valid_cosines),
            'mean_cosine': float(np.mean(valid_cosines)) if valid_cosines else None,
            'split_cross_gradients_exactly_zero': all(x['gate_to_density_encoder_norm']==0 and
                      x['density_to_gate_encoder_norm']==0 for x in records),
            'limitation': 'small fixed training sample at selected checkpoint;not proof of cause of validation difference'}
