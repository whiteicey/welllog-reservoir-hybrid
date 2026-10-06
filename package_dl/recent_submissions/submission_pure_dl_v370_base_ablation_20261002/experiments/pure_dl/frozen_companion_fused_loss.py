"""Train a fresh B through the final equal A/B/D predictive mixture; A/D frozen."""
from contextlib import nullcontext
from pathlib import Path
import torch
from torch.nn import functional as F
import train_invariant as base
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from paired_detached_router_sequence import routed_merge
from global_cosine_sequence import Network as SingleNetwork
from por_t3_patch_loss import components, head_loss


def channel_loss(output, target, default, mask, weights):
    g, r = output
    g, r, d = g[mask].float(), r[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    bce = F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
    nll = -(r[:, :, 0].log_softmax(-1)+components(r, y)).logsumexp(-1).masked_fill(d, 0)
    w = weights[mask].detach().float()[:, None]
    return (w*(bce+nll)).mean(0), (w*bce).mean(0), (w*nll).mean(0)


def equal_merge(outputs):
    gates = torch.stack([g.float() for g, _ in outputs])
    survival_each = F.logsigmoid(-gates)
    survival = survival_each.logsumexp(0)
    gate = F.logsigmoid(gates).logsumexp(0)-survival
    shares = survival_each-survival[None]
    weights = torch.cat([r[..., 0, :].float().log_softmax(-1)+shares[i, ..., None]
                         for i, (_, r) in enumerate(outputs)], -1)
    means = torch.cat([r[..., 1, :].float() for _, r in outputs], -1)
    scales = torch.cat([r[..., 2, :].float() for _, r in outputs], -1)
    return gate, torch.stack([weights, means, scales], -2)


class FrozenCompanions:
    def __init__(self, source_checks):
        self.models = []
        self.snapshots = []
        for item in source_checks:
            cp = torch.load(base.ROOT/Path(item['folder'])/'fold_5.pt', map_location='cpu', weights_only=False)
            names = cp['selected_feature_names']
            with torch.random.fork_rng(devices=[]):
                if item['role'] == 'A':
                    numeric, conditioning = calibration_indices(names)
                    model = Network(retain_indices=retained_indices(names), numeric_indices=numeric,
                                    conditioning_indices=conditioning)
                else:
                    assert item['role'] == 'D'
                    model = SingleNetwork()
            model.load_state_dict(cp['state_dict'], strict=True)
            model.cuda().eval().requires_grad_(False)
            self.models.append(model)
            self.snapshots.append(cp['state_dict'])

    def forward_loss(self, b, x, relative, exists, absolute, target, default, mask, weights, epoch):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            branches = [member(x, relative, exists, True, absolute) for member in b.members]
            mains = [out[0] for out in branches]
            # Main fused loss differentiates through BOTH B backbones and B router.
            bmain = routed_merge(mains, b.route_logits(x, exists, mains))
            with torch.no_grad():
                a, d = [model(x, relative, exists, absolute_depth=absolute) for model in self.models]
        fused = equal_merge([a, bmain, d])
        channels, bce, nll = channel_loss(fused, target, default, mask, weights)
        aux = channels.sum()*0
        factor = max(0., 1.-epoch/40.)
        if factor:
            aux = sum(factor*(.2*head_loss(out[1], target, default, mask, weights)+
                              .1*head_loss(out[2], target, default, mask, weights)) for out in branches)/2
        return channels.mean()+aux, channels, bce, nll, aux

    def verify(self):
        for model, snapshot in zip(self.models, self.snapshots):
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            assert all(torch.equal(v.detach().cpu(), snapshot[k]) for k, v in model.state_dict().items())


def channel_gradient_snapshot(channels, model):
    params = [p for name, p in model.named_parameters() if not any(
        part in name for part in ('main_head.', 'aux_mid.', 'aux_coarse.', 'router.'))]
    rows = []
    for j in range(3):
        gradients = torch.autograd.grad(channels[j], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([g.detach().float().flatten() if g is not None else torch.zeros_like(p).flatten()
                               for g, p in zip(gradients, params)]))
    matrix = torch.stack(rows)
    dot = matrix@matrix.T
    norms = dot.diag().clamp_min(0).sqrt()
    cosine = dot/(norms[:, None]*norms[None, :]).clamp_min(1e-30)
    return {'scope': 'first random training batch, final fused main loss, B shared params only, before clipping/Adam',
            'norm': norms.cpu().tolist(), 'cosine': cosine.cpu().tolist()}
