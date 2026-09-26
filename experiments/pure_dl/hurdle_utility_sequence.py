"""Gated conditional expected utility, with a proper soft-BCE action loss.

g(x)=P(default|x), u(a,x)=E[S(a,Y)|x,nondefault]. The final expected
official utility is g*S(a,default)+(1-g)*u. Conditional u uses sigmoids,
not a softmax, so no unknown row-dependent scale enters this mixture.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as Backbone
from score_aligned_loss import reward, TARGETS, WEIGHTS

DEFAULTS = (.1, .01, 99.9)


class HurdleHead(nn.Module):
    def __init__(self, width, sizes, conditional_prior=None, default_prior=None):
        super().__init__()
        self.sizes = tuple(sizes)
        self.gate = nn.Conv1d(width, 3, 1)
        self.action = nn.Conv1d(width, sum(sizes), 1)
        nn.init.normal_(self.action.weight, 0, .005)
        with torch.no_grad():
            gp = torch.full((3,), .5) if default_prior is None else torch.as_tensor(default_prior).float()
            self.gate.bias.copy_(torch.logit(gp.clamp(1e-4, 1-1e-4)))
            cp = torch.full((sum(sizes),), .5) if conditional_prior is None else torch.cat([torch.as_tensor(p).float() for p in conditional_prior])
            assert cp.shape == (sum(sizes),)
            self.action.bias.copy_(torch.logit(cp.clamp(1e-4, 1-1e-4)))

    def forward(self, x):
        gate = self.gate(x).transpose(1, 2)
        conditional = self.action(x).transpose(1, 2).split(self.sizes, dim=-1)
        return gate, conditional


class Network(Backbone):
    def __init__(self, sizes, conditional_prior=None, default_prior=None, features=206):
        super().__init__(features=features)
        self.main_head = HurdleHead(128, sizes, conditional_prior, default_prior)
        del self.aux_mid, self.aux_coarse

    def forward(self, features, relative_depth, valid, auxiliary=False):
        assert not auxiliary
        return super().forward(features, relative_depth, valid, auxiliary=False)


def objective(output, truth, defaults, supervised, actions, normalizers):
    gate, conditional = output
    assert bool(supervised.any())
    y = truth[supervised]; d = defaults[supervised].float()
    loss = F.binary_cross_entropy_with_logits(gate[supervised].float(), d)
    for j, target in enumerate(TARGETS):
        logits = conditional[j][supervised].float()
        active = d[:, j] < .5
        if bool(active.any()):
            gain = reward(actions[j], y[active, j], target)
            assert normalizers[j] > 0
            conditional_loss = F.binary_cross_entropy_with_logits(logits[active], gain, reduction='none').sum(-1).mean()/normalizers[j]
            loss = loss+WEIGHTS[j]*conditional_loss
        else:
            loss = loss+logits.sum()*0
    return loss


def expected_utilities(output, actions):
    logits, conditional = output
    gate = logits.float().sigmoid()
    result = []
    for j, target in enumerate(TARGETS):
        atom = reward(actions[j], torch.tensor([DEFAULTS[j]], device=logits.device), target)
        g = gate[..., j, None]
        result.append(g*atom+(1-g)*conditional[j].float().sigmoid())
    return result


@torch.no_grad()
def fit_priors(training_truth, defaults, actions):
    conditional = []; normalizers = []
    for j, target in enumerate(TARGETS):
        y = training_truth[~defaults[:, j].bool(), j]
        assert len(y)
        total = torch.zeros(len(actions[j]), device=y.device, dtype=torch.float64)
        for begin in range(0, len(y), 4096):
            total += reward(actions[j], y[begin:begin+4096], target).sum(0, dtype=torch.float64)
        mean = (total/len(y)).float()
        conditional.append(mean.cpu().numpy()); normalizers.append(float(mean.sum()))
    assert all(c > 0 for c in normalizers)
    return defaults.float().mean(0).cpu().numpy(), conditional, normalizers


def self_test():
    torch.set_num_threads(2); torch.manual_seed(83)
    # Proper BCE learns an expected soft reward, without softmax scale ambiguity.
    values = torch.tensor([[.2, .8, .4], [.7, .1, .3]], dtype=torch.float64)
    weights = torch.tensor([.3, .7], dtype=torch.float64)
    expected = weights@values
    z = torch.logit(expected).requires_grad_(True)
    risk = (weights[:, None]*F.binary_cross_entropy_with_logits(z[None].expand_as(values), values, reduction='none')).sum()
    risk.backward(); assert z.grad.abs().max() < 1e-12
    g = .65; atom = torch.tensor([.1, .3, 1.], dtype=torch.float64)
    combined = g*atom+(1-g)*expected
    full = torch.cat([torch.tensor([g], dtype=torch.float64), (1-g)*weights])@torch.cat([atom[None], values])
    assert torch.allclose(combined, full, atol=1e-12)
    actions = [torch.tensor([0., .1, 1., 10.]), torch.tensor([.01, .1, 10., 100.]), torch.tensor([8.305, 30., 70., 99.9])]
    y = torch.tensor([[[.1, .01, 99.9], [1., 10., 30.]]]).repeat(2, 128, 1)
    default = torch.isclose(y, torch.tensor(DEFAULTS)[None, None])
    mask = torch.zeros(2, 256, dtype=torch.bool); mask[:, 64:192] = True
    gp, cp, constants = fit_priors(y[mask], default[mask], actions)
    model = Network([4, 4, 4], cp, gp, features=12).eval()
    x = torch.randn(2, 256, 12); depth = torch.randn(2, 256); valid = torch.ones(2, 256, dtype=torch.bool)
    output = model(x, depth, valid); loss = objective(output, y, default, mask, actions, constants)
    changed = y.clone(); changed[~mask] = 1000
    assert torch.equal(loss, objective(output, changed, default, mask, actions, constants))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    u = expected_utilities(output, actions)
    assert all(torch.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in u)
    saturated = (torch.full_like(output[0], 1000), output[1])
    atoms = expected_utilities(saturated, actions)
    for j, target in enumerate(TARGETS):
        assert torch.allclose(atoms[j], reward(actions[j], torch.tensor([DEFAULTS[j]]), target))
    return {'passed': True, 'soft_bce_population_optimum_is_expected_reward': True,
            'gate_mixture_matches_total_expected_reward': True,
            'masked_targets_do_not_change_loss': True, 'finite_gradients_and_bounded_utilities': True,
            'default_limit_recovers_exact_official_reward': True, 'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
