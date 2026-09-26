"""Fixed CRPS auxiliary term for the existing dense Gaussian-mixture network.

Keeps baseline BCE/NLL, conditional target balancing and deep supervision.
CRPS is evaluated only for supervised, nondefault labels in train-scaled units.
"""
import json
import torch
from torch.nn import functional as F
from gaussian_crps import mixture_crps, self_test as integral_test
from multiscale_sequence import head_loss as baseline_head_loss

CRPS_WEIGHT = .2


def head_loss(output, target_z, target_default, supervised):
    loss = baseline_head_loss(output, target_z, target_default, supervised)
    _, raw = output
    if raw.shape[1] != target_z.shape[1]:
        repeat = target_z.shape[1] // raw.shape[1]
        assert repeat * raw.shape[1] == target_z.shape[1]
        raw = raw.repeat_interleave(repeat, dim=1)
    raw = raw[supervised].float()
    truth = target_z[supervised]
    defaults = target_default[supervised]
    terms = []
    for j in range(3):
        active = ~defaults[:, j].bool()
        values = raw[active, j]
        if bool(active.any()):
            terms.append(mixture_crps(values[:, 0].log_softmax(-1), values[:, 1],
                                      F.softplus(values[:, 2]) + .025,
                                      truth[active, j]).mean())
        else:
            terms.append(raw[:, j].sum() * 0)
    return loss + CRPS_WEIGHT * sum(terms) / 3


def objective(outputs, target_z, target_default, supervised, deep_supervision=False):
    if not deep_supervision:
        return head_loss(outputs, target_z, target_default, supervised)
    return sum(weight * head_loss(output, target_z, target_default, supervised)
               for weight, output in zip((1., .2, .1), outputs))


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(107)
    checks = integral_test()
    gate = torch.randn(2, 16, 3, requires_grad=True)
    raw = torch.randn(2, 16, 3, 3, 8, requires_grad=True)
    truth = torch.randn(2, 16, 3)
    defaults = torch.rand(2, 16, 3) > .5
    supervised = torch.ones(2, 16, dtype=torch.bool)
    supervised[:, :4] = False
    loss = objective((gate, raw), truth, defaults, supervised)
    altered = truth.clone(); altered[~supervised] = float('nan')
    assert torch.equal(loss, objective((gate, raw), altered, defaults, supervised))
    loss.backward()
    assert torch.isfinite(raw.grad).all() and torch.isfinite(gate.grad).all()
    assert (raw.grad[~supervised] == 0).all()
    assert (gate.grad[~supervised] == 0).all()
    assert (raw.grad[defaults] == 0).all()
    all_default = torch.ones_like(defaults)
    a = head_loss((gate, raw), truth, all_default, supervised)
    b = baseline_head_loss((gate, raw), truth, all_default, supervised)
    assert torch.equal(a, b)
    checks.update(masked_labels_and_gradients=True, default_density_gradient_zero=True,
                  all_default_batch_equals_baseline=True, crps_weight=CRPS_WEIGHT)
    return checks


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
