"""Unnormalized utility-weighted cross entropy for direct action prediction.

For a fixed input x, the population optimum satisfies
p(a|x) proportional to E[official_reward(a,Y)|x]. Thus argmax p is the
Bayes-optimal action on the fixed grid. Per-label normalization would change
this objective and is intentionally forbidden. This is NOT a label density.
"""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

TARGETS = ('POR', 'PERM', 'SW')
WEIGHTS = (.3, .35, .35)


def reward(actions, truth, target):
    a, y = actions[None], truth[:, None]
    if target == 'PERM':
        ratio = (a/y.clamp_min(1e-5)).clamp_min(1e-5)
        return (1-ratio.log10().abs()).clamp_min(0)
    tolerance = .08 if target == 'POR' else .05
    return (1-(a-y).abs()/(tolerance*(y.abs()+1e-5))).clamp_min(0)


def utility_cross_entropy(logits, truth, supervised, actions, normalizers):
    assert len(logits) == len(actions) == len(normalizers) == 3
    assert bool(supervised.any())
    y = truth[supervised]
    losses = []
    for j, target in enumerate(TARGETS):
        z = logits[j][supervised].float()
        gain = reward(actions[j], y[:, j], target)
        # The scalar is fitted once from training labels; never normalize rows.
        constant = torch.as_tensor(normalizers[j], device=z.device, dtype=z.dtype)
        assert constant.ndim == 0 and bool(constant > 0)
        losses.append(-(gain*z.log_softmax(-1)).sum(-1).mean()/constant)
    return sum(w*loss for w, loss in zip(WEIGHTS, losses))


def self_test():
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[2]
    source = root.parent/'catboost_hybrid_baseline/submission_code_sota/src/metrics.py'
    spec = importlib.util.spec_from_file_location('original_official', source)
    official = importlib.util.module_from_spec(spec); spec.loader.exec_module(official)
    values = ([0., .1, 1., 10., 50.], [.01, .1, 1., 100., 10000.], [8.305, 30., 70., 99., 99.9])
    for j, target in enumerate(TARGETS):
        actions = torch.tensor(values[j], dtype=torch.float64)
        got = reward(actions, actions, target).numpy()
        for row, truth in enumerate(values[j]):
            for col, action in enumerate(values[j]):
                expected = official.official_target_score(np.array([truth]), np.array([action]), target)
                assert abs(got[row, col]-expected) < 1e-12
    # Unequal reward row sums: normalization per observed label can reverse the action.
    gain = torch.tensor([[1., .8, .8], [0., 1., 0.]], dtype=torch.float64)
    truth_probability = torch.tensor([.9, .1], dtype=torch.float64)
    expected_reward = truth_probability@gain
    optimum = expected_reward/expected_reward.sum()
    assert optimum.argmax() == expected_reward.argmax() == 0
    wrong = truth_probability@(gain/gain.sum(-1, keepdim=True))
    assert wrong.argmax() == 1
    z = optimum.log().detach().requires_grad_(True)
    loss = -(truth_probability[:, None]*gain*z.log_softmax(-1)[None]).sum()
    loss.backward(); assert z.grad.abs().max() < 1e-12
    torch.manual_seed(19)
    logits = [torch.randn(2, 4, 5, requires_grad=True) for _ in range(3)]
    truth = torch.tensor([values[j][:4] for j in range(3)]).T[None].repeat(2, 1, 1)
    actions = [torch.tensor(v) for v in values]
    mask = torch.tensor([[True, False, True, False], [True, True, False, False]])
    before = utility_cross_entropy(logits, truth, mask, actions, [1., 2., 3.])
    altered = truth.clone(); altered[~mask] = 1e6
    assert torch.equal(before, utility_cross_entropy(logits, altered, mask, actions, [1., 2., 3.]))
    before.backward()
    assert all(torch.isfinite(z.grad).all() and (z.grad[~mask] == 0).all() for z in logits)
    return {'passed': True, 'official_metric_kernel_exact': True,
            'population_optimum_selects_maximum_expected_official_reward': True,
            'per_label_normalization_counterexample_verified': True,
            'masked_labels_and_gradients_isolated': True,
            'not_label_probability_density': True, 'gpu_experiment_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
