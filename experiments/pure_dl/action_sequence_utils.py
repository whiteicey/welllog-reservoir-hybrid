"""Train-partition-only action grids and fixed loss normalization constants."""
import numpy as np
import torch
from score_aligned_loss import reward, TARGETS


def make_actions(training_truth):
    y = np.asarray(training_truth, dtype=np.float64)
    assert y.ndim == 2 and y.shape[1] == 3 and np.isfinite(y).all()
    positive = y[y[:, 0] > 0, 0]
    assert len(positive)
    por = np.geomspace(max(positive.min()*.9, 1e-5), min(y[:, 0].max()*1.1, 50), 800)
    perm = np.geomspace(max(y[:, 1].min()*.9, .01), min(y[:, 1].max()*1.1, 10000), 800)
    sw = np.linspace(max(y[:, 2].min(), 8.305), min(y[:, 2].max(), 99.9), 800)
    grids = [np.unique(np.r_[0., .1, por]), np.unique(np.r_[.01, perm]), np.unique(np.r_[99.9, sw])]
    assert all(len(a) > 1 and np.isfinite(a).all() and (np.diff(a) > 0).all() for a in grids)
    return grids


@torch.no_grad()
def fit_normalizers(training_truth, actions):
    constants = []
    for j, name in enumerate(TARGETS):
        total = torch.zeros((), device=training_truth.device, dtype=torch.float64)
        minimum_best = torch.ones((), device=training_truth.device)
        for start in range(0, len(training_truth), 4096):
            scores = reward(actions[j], training_truth[start:start+4096, j], name)
            total += scores.sum(dtype=torch.float64)
            minimum_best = torch.minimum(minimum_best, scores.max(-1).values.min())
        assert bool(minimum_best > 0), 'Action grid gives zero reward to a training label'
        constants.append(float(total/len(training_truth)))
    assert all(np.isfinite(c) and c > 0 for c in constants)
    return constants
