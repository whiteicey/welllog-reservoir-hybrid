"""Candidate: fold-local piecewise-linear numeric encoding for neural retrieval."""
import json
import numpy as np
import torch
from torch import nn
from neighborhood_density import NeighborhoodDensity as RawNeighborhoodDensity
from train_invariant import NumericEncoding, FEATURES, SEGMENTS, PADDED_ENCODED


class NeighborhoodDensity(RawNeighborhoodDensity):
    def __init__(self, edges=None, inputs=206, hidden=128, embedding=32):
        assert inputs == FEATURES
        super().__init__(inputs, hidden, embedding)
        if edges is None:
            edges = np.tile(np.linspace(-3, 3, SEGMENTS+1), (FEATURES, 1))
        self.encoder = nn.Sequential(NumericEncoding(edges), nn.Linear(PADDED_ENCODED, hidden),
                                     nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, 3*embedding))


def self_test():
    torch.set_num_threads(2); torch.manual_seed(43)
    edges = np.tile(np.linspace(-3, 3, SEGMENTS+1), (FEATURES, 1))
    edges[0] = 0  # Constant input columns must not create NaN/division by zero.
    model = NeighborhoodDensity(edges=edges).eval()
    q = torch.randn(4, FEATURES); m = torch.randn(32, FEATURES)
    qw = torch.arange(4); mw = torch.arange(4).repeat_interleave(8)
    defaults = torch.zeros(32, 3, dtype=torch.bool); defaults[::2] = True
    loga = model.log_attention(q, m, qw, mw)
    gate, lw = model.conditional(loga, defaults)
    loss = model.conditional_nll(lw, torch.randn(4, 3), torch.randn(32, 3)).mean()+gate.mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.encoder[1].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(loga.exp().permute(0, 2, 1)[qw[:, None] == mw]) == 0
    kernels = torch.rand(32, 3, 256); kernels /= kernels.sum(-1, keepdim=True)
    sg, sp = model.streaming_posterior(model.encode(q), model.encode(m), qw, mw, defaults, kernels, 3)
    expected = torch.bmm(lw.exp().permute(1, 0, 2), kernels.permute(1, 0, 2)).permute(1, 0, 2)
    assert torch.allclose(sg, gate, atol=1e-6) and torch.allclose(sp, expected, atol=1e-6)
    return {'passed': True, 'parameters': sum(p.numel() for p in model.parameters()),
            'constant_column_safe': True, 'encoder_gradients_finite': True,
            'whole_well_mask_and_streaming_equivalence': True, 'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
