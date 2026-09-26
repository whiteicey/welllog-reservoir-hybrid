"""Neural cross-well attention module for a future fold-local retrieval experiment.

Memory may contain only the current outer training wells. Training queries mask
their own well; evaluation queries must be from wells outside that memory.
This module performs no data loading, training launch, or experiment selection.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class CrossWellAttention(nn.Module):
    def __init__(self, hidden_size=256, key_size=32, outputs=75):
        super().__init__()
        self.key = nn.Sequential(nn.Linear(hidden_size, 64), nn.GELU(), nn.Linear(64, key_size))
        self.value = nn.Linear(hidden_size, 16)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(20.0)))
        self.head = nn.Sequential(nn.Linear(hidden_size + 25, 128), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(128, 128), nn.GELU(),
                                  nn.Linear(128, outputs))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, query_hidden, memory_hidden, memory_values,
                memory_wells, query_wells, *, return_attention=False):
        if memory_values.shape != (len(memory_hidden), 9):
            raise ValueError('Memory values must be three gates, masked targets, and masked squared targets')
        if query_wells.shape != (len(query_hidden),) or memory_wells.shape != (len(memory_hidden),):
            raise ValueError('Every query and memory row needs a well identity')
        allowed = query_wells[:, None] != memory_wells[None, :]
        if not bool(allowed.any(1).all()):
            raise ValueError('A query has no memory from another training well')
        q = F.normalize(self.key(query_hidden), dim=-1)
        k = F.normalize(self.key(memory_hidden), dim=-1)
        logits = (q @ k.T).float() * self.log_temperature.exp().clamp(1, 100)
        weights = logits.masked_fill(~allowed, -torch.inf).softmax(-1)
        values = torch.cat([memory_values.float(), self.value(memory_hidden).float()], dim=-1)
        context = weights @ values
        correction = self.head(torch.cat([query_hidden.float(), context], -1))
        return (correction, weights) if return_attention else correction


def validate_memory_split(memory_rows, outer_train_rows, outer_validation_rows):
    """Fail closed if a held-out row could enter the label-bearing memory."""
    memory = set(map(int, memory_rows))
    train = set(map(int, outer_train_rows))
    validation = set(map(int, outer_validation_rows))
    if train & validation or not memory.issubset(train) or memory & validation:
        raise ValueError('Outer validation leakage in retrieval memory')


def self_check():
    torch.manual_seed(17)
    model = CrossWellAttention(hidden_size=12, key_size=8).eval()
    q = torch.randn(4, 12)
    h = torch.randn(8, 12)
    values = torch.randn(8, 9)
    mw = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    qw = torch.arange(4)
    out, attention = model(q, h, values, mw, qw, return_attention=True)
    assert out.shape == (4, 75) and torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))
    assert torch.allclose(attention.sum(-1), torch.ones(4))
    assert torch.count_nonzero(attention[qw[:, None] == mw[None, :]]) == 0
    # Changing excluded same-well labels cannot affect this query's context.
    changed = values.clone(); changed[mw == 0] = 1000
    nn.init.normal_(model.head[-1].weight, std=.01)
    before = model(q, h, values, mw, qw)
    after = model(q, h, changed, mw, qw)
    assert torch.equal(before[0], after[0])
    model.train(); model(q, h, values, mw, qw).square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    validate_memory_split([0, 2], [0, 1, 2], [3, 4])
    try:
        validate_memory_split([0, 3], [0, 1, 2], [3, 4])
    except ValueError:
        pass
    else:
        raise AssertionError('Held-out memory row was not rejected')
    print('PASS: same-well labels masked; held-out memory rejected; finite forward/backward')


if __name__ == '__main__':
    self_check()
