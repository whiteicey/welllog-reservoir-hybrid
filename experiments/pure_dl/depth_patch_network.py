"""Small depth-window Transformer, inspired by patching, not a PatchTST replica."""
import numpy as np
import torch
from torch import nn
import train_invariant as inv

CURVES = ['GR', 'AC', 'DEN', 'CNL', 'RT', 'RXO', 'PE', 'SP', 'CAL']


class Network(inv.Network):
    def __init__(self, edges=None, initial_means=None):
        super().__init__(edges, initial_means)
        self.patch = nn.Linear(4*11, 64)
        self.query = nn.Linear(256, 64)
        self.position = nn.Parameter(torch.randn(17, 64)*.02)
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=.1,
                    activation='gelu', batch_first=True, norm_first=True)
        self.sequence = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(64), enable_nested_tensor=False)
        self.context_projection = nn.Linear(64, 256)
        self.context_gain = nn.Parameter(torch.full((256,), .1))

    def bind_context(self, point, raw, depth, mapping, lower, upper):
        # Runtime inputs are deliberately not checkpoint buffers or parameters.
        self.point, self.raw, self.depth = point, raw, depth
        self.mapping, self.lower, self.upper = mapping, lower, upper

    def window(self, rows):
        center = self.mapping[rows]
        index = center[:, None]+torch.arange(-32, 32, device=rows.device)
        valid = (index >= self.lower[center, None]) & (index < self.upper[center, None])
        safe = torch.maximum(self.lower[center, None], torch.minimum(index, self.upper[center, None]-1))
        values = self.raw[safe].float()
        relative_depth = torch.tanh((self.depth[safe]-self.depth[center, None])/10)
        tokens = torch.cat([values, relative_depth[:, :, None], valid[:, :, None].float()], -1)
        tokens = tokens*valid[:, :, None]
        padding = ~valid.reshape(len(rows), 16, 4).any(-1)
        return tokens.reshape(len(rows), 16, 44), padding, safe, valid

    def forward(self, rows):
        point = self.point[rows]
        h = self.norm(self.blocks(self.stem(self.encoding(point))))
        windows, padding, _, _ = self.window(rows)
        token = torch.cat([self.query(h)[:, None], self.patch(windows)], 1)
        mask = torch.cat([torch.zeros(len(rows), 1, device=rows.device, dtype=torch.bool), padding], 1)
        hidden = self.sequence(token+self.position[None].to(token.dtype), src_key_padding_mask=mask)
        h = h+self.context_gain*self.context_projection(hidden[:, 0])
        return self.gate(h), self.density(h).reshape(-1, 3, 3, inv.MIX)


def bind_real(model, point, rec, valid, builder, fm, fs, feature_names):
    selected = [feature_names.index(c) for c in CURVES]
    original = [builder.enhanced_feature_names_.index(c) for c in CURVES]
    curves, depths, mapping, lower, upper = [], [], [], [], []
    start = 0
    for i, (_, frame) in enumerate(rec):
        n = len(frame)
        values = builder.transform(frame).enhanced[:, original]
        curves.append(np.clip((values-fm[selected])/fs[selected], -8, 8).astype(np.float16))
        depths.append(frame['DEPTH'].to_numpy(np.float32))
        mapping.append(np.flatnonzero(valid[i])+start)
        lower.append(np.full(n, start, np.int64)); upper.append(np.full(n, start+n, np.int64))
        start += n
    arrays = [np.concatenate(v) for v in [curves, depths, mapping, lower, upper]]
    assert all(np.isfinite(v).all() for v in arrays)
    model.bind_context(point, *(torch.as_tensor(v, device=point.device) for v in arrays))


def bind_synthetic(model, point):
    n = len(point); device = point.device
    model.bind_context(point, point[:, :9], torch.arange(n, device=device)*.125,
                       torch.arange(n, device=device), torch.zeros(n, device=device, dtype=torch.long),
                       torch.full((n,), n, device=device, dtype=torch.long))


def self_test():
    model = Network().eval()
    point = torch.randn(10, inv.FEATURES)
    lower = torch.tensor([0]*5+[5]*5); upper = lower+5
    model.bind_context(point, torch.randn(10, 9), torch.arange(10).float(), torch.arange(10), lower, upper)
    rows = torch.tensor([0, 4, 5, 9])
    _, _, index, valid = model.window(rows)
    assert ((index < 5) == (rows[:, None] < 5)).all()
    assert (valid.sum(1) == 5).all()
    a, b = model(rows)
    (a.square().mean()+b.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with torch.no_grad():
        original = model(torch.tensor([0]))
        model.raw[5:] += 1000
        changed = model(torch.tensor([0]))
        assert all(torch.equal(x, y) for x, y in zip(original, changed))
    return {'passed': True, 'windows_never_cross_wells': True,
            'other_well_perturbation_invariant': True, 'padding_and_gradients_passed': True}
