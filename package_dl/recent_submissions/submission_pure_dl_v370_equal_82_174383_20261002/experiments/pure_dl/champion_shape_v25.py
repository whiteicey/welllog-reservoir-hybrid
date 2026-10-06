"""Champion preserved; train only two small input-conditioned location heads.

The raw side input never enters either backbone or router. Only component
locations change. All previous weights, including the old point32 residual,
remain fixed. The two modes have identical parameter tensors at initialization.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from independent_channels_v8 import Network as PrivateNetwork
from champion_scaling_v21 import PRIVATE, HEADS, REFERENCE, load_data as load_base_data

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/champion_shape_v25_20260928'
FEATURES = ROOT/'results/deep_learning/feature_structure_v20_20260928'
MODES = ('amplitude', 'shape')


class SideLocation(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.side = None
        self.enabled = True
        self.delta_in = nn.Conv1d(128+63, 32, 1)
        self.delta_out = nn.Conv1d(32, 8, 1)
        nn.init.zeros_(self.delta_out.weight)
        nn.init.zeros_(self.delta_out.bias)
        self.bound_fraction = .25

    def forward(self, h, valid, raw_scale):
        old = self.original(h, valid, raw_scale)
        if not self.enabled:
            return old
        if self.side is None or self.side.shape != (h.shape[0], h.shape[2], 63):
            raise RuntimeError('Missing aligned input-only side view')
        side = self.side.transpose(1, 2).to(h.dtype).masked_fill(~valid[:, None], 0)
        hidden = F.gelu(self.delta_in(torch.cat((h, side), 1)))
        proposal = self.delta_out(hidden).float()
        bound = self.bound_fraction*(F.softplus(raw_scale.detach().float())+.025)
        shift = (bound*torch.tanh(proposal/bound)).masked_fill(~valid[:, None], 0)
        return old+shift


class Network(PrivateNetwork):
    def __init__(self, master, previous, champion, mode):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(master, previous, 'POR', 'dual')
        self.load_state_dict(champion['state_dict'], strict=True)
        self.requires_grad_(False)
        self.mode = mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261101)
            for member in self.members:
                member.main_head.residual = SideLocation(member.main_head.residual)
        self.trainable_names = {n for n,p in self.named_parameters() if p.requires_grad}
        assert self.trainable_names and all('.delta_' in n for n in self.trainable_names)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == 12816
        assert sum(p.numel() for p in self.parameters() if not p.requires_grad) == 1290490

    def forward(self, x, relative, valid, absolute_depth=None):
        if x.shape[-1] != 269:
            raise ValueError('Expected unchanged206 + aligned side63')
        for member in self.members:
            member.main_head.residual.side = x[...,206:]
        try:
            return super().forward(x[...,:206].contiguous(), relative, valid, absolute_depth)
        finally:
            for member in self.members:
                member.main_head.residual.side = None

    def train(self, mode=True):
        # The old predictive function is always evaluated as originally retained.
        super().train(False)
        self.training = mode
        for member in self.members:
            member.main_head.residual.delta_in.train(mode)
            member.main_head.residual.delta_out.train(mode)
        return self


def make_model(mode, master):
    previous = torch.load(HEADS, map_location='cpu', weights_only=False)
    champion = torch.load(PRIVATE, map_location='cpu', weights_only=False)
    return Network(master, previous, champion, mode)


def load_data(mode):
    if mode not in MODES:
        raise ValueError(mode)
    data, master = load_base_data()
    meta = json.loads((FEATURES/'feature_manifest.json').read_text(encoding='utf-8'))
    train_wells = set(np.unique(data['wells'][data['fold'] != 4]).tolist())
    assert train_wells == set(meta['fit_wells'])
    assert not meta['labels_used_to_construct_features']
    assert len(data['validation']) == meta['monitor_points'] == 144665
    with np.load(FEATURES/'side_views.npz') as archive:
        values = archive[mode]
    assert values.shape == (len(data['x']),63) and np.isfinite(values).all()
    side = torch.as_tensor(values, device='cuda', dtype=data['x'].dtype)
    data['x'] = torch.cat((data['x'],side), -1)
    return data, master


def state_signature(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
