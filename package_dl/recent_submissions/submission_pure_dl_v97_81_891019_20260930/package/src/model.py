"""Model definitions and loaders.

CatBoost : default-value gates, quantile-bin distribution models, 4-state reservoir-state model
WellNet  : whole-well 1-D dilated residual convolutional network with a gate head and a bin head per property
level-2  : feature assembly for the optional stacked state model
"""
from __future__ import annotations

import numpy as np

from .data import TARGET_COLUMNS

TARGETS = ("POR", "PERM", "SW")


# ----------------------------------------------------------------------------- CatBoost
def load_cb(model_dir, files):
    from catboost import CatBoostClassifier
    out = []
    for f in files:
        m = CatBoostClassifier(); m.load_model(model_dir / f); out.append(m)
    return out


def mean_proba(models, X, classes=None):
    """Seed-averaged class probabilities; columns re-ordered to 0..K-1 when `classes` is given."""
    acc = 0
    for m in models:
        p = m.predict_proba(X)
        if classes is not None:
            order = np.argsort(m.classes_.astype(int)); p = p[:, order]
        acc = acc + p
    return acc / len(models)


# ----------------------------------------------------------------------------- sequence network
CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI"]
LOG_CURVES = {"RXO", "RT"}
NBINS = {"POR": 80, "PERM": 60, "SW": 60}


def prep_well(frame, gstats):
    """Return the (C, L) float32 input tensor of one well: per curve a globally robust-scaled and a
    within-well robust-scaled channel, missing-value masks, relative and absolute depth."""
    chans, masks = [], []
    for c in CURVES:
        s = frame[c].astype(np.float64)
        miss = s.isna().to_numpy()
        s = s.interpolate(limit_direction="both")
        if s.isna().all():
            s = s.fillna(gstats[c]["median"])
        v = s.to_numpy()
        if c in LOG_CURVES:
            v = np.log10(np.clip(v, 0.01, None))
        g = (v - gstats[c]["median"]) / gstats[c]["iqr"]
        med, q1, q3 = np.median(v), np.percentile(v, 25), np.percentile(v, 75)
        w = (v - med) / max(q3 - q1, 1e-3)
        chans += [g, w]
        masks.append(miss.astype(np.float64))
    depth = frame["DEPTH"].to_numpy(np.float64)
    rel = (depth - depth.min()) / max(depth.max() - depth.min(), 1.0)
    chans += [rel, (depth - 1800.0) / 600.0]
    x = np.stack(chans + masks).astype(np.float32)
    return np.nan_to_num(x)


def append_base_channels(x, base_matrix, fstats):
    """Optional variant: append the 113 base features (robust-scaled with training statistics) as channels."""
    fmed = np.asarray(fstats["median"], np.float32)[:, None]
    fsc = np.asarray(fstats["scale"], np.float32)[:, None]
    return np.concatenate([x, np.clip((base_matrix.T.astype(np.float32) - fmed) / fsc, -8, 8)])


def build_wellnet(spec, device):
    """Instantiate WellNet from a model_meta.json entry."""
    import torch
    net = WellNet(spec["n_in"], ch=spec["channels"], drop=spec["dropout"], k=spec.get("kernel", 5),
                  nbins=spec.get("nbins", NBINS)).to(device)
    return net


def load_wellnets(model_dir, spec, device):
    import torch
    nets = []
    for f in spec["files"]:
        net = build_wellnet(spec, device)
        net.load_state_dict(torch.load(model_dir / f, map_location=device, weights_only=True))
        net.eval()
        nets.append(net)
    return nets


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        def __init__(self, ch, dil, k=5, drop=0.1):
            super().__init__()
            self.c1 = nn.Conv1d(ch, ch, k, padding=dil * (k // 2), dilation=dil)
            self.c2 = nn.Conv1d(ch, ch, k, padding=dil * (k // 2), dilation=dil)
            self.n1 = nn.GroupNorm(8, ch); self.n2 = nn.GroupNorm(8, ch)
            self.drop = nn.Dropout(drop)
            self.se = nn.Sequential(nn.Linear(ch, ch // 4), nn.GELU(), nn.Linear(ch // 4, ch), nn.Sigmoid())

        def forward(self, x):
            h = self.c1(F.gelu(self.n1(x)))
            h = self.c2(self.drop(F.gelu(self.n2(h))))
            s = self.se(h.mean(-1))[:, :, None]
            return x + h * s

    class WellNet(nn.Module):
        def __init__(self, n_in, ch=128, dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32), drop=0.1, k=5, nbins=None):
            super().__init__()
            nbins = NBINS if nbins is None else nbins
            self.inp = nn.Conv1d(n_in, ch, 7, padding=3)
            self.blocks = nn.ModuleList([ResBlock(ch, d, k=k, drop=drop) for d in dils])
            self.ctx = nn.Sequential(nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, ch))
            self.mix = nn.Conv1d(2 * ch, ch, 1)
            self.heads = nn.ModuleDict({t: nn.Conv1d(ch, 1 + nbins[t], 1) for t in TARGET_COLUMNS})

        def forward(self, x):
            h = self.inp(x)
            for b in self.blocks:
                h = b(h)
            g = self.ctx(h.mean(-1))[:, :, None].expand(-1, -1, h.shape[-1])
            h = F.gelu(self.mix(torch.cat([h, g], 1)))
            return {t: self.heads[t](h) for t in TARGET_COLUMNS}

    @torch.no_grad()
    def predict_well(net, x, device):
        """One forward pass over the whole well. Returns {target: (P(default) (L,), bin probabilities (L, K))}."""
        net.eval()
        L = x.shape[1]
        pad = (-L) % 32
        xt = torch.from_numpy(np.pad(x, ((0, 0), (0, pad)))[None]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(xt)
        res = {}
        for t in TARGET_COLUMNS:
            o = out[t].float()[0, :, :L]
            res[t] = (torch.sigmoid(o[0]).cpu().numpy(), torch.softmax(o[1:], 0).T.cpu().numpy())
        return res

except ImportError:  # torch is only needed when model_meta.json lists sequence networks
    WellNet = None
    predict_well = None


# ----------------------------------------------------------------------------- level-2 stack (optional)
def dist_summary(prob, centers):
    prob = prob.astype(np.float64); prob /= prob.sum(1, keepdims=True)
    mean = prob @ centers
    std = np.sqrt(np.maximum(prob @ centers ** 2 - mean ** 2, 0))
    ent = -(prob * np.log(prob + 1e-12)).sum(1)
    return np.c_[mean, std, ent, prob.max(1)].astype(np.float32)


def level2_features(static, gates3, post, nn_outputs, tree_summ, nn_order):
    """static (n, 294); gates3 (n, 3) POR/PERM/SW default probs; post (n, 4) state posterior;
    nn_outputs {name: (gates (n,3), {t: summary (n,4)})}; tree_summ {t: (n,4)}; nn_order tuple of names."""
    cols = [static, gates3, post]
    for name in nn_order:
        g, summ = nn_outputs[name]
        cols.append(g)
        for t in TARGETS:
            cols.append(summ[t])
    for t in TARGETS:
        cols.append(tree_summ[t])
    return np.column_stack(cols).astype(np.float32)
