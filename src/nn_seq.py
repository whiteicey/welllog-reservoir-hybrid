"""Whole-well 1D sequence network with metric-aligned discrete heads.

Differences from the earlier "plain TCN replacement" that failed:
  * outputs are default-gate logits + quantile-bin class logits per target, decoded with the
    official-kernel Bayes action (not a point regression head);
  * per-well robust normalisation channels + global-context pooling let the net infer
    well-level calibration (Rw, matrix density) from the whole well;
  * calibration-shift augmentation (per-curve gain/offset jitter) targets cross-well drift;
  * it is trained/evaluated under the same 5 well folds and written to the same OOF format,
    so it is blended with CatBoost rather than replacing it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "ref" / "solution_v2"))
from src.data import load_split, TARGET_COLUMNS  # noqa: E402
from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, acc_vec, breakdown, default_mask,
                     load_or_build_cache, paired_well_bootstrap, quantile_bins, utility_action, total)

CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT", "DEVI"]
LOG_CURVES = {"RXO", "RT"}
NBINS = {"POR": 80, "PERM": 60, "SW": 60}
TFWD = {"POR": np.log1p, "PERM": np.log10, "SW": lambda v: v}
TINV = {"POR": np.expm1, "PERM": lambda v: 10 ** v, "SW": lambda v: v}


# ----------------------------------------------------------------------------- data
def prep_well(frame, gstats):
    """Return (C, L) float32 input tensor for one well and the row mask of valid inputs."""
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


def global_stats(frames):
    st = {}
    for c in CURVES:
        v = np.concatenate([f[c].dropna().to_numpy(np.float64) for f in frames])
        if c in LOG_CURVES:
            v = np.log10(np.clip(v, 0.01, None))
        q1, q3 = np.percentile(v, [25, 75])
        st[c] = {"median": float(np.median(v)), "iqr": float(max(q3 - q1, 1e-3))}
    return st


class WellData:
    def __init__(self, data_dir: Path, use_base=False, medians=None, fstats=None):
        self.train = load_split(data_dir, "train")
        self.test = load_split(data_dir, "test")
        self.gstats = global_stats([f for _, f in self.train])
        self.xtr = [prep_well(f, self.gstats) for _, f in self.train]
        self.xte = [prep_well(f, self.gstats) for _, f in self.test]
        self.use_base = use_base
        self.fstats = fstats
        if use_base:
            from src.features import FeatureBuilder
            b = FeatureBuilder()
            if medians is None:
                b.fit([f for _, f in self.train])
            else:
                b.medians_ = dict(medians); b.base_feature_names_ = []; b.enhanced_feature_names_ = []
                m0 = b._transform_frames(self.train[0][1]); b.base_feature_names_ = m0[0].columns.tolist(); b.enhanced_feature_names_ = m0[1].columns.tolist()
            ftr = [b.transform(f).base.T.astype(np.float32) for _, f in self.train]  # (113, L)
            fte = [b.transform(f).base.T.astype(np.float32) for _, f in self.test]
            if self.fstats is None:
                allf = np.concatenate(ftr, 1)
                med = np.median(allf, 1); q1, q3 = np.percentile(allf, 25, axis=1), np.percentile(allf, 75, axis=1)
                self.fstats = {"median": med.tolist(), "scale": np.maximum(q3 - q1, 1e-3).tolist()}
            med = np.asarray(self.fstats["median"], np.float32)[:, None]; sc = np.asarray(self.fstats["scale"], np.float32)[:, None]
            self.xtr = [np.concatenate([x, np.clip((f - med) / sc, -8, 8)]) for x, f in zip(self.xtr, ftr)]
            self.xte = [np.concatenate([x, np.clip((f - med) / sc, -8, 8)]) for x, f in zip(self.xte, fte)]
        self.ytr = [f[TARGET_COLUMNS].to_numpy(np.float64) for _, f in self.train]  # NaN where unlabeled
        self.n_in = self.xtr[0].shape[0]


# ----------------------------------------------------------------------------- model
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
    def __init__(self, n_in, ch=128, dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32), drop=0.1, k=5):
        super().__init__()
        self.inp = nn.Conv1d(n_in, ch, 7, padding=3)
        self.blocks = nn.ModuleList([ResBlock(ch, d, k=k, drop=drop) for d in dils])
        self.ctx = nn.Sequential(nn.Linear(ch, ch), nn.GELU(), nn.Linear(ch, ch))
        self.mix = nn.Conv1d(2 * ch, ch, 1)
        self.heads = nn.ModuleDict({t: nn.Conv1d(ch, 1 + NBINS[t], 1) for t in TARGET_COLUMNS})

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        g = self.ctx(h.mean(-1))[:, :, None].expand(-1, -1, h.shape[-1])
        h = F.gelu(self.mix(torch.cat([h, g], 1)))
        return {t: self.heads[t](h) for t in TARGET_COLUMNS}


# ----------------------------------------------------------------------------- training
def make_bins(ys, fold_train_idx):
    """Per-target quantile bins from training-fold labels (non-default only)."""
    bins = {}
    for ti, t in enumerate(TARGET_COLUMNS):
        v = np.concatenate([ys[i][:, ti] for i in fold_train_idx])
        v = v[np.isfinite(v)]
        v = v[~default_mask(v, t)]
        labels, centers, edges = quantile_bins(TFWD[t](v), NBINS[t])
        assert len(centers) == NBINS[t], (t, len(centers))
        bins[t] = {"centers": centers, "edges": edges}
    return bins


def targets_for_well(y, bins):
    """gate (1 default / 0 nondefault / -1 unlabeled) and bin label (-1 where default/unlabeled)."""
    out = {}
    for ti, t in enumerate(TARGET_COLUMNS):
        v = y[:, ti]
        lab = np.isfinite(v)
        d = default_mask(np.where(lab, v, 0.0), t) & lab
        gate = np.where(lab, d.astype(np.int64), -1)
        edges = bins[t]["edges"]
        b = np.clip(np.searchsorted(edges, TFWD[t](np.where(lab & ~d, v, 1.0)), side="right") - 1, 0, NBINS[t] - 1)
        cls = np.where(lab & ~d, b, -1)
        out[t] = (gate, cls)
    return out


def augment(x, rng, gain=0.08, shift=0.15):
    """Per-curve gain/offset jitter on the normalised channels (first 2*len(CURVES) channels)."""
    x = x.copy()
    nc = 2 * len(CURVES)
    g = 1.0 + rng.normal(0, gain, (nc, 1)).astype(np.float32)
    o = rng.normal(0, shift, (nc, 1)).astype(np.float32)
    x[:nc] = x[:nc] * g + o
    return x


def train_fold(D, bins, train_idx, args, device, seed, pseudo=None):
    """pseudo: optional dict {key: (x, y_pseudo)} of extra wells trained with weight args.pseudo_weight."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    net = WellNet(D.n_in, ch=args.channels, drop=args.dropout, k=getattr(args, "kernel", 5)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * args.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    pool = [(D.xtr[i], targets_for_well(D.ytr[i], bins), 1.0) for i in train_idx]
    pw_ = float(getattr(args, "pseudo_weight", 0.5))
    if pseudo:
        pool += [(x, targets_for_well(yp, bins), pw_) for x, yp in pseudo.values()]
    lengths = np.array([q[0].shape[1] for q in pool], dtype=np.float64)
    pw = lengths / lengths.sum()
    L = args.crop
    step = 0
    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for _ in range(args.steps_per_epoch):
            xb, wb, gb, cb = [], [], {t: [] for t in TARGET_COLUMNS}, {t: [] for t in TARGET_COLUMNS}
            for _ in range(args.batch):
                xw, tgw, ww = pool[rng.choice(len(pool), p=pw)]
                n = xw.shape[1]
                s = rng.integers(0, max(n - L, 0) + 1)
                x = xw[:, s:s + L]
                if x.shape[1] < L:
                    x = np.pad(x, ((0, 0), (0, L - x.shape[1])))
                xb.append(augment(x, rng) if args.augment else x); wb.append(ww)
                for t in TARGET_COLUMNS:
                    g, c = tgw[t]
                    g, c = g[s:s + L], c[s:s + L]
                    if len(g) < L:
                        g = np.pad(g, (0, L - len(g)), constant_values=-1); c = np.pad(c, (0, L - len(c)), constant_values=-1)
                    gb[t].append(g); cb[t].append(c)
            xb = torch.from_numpy(np.stack(xb)).to(device)
            wt = torch.tensor(wb, dtype=torch.float32, device=device)[:, None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xb)
            loss = 0.0
            for t in TARGET_COLUMNS:
                o = out[t].float()
                g = torch.from_numpy(np.stack(gb[t])).to(device); c = torch.from_numpy(np.stack(cb[t])).to(device)
                lg = F.binary_cross_entropy_with_logits(o[:, 0], (g == 1).float(), reduction="none") * wt
                loss = loss + (lg * (g >= 0)).sum() / ((g >= 0) * wt).sum().clamp(min=1)
                lc = F.cross_entropy(o[:, 1:], c.clamp(min=0), reduction="none", label_smoothing=0.0) * wt
                loss = loss + args.bin_weight * (lc * (c >= 0)).sum() / ((c >= 0) * wt).sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step(); sched.step(); step += 1
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  seed {seed} epoch {ep + 1}/{args.epochs} loss {tot / args.steps_per_epoch:.4f}", flush=True)
    return net


@torch.no_grad()
def predict_well(net, x, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="nn_seq")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--bin-weight", type=float, default=1.0)
    ap.add_argument("--augment", type=int, default=1)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--cv-dir", type=Path, default=Path("outputs/cv_v2"))
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--use-base", type=int, default=0)
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--pseudo-oof", type=Path, default=None, help="final_oof.npy (rows aligned with cache) used as pseudo labels for the held-out wells of each fold")
    ap.add_argument("--pseudo-weight", type=float, default=0.5)
    args = ap.parse_args()
    device = torch.device("cuda")
    out_dir = Path("outputs") / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    run_folds = [int(f) for f in args.folds.split(",")]
    t0 = time.time()
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells, pos = C["tr_y"], C["tr_fold"], C["tr_well"], C["tr_pos"]
    n, nt = len(y), len(C["te_well"])
    if args.eval_only:
        oof = dict(np.load(out_dir / "oof.npz")); log = []
        evaluate(args, out_dir, oof, run_folds, log, C)
        return
    D = WellData(Path("data"), use_base=bool(args.use_base))
    print("prepared wells", len(D.xtr), "channels", D.n_in, f"{time.time() - t0:.1f}s", flush=True)
    # containers in the shared OOF layout (rows aligned with cache order)
    oof = {f"{t}_gate": np.zeros(n) for t in TARGET_COLUMNS}
    oof.update({f"{t}_prob": np.zeros((n, NBINS[t]), np.float32) for t in TARGET_COLUMNS})
    oof.update({f"{t}_centers": np.zeros((N_FOLDS, NBINS[t])) for t in TARGET_COLUMNS})
    te = {f"{t}_gate": np.zeros((N_FOLDS, nt)) for t in TARGET_COLUMNS}
    te.update({f"{t}_prob": np.zeros((N_FOLDS, nt, NBINS[t]), np.float32) for t in TARGET_COLUMNS})
    well_fold = {int(w): int(folds[wells == w][0]) for w in np.unique(wells)}
    pseudo_oof = np.load(args.pseudo_oof) if args.pseudo_oof else None
    log = []
    for f in run_folds:
        tf = time.time()
        train_idx = [w for w in range(len(D.xtr)) if well_fold.get(w, -1) != f]
        val_idx = [w for w in range(len(D.xtr)) if well_fold.get(w, -1) == f]
        bins = make_bins(D.ytr, train_idx)
        for t in TARGET_COLUMNS:
            oof[f"{t}_centers"][f] = bins[t]["centers"]
        pseudo = None
        if pseudo_oof is not None:
            pseudo = {}
            for w in val_idx:
                rows = np.flatnonzero(wells == w)
                yp = np.full(D.ytr[w].shape, np.nan)
                yp[pos[rows]] = pseudo_oof[rows]
                pseudo[w] = (D.xtr[w], yp)
            print(f"  fold {f}: {len(pseudo)} pseudo-labelled held-out wells (weight {args.pseudo_weight})", flush=True)
        nets = [train_fold(D, bins, train_idx, args, device, seed, pseudo=pseudo) for seed in seeds]
        for w in val_idx:
            rows = np.flatnonzero(wells == w); p = pos[rows]
            acc = None
            for net in nets:
                r = predict_well(net, D.xtr[w], device)
                acc = r if acc is None else {t: (acc[t][0] + r[t][0], acc[t][1] + r[t][1]) for t in TARGET_COLUMNS}
            for t in TARGET_COLUMNS:
                oof[f"{t}_gate"][rows] = acc[t][0][p] / len(nets)
                oof[f"{t}_prob"][rows] = acc[t][1][p] / len(nets)
        for w in range(len(D.xte)):
            rows = np.flatnonzero(C["te_well"] == w)
            acc = None
            for net in nets:
                r = predict_well(net, D.xte[w], device)
                acc = r if acc is None else {t: (acc[t][0] + r[t][0], acc[t][1] + r[t][1]) for t in TARGET_COLUMNS}
            for t in TARGET_COLUMNS:
                te[f"{t}_gate"][f, rows] = acc[t][0] / len(nets)
                te[f"{t}_prob"][f, rows] = acc[t][1] / len(nets)
        log.append({"fold": f, "seconds": time.time() - tf})
        print(log[-1], flush=True)
        np.savez(out_dir / "oof.npz", **oof); np.savez(out_dir / "test.npz", **te)
    evaluate(args, out_dir, oof, run_folds, log, C)


def evaluate(args, out_dir, oof, run_folds, log, C):
    y, folds, wells = C["tr_y"], C["tr_fold"], C["tr_well"]
    n = len(y)
    done = np.isin(folds, run_folds)
    if not (args.cv_dir / "oof.npz").exists():
        print("no v2 OOF available, skipping evaluation"); return
    O = np.load(args.cv_dir / "oof.npz")
    rep = {"log": log, "args": {k: str(v) for k, v in vars(args).items()}}
    rep["transductive"] = bool(getattr(args, "pseudo_oof", None))
    scores_nn, scores_v2 = {}, {}
    for ti, t in enumerate(TARGET_COLUMNS):
        cont = np.zeros(n)
        for f in run_folds:
            va = folds == f
            pp = oof[f"{t}_prob"][va].astype(np.float64); pp /= pp.sum(1, keepdims=True)
            sup = TINV[t](oof[f"{t}_centers"][f])
            dense = np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))
            cont[va] = utility_action(pp, sup, t, actions=dense)
        g = oof[f"{t}_gate"]
        d = default_mask(y[:, ti], t)
        pred_nn = np.where(g >= 0.5, DEFAULTS[t], np.clip(cont, *BOUNDS[t]))
        # v2 reference
        gk = {"POR": "por_gate", "PERM": "perm_gate", "SW": "sw_gate"}[t]
        if t == "POR":
            v2c = np.zeros(n)
            for f in range(N_FOLDS):
                va = folds == f
                v2c[va] = np.expm1(O["por_centers"][f])[np.argmax(O["por_prob"][va], 1)]
        elif t == "PERM":
            v2c = np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"])
        else:
            v2c = np.clip(O["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"])
        pred_v2 = np.where(O[gk] >= V2_CAL[t]["threshold"], DEFAULTS[t], v2c)
        pred_nn_v2gate = np.where(O[gk] >= V2_CAL[t]["threshold"], DEFAULTS[t], np.clip(cont, *BOUNDS[t]))
        joint = np.zeros(n)
        for f in run_folds:
            va = folds == f
            pp = oof[f"{t}_prob"][va].astype(np.float64); pp /= pp.sum(1, keepdims=True)
            sup = TINV[t](oof[f"{t}_centers"][f]); gg = g[va]
            dense = np.r_[DEFAULTS[t], np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))]
            joint[va] = utility_action(np.c_[gg, (1 - gg)[:, None] * pp], np.r_[DEFAULTS[t], sup], t, actions=dense)
        joint = np.clip(joint, *BOUNDS[t])
        rep[t] = {
            "nondefault_only": {"nn": float(acc_vec(cont[done & ~d], y[done & ~d, ti], t).mean()), "v2": float(acc_vec(v2c[done & ~d], y[done & ~d, ti], t).mean())},
            "gate_acc": {"nn@0.5": float(((g >= 0.5) == d)[done].mean()), "v2": float(((O[gk] >= V2_CAL[t]["threshold"]) == d)[done].mean())},
            "full": {"nn": float(acc_vec(pred_nn[done], y[done, ti], t).mean()), "nn_cont_v2gate": float(acc_vec(pred_nn_v2gate[done], y[done, ti], t).mean()),
                     "nn_joint": float(acc_vec(joint[done], y[done, ti], t).mean()),
                     "v2": float(acc_vec(pred_v2[done], y[done, ti], t).mean())},
            "paired_nn_cont_v2gate_vs_v2": paired_well_bootstrap(acc_vec(pred_nn_v2gate[done], y[done, ti], t), acc_vec(pred_v2[done], y[done, ti], t), wells[done]),
        }
        scores_nn[t] = rep[t]["full"]["nn"]; scores_v2[t] = rep[t]["full"]["v2"]
    rep["TOTAL"] = {"nn": total(scores_nn), "v2": total(scores_v2)}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({t: rep[t] for t in TARGET_COLUMNS} | {"TOTAL": rep["TOTAL"]}, indent=1))


if __name__ == "__main__":
    main()
