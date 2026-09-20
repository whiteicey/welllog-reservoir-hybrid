"""Type-well selection criteria compared (leave-well-out, LightGBM on 13 curves + depth, non-default samples).

For each of 16 evaluation wells and K in {12, 30}: train on the K nearest training wells chosen by
(a) median-curve distance, (b) KL divergence, (c) GR-DTW, (d) composite-DTW; plus all 79 wells; plus
similarity-weighted training on all 79 wells (weights exp(-d/scale)). Extends the earlier diag_wells table.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from harness import load_or_build_cache, acc_vec, default_mask

C = load_or_build_cache(Path("data"), Path("cache"))
y, wells, raw = C["tr_y"], C["tr_well"], C["tr_raw"]
cols = C["meta"]["raw_cols"]
X = raw.copy()
for c in ("RXO", "RT"):
    j = cols.index(c); X[:, j] = np.log10(np.clip(X[:, j], 0.01, None))
X = np.nan_to_num(np.c_[X, C["tr_depth"]])
ws = np.unique(wells); n_tr = len(ws)
S = {n: np.load(f"cache/sim_{n}.npy")[:n_tr, :n_tr] for n in ("median", "kl", "dtw_gr", "dtw_comp")}
rng = np.random.default_rng(0)
eval_wells = rng.choice(ws, 16, replace=False)


def fit_pred(Xtr, ytr, Xte, wtr=None):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, objective="l1", verbose=-1, n_jobs=6)
    return m.fit(Xtr, ytr, sample_weight=wtr).predict(Xte)


out = {}
for ti, t in [(0, "POR"), (2, "SW")]:
    nd = ~default_mask(y[:, ti], t)
    for crit in S:
        for Kn in (12, 30):
            accs = []
            for w in eval_wells:
                d = S[crit][w].copy(); d[w] = np.inf; nb = np.argsort(d)[:Kn]
                trm = np.isin(wells, nb) & nd; tem = (wells == w) & nd
                accs.append(acc_vec(fit_pred(X[trm], y[trm, ti], X[tem]), y[tem, ti], t).mean())
            out[f"{t}_{crit}_K{Kn}"] = float(np.mean(accs)); print(t, crit, Kn, out[f"{t}_{crit}_K{Kn}"], flush=True)
        # similarity-weighted training on all other wells (scale = median distance)
        accs = []
        for w in eval_wells:
            d = S[crit][w].copy(); d[w] = np.inf; scale = np.median(d[np.isfinite(d)])
            wt = np.exp(-d / max(scale, 1e-9)); trm = (wells != w) & nd; tem = (wells == w) & nd
            accs.append(acc_vec(fit_pred(X[trm], y[trm, ti], X[tem], wtr=wt[wells[trm]]), y[tem, ti], t).mean())
        out[f"{t}_{crit}_weighted_all"] = float(np.mean(accs)); print(t, crit, "weighted", out[f"{t}_{crit}_weighted_all"], flush=True)
    accs = []
    for w in eval_wells:
        trm = (wells != w) & nd; tem = (wells == w) & nd
        accs.append(acc_vec(fit_pred(X[trm], y[trm, ti], X[tem]), y[tem, ti], t).mean())
    out[f"{t}_all79"] = float(np.mean(accs)); print(t, "all79", out[f"{t}_all79"], flush=True)
out["eval_wells"] = [int(w) for w in eval_wells]
Path("outputs/diag_typewell.json").write_text(json.dumps(out, indent=1))
