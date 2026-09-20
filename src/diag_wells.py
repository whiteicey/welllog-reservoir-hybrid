"""Cross-well diagnostics: within-well split vs leave-well-out, nearest-well training, per-well oracle scale."""
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
    j = cols.index(c)
    X[:, j] = np.log10(np.clip(X[:, j], 0.01, None))
X = np.nan_to_num(np.c_[X, C["tr_depth"]])
out = {}


def fit_pred(Xtr, ytr, Xte, obj="l1"):
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=63, objective=obj, verbose=-1, n_jobs=16)
    return m.fit(Xtr, ytr).predict(Xte)


ident = lambda v: v  # noqa: E731
for ti, (t, tr_fn, inv) in enumerate([("POR", ident, ident), ("PERM", np.log10, lambda v: 10 ** v), ("SW", ident, ident)]):
    res = []
    for w in np.unique(wells)[:40]:
        m = (wells == w) & ~default_mask(y[:, ti], t)
        if m.sum() < 400:
            continue
        Xw, yw = X[m], y[m, ti]
        half = np.arange(len(yw)) < len(yw) // 2
        p1 = inv(fit_pred(Xw[half], tr_fn(yw[half]), Xw[~half]))
        p2 = inv(fit_pred(Xw[~half], tr_fn(yw[~half]), Xw[half]))
        res.append(0.5 * (acc_vec(p1, yw[~half], t).mean() + acc_vec(p2, yw[half], t).mean()))
    out[f"within_well_halfsplit_{t}"] = float(np.mean(res))
    print(t, "within-well half split", out[f"within_well_halfsplit_{t}"], flush=True)

ws = np.unique(wells)
nf = X.shape[1] - 1
stats = np.array([[np.nanmedian(X[wells == w, j]) for j in range(nf)] +
                  [np.nanpercentile(X[wells == w, j], 90) - np.nanpercentile(X[wells == w, j], 10) for j in range(nf)] for w in ws])
stats = (stats - stats.mean(0)) / (stats.std(0) + 1e-9)
rng = np.random.default_rng(0)
eval_wells = rng.choice(ws, 16, replace=False)
for ti, t in [(0, "POR"), (2, "SW")]:
    for K in (79, 30, 12):
        accs = []
        for w in eval_wells:
            d = np.sqrt(((stats - stats[w]) ** 2).sum(1))
            d[w] = np.inf
            nb = np.argsort(d)[:K]
            trm = np.isin(wells, nb) & ~default_mask(y[:, ti], t)
            tem = (wells == w) & ~default_mask(y[:, ti], t)
            p = fit_pred(X[trm], y[trm, ti], X[tem])
            accs.append(acc_vec(p, y[tem, ti], t).mean())
        out[f"lwo_{t}_K{K}"] = float(np.mean(accs))
        print(t, "leave-well-out K=", K, out[f"lwo_{t}_K{K}"], flush=True)

for ti, t in [(0, "POR"), (2, "SW")]:
    accs, accs_s = [], []
    for w in eval_wells:
        trm = ~np.isin(wells, eval_wells) & ~default_mask(y[:, ti], t)
        tem = (wells == w) & ~default_mask(y[:, ti], t)
        p = fit_pred(X[trm], y[trm, ti], X[tem])
        accs.append(acc_vec(p, y[tem, ti], t).mean())
        accs_s.append(max(acc_vec(p * s, y[tem, ti], t).mean() for s in np.linspace(0.8, 1.25, 46)))
    out[f"lwo_{t}_plain"] = float(np.mean(accs))
    out[f"lwo_{t}_oracle_scale"] = float(np.mean(accs_s))
    print(t, "leave-well-out plain", out[f"lwo_{t}_plain"], "oracle per-well scale", out[f"lwo_{t}_oracle_scale"], flush=True)
Path("outputs").mkdir(exist_ok=True)
Path("outputs/diag_wells.json").write_text(json.dumps(out, indent=1))
