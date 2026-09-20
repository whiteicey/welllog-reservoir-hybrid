"""High-frequency structure of labels vs curves inside non-default segments.

Question: is the sample-to-sample variation of POR/SW explained by the sample-to-sample variation of
any input curve (possibly at a ±1..3 lag)? If not, that variation is irreducible for any model.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from harness import load_or_build_cache, default_mask, acc_vec

C = load_or_build_cache(Path("data"), Path("cache"))
y, wells, pos, raw = C["tr_y"], C["tr_well"], C["tr_pos"], C["tr_raw"].copy()
cols = C["meta"]["raw_cols"]
for c in ("RT", "RXO"):
    raw[:, cols.index(c)] = np.log10(np.clip(raw[:, cols.index(c)], 0.01, None))
raw = np.nan_to_num(raw)
brk = np.flatnonzero((np.diff(wells) != 0) | (np.diff(pos) != 1)) + 1
segs = np.split(np.arange(len(y)), brk)
R = {}
for ti, t, fwd in [(0, "POR", lambda v: v), (2, "SW", np.log10), (1, "PERM", np.log10)]:
    d = default_mask(y[:, ti], t)
    dy, dx = [], []
    for idx in segs:
        m = ~d[idx]
        # contiguous nondefault sub-runs of length >= 8
        ch = np.r_[True, m[1:] != m[:-1]]; st = np.flatnonzero(ch); ln = np.diff(np.r_[st, len(idx)])
        for s0, l0 in zip(st, ln):
            if not m[s0] or l0 < 8:
                continue
            rows = idx[s0:s0 + l0]
            yy = fwd(y[rows, ti]); xx = raw[rows]
            dy.append(np.diff(yy))
            # lagged diffs of curves: lags -3..3
            feats = []
            for lag in range(-3, 4):
                sh = np.roll(xx, lag, axis=0)
                dd = np.diff(sh, axis=0)
                if lag != 0:
                    dd[:max(0, -lag)] = 0; dd[len(dd) - max(0, lag):] = 0
                feats.append(dd)
            dx.append(np.concatenate(feats, 1))
    dy = np.concatenate(dy); dx = np.concatenate(dx)
    res = {"n_diffs": int(len(dy)), "var_dy": float(dy.var())}
    # R2 of dy on all lagged curve diffs (linear, global)
    A = np.c_[dx, np.ones(len(dx))]
    coef, *_ = np.linalg.lstsq(A, dy, rcond=None)
    pred = A @ coef
    res["R2_dy_on_all_lagged_curve_diffs"] = float(1 - ((dy - pred) ** 2).sum() / ((dy - dy.mean()) ** 2).sum())
    # lag-0 only
    A0 = np.c_[dx[:, 3 * len(cols):4 * len(cols)], np.ones(len(dx))]
    coef0, *_ = np.linalg.lstsq(A0, dy, rcond=None)
    res["R2_dy_on_lag0_curve_diffs"] = float(1 - ((dy - A0 @ coef0) ** 2).sum() / ((dy - dy.mean()) ** 2).sum())
    # per-curve, best lag correlation
    per = {}
    for j, c in enumerate(cols):
        best = max(((lag, float(np.corrcoef(dy, dx[:, (lag + 3) * len(cols) + j])[0, 1])) for lag in range(-3, 4)), key=lambda z: abs(z[1]) if np.isfinite(z[1]) else 0)
        per[c] = {"best_lag": best[0], "corr": best[1]}
    res["per_curve_best_lag_corr"] = per
    # implied ceiling: if only the linear-explained part of dy is predictable, how does a predictor with
    # residual noise of that size score? simulate y_hat = y - (unexplained diff component)/2
    resid = dy - pred
    res["resid_sd_of_dy"] = float(resid.std())
    res["sd_dy"] = float(dy.std())
    R[t] = res
    print(t, {k: v for k, v in res.items() if k != "per_curve_best_lag_corr"}, flush=True)
    print("   ", {c: (v["best_lag"], round(v["corr"], 3)) for c, v in per.items()})
Path("outputs/diag_hf.json").write_text(json.dumps(R, indent=1))
