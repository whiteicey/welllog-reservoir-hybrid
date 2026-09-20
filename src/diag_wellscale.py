"""Per-well bias: is the OOF-optimal multiplicative scale of a well predictable from label-free well statistics?

If yes, a test well's scale can be predicted from its own curves (legal). Evaluated leave-one-well-out.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from harness import load_or_build_cache, default_mask, acc_vec, BOUNDS, N_FOLDS

C = load_or_build_cache(Path("data"), Path("cache"))
y, wells, raw = C["tr_y"], C["tr_well"], C["tr_raw"]
folds = C["tr_fold"]
cols = C["meta"]["raw_cols"]
O = np.load("outputs/cv_v2/oof.npz")
ws = np.unique(wells)
# label-free well descriptors: median / IQR / p10 / p90 of each curve + depth range + length
X = raw.copy()
for c in ("RT", "RXO"):
    X[:, cols.index(c)] = np.log10(np.clip(X[:, cols.index(c)], 0.01, None))
desc = []
for w in ws:
    m = wells == w; xw = X[m]
    d = [np.nanmedian(xw[:, j]) for j in range(xw.shape[1])] + [np.nanpercentile(xw[:, j], 75) - np.nanpercentile(xw[:, j], 25) for j in range(xw.shape[1])]
    d += [np.nanpercentile(xw[:, j], 10) for j in range(xw.shape[1])] + [np.nanpercentile(xw[:, j], 90) for j in range(xw.shape[1])]
    dep = C["tr_depth"][m]; d += [dep.min(), dep.max(), len(dep)]
    desc.append(d)
desc = np.nan_to_num(np.array(desc))
R = {}
scales = np.linspace(0.80, 1.25, 91)
for ti, t, cont in [(0, "POR", None), (2, "SW", np.clip(O["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"])), (1, "PERM", np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"]))]:
    if t == "POR":
        cont = np.zeros(len(y))
        for f in range(N_FOLDS):
            va = folds == f; cont[va] = np.expm1(O["por_centers"][f])[np.argmax(O["por_prob"][va], 1)]
    d = default_mask(y[:, ti], t)
    best_s = np.zeros(len(ws)); base = np.zeros(len(ws)); oracle = np.zeros(len(ws)); cnt = np.zeros(len(ws))
    for i, w in enumerate(ws):
        m = (wells == w) & ~d
        cnt[i] = m.sum()
        sc = [acc_vec(cont[m] * s, y[m, ti], t).mean() for s in scales]
        best_s[i] = scales[int(np.argmax(sc))]; oracle[i] = max(sc); base[i] = acc_vec(cont[m], y[m, ti], t).mean()
    # LOO prediction of log-scale from descriptors
    Z = (desc - desc.mean(0)) / (desc.std(0) + 1e-9)
    pred_s = np.zeros(len(ws))
    for i in range(len(ws)):
        tr = np.arange(len(ws)) != i
        model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(Z[tr], np.log(best_s[tr]))
        pred_s[i] = float(np.exp(model.predict(Z[i:i + 1])[0]))
    pred_s = np.clip(pred_s, scales[0], scales[-1])
    applied = np.array([acc_vec(cont[(wells == w) & ~d] * pred_s[i], y[(wells == w) & ~d, ti], t).mean() for i, w in enumerate(ws)])
    # shrunk version (half-way to 1)
    half = np.sqrt(pred_s)
    applied_half = np.array([acc_vec(cont[(wells == w) & ~d] * half[i], y[(wells == w) & ~d, ti], t).mean() for i, w in enumerate(ws)])
    wsum = cnt / cnt.sum()
    R[t] = {"nondefault_weighted_score": {"base": float((base * wsum).sum()), "oracle_scale": float((oracle * wsum).sum()),
                                          "loo_predicted_scale": float((applied * wsum).sum()), "loo_predicted_scale_half": float((applied_half * wsum).sum())},
            "best_scale_quantiles": np.quantile(best_s, [.1, .25, .5, .75, .9]).tolist(),
            "corr_pred_vs_best_logscale": float(np.corrcoef(np.log(pred_s), np.log(best_s))[0, 1]),
            "wells_improved_by_loo_scale": int((applied > base).sum())}
    print(t, R[t], flush=True)
Path("outputs/diag_wellscale.json").write_text(json.dumps(R, indent=1))
