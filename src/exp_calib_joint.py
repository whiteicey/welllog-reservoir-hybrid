"""Nested gate-probability calibration + joint (default-mass + continuous distribution) Bayes decoding.

The joint action is only Bayes-optimal if P(default) is calibrated. v2 gates are trained with logloss but
averaged over seeds and thresholded far from 0.5, so we test: temperature scaling / isotonic fitted on the
4 inner folds, applied to the outer fold, then joint utility decoding without any threshold.
Also: expected-score-aware gating: predict default iff  p*U_dd + (1-p)*E[U_dc] >= p*U_cd + (1-p)*E[U_cc],
where E[U_cc] is the expected continuous accuracy from the predictive distribution (self-assessed).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, acc_vec, breakdown, default_mask, kernel_matrix,
                     load_or_build_cache, paired_well_bootstrap, utility_action)

C = load_or_build_cache(Path("data"), Path("cache"))
y, folds, wells = C["tr_y"], C["tr_fold"], C["tr_well"]
n = len(y)
O = np.load("outputs/cv_v2/oof.npz")
SRC = {
    "POR": {"gate": O["por_gate"], "dist": ("outputs/cv_v2/oof.npz", "por_prob", "por_centers", np.expm1)},
    "PERM": {"gate": O["perm_gate"], "dist": ("outputs/perm_bins60_base/oof.npz", "prob", "centers", lambda v: 10 ** v)},
    "SW": {"gate": O["sw_gate"], "dist": ("outputs/sw_bins60_enh/oof.npz", "prob", "centers", lambda v: v)},
}
R = {}
for ti, t in enumerate(("POR", "PERM", "SW")):
    d = default_mask(y[:, ti], t).astype(float)
    g = SRC[t]["gate"]
    path, pk, ck, inv = SRC[t]["dist"]
    if not Path(path).exists():
        print("skip", t); continue
    D = np.load(path); prob = D[pk].astype(np.float64); prob /= prob.sum(1, keepdims=True)
    # nested calibration
    g_iso = np.zeros(n); g_temp = np.zeros(n)
    for f in range(N_FOLDS):
        inner, outer = folds != f, folds == f
        iso = IsotonicRegression(out_of_bounds="clip").fit(g[inner], d[inner])
        g_iso[outer] = iso.predict(g[outer])
        # temperature + bias on logit
        z = np.log(np.clip(g[inner], 1e-6, 1 - 1e-6) / (1 - np.clip(g[inner], 1e-6, 1 - 1e-6)))
        best = (1e9, None)
        for a in np.linspace(0.5, 3.0, 26):
            for b in np.linspace(-2, 2, 41):
                p = 1 / (1 + np.exp(-(a * z + b)))
                ll = -np.mean(d[inner] * np.log(np.clip(p, 1e-6, 1)) + (1 - d[inner]) * np.log(np.clip(1 - p, 1e-6, 1)))
                if ll < best[0]:
                    best = (ll, (a, b))
        a, b = best[1]
        zo = np.log(np.clip(g[outer], 1e-6, 1 - 1e-6) / (1 - np.clip(g[outer], 1e-6, 1 - 1e-6)))
        g_temp[outer] = 1 / (1 + np.exp(-(a * zo + b)))
    ll = lambda p: float(-np.mean(d * np.log(np.clip(p, 1e-6, 1)) + (1 - d) * np.log(np.clip(1 - p, 1e-6, 1))))
    res = {"logloss": {"raw": ll(g), "isotonic": ll(g_iso), "temperature": ll(g_temp)}}
    # continuous decode (utility) + reference threshold rule
    cont = np.zeros(n); e_cont = np.zeros(n)
    for f in range(N_FOLDS):
        va = folds == f
        sup = inv(D[ck][f])
        grid = np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))
        K = kernel_matrix(grid, sup, t)
        util = prob[va] @ K.T
        j = np.argmax(util, 1)
        cont[va] = grid[j]; e_cont[va] = util[np.arange(va.sum()), j]  # self-assessed expected accuracy if truly nondefault
    cont = np.clip(cont, *BOUNDS[t])
    acc_c = acc_vec(cont, y[:, ti], t); acc_d = acc_vec(np.full(n, DEFAULTS[t]), y[:, ti], t)
    ref = np.where(g >= V2_CAL[t]["threshold"], DEFAULTS[t], cont)
    res["threshold_rule_v2th"] = float(acc_vec(ref, y[:, ti], t).mean())
    # joint decode for each calibration
    for name, gg in (("raw", g), ("isotonic", g_iso), ("temperature", g_temp)):
        joint = np.zeros(n)
        for f in range(N_FOLDS):
            va = folds == f
            sup = inv(D[ck][f])
            grid = np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))
            pj = np.c_[gg[va], (1 - gg[va])[:, None] * prob[va]]
            joint[va] = utility_action(pj, np.r_[DEFAULTS[t], sup], t, actions=np.r_[DEFAULTS[t], grid])
        joint = np.clip(joint, *BOUNDS[t])
        a = acc_vec(joint, y[:, ti], t)
        # expected-score gating: default iff p >= e_cont/(1+e_cont-U_cd) with U_cd ~ acc of continuous when truth default ~0
        u_dc = np.zeros(n)  # acc of predicting default when truth nondefault: only SW near 99.9 gets partial credit; use kernel of default vs support mean
        esg = np.where(gg * 1.0 >= (1 - gg) * e_cont, DEFAULTS[t], cont)
        res[f"joint_{name}"] = {"score": float(a.mean()), "paired_vs_v2rule": paired_well_bootstrap(a, acc_vec(ref, y[:, ti], t), wells),
                                "expected_score_gate": float(acc_vec(esg, y[:, ti], t).mean())}
    R[t] = res
    print(t, json.dumps({k: (v if not isinstance(v, dict) else {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "paired_vs_v2rule"}) for k, v in res.items()}), flush=True)
    for name in ("raw", "isotonic", "temperature"):
        print("   ", name, "paired", R[t][f"joint_{name}"]["paired_vs_v2rule"])
Path("outputs/exp_calib_joint.json").write_text(json.dumps(R, indent=1))
