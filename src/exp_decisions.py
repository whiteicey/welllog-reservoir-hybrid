"""Decision-rule and gate-smoothing experiments on cached v2 OOF outputs (no retraining).

E1  POR: argmax vs Bayes utility action vs utility with a dense action grid.
E2  PERM/SW: nothing changes here (regressors), kept for the total.
E3  Gate smoothing: median filter / HMM (forward-backward with sticky transition) on the
    default-probability sequences of each well, then re-threshold.
All thresholds are evaluated both at the historical v2 values and via nested (leave-one-fold-out) selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, acc_vec, breakdown, default_mask,
                     load_or_build_cache, paired_well_bootstrap, total, utility_action)


def by_well_sequences(wells, pos):
    """Yield index arrays of contiguous runs (same well, consecutive original positions)."""
    order = np.arange(len(wells))
    brk = np.flatnonzero((np.diff(wells) != 0) | (np.diff(pos) != 1)) + 1
    return np.split(order, brk)


def smooth_prob(p, wells, pos, method, size):
    out = p.copy()
    for idx in by_well_sequences(wells, pos):
        if len(idx) < 3:
            continue
        if method == "median":
            out[idx] = median_filter(p[idx], size=size, mode="nearest")
        elif method == "hmm":
            out[idx] = hmm_posterior(p[idx], stay=size)
    return out


def hmm_posterior(p, stay=0.98, prior=None):
    """Two-state forward-backward. Emission likelihood ratio taken from the classifier prob (treated as posterior
    under an approximately flat prior): e1 = p, e0 = 1-p. Returns smoothed P(default)."""
    n = len(p)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    e = np.stack([1 - p, p], 1)
    A = np.array([[stay, 1 - stay], [1 - stay, stay]])
    alpha = np.zeros((n, 2)); beta = np.ones((n, 2))
    alpha[0] = e[0] * (0.5 if prior is None else prior); alpha[0] /= alpha[0].sum()
    for t in range(1, n):
        alpha[t] = e[t] * (alpha[t - 1] @ A); alpha[t] /= alpha[t].sum()
    for t in range(n - 2, -1, -1):
        beta[t] = A @ (e[t + 1] * beta[t + 1]); beta[t] /= beta[t].sum()
    post = alpha * beta; post /= post.sum(1, keepdims=True)
    return post[:, 1]


def best_threshold(gate, cont, y, target, grid):
    best, bt = -1, None
    for th in grid:
        s = acc_vec(np.where(gate >= th, DEFAULTS[target], cont), y, target).mean()
        if s > best:
            best, bt = s, th
    return bt, best


def nested_threshold(gate, cont, y, target, folds, grid):
    """Pick threshold on 4 folds, apply to the 5th. Returns predictions and the chosen thresholds."""
    pred = np.empty_like(y); chosen = []
    for f in range(N_FOLDS):
        va = folds == f
        th, _ = best_threshold(gate[~va], cont[~va], y[~va], target, grid)
        chosen.append(float(th))
        pred[va] = np.where(gate[va] >= th, DEFAULTS[target], cont[va])
    return pred, chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", type=Path, default=Path("outputs/cv_v2"))
    ap.add_argument("--out", type=Path, default=Path("outputs/exp_decisions.json"))
    args = ap.parse_args()
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells, pos = C["tr_y"], C["tr_fold"], C["tr_well"], C["tr_pos"]
    O = np.load(args.cv_dir / "oof.npz")
    R = {}

    # ---------- POR continuous decision ----------
    por_prob = O["por_prob"].astype(np.float64)
    por_prob /= por_prob.sum(1, keepdims=True)
    cont = {"argmax": np.zeros(len(y)), "utility": np.zeros(len(y)), "utility_dense": np.zeros(len(y)), "mean": np.zeros(len(y))}
    for f in range(N_FOLDS):
        va = folds == f
        sup = np.expm1(O["por_centers"][f])
        pp = por_prob[va]
        cont["argmax"][va] = sup[np.argmax(pp, 1)]
        cont["utility"][va] = utility_action(pp, sup, "POR")
        dense = np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 600))
        cont["utility_dense"][va] = utility_action(pp, sup, "POR", actions=dense)
        cont["mean"][va] = np.expm1(pp @ O["por_centers"][f])
    gate = O["por_gate"]
    nd = ~default_mask(y[:, 0], "POR")
    R["POR_continuous_on_true_nondefault"] = {k: float(acc_vec(v[nd], y[nd, 0], "POR").mean()) for k, v in cont.items()}
    th = V2_CAL["POR"]["threshold"]
    R["POR_full_v2gate"] = {k: breakdown(np.where(gate >= th, 0.1, v), y[:, 0], "POR", wells, folds) for k, v in cont.items()}
    base_acc = acc_vec(np.where(gate >= th, 0.1, cont["argmax"]), y[:, 0], "POR")
    for k in ("utility", "utility_dense"):
        R["POR_full_v2gate"][k]["paired_vs_argmax"] = paired_well_bootstrap(acc_vec(np.where(gate >= th, 0.1, cont[k]), y[:, 0], "POR"), base_acc, wells)
    grid = np.linspace(0.1, 0.7, 61)
    for k in ("argmax", "utility_dense"):
        pred, chosen = nested_threshold(gate, cont[k], y[:, 0], "POR", folds, grid)
        R[f"POR_nested_threshold_{k}"] = {"micro": float(acc_vec(pred, y[:, 0], "POR").mean()), "thresholds": chosen}
    # Joint action (default mass + continuous distribution)
    joint = np.zeros(len(y))
    for f in range(N_FOLDS):
        va = folds == f
        sup = np.r_[0.1, np.expm1(O["por_centers"][f])]
        pj = np.c_[gate[va], (1 - gate[va])[:, None] * por_prob[va]]
        joint[va] = utility_action(pj, sup, "POR")
    R["POR_joint_action"] = breakdown(joint, y[:, 0], "POR", wells, folds)

    # ---------- gate quality / oracle ----------
    for ti, (t, g, c) in enumerate([("POR", gate, cont["utility_dense"]), ("PERM", O["perm_gate"], np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"])),
                                    ("SW", O["sw_gate"], np.clip(O["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"]))]):
        d = default_mask(y[:, ti], t)
        th = V2_CAL[t]["threshold"]
        R[f"{t}_gate"] = {
            "v2_threshold": th,
            "pred_default_rate": float((g >= th).mean()), "true_default_rate": float(d.mean()),
            "gate_accuracy": float(((g >= th) == d).mean()),
            "false_default": float(((g >= th) & ~d).mean()), "missed_default": float(((g < th) & d).mean()),
            "score_v2": float(acc_vec(np.where(g >= th, DEFAULTS[t], c), y[:, ti], t).mean()),
            "score_oracle_gate": float(acc_vec(np.where(d, DEFAULTS[t], c), y[:, ti], t).mean()),
            "score_oracle_continuous": float(acc_vec(np.where(g >= th, DEFAULTS[t], y[:, ti]), y[:, ti], t).mean()),
        }
        # smoothing of the gate probability
        sm = {}
        for method, sizes in (("median", (3, 5, 7, 9, 13)), ("hmm", (0.9, 0.95, 0.98, 0.99, 0.995))):
            for s in sizes:
                gs = smooth_prob(g, wells, pos, method, s)
                sc_fixed = float(acc_vec(np.where(gs >= th, DEFAULTS[t], c), y[:, ti], t).mean())
                pred, chosen = nested_threshold(gs, c, y[:, ti], t, folds, np.linspace(0.05, 0.8, 76))
                sm[f"{method}_{s}"] = {"score_v2_threshold": sc_fixed, "score_nested_threshold": float(acc_vec(pred, y[:, ti], t).mean()),
                                       "gate_accuracy": float(((gs >= th) == d).mean()), "nested_thresholds": chosen}
        R[f"{t}_gate"]["smoothing"] = sm
        pred, chosen = nested_threshold(g, c, y[:, ti], t, folds, np.linspace(0.05, 0.8, 76))
        R[f"{t}_gate"]["nested_threshold_no_smoothing"] = {"score": float(acc_vec(pred, y[:, ti], t).mean()), "thresholds": chosen}

    # ---------- continuous-segment error anatomy ----------
    for ti, (t, c) in enumerate([("POR", cont["utility_dense"]), ("PERM", np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"])), ("SW", np.clip(O["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"]))]):
        d = default_mask(y[:, ti], t)
        yt, pt = y[~d, ti], c[~d]
        if t == "PERM":
            err = np.log10(pt) - np.log10(yt)
        else:
            err = (pt - yt) / yt
        qs = np.quantile(yt, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        R[f"{t}_continuous_anatomy"] = {
            "signed_err_quantiles": {str(q): float(np.quantile(err, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "median_signed_err_by_true_quintile": [float(np.median(err[(yt >= qs[i]) & (yt <= qs[i + 1])])) for i in range(5)],
            "score_by_true_quintile": [float(acc_vec(pt[(yt >= qs[i]) & (yt <= qs[i + 1])], yt[(yt >= qs[i]) & (yt <= qs[i + 1])], t).mean()) for i in range(5)],
            "true_quintile_edges": qs.tolist(),
            "oracle_global_scale": max(((s, float(acc_vec(pt * s, yt, t).mean())) for s in np.linspace(0.85, 1.15, 61)), key=lambda z: z[1]),
        }
    args.out.write_text(json.dumps(R, indent=1))
    keep = {k: v for k, v in R.items() if "anatomy" in k or k.endswith("_gate") or "continuous" in k}
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "smoothing"} for k, v in keep.items()}, indent=1)[:6000])
    print("POR full v2gate:", {k: round(v["micro"], 5) for k, v in R["POR_full_v2gate"].items()})
    print("POR joint:", round(R["POR_joint_action"]["micro"], 5))
    for t in ("POR", "PERM", "SW"):
        best = max(R[f"{t}_gate"]["smoothing"].items(), key=lambda kv: kv[1]["score_nested_threshold"])
        print(t, "best smoothing", best[0], round(best[1]["score_nested_threshold"], 5), "v2", round(R[f"{t}_gate"]["score_v2"], 5), "nested-no-smooth", round(R[f"{t}_gate"]["nested_threshold_no_smoothing"]["score"], 5))


if __name__ == "__main__":
    main()
