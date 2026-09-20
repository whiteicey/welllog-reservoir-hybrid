"""Joint 4-state reservoir-state model replacing three independent default gates.

States (from training labels, 99.98% consistent):
  0 = all defaults (non-reservoir)          1 = all three non-default (reservoir, hydrocarbon-bearing)
  2 = POR/PERM non-default, SW = 99.9       3 = POR non-default, PERM = 0.01 (tight)
A CatBoost MultiClass model predicts the state posterior per row; per-target default probabilities are
marginals (POR: P0, PERM: P0+P3, SW: P0+P2). Optionally the posterior sequence is smoothed by a 4-state
forward-backward pass with the empirical transition matrix. Evaluated with nested thresholds against the
v2 gates using the v2 continuous predictions, so only the gate changes.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from exp_decisions import by_well_sequences, nested_threshold
from extra_features import build_extra
from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, acc_vec, breakdown, default_mask,
                     load_or_build_cache, paired_well_bootstrap)


def states(y):
    return np.select([y[:, 0] <= 0.1, y[:, 2] >= 99.9, y[:, 1] <= 0.01], [0, 2, 3], 1)


def forward_backward(post, A, temper=1.0):
    """Smooth per-row posteriors (n, S) with transition matrix A (S, S). Emissions ~ posterior / prior."""
    n, S = post.shape
    prior = post.mean(0) + 1e-6
    e = np.clip(post / prior, 1e-6, None) ** temper
    alpha = np.zeros((n, S)); beta = np.ones((n, S))
    alpha[0] = e[0] * prior; alpha[0] /= alpha[0].sum()
    for t in range(1, n):
        alpha[t] = e[t] * (alpha[t - 1] @ A); alpha[t] /= alpha[t].sum()
    for t in range(n - 2, -1, -1):
        beta[t] = A @ (e[t + 1] * beta[t + 1]); beta[t] /= beta[t].sum()
    p = alpha * beta
    return p / p.sum(1, keepdims=True)


def smooth_states(post, wells, pos, A, temper=1.0):
    out = post.copy()
    for idx in by_well_sequences(wells, pos):
        if len(idx) >= 3:
            out[idx] = forward_backward(post[idx], A, temper)
    return out


def marginals(post):
    return {"POR": post[:, 0], "PERM": post[:, 0] + post[:, 3], "SW": post[:, 0] + post[:, 2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="states_base_extra")
    ap.add_argument("--features", default="base+extra")
    ap.add_argument("--iterations", type=int, default=900)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--seeds", default="42,2026")
    ap.add_argument("--cv-dir", type=Path, default=Path("outputs/cv_v2"))
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path("outputs") / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells, pos = C["tr_y"], C["tr_fold"], C["tr_well"], C["tr_pos"]
    n, nt = len(y), len(C["te_well"])
    parts = args.features.split("+")
    Xtr = C["tr_enh"] if parts[0] == "enh" else C["tr_base"]
    Xte = C["te_enh"] if parts[0] == "enh" else C["te_base"]
    if "extra" in parts:
        etr, ete, _ = build_extra(Path("data"), Path("cache"), C["meta"]["medians"])
        Xtr, Xte = np.c_[Xtr, etr], np.c_[Xte, ete]
    aln_tr = aln_te = None
    if "aln" in parts:
        aln_tr = [np.load(f"cache/aln_fold{f}_tr.npy") for f in range(N_FOLDS)]
        aln_te = [np.load(f"cache/aln_fold{f}_te.npy") for f in range(N_FOLDS)]
    S = states(y)
    if args.eval_only:
        oof_post = np.load(out_dir / "oof.npz")["post"]
    else:
        oof_post = np.zeros((n, 4)); te_post = np.zeros((N_FOLDS, nt, 4))
        for f in range(N_FOLDS):
            t0 = time.time(); tr, va = folds != f, folds == f
            Xtr_f = Xtr if aln_tr is None else np.c_[Xtr, aln_tr[f]]
            Xte_f = Xte if aln_te is None else np.c_[Xte, aln_te[f]]
            for seed in seeds:
                m = CatBoostClassifier(iterations=args.iterations, loss_function="MultiClass", depth=args.depth, learning_rate=0.05,
                                       task_type="GPU", devices="0", max_bin=128, random_seed=seed, bootstrap_type="Bayesian",
                                       allow_writing_files=False, verbose=0, gpu_ram_part=0.4)
                m.fit(Pool(Xtr_f[tr], label=S[tr]))
                order = np.argsort(m.classes_.astype(int)); assert (m.classes_.astype(int)[order] == np.arange(4)).all()
                oof_post[va] += m.predict_proba(Xtr_f[va])[:, order] / len(seeds)
                te_post[f] += m.predict_proba(Xte_f)[:, order] / len(seeds)
            print({"fold": f, "seconds": time.time() - t0}, flush=True)
        np.savez(out_dir / "oof.npz", post=oof_post); np.savez(out_dir / "test.npz", post=te_post)
    # empirical transition matrix from training labels (all wells; only used as a smoothing prior)
    A = np.zeros((4, 4))
    for idx in by_well_sequences(wells, pos):
        s = S[idx]; np.add.at(A, (s[:-1], s[1:]), 1)
    A = A / A.sum(1, keepdims=True)
    # v2 continuous parts and gates
    O = np.load(args.cv_dir / "oof.npz")
    v2c = {"PERM": np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"]), "SW": np.clip(O["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"])}
    pc = np.zeros(n)
    for f in range(N_FOLDS):
        va = folds == f; pc[va] = np.expm1(O["por_centers"][f])[np.argmax(O["por_prob"][va], 1)]
    v2c["POR"] = pc
    v2g = {"POR": O["por_gate"], "PERM": O["perm_gate"], "SW": O["sw_gate"]}
    grid = np.linspace(0.05, 0.8, 76)
    R = {"state_accuracy_argmax": float((oof_post.argmax(1) == S).mean()), "transition": A.round(5).tolist(), "args": vars(args) | {"cv_dir": str(args.cv_dir)}}
    variants = {"raw": oof_post}
    for temper in (1.0, 0.7):
        for sticky in (1.0, 0.5):
            Ak = A.copy()
            if sticky != 1.0:  # shrink toward the identity to strengthen persistence
                Ak = sticky * Ak + (1 - sticky) * np.eye(4)
            variants[f"fb_t{temper}_s{sticky}"] = smooth_states(oof_post, wells, pos, Ak, temper)
    for vname, post in variants.items():
        marg = marginals(post)
        R[vname] = {"state_accuracy": float((post.argmax(1) == S).mean())}
        for ti, t in enumerate(("POR", "PERM", "SW")):
            d = default_mask(y[:, ti], t)
            base_pred = np.where(v2g[t] >= V2_CAL[t]["threshold"], DEFAULTS[t], v2c[t])
            base_acc = acc_vec(base_pred, y[:, ti], t)
            for gname, g in (("state", marg[t]), ("avg_state_v2", 0.5 * (marg[t] + v2g[t]))):
                pred, chosen = nested_threshold(g, v2c[t], y[:, ti], t, folds, grid)
                a = acc_vec(pred, y[:, ti], t)
                R[vname][f"{t}_{gname}"] = {"nested_score": float(a.mean()), "v2_score": float(base_acc.mean()),
                                             "gate_acc_at_0.5": float(((g >= 0.5) == d).mean()), "v2_gate_acc": float(((v2g[t] >= V2_CAL[t]["threshold"]) == d).mean()),
                                             "thresholds": chosen, "paired_vs_v2": paired_well_bootstrap(a, base_acc, wells)}
        print(vname, {k: (round(v["nested_score"], 5), round(v["v2_score"], 5)) for k, v in R[vname].items() if isinstance(v, dict)}, flush=True)
    (out_dir / "report.json").write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
