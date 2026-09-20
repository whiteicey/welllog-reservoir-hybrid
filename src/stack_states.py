"""Level-2 (stacked) reservoir-state model.

Features = base (113) + extra (181) + first-level outputs:
  v2-style gates (POR, PERM, SW), 4-state posterior, per sequence-net group [3 gates + 3x4 bin summaries],
  per target 4 tree bin-distribution summaries (mean, std, entropy, max prob in transformed space).
CV: trained on the inner folds' OOF rows, predicts the outer fold. Test: applied to each first-level fold
model's test output and averaged. The same `level2_features` function is used by predict.py so that the
column order is identical at inference time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from extra_features import build_extra
from harness import N_FOLDS, load_or_build_cache

TARGETS = ("POR", "PERM", "SW")
DEFAULT_NN = ("nn_gate_b", "nn_base_c")
TREE_DIST = {"POR": "por_bins80_base_extra", "PERM": "perm_bins60_base", "SW": "sw_bins60_enh"}


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


def states(y):
    return np.select([y[:, 0] <= 0.1, y[:, 2] >= 99.9, y[:, 1] <= 0.01], [0, 2, 3], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stack_states")
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--seeds", default="42,2026")
    ap.add_argument("--nn-sources", default=",".join(DEFAULT_NN))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    nn_order = tuple(args.nn_sources.split(","))
    out_dir = Path("outputs") / args.name; out_dir.mkdir(parents=True, exist_ok=True)
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells = C["tr_y"], C["tr_fold"], C["tr_well"]
    n, nt = len(y), len(C["te_well"])
    etr, ete, _ = build_extra(Path("data"), Path("cache"), C["meta"]["medians"])
    static_tr = np.c_[C["tr_base"], etr]; static_te = np.c_[C["te_base"], ete]
    V2, V2T = np.load("outputs/cv_v2/oof.npz"), np.load("outputs/cv_v2/test.npz")
    S, ST = np.load("outputs/states_base_extra/oof.npz"), np.load("outputs/states_base_extra/test.npz")
    gates_tr = np.c_[V2["por_gate"], V2["perm_gate"], V2["sw_gate"]]
    nn_tr, nn_te = {}, [{} for _ in range(N_FOLDS)]
    for name in nn_order:
        O, T = np.load(f"outputs/{name}/oof.npz"), np.load(f"outputs/{name}/test.npz")
        g = np.c_[[O[f"{t}_gate"] for t in TARGETS]].T
        summ = {}
        for t in TARGETS:
            s = np.zeros((n, 4), np.float32)
            for f in range(N_FOLDS):
                va = folds == f
                s[va] = dist_summary(O[f"{t}_prob"][va], O[f"{t}_centers"][f])
            summ[t] = s
        nn_tr[name] = (g, summ)
        for f in range(N_FOLDS):
            nn_te[f][name] = (np.c_[[T[f"{t}_gate"][f] for t in TARGETS]].T, {t: dist_summary(T[f"{t}_prob"][f], O[f"{t}_centers"][f]) for t in TARGETS})
    tree_tr, tree_te = {}, [{} for _ in range(N_FOLDS)]
    for t, name in TREE_DIST.items():
        O, T = np.load(f"outputs/{name}/oof.npz"), np.load(f"outputs/{name}/test.npz")
        s = np.zeros((n, 4), np.float32)
        for f in range(N_FOLDS):
            va = folds == f
            s[va] = dist_summary(O["prob"][va], O["centers"][f])
        tree_tr[t] = s
        for f in range(N_FOLDS):
            tree_te[f][t] = dist_summary(T["prob"][f], O["centers"][f])
    Xtr = level2_features(static_tr, gates_tr, S["post"], nn_tr, tree_tr, nn_order)
    Xte = [level2_features(static_te, np.c_[V2T["por_gate"][f], V2T["perm_gate"][f], V2T["sw_gate"][f]], ST["post"][f], nn_te[f], tree_te[f], nn_order) for f in range(N_FOLDS)]
    print("level-2 features", Xtr.shape, flush=True)
    Sy = states(y)
    oof_post = np.zeros((n, 4)); te_post = np.zeros((N_FOLDS, nt, 4))
    for f in range(N_FOLDS):
        t0 = time.time(); tr, va = folds != f, folds == f
        for seed in seeds:
            m = CatBoostClassifier(iterations=args.iterations, loss_function="MultiClass", depth=8, learning_rate=0.05,
                                   task_type="GPU", devices="0", max_bin=128, random_seed=seed, bootstrap_type="Bayesian",
                                   allow_writing_files=False, verbose=0, gpu_ram_part=0.4)
            m.fit(Pool(Xtr[tr], label=Sy[tr]))
            order = np.argsort(m.classes_.astype(int)); assert (m.classes_.astype(int)[order] == np.arange(4)).all()
            oof_post[va] += m.predict_proba(Xtr[va])[:, order] / len(seeds)
            for g in range(N_FOLDS):
                te_post[f] += m.predict_proba(Xte[g])[:, order] / (len(seeds) * N_FOLDS)
        print({"fold": f, "seconds": time.time() - t0, "state_acc_fold": float((oof_post[va].argmax(1) == Sy[va]).mean())}, flush=True)
    np.savez(out_dir / "oof.npz", post=oof_post); np.savez(out_dir / "test.npz", post=te_post)
    rep = {"state_accuracy": float((oof_post.argmax(1) == Sy).mean()), "first_level_state_accuracy": float((S["post"].argmax(1) == Sy).mean()),
           "n_features": int(Xtr.shape[1]), "nn_order": list(nn_order), "args": vars(args)}
    for t in TARGETS:
        d = {"POR": Sy == 0, "PERM": np.isin(Sy, [0, 3]), "SW": np.isin(Sy, [0, 2])}[t]
        marg = {"POR": oof_post[:, 0], "PERM": oof_post[:, 0] + oof_post[:, 3], "SW": oof_post[:, 0] + oof_post[:, 2]}[t]
        marg1 = {"POR": S["post"][:, 0], "PERM": S["post"][:, 0] + S["post"][:, 3], "SW": S["post"][:, 0] + S["post"][:, 2]}[t]
        rep[f"{t}_gate_acc_0.5"] = {"level2": float(((marg >= 0.5) == d).mean()), "level1_states": float(((marg1 >= 0.5) == d).mean())}
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
