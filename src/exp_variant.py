"""Parameterised CatBoost variant experiment producing OOF + per-fold test outputs.

Examples
  python exp_variant.py --name sw_bins60_enh --target SW --model bins --bins 60 --features enh
  python exp_variant.py --name por_bins80_base_extra --target POR --model bins --bins 80 --features base+extra
  python exp_variant.py --name sw_bins60_enh_extra_stack --target SW --model bins --bins 60 --features enh+extra+stack
Feature sets:  base | enh | +extra (lag features) | +stack (OOF POR distribution summaries from a cv dir)
Models:        bins (MultiClass over quantile bins, Bayes utility decoding) | reg_mae | reg_rmse | gate (binary default)
The default gate for the target is NOT retrained here unless --model gate; final scoring uses the v2 gate OOF.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from extra_features import build_extra
from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, acc_vec, breakdown, default_mask,
                     load_or_build_cache, paired_well_bootstrap, quantile_bins, utility_action)

TI = {"POR": 0, "PERM": 1, "SW": 2}


def target_transform(t):
    if t == "POR":
        return np.log1p, np.expm1
    if t == "PERM":
        return np.log10, lambda v: 10 ** v
    return (lambda v: v), (lambda v: v)


def stack_features(cv_dir: Path, folds, n_test):
    """Summaries of the v2 POR OOF distribution: gate prob, Bayes action, mean, std, entropy."""
    O = np.load(cv_dir / "oof.npz"); T = np.load(cv_dir / "test.npz")
    def summarise(prob, centers, gate):
        prob = prob.astype(np.float64); prob /= prob.sum(1, keepdims=True)
        sup = np.expm1(centers)
        act = utility_action(prob, sup, "POR")
        mean_l = prob @ centers
        std_l = np.sqrt(np.maximum(prob @ centers ** 2 - mean_l ** 2, 0))
        ent = -(prob * np.log(prob + 1e-12)).sum(1)
        return np.c_[gate, act, np.log1p(act), mean_l, std_l, ent, prob.max(1)].astype(np.float32)
    n = len(folds)
    tr = np.zeros((n, 7), np.float32)
    for f in range(N_FOLDS):
        va = folds == f
        tr[va] = summarise(O["por_prob"][va], O["por_centers"][f], O["por_gate"][va])
    te = np.zeros((N_FOLDS, n_test, 7), np.float32)
    for f in range(N_FOLDS):
        te[f] = summarise(T["por_prob"][f], O["por_centers"][f], T["por_gate"][f])
    return tr, te


def gpu(seed, depth=8, learning_rate=0.04, **kw):
    return dict(task_type="GPU", devices="0", max_bin=128, random_seed=seed, learning_rate=learning_rate, depth=depth,
                bootstrap_type="Bayesian", allow_writing_files=False, verbose=0, gpu_ram_part=0.4, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--target", required=True, choices=list(TI))
    ap.add_argument("--model", required=True, choices=["bins", "reg_mae", "reg_rmse", "gate"])
    ap.add_argument("--features", default="base")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--iterations", type=int, default=1200)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--seeds", default="42,2026")
    ap.add_argument("--cv-dir", type=Path, default=Path("outputs/cv_v2"))
    ap.add_argument("--l2", type=float, default=3.0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path("outputs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells = C["tr_y"], C["tr_fold"], C["tr_well"]
    ti, t = TI[args.target], args.target
    parts = args.features.split("+")
    Xtr = C["tr_enh"] if parts[0] == "enh" else C["tr_base"]
    Xte = C["te_enh"] if parts[0] == "enh" else C["te_base"]
    Xte_fold = None
    if "extra" in parts:
        etr, ete, _ = build_extra(Path("data"), Path("cache"), C["meta"]["medians"])
        Xtr, Xte = np.c_[Xtr, etr], np.c_[Xte, ete]
    if "stack" in parts:
        s_tr, s_te = stack_features(args.cv_dir, folds, len(Xte))
        Xtr = np.c_[Xtr, s_tr]
        Xte_fold = [np.c_[Xte, s_te[f]] for f in range(N_FOLDS)]
    aln_tr = aln_te = None
    if "aln" in parts:  # cross-fitted DTW alignment features, one matrix per fold (see dtw_align.py)
        aln_tr = [np.load(f"cache/aln_fold{f}_tr.npy") for f in range(N_FOLDS)]
        aln_te = [np.load(f"cache/aln_fold{f}_te.npy") for f in range(N_FOLDS)]
        if Xte_fold is None:
            Xte_fold = [Xte for _ in range(N_FOLDS)]
        Xte_fold = [np.c_[Xte_fold[f], aln_te[f]] for f in range(N_FOLDS)]
    print("features", Xtr.shape, "+aln" if aln_tr is not None else "", flush=True)
    fwd, inv = target_transform(t)
    yt = fwd(y[:, ti])
    dmask = default_mask(y[:, ti], t)
    n, nt = len(y), len(Xte)
    res = {"folds": []}
    if args.model == "bins":
        oof_prob = np.zeros((n, args.bins), np.float32); te_prob = np.zeros((N_FOLDS, nt, args.bins), np.float32)
        centers_all = np.zeros((N_FOLDS, args.bins))
    else:
        oof_val = np.zeros(n); te_val = np.zeros((N_FOLDS, nt))
    for f in range(N_FOLDS):
        t0 = time.time()
        tr, va = folds != f, folds == f
        Xte_f = Xte if Xte_fold is None else Xte_fold[f]
        Xtr_f = Xtr if aln_tr is None else np.c_[Xtr, aln_tr[f]]
        if args.model == "gate":
            fit_rows = tr
        else:
            fit_rows = tr & ~dmask
        if args.model == "bins":
            labels, centers, _ = quantile_bins(yt[fit_rows], args.bins)
            assert len(centers) == args.bins, (len(centers), args.bins)
            centers_all[f] = centers
            acc_va, acc_te = 0, 0
            for seed in seeds:
                m = CatBoostClassifier(iterations=args.iterations, loss_function="MultiClass", l2_leaf_reg=args.l2,
                                       **gpu(seed + 100, depth=args.depth, learning_rate=args.lr))
                m.fit(Pool(Xtr_f[fit_rows], label=labels))
                order = np.argsort(m.classes_.astype(int))
                acc_va = acc_va + m.predict_proba(Xtr_f[va])[:, order]
                acc_te = acc_te + m.predict_proba(Xte_f)[:, order]
            oof_prob[va] = acc_va / len(seeds); te_prob[f] = acc_te / len(seeds)
        elif args.model == "gate":
            acc_va, acc_te = 0, 0
            for seed in seeds:
                m = CatBoostClassifier(iterations=args.iterations, loss_function="Logloss", l2_leaf_reg=args.l2,
                                       **gpu(seed, depth=args.depth, learning_rate=args.lr))
                m.fit(Pool(Xtr_f[fit_rows], label=dmask[fit_rows].astype(np.int8)))
                acc_va = acc_va + m.predict_proba(Xtr_f[va])[:, 1]; acc_te = acc_te + m.predict_proba(Xte_f)[:, 1]
            oof_val[va] = acc_va / len(seeds); te_val[f] = acc_te / len(seeds)
        else:
            loss = "MAE" if args.model == "reg_mae" else "RMSE"
            acc_va, acc_te = 0, 0
            for seed in seeds:
                m = CatBoostRegressor(iterations=args.iterations, loss_function=loss, l2_leaf_reg=args.l2,
                                      **gpu(seed + 300, depth=args.depth, learning_rate=args.lr))
                m.fit(Pool(Xtr_f[fit_rows], label=yt[fit_rows]))
                acc_va = acc_va + m.predict(Xtr_f[va]); acc_te = acc_te + m.predict(Xte_f)
            oof_val[va] = acc_va / len(seeds); te_val[f] = acc_te / len(seeds)
        res["folds"].append({"fold": f, "seconds": time.time() - t0})
        print(res["folds"][-1], flush=True)

    # ----- evaluation against the v2 gate OOF -----
    O = np.load(args.cv_dir / "oof.npz")
    gate_key = {"POR": "por_gate", "PERM": "perm_gate", "SW": "sw_gate"}[t]
    gate = O[gate_key]; th = V2_CAL[t]["threshold"]
    if args.model == "bins":
        np.savez(out_dir / "oof.npz", prob=oof_prob, centers=centers_all)
        np.savez(out_dir / "test.npz", prob=te_prob)
        cont = {"argmax": np.zeros(n), "utility": np.zeros(n)}
        for f in range(N_FOLDS):
            va = folds == f
            pp = oof_prob[va].astype(np.float64); pp /= pp.sum(1, keepdims=True)
            sup = inv(centers_all[f])
            cont["argmax"][va] = sup[np.argmax(pp, 1)]
            dense = np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))
            cont["utility"][va] = utility_action(pp, sup, t, actions=dense)
    elif args.model == "gate":
        np.savez(out_dir / "oof.npz", gate=oof_val); np.savez(out_dir / "test.npz", gate=te_val)
        cont = None
    else:
        np.savez(out_dir / "oof.npz", val=oof_val); np.savez(out_dir / "test.npz", val=te_val)
        cont = {"point": np.clip(inv(oof_val), *BOUNDS[t])}
    # reference: v2 continuous part
    if t == "POR":
        v2c = np.zeros(n)
        for f in range(N_FOLDS):
            va = folds == f
            v2c[va] = np.expm1(O["por_centers"][f])[np.argmax(O["por_prob"][va], 1)]
    elif t == "PERM":
        v2c = np.clip(10 ** (O["perm_log"] * 1.03), *BOUNDS["PERM"])
    else:
        v2c = np.clip(O["sw_reg"] * 1.05 - 3.0, *BOUNDS["SW"])
    if args.model == "bins":
        # joint Bayes action over [default mass, (1-gate) * bins] -> no threshold at all
        joint = np.zeros(n)
        for f in range(N_FOLDS):
            va = folds == f
            pp = oof_prob[va].astype(np.float64); pp /= pp.sum(1, keepdims=True)
            sup = inv(centers_all[f]); g = gate[va]
            pj = np.c_[g, (1 - g)[:, None] * pp]; supj = np.r_[DEFAULTS[t], sup]
            dense = np.r_[DEFAULTS[t], np.linspace(sup[0], sup[-1], 800) if t == "SW" else np.exp(np.linspace(np.log(sup[0] * 0.9), np.log(sup[-1] * 1.1), 800))]
            joint[va] = utility_action(pj, supj, t, actions=dense)
        res["joint_full"] = breakdown(np.clip(joint, *BOUNDS[t]), y[:, ti], t, wells, folds)
        res["joint_full"]["paired_vs_v2"] = paired_well_bootstrap(acc_vec(np.clip(joint, *BOUNDS[t]), y[:, ti], t), acc_vec(np.where(gate >= th, DEFAULTS[t], v2c), y[:, ti], t), wells)
        res["joint_full"].pop("per_well", None)
        print("joint full", round(res["joint_full"]["micro"], 5), res["joint_full"]["paired_vs_v2"])
    if cont is not None:
        res["nondefault_only"] = {"v2": float(acc_vec(v2c[~dmask], y[~dmask, ti], t).mean())}
        res["full_v2gate"] = {"v2": breakdown(np.where(gate >= th, DEFAULTS[t], v2c), y[:, ti], t, wells, folds)}
        base_acc = acc_vec(np.where(gate >= th, DEFAULTS[t], v2c), y[:, ti], t)
        for k, v in cont.items():
            res["nondefault_only"][k] = float(acc_vec(v[~dmask], y[~dmask, ti], t).mean())
            full = np.where(gate >= th, DEFAULTS[t], np.clip(v, *BOUNDS[t]))
            res["full_v2gate"][k] = breakdown(full, y[:, ti], t, wells, folds)
            res["full_v2gate"][k]["paired_vs_v2"] = paired_well_bootstrap(acc_vec(full, y[:, ti], t), base_acc, wells)
        print(json.dumps({"nondefault_only": res["nondefault_only"],
                          "full": {k: (round(v["micro"], 5), round(v["macro"], 5)) for k, v in res["full_v2gate"].items()},
                          "paired": {k: v.get("paired_vs_v2") for k, v in res["full_v2gate"].items() if "paired_vs_v2" in v}}, indent=1))
    else:
        d = dmask
        res["gate"] = {"v2_gate_acc_at_v2th": float(((gate >= th) == d).mean()),
                       "new_gate_acc_at_0.5": float(((oof_val >= 0.5) == d).mean()),
                       "new_gate_acc_at_v2th": float(((oof_val >= th) == d).mean()),
                       "v2_logloss": float(-np.mean(d * np.log(np.clip(gate, 1e-6, 1)) + (~d) * np.log(np.clip(1 - gate, 1e-6, 1)))),
                       "new_logloss": float(-np.mean(d * np.log(np.clip(oof_val, 1e-6, 1)) + (~d) * np.log(np.clip(1 - oof_val, 1e-6, 1))))}
        print(json.dumps(res["gate"], indent=1))
    res["args"] = vars(args) | {"cv_dir": str(args.cv_dir)}
    (out_dir / "report.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
