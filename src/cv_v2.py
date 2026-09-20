"""Reproduce the v2 ensemble under 5-fold GroupKFold and cache OOF + per-fold test outputs.

Everything downstream (decision rules, gate smoothing, blending) is evaluated on
these cached out-of-fold outputs so that all comparisons share one split.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, default_mask, load_or_build_cache,
                     make_por_bins, breakdown, total)


def gpu(seed: int, **kw):
    return dict(task_type="GPU", devices="0", max_bin=128, random_seed=seed, learning_rate=0.04,
                depth=8, bootstrap_type="Bayesian", allow_writing_files=False, verbose=0, **kw)


def fit_clf(X, y, iters, seed):
    m = CatBoostClassifier(iterations=iters, loss_function="Logloss", **gpu(seed))
    m.fit(Pool(X, label=y.astype(np.int8)))
    return m


def fit_reg(X, y, iters, loss, seed):
    m = CatBoostRegressor(iterations=iters, loss_function=loss, l2_leaf_reg=4.0, **gpu(seed))
    m.fit(Pool(X, label=y))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cv_v2"))
    ap.add_argument("--seeds", default="42,2026")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C = load_or_build_cache(args.data_dir, args.cache_dir)
    y, folds, wells = C["tr_y"], C["tr_fold"], C["tr_well"]
    n, nt = len(y), len(C["te_base"])
    oof = {
        "por_gate": np.zeros(n), "por_prob": np.zeros((n, 80), np.float32), "por_centers": np.zeros((N_FOLDS, 80)),
        "perm_gate": np.zeros(n), "perm_log": np.zeros(n),
        "sw_gate": np.zeros(n), "sw_reg": np.zeros(n),
    }
    test = {
        "por_gate": np.zeros((N_FOLDS, nt)), "por_prob": np.zeros((N_FOLDS, nt, 80), np.float32),
        "perm_gate": np.zeros((N_FOLDS, nt)), "perm_log": np.zeros((N_FOLDS, nt)),
        "sw_gate": np.zeros((N_FOLDS, nt)), "sw_reg": np.zeros((N_FOLDS, nt)),
    }
    log = []
    for f in range(N_FOLDS):
        t0 = time.time()
        tr, va = folds != f, folds == f
        Xb, Xe = C["tr_base"][tr], C["tr_enh"][tr]
        Xb_va, Xe_va = C["tr_base"][va], C["tr_enh"][va]
        Tb, Te = C["te_base"], C["te_enh"]
        por, perm, sw = y[tr, 0], y[tr, 1], y[tr, 2]
        nd, cls, centers, _ = make_por_bins(por, 80)
        assert len(centers) == 80, len(centers)
        oof["por_centers"][f] = centers
        acc = {k: 0 for k in ("por_gate", "por_prob", "perm_gate", "perm_log", "sw_gate", "sw_reg")}
        acct = {k: 0 for k in acc}
        for seed in seeds:
            m = fit_clf(Xb, default_mask(por, "POR"), 500, seed)
            acc["por_gate"] = acc["por_gate"] + m.predict_proba(Xb_va)[:, 1]
            acct["por_gate"] = acct["por_gate"] + m.predict_proba(Tb)[:, 1]
            m = CatBoostClassifier(iterations=1200, loss_function="MultiClass", **gpu(seed + 100))
            m.fit(Pool(Xb[nd], label=cls[nd]))
            order = np.argsort(m.classes_.astype(int))
            assert (m.classes_.astype(int)[order] == np.arange(80)).all()
            acc["por_prob"] = acc["por_prob"] + m.predict_proba(Xb_va)[:, order]
            acct["por_prob"] = acct["por_prob"] + m.predict_proba(Tb)[:, order]
            pdm = default_mask(perm, "PERM")
            m = fit_clf(Xb, pdm, 800, seed + 200)
            acc["perm_gate"] = acc["perm_gate"] + m.predict_proba(Xb_va)[:, 1]
            acct["perm_gate"] = acct["perm_gate"] + m.predict_proba(Tb)[:, 1]
            m = fit_reg(Xb[~pdm], np.log10(perm[~pdm]), 370, "MAE", seed + 300)
            acc["perm_log"] = acc["perm_log"] + m.predict(Xb_va)
            acct["perm_log"] = acct["perm_log"] + m.predict(Tb)
            sdm = default_mask(sw, "SW")
            m = fit_clf(Xe, sdm, 950, seed + 400)
            acc["sw_gate"] = acc["sw_gate"] + m.predict_proba(Xe_va)[:, 1]
            acct["sw_gate"] = acct["sw_gate"] + m.predict_proba(Te)[:, 1]
            m = fit_reg(Xe[~sdm], sw[~sdm], 500, "RMSE", seed + 500)
            acc["sw_reg"] = acc["sw_reg"] + m.predict(Xe_va)
            acct["sw_reg"] = acct["sw_reg"] + m.predict(Te)
        for k in acc:
            oof[k][va] = acc[k] / len(seeds)
            test[k][f] = acct[k] / len(seeds)
        log.append({"fold": f, "seconds": time.time() - t0, "val_rows": int(va.sum())})
        print(log[-1], flush=True)
    np.savez(args.out_dir / "oof.npz", **oof)
    np.savez(args.out_dir / "test.npz", **test)
    # v2 decision rule on OOF
    pred = np.zeros_like(y)
    for f in range(N_FOLDS):
        va = folds == f
        c = np.expm1(oof["por_centers"][f])
        pc = c[np.argmax(oof["por_prob"][va], axis=1)]
        pred[va, 0] = np.where(oof["por_gate"][va] >= V2_CAL["POR"]["threshold"], DEFAULTS["POR"], np.clip(pc, *BOUNDS["POR"]))
    pk = np.clip(10 ** (oof["perm_log"] * V2_CAL["PERM"]["scale"]), *BOUNDS["PERM"])
    pred[:, 1] = np.where(oof["perm_gate"] >= V2_CAL["PERM"]["threshold"], DEFAULTS["PERM"], pk)
    ps = np.clip(oof["sw_reg"] * V2_CAL["SW"]["scale"] + V2_CAL["SW"]["offset"], *BOUNDS["SW"])
    pred[:, 2] = np.where(oof["sw_gate"] >= V2_CAL["SW"]["threshold"], DEFAULTS["SW"], ps)
    rep = {t: breakdown(pred[:, i], y[:, i], t, wells, folds) for i, t in enumerate(("POR", "PERM", "SW"))}
    rep["TOTAL"] = total({t: rep[t]["micro"] for t in ("POR", "PERM", "SW")})
    rep["log"] = log
    (args.out_dir / "report_v2_rule.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v if k == "TOTAL" else {kk: v[kk] for kk in ("micro", "macro", "default_seg", "nondefault_seg")})
                      for k, v in rep.items() if k != "log"}, indent=1))


if __name__ == "__main__":
    main()
