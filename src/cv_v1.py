"""Reproduce the v1 submission (unified two-stage CatBoost, first-round entry) under the shared 5 well folds.

v1 as actually submitted: per target one Logloss classifier (default vs not) and one MAE regressor on the
transformed target (log1p POR, log10 PERM, raw SW), 113 base features, depth 8, lr 0.05, seed 42/43,
final models trained for a fixed 500 iterations (no early stopping), thresholds/scale/offset from a 3-fold OOF
grid search: POR th 0.20; PERM th 0.35; SW th 0.10, scale 1.05, offset -2.0.
Here the same fixed-500-iteration models are trained on each training fold, so the OOF matches the
submitted estimator; the submitted post-processing parameters are applied (and a nested re-selection is reported).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from exp_decisions import nested_threshold
from harness import BOUNDS, DEFAULTS, N_FOLDS, acc_vec, breakdown, default_mask, load_or_build_cache, total

V1_PARAMS = {"POR": {"threshold": 0.2, "scale": 1.0, "offset": 0.0},
             "PERM": {"threshold": 0.35, "scale": 1.0, "offset": 0.0},
             "SW": {"threshold": 0.1, "scale": 1.05, "offset": -2.0}}
TI = {"POR": 0, "PERM": 1, "SW": 2}
FWD = {"POR": np.log1p, "PERM": np.log10, "SW": lambda v: v}
INV = {"POR": np.expm1, "PERM": lambda v: 10 ** v, "SW": lambda v: v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/cv_v1"))
    ap.add_argument("--iterations", type=int, default=500)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds, wells = C["tr_y"], C["tr_fold"], C["tr_well"]
    n, nt = len(y), len(C["te_base"])
    Xall, Xte = C["tr_base"], C["te_base"]
    oof = {f"{t}_gate": np.zeros(n) for t in TI}; oof.update({f"{t}_reg": np.zeros(n) for t in TI})
    te = {f"{t}_gate": np.zeros((N_FOLDS, nt)) for t in TI}; te.update({f"{t}_reg": np.zeros((N_FOLDS, nt)) for t in TI})
    common = dict(iterations=args.iterations, learning_rate=0.05, depth=8, task_type="GPU", devices="0", max_bin=128,
                  bootstrap_type="Bayesian", verbose=0, allow_writing_files=False, gpu_ram_part=0.4)
    log = []
    for f in range(N_FOLDS):
        t0 = time.time(); tr, va = folds != f, folds == f
        for t in TI:
            ti = TI[t]; d = default_mask(y[:, ti], t)
            clf = CatBoostClassifier(loss_function="Logloss", random_seed=42, **common)
            clf.fit(Pool(Xall[tr], label=d[tr].astype(np.int8)))
            oof[f"{t}_gate"][va] = clf.predict_proba(Xall[va])[:, 1]; te[f"{t}_gate"][f] = clf.predict_proba(Xte)[:, 1]
            reg = CatBoostRegressor(loss_function="MAE", random_seed=43, **common)
            m = tr & ~d
            reg.fit(Pool(Xall[m], label=FWD[t](y[m, ti])))
            oof[f"{t}_reg"][va] = reg.predict(Xall[va]); te[f"{t}_reg"][f] = reg.predict(Xte)
        log.append({"fold": f, "seconds": time.time() - t0}); print(log[-1], flush=True)
    np.savez(args.out_dir / "oof.npz", **oof); np.savez(args.out_dir / "test.npz", **te)
    pred = np.zeros_like(y); rep = {"log": log}
    for t in TI:
        ti = TI[t]; p = V1_PARAMS[t]
        cont = np.clip(INV[t](oof[f"{t}_reg"] * p["scale"] + p["offset"]), *BOUNDS[t])
        pred[:, ti] = np.where(oof[f"{t}_gate"] >= p["threshold"], DEFAULTS[t], cont)
        rep[t] = breakdown(pred[:, ti], y[:, ti], t, wells, folds); rep[t].pop("per_well", None)
        nested, chosen = nested_threshold(oof[f"{t}_gate"], cont, y[:, ti], t, folds, np.linspace(0.05, 0.8, 76))
        rep[t]["nested_threshold_score"] = float(acc_vec(nested, y[:, ti], t).mean()); rep[t]["nested_thresholds"] = chosen
    rep["TOTAL"] = total({t: rep[t]["micro"] for t in TI})
    rep["TOTAL_nested_threshold"] = total({t: rep[t]["nested_threshold_score"] for t in TI})
    np.save(args.out_dir / "final_oof.npy", pred)
    (args.out_dir / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({t: {k: round(rep[t][k], 5) for k in ("micro", "macro", "default_seg", "nondefault_seg", "nested_threshold_score")} for t in TI} | {"TOTAL": rep["TOTAL"], "TOTAL_nested": rep["TOTAL_nested_threshold"]}, indent=1))


if __name__ == "__main__":
    main()
