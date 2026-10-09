"""Drift-augmented SW head, v3-style bins recipe (60 linear bins MultiClass).

Same protocol as drift_sw_train.py (aug/noaug arms, same eval perturbations,
2-pass decode on clean OOF), but the continuous head is a 60-bin MultiClass
distribution model trained on non-default rows (v3's sw_bins60_enh style),
decoded by expectation (probs @ centers), plus the same Logloss gate.

Usage: python drift_sw_bins.py --arm both --copies 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, default_mask, load_or_build_cache, score
from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder
from drift_sw_train import (CURVES, bounds_and_iqr, perturb_frame, rng_for)

MISSING = -99999.0


def gpu(seed: int, **kw):
    return dict(task_type="GPU", devices="0", max_bin=128, random_seed=seed, learning_rate=0.04,
                depth=8, bootstrap_type="Bayesian", allow_writing_files=False, verbose=0, **kw)


def make_sw_bins(values: np.ndarray, bins: int = 60):
    """v3-style linear quantile bins for SW on non-default rows."""
    qs = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    edges = np.unique(qs)
    centers = np.concatenate([[values.min()], (edges[:-1] + edges[1:]) / 2.0, [values.max()]])
    cls = np.searchsorted(edges, values, side="right")
    return cls, centers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/drift_sw_bins"))
    ap.add_argument("--arm", choices=("noaug", "aug", "both"), default="both")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--seeds", default="42,2026")
    ap.add_argument("--gate-iters", type=int, default=950)
    ap.add_argument("--bins-iters", type=int, default=1200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t00 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    folds_n = 1 if args.quick else N_FOLDS
    if args.quick:
        seeds = seeds[:1]
        args.gate_iters //= 4
        args.bins_iters //= 4

    C = load_or_build_cache(args.data_dir, args.cache_dir)
    y, folds = C["tr_y"], C["tr_fold"]
    meta = C["meta"]
    train_wells = meta["train_wells"]
    records = dict(load_split(args.data_dir, "train"))
    Xe_cache = C["tr_enh"]

    iqr, pools, bounds = bounds_and_iqr(records, train_wells)
    builder = FeatureBuilder().fit([records[w] for w in train_wells])

    starts = np.concatenate([[0], np.cumsum([int((C["tr_well"] == w).sum()) for w in range(len(train_wells))])])
    row_ptr = {w: (int(starts[i]), int(starts[i + 1])) for i, w in enumerate(train_wells)}

    def xe_of(frame, valid):
        m = builder.transform(frame)
        return np.asarray(m.enhanced[valid], dtype=np.float32)

    Xe_eval = np.zeros_like(Xe_cache)
    for w in train_wells:
        frame = records[w]
        valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        pf = perturb_frame(frame, w, "eval", iqr, bounds, 0.03, 0.20)
        Xe_eval[row_ptr[w][0]:row_ptr[w][1]] = xe_of(pf, valid)

    K = args.copies
    Xe_aug = [np.zeros_like(Xe_cache) for _ in range(K)]
    for k in range(K):
        for w in train_wells:
            frame = records[w]
            valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
            pf = perturb_frame(frame, w, f"train{k}", iqr, bounds, 0.03, 0.20)
            Xe_aug[k][row_ptr[w][0]:row_ptr[w][1]] = xe_of(pf, valid)
    print(f"[prep] eval + {K} aug copies built ({time.time()-t00:.0f}s)", flush=True)

    results = {}
    for arm in (("noaug", "aug") if args.arm == "both" else (args.arm,)):
        t0 = time.time()
        gate_oof = np.zeros(len(y)); exp_oof = np.zeros(len(y))
        gate_drift = np.zeros(len(y)); exp_drift = np.zeros(len(y))
        for f in range(folds_n):
            va = folds == f
            tr = ~va
            if arm == "aug":
                Xtr = np.concatenate([Xe_cache[tr]] + [Xe_aug[k][tr] for k in range(K)])
                ytr = np.tile(y[tr], (1 + K, 1))
            else:
                Xtr, ytr = Xe_cache[tr], y[tr]
            sw = ytr[:, 2]
            sdm = default_mask(sw, "SW")
            cls_tr, centers = make_sw_bins(sw[~sdm])
            acc_g = np.zeros(len(y)); acc_e = np.zeros(len(y))
            acc_gd = np.zeros(len(y)); acc_ed = np.zeros(len(y))
            for seed in seeds:
                m = CatBoostClassifier(iterations=args.gate_iters, loss_function="Logloss", **gpu(seed + 400))
                m.fit(Pool(Xtr, label=sdm.astype(np.int8)))
                acc_g[va] += m.predict_proba(Xe_cache[va])[:, 1]
                acc_gd[va] += m.predict_proba(Xe_eval[va])[:, 1]
                mb = CatBoostClassifier(iterations=args.bins_iters, loss_function="MultiClass", **gpu(seed + 600))
                mb.fit(Pool(Xtr[~sdm], label=cls_tr))
                order = np.argsort(mb.classes_.astype(int))
                pr = mb.predict_proba(Xe_cache[va])[:, order] @ centers
                acc_e[va] += pr
                prd = mb.predict_proba(Xe_eval[va])[:, order] @ centers
                acc_ed[va] += prd
            ns = len(seeds)
            gate_oof[va] = acc_g[va] / ns; exp_oof[va] = acc_e[va] / ns
            gate_drift[va] = acc_gd[va] / ns; exp_drift[va] = acc_ed[va] / ns
            print(f"[{arm}] fold {f} done ({time.time()-t0:.0f}s)", flush=True)

        pm = np.zeros(len(y), dtype=bool)
        for f in range(folds_n):
            pm |= (folds == f)
        ym = y[pm, 2]
        best = (-1.0, None)
        for thr in np.arange(0.02, 0.61, 0.01):
            for scale in (0.98, 1.0, 1.02):
                for offset in (-1.5, -1.0, -0.5, 0.0):
                    p = np.where(gate_oof[pm] >= thr, DEFAULTS["SW"],
                                 np.clip(exp_oof[pm] * scale + offset, *BOUNDS["SW"]))
                    s = score(p, ym, "SW") * 100
                    if s > best[0]:
                        best = (s, (round(float(thr), 2), scale, offset))
        thr, scale, offset = best[1]
        p_clean = np.where(gate_oof[pm] >= thr, DEFAULTS["SW"],
                           np.clip(exp_oof[pm] * scale + offset, *BOUNDS["SW"]))
        p_drift = np.where(gate_drift[pm] >= thr, DEFAULTS["SW"],
                           np.clip(exp_drift[pm] * scale + offset, *BOUNDS["SW"]))
        clean = score(p_clean, ym, "SW") * 100
        drift = score(p_drift, ym, "SW") * 100
        results[arm] = {
            "clean_oof": round(clean, 4), "drift_oof": round(drift, 4),
            "drift_delta": round(drift - clean, 4),
            "decode": {"threshold": thr, "scale": scale, "offset": offset},
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[{arm}] clean {clean:.4f}  drift {drift:.4f}  Δ {drift-clean:+.4f}  "
              f"thr={thr} scale={scale} off={offset}", flush=True)
        np.savez(args.out_dir / f"oof_{arm}.npz", gate_oof=gate_oof, exp_oof=exp_oof,
                 gate_drift=gate_drift, exp_drift=exp_drift, centers=centers)

    baselines = {"v97_sw_clean": 75.5364, "v97_sw_drift": 73.9284,
                 "v2recipe_aug_drift": 74.7058, "v3_assembly_sw_clean": 75.2425}
    report = {"results": results, "baselines": baselines,
              "total_seconds": round(time.time() - t00, 1)}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
