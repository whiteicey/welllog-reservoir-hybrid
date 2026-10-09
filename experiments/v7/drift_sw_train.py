"""Drift-augmented SW head retrain (v3 pipeline, v2-recipe SW head).

Two arms, identical recipe (SW gate: Logloss on Xe 950 iters; SW continuous:
RMSE regressor on Xe non-default rows, 500 iters; 2 seeds averaged; 5-fold
GroupKFold reusing cache/tr_fold):
  - noaug : train on original 80 wells only
  - aug   : train on originals + K drift-augmented copies per well
    (label-preserving per-well per-curve shifts in pooled-IQR units +
     small scale jitter; RT/RXO also get log-domain jitter)

Evaluation:
  - clean OOF   : held-out fold, original rows (official kernel)
  - drift OOF   : held-out fold, freshly-perturbed rows (different seed;
                  the drift the model never saw in training)
  - decode      : two-pass — per-fold models produce clean OOF; a single
                  global (threshold, scale, offset) is tuned on clean OOF,
                  then applied unchanged to both clean and drift rows.
Baselines reported: cv_v2 SW (same recipe, original inputs, from
outputs/cv_v2/report_v2_rule.json), v3 assembly SW, v97 SW.

Usage:
  python drift_sw_train.py --arm both --copies 2 [--calibrate-only] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (BOUNDS, DEFAULTS, N_FOLDS, V2_CAL, default_mask, load_or_build_cache, score)
from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder

MISSING = -99999.0
CURVES = ["GR", "PE", "SP", "CAL", "AC", "DEN", "CNL", "RXO", "RT"]
# per-curve drift-amplitude multipliers (observed: CAL/RT/PE drift hardest)
CURVE_W = {"GR": 1.0, "PE": 1.3, "SP": 1.0, "CAL": 1.6, "AC": 0.9, "DEN": 1.0,
           "CNL": 0.9, "RXO": 1.2, "RT": 1.4}
LOG_CURVES = ("RT", "RXO")


def gpu(seed: int, **kw):
    return dict(task_type="GPU", devices="0", max_bin=128, random_seed=seed, learning_rate=0.04,
                depth=8, bootstrap_type="Bayesian", allow_writing_files=False, verbose=0, **kw)


def rng_for(well: str, tag: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{well}:{tag}".encode()) & 0xFFFFFFFF)


def perturb_frame(frame: pd.DataFrame, well: str, tag: str, iqr: dict, bounds: dict,
                  sigma_lo: float, sigma_hi: float) -> pd.DataFrame:
    """Label-preserving per-well curve perturbation (shift in pooled-IQR units + scale jitter)."""
    rng = rng_for(well, tag)
    sigma_w = rng.uniform(sigma_lo, sigma_hi)
    out = frame.copy()
    for c in CURVES:
        s = out[c].to_numpy(np.float64).copy()
        valid = s > MISSING / 2
        z = rng.normal(0.0, sigma_w * CURVE_W[c])
        scale = float(rng.normal(1.0, 0.04))
        sv = s[valid]
        sv = sv * scale + z * iqr[c]
        if c in LOG_CURVES:
            dz = rng.normal(0.0, 0.18 * sigma_w * CURVE_W[c])
            sv = 10.0 ** (np.log10(np.maximum(sv, 1e-3)) + dz)
        lo, hi = bounds[c]
        s[valid] = np.clip(sv, lo, hi)
        out[c] = s
    return out


def ks_vs_pool(x: np.ndarray, pool: np.ndarray) -> float:
    from scipy.stats import ks_2samp
    a = x[x > MISSING / 2]
    a = a[np.isfinite(a)]
    if len(a) < 50:
        return float("nan")
    return float(ks_2samp(a, pool).statistic)


def bounds_and_iqr(records: dict, train_wells: list) -> tuple[dict, dict, dict]:
    """Pooled IQR per curve + pooled samples + perturbation bounds (shared calibration)."""
    iqr: dict[str, float] = {}
    pools: dict[str, np.ndarray] = {}
    for c in CURVES:
        allv = np.concatenate([records[w][c].to_numpy(np.float64) for w in train_wells])
        allv = allv[allv > MISSING / 2]
        pools[c] = allv
        q75, q25 = np.percentile(allv, [75, 25])
        iqr[c] = max(float(q75 - q25), 1e-6)
    FLOOR = {"GR": 1.0, "RT": 0.05, "RXO": 0.05, "DEN": 1.6}
    bounds: dict[str, tuple[float, float]] = {}
    for c in CURVES:
        q_lo, q_hi = np.percentile(pools[c], [0.1, 99.9])
        bounds[c] = (max(FLOOR.get(c, -np.inf), float(q_lo)), float(q_hi) + 0.5 * iqr[c])
    return iqr, pools, bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/drift_sw"))
    ap.add_argument("--arm", choices=("noaug", "aug", "both"), default="both")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--sigma-lo", type=float, default=0.03)
    ap.add_argument("--sigma-hi", type=float, default=0.20)
    ap.add_argument("--seeds", default="42,2026")
    ap.add_argument("--gate-iters", type=int, default=950)
    ap.add_argument("--reg-iters", type=int, default=500)
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--quick", action="store_true", help="1 seed, 1 fold, quarter iters (smoke test)")
    args = ap.parse_args()
    t00 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",")]
    folds_n = 1 if args.quick else N_FOLDS
    if args.quick:
        seeds = seeds[:1]
        args.gate_iters //= 4
        args.reg_iters //= 4

    # ---------- load cache + raw frames ----------
    C = load_or_build_cache(args.data_dir, args.cache_dir)
    y, folds, wells_arr = C["tr_y"], C["tr_fold"], C["tr_well"]
    meta = C["meta"]
    train_wells = meta["train_wells"]
    records = dict(load_split(args.data_dir, "train"))
    assert set(records) == set(train_wells)

    # pooled IQR per curve (from ORIGINAL wells; frozen for perturbation)
    iqr: dict[str, float] = {}
    pools: dict[str, np.ndarray] = {}
    for c in CURVES:
        allv = np.concatenate([
            records[w][c].to_numpy(np.float64) for w in train_wells
        ])
        allv = allv[allv > MISSING / 2]
        pools[c] = allv
        q75, q25 = np.percentile(allv, [75, 25])
        iqr[c] = max(float(q75 - q25), 1e-6)

    # per-curve perturbation bounds: data-driven (pool q0.1/q99.9) + physical floors
    FLOOR = {"GR": 1.0, "RT": 0.05, "RXO": 0.05, "DEN": 1.6}
    bounds: dict[str, tuple[float, float]] = {}
    for c in CURVES:
        q_lo, q_hi = np.percentile(pools[c], [0.1, 99.9])
        bounds[c] = (max(FLOOR.get(c, -np.inf), float(q_lo)), float(q_hi) + 0.5 * iqr[c])

    # builder fit on ORIGINALS only (same as cache build)
    builder = FeatureBuilder().fit([records[w] for w in train_wells])

    # per-well valid masks + Xe rows for originals; sanity vs cache
    Xe_cache = C["tr_enh"]
    row_ptr = {}  # well -> (start, end) in cache row layout
    starts = np.concatenate([[0], np.cumsum([int((wells_arr == w).sum()) for w in range(len(train_wells))])])
    for wi, w in enumerate(train_wells):
        row_ptr[w] = (int(starts[wi]), int(starts[wi + 1]))

    def xe_of(frame: pd.DataFrame, valid: np.ndarray) -> np.ndarray:
        m = builder.transform(frame)
        return np.asarray(m.enhanced[valid], dtype=np.float32)

    Xe_eval = np.zeros_like(Xe_cache)      # freshly-perturbed rows (drift eval)
    ks_rows = []
    ks_orig_rows = []
    max_dev = 0.0
    for wi, w in enumerate(train_wells):
        frame = records[w]
        valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
        # sanity: recomputed originals match the cache
        dev = float(np.max(np.abs(xe_of(frame, valid) - Xe_cache[row_ptr[w][0]:row_ptr[w][1]])))
        max_dev = max(max_dev, dev)
        # leave-one-out baseline: original well vs pool of the OTHER 79
        ko = {}
        for c in CURVES:
            others = np.concatenate([
                records[u][c].to_numpy(np.float64) for u in train_wells if u != w
            ])
            others = others[others > MISSING / 2]
            ko[c] = round(ks_vs_pool(frame[c].to_numpy(np.float64), others), 3)
        ks_orig_rows.append({"well": w[:8], **ko,
                             "mean_ks": round(float(np.mean(list(ko.values()))), 3)})
        pf = perturb_frame(frame, w, "eval", iqr, bounds, args.sigma_lo, args.sigma_hi)
        Xe_eval[row_ptr[w][0]:row_ptr[w][1]] = xe_of(pf, valid)
        ks_rows.append({"well": w[:8], "sigma": None,
                        **{c: round(ks_vs_pool(pf[c].to_numpy(np.float64), pools[c]), 3) for c in CURVES}})
    for r in ks_rows:
        r["mean_ks"] = round(float(np.mean([r[c] for c in CURVES])), 3)
    orig_ks = [r["mean_ks"] for r in ks_orig_rows]
    calib = {
        "recomputed_original_max_dev": max_dev,
        "loo_original_well_mean_ks": {"min": round(min(orig_ks), 3),
                                      "median": round(float(np.median(orig_ks)), 3),
                                      "max": round(max(orig_ks), 3)},
        "eval_well_mean_ks": [r["mean_ks"] for r in ks_rows],
        "eval_wells": ks_rows,
        "loo_original_wells": ks_orig_rows,
        "observed_test_mean_ks_range": [0.114, 0.240],
    }
    print(f"[calib] originals recomputed max|dev| vs cache = {max_dev:.3e}", flush=True)
    print(f"[calib] LOO original wells mean-KS: min {min(orig_ks):.3f} "
          f"median {float(np.median(orig_ks)):.3f} max {max(orig_ks):.3f} "
          f"(test wells: 0.114-0.240)", flush=True)
    print(f"[calib] eval perturbed wells mean-KS: min {min(calib['eval_well_mean_ks']):.3f} "
          f"median {float(np.median(calib['eval_well_mean_ks'])):.3f} "
          f"max {max(calib['eval_well_mean_ks']):.3f} (target ~0.11-0.24)", flush=True)
    (args.out_dir / "calibration.json").write_text(json.dumps(calib, indent=1))
    if args.calibrate_only:
        return
    assert max_dev < 1e-3, f"feature reproduction mismatch: {max_dev}"

    # ---------- augmented copies ----------
    K = args.copies
    Xe_aug = [np.zeros_like(Xe_cache) for _ in range(K)]
    for k in range(K):
        for wi, w in enumerate(train_wells):
            frame = records[w]
            valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
            pf = perturb_frame(frame, w, f"train{k}", iqr, bounds, args.sigma_lo, args.sigma_hi)
            Xe_aug[k][row_ptr[w][0]:row_ptr[w][1]] = xe_of(pf, valid)
    print(f"[aug] {K} training copies built ({time.time()-t00:.0f}s)", flush=True)

    # ---------- train both arms ----------
    results = {}
    for arm in (("noaug", "aug") if args.arm == "both" else (args.arm,)):
        t0 = time.time()
        gate_oof = np.zeros(len(y))
        reg_oof = np.zeros(len(y))
        gate_drift = np.zeros(len(y))
        reg_drift = np.zeros(len(y))
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
            acc_g = np.zeros(len(y)); acc_r = np.zeros(len(y))
            acc_gd = np.zeros(len(y)); acc_rd = np.zeros(len(y))
            for seed in seeds:
                m = CatBoostClassifier(iterations=args.gate_iters, loss_function="Logloss", **gpu(seed + 400))
                m.fit(Pool(Xtr, label=sdm.astype(np.int8)))
                acc_g[va] += m.predict_proba(Xe_cache[va])[:, 1]
                acc_gd[va] += m.predict_proba(Xe_eval[va])[:, 1]
                m = CatBoostRegressor(iterations=args.reg_iters, loss_function="RMSE", l2_leaf_reg=4.0, **gpu(seed + 500))
                m.fit(Pool(Xtr[~sdm], label=sw[~sdm]))
                acc_r[va] += m.predict(Xe_cache[va])
                acc_rd[va] += m.predict(Xe_eval[va])
            ns = len(seeds)
            gate_oof[va] = acc_g[va] / ns; reg_oof[va] = acc_r[va] / ns
            gate_drift[va] = acc_gd[va] / ns; reg_drift[va] = acc_rd[va] / ns
            print(f"[{arm}] fold {f} done ({time.time()-t0:.0f}s)", flush=True)
        # rows actually predicted (quick mode covers fewer folds)
        pm = np.zeros(len(y), dtype=bool)
        for f in range(folds_n):
            pm |= (folds == f)
        ym = y[pm, 2]
        # two-pass decode: tune (thr, scale, offset) on clean OOF only
        best = (-1.0, None)
        for thr in np.arange(0.02, 0.61, 0.01):
            for scale in (1.0, 1.05):
                for offset in (-3.0, -2.0, -1.0, 0.0):
                    p = np.where(gate_oof[pm] >= thr, DEFAULTS["SW"],
                                 np.clip(reg_oof[pm] * scale + offset, *BOUNDS["SW"]))
                    s = score(p, ym, "SW") * 100
                    if s > best[0]:
                        best = (s, (round(float(thr), 2), scale, offset))
        thr, scale, offset = best[1]
        p_clean = np.where(gate_oof[pm] >= thr, DEFAULTS["SW"],
                           np.clip(reg_oof[pm] * scale + offset, *BOUNDS["SW"]))
        p_drift = np.where(gate_drift[pm] >= thr, DEFAULTS["SW"],
                           np.clip(reg_drift[pm] * scale + offset, *BOUNDS["SW"]))
        clean = score(p_clean, ym, "SW") * 100
        drift = score(p_drift, ym, "SW") * 100
        # v2-rule reference decode
        v2 = V2_CAL["SW"]
        p_v2 = np.where(gate_oof[pm] >= v2["threshold"], DEFAULTS["SW"],
                        np.clip(reg_oof[pm] * v2["scale"] + v2["offset"], *BOUNDS["SW"]))
        results[arm] = {
            "clean_oof": round(clean, 4), "drift_oof": round(drift, 4),
            "drift_delta": round(drift - clean, 4),
            "decode": {"threshold": thr, "scale": scale, "offset": offset},
            "v2rule_clean_oof": round(score(p_v2, ym, "SW") * 100, 4),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[{arm}] clean {clean:.4f}  drift {drift:.4f}  Δ {drift-clean:+.4f}  "
              f"decode thr={thr} scale={scale} off={offset}", flush=True)
        np.savez(args.out_dir / f"oof_{arm}.npz", gate_oof=gate_oof, reg_oof=reg_oof,
                 gate_drift=gate_drift, reg_drift=reg_drift)

    # baselines
    baselines = {"v3_assembly_sw": 75.2425, "v97_sw": 75.5354}
    cv2_rep = Path("outputs/cv_v2/report_v2_rule.json")
    if cv2_rep.exists():
        baselines["cv_v2_recipe_sw"] = round(json.loads(cv2_rep.read_text())["SW"]["micro"] * 100, 4)
    report = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "results": results, "baselines": baselines,
              "calibration_summary": {"eval_mean_ks_median": float(np.median(calib["eval_well_mean_ks"]))},
              "total_seconds": round(time.time() - t00, 1)}
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report["results"], indent=1), flush=True)
    print("baselines:", baselines, flush=True)


if __name__ == "__main__":
    main()
