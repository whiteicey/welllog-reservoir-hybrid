"""Train the final drift-augmented SW head on ALL wells and build the v7 candidate:
v97 base (A-board anchor) + SW head swap.

Train: gate (Logloss, 950 iters) + regressor (RMSE, 500 iters) on originals +
K=2 drift-augmented copies, seeds 42/2026, predictions averaged (v3 convention).
Decode: threshold 0.09, scale 1.05, offset -3.0 (tuned on clean OOF in
drift_sw_train.py's aug arm, before any test data).

Outputs:
  outputs/drift_sw/final/{SW_gate_seed*.cbm, SW_reg_seed*.cbm, meta.json}
  outputs/drift_sw/final/test_sw.npy        (95,948 decoded SW values)
  outputs/drift_sw/final/candidate_v7.json  (v97 base + SW swap, official format)
  outputs/drift_sw/final/swap_stats.json
"""
from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BOUNDS, DEFAULTS, default_mask, load_or_build_cache
from src.data import TARGET_COLUMNS, load_split
from src.features import FeatureBuilder
from drift_sw_train import bounds_and_iqr, perturb_frame

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/drift_sw/final"
DECODE = {"threshold": 0.09, "scale": 1.05, "offset": -3.0}


def gpu(seed: int, **kw):
    return dict(task_type="GPU", devices="0", max_bin=128, random_seed=seed, learning_rate=0.04,
                depth=8, bootstrap_type="Bayesian", allow_writing_files=False, verbose=0, **kw)


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    C = load_or_build_cache(Path("data"), Path("cache"))
    y, folds = C["tr_y"], C["tr_fold"]
    meta = C["meta"]
    train_wells = meta["train_wells"]
    records = dict(load_split(Path("data"), "train"))
    Xe_cache, Xe_test = C["tr_enh"], C["te_enh"]

    iqr, pools, bounds = bounds_and_iqr(records, train_wells)
    builder = FeatureBuilder().fit([records[w] for w in train_wells])
    starts = np.concatenate([[0], np.cumsum([int((C["tr_well"] == w).sum()) for w in range(len(train_wells))])])
    row_ptr = {w: (int(starts[i]), int(starts[i + 1])) for i, w in enumerate(train_wells)}

    def xe_of(frame, valid):
        m = builder.transform(frame)
        return np.asarray(m.enhanced[valid], dtype=np.float32)

    K = 2
    Xe_aug = [np.zeros_like(Xe_cache) for _ in range(K)]
    for k in range(K):
        for w in train_wells:
            frame = records[w]
            valid = frame[TARGET_COLUMNS].notna().all(axis=1).to_numpy()
            pf = perturb_frame(frame, w, f"train{k}", iqr, bounds, 0.03, 0.20)
            Xe_aug[k][row_ptr[w][0]:row_ptr[w][1]] = xe_of(pf, valid)
    print(f"[prep] aug copies built ({time.time()-t0:.0f}s)", flush=True)

    Xtr = np.concatenate([Xe_cache] + Xe_aug)
    ytr = np.tile(y, (1 + K, 1))
    sw = ytr[:, 2]
    sdm = default_mask(sw, "SW")

    gate_test = np.zeros(len(Xe_test))
    reg_test = np.zeros(len(Xe_test))
    for seed in (42, 2026):
        m = CatBoostClassifier(iterations=950, loss_function="Logloss", **gpu(seed + 400))
        m.fit(Pool(Xtr, label=sdm.astype(np.int8)))
        m.save_model(str(OUT / f"SW_gate_seed{seed}.cbm"))
        gate_test += m.predict_proba(Xe_test)[:, 1]
        m = CatBoostRegressor(iterations=500, loss_function="RMSE", l2_leaf_reg=4.0, **gpu(seed + 500))
        m.fit(Pool(Xtr[~sdm], label=sw[~sdm]))
        m.save_model(str(OUT / f"SW_reg_seed{seed}.cbm"))
        reg_test += m.predict(Xe_test)
        print(f"[train] seed {seed} done ({time.time()-t0:.0f}s)", flush=True)
    gate_test /= 2
    reg_test /= 2

    sw_new = np.where(gate_test >= DECODE["threshold"], DEFAULTS["SW"],
                      np.clip(reg_test * DECODE["scale"] + DECODE["offset"], *BOUNDS["SW"]))
    np.save(OUT / "test_sw.npy", sw_new)

    meta_out = {
        "name": "drift-augmented SW head (v2-recipe regressor, K=2 copies)",
        "decode": DECODE,
        "features": "cache enhanced (237 dims, FeatureBuilder fit on originals)",
        "augmentation": {"copies": K, "sigma_lo": 0.03, "sigma_hi": 0.20,
                         "perturb_tag": "train{k}", "label_preserving": True},
        "seeds": [42, 2026],
        "oof_evidence": {
            "clean_oof": 74.7392, "drift_oof": 74.7058, "drift_delta": -0.0334,
            "v97_sw_clean": 75.5364, "v97_sw_drift": 73.9284, "v97_drift_delta": -1.608,
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta_out, indent=1))

    # ---------- build candidate: v97 base + SW swap ----------
    import zipfile as zf
    with zf.ZipFile(ROOT / "dl_method/最高分/result.zip") as z:
        v97 = json.loads(z.read("result.json"))
    test_wells = meta["test_wells"]
    te_well = C["te_well"]
    # map test row index ranges per well in resultData order (sorted files, same order)
    assert [w["logId"] for w in v97["resultData"]] == test_wells, "well order mismatch"
    starts_t = np.concatenate([[0], np.cumsum([int((te_well == i).sum()) for i in range(len(test_wells))])])

    out = {"resultData": []}
    changed = 0
    flips = 0
    band_cross = 0
    n_rows = 0
    for wi, w97 in enumerate(v97["resultData"]):
        rows = []
        s0, s1 = int(starts_t[wi]), int(starts_t[wi + 1])
        assert len(w97["predictions"]) == s1 - s0
        for j, p in enumerate(w97["predictions"]):
            new_sw = round(float(sw_new[s0 + j]), 6)
            old_sw = p["SW"]
            d = abs(new_sw - old_sw)
            if d > 1e-9:
                changed += 1
                if (abs(new_sw - DEFAULTS["SW"]) < 1e-9) != (abs(old_sw - DEFAULTS["SW"]) < 1e-9):
                    flips += 1
                elif d > 0.05 * max(abs(new_sw), abs(old_sw)):
                    band_cross += 1
            rows.append({"depth": p["depth"], "POR": p["POR"], "PERM": p["PERM"], "SW": new_sw})
            n_rows += 1
        out["resultData"].append({"logId": w97["logId"], "predictions": rows})
    (OUT / "candidate_v7.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with zf.ZipFile(OUT / "candidate_v7_result.zip", "w", zf.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(OUT / "candidate_v7.json", "result.json")

    stats = {"rows": n_rows, "sw_changed_vs_v97": changed,
             "sw_default_flips": flips, "sw_band_crossings": band_cross,
             "sw_default_frac_new": round(float(np.mean(np.abs(sw_new - DEFAULTS["SW"]) < 1e-9)), 4),
             "sw_default_frac_v97": round(float(np.mean([abs(p["SW"] - 99.9) < 1e-9
                 for w in v97["resultData"] for p in w["predictions"]])), 4),
             "total_seconds": round(time.time() - t0, 1)}
    (OUT / "swap_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
