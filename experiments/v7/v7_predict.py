"""v7 submission: Drift-Augmented Gate Transplant on an Eighteen-Stage Cascaded Backbone.

Pipeline (all from raw test data, one command):
  1. runtime/backbone/predict.py  -- the eighteen-stage cascaded backbone
     (dual U-Net, 5-fold ensemble, A-board 83.9329 anchor) produces POR/PERM
     for every row and its own SW values.
  2. feature/  -- the tree-sequence-hybrid FeatureBuilder (237 enhanced
     features, medians frozen in feature/meta.json; no training data read).
  3. sw_head/  -- drift-augmented SW gate (CatBoost Logloss, 2 seeds averaged)
     + fallback regressor, trained on 80 wells + 2 label-preserving
     drift-augmented copies per well.
  4. Composition per row (B-prime):
       gate says default      -> SW = 99.9
       gate says continuous
         and backbone too     -> SW = backbone's value (unchanged)
         but backbone default -> SW = head regressor value (x1.05 - 3.0,
                                 clipped to [8.305, 99.9])
     POR / PERM / depth       -> backbone's values, untouched.

Usage:
    python predict.py --data_dir ./data --output ./result.json [--device cuda|cpu|auto]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "feature"))
from src.data import read_curve  # noqa: E402
from src.feature import FeatureBuilder  # noqa: E402

BOUNDS_SW = (8.305, 99.9)
DEFAULT_SW = 99.9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", "--data-dir", dest="data_dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--batch_windows", "--batch-windows", type=int, default=32)
    ap.add_argument("--skip_pinned_check", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    prov = json.loads((ROOT / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
    head_meta = json.loads((ROOT / "sw_head/meta.json").read_text(encoding="utf-8"))
    dec = head_meta["decode"]

    # ---- 1. backbone inference ----
    with tempfile.TemporaryDirectory(prefix="v7_") as tmp:
        tmp = Path(tmp)
        base_out = tmp / "base_result.json"
        cmd = [sys.executable, str(ROOT / "runtime/backbone/predict.py"),
               "--data_dir", str(args.data_dir.resolve()),
               "--output", str(base_out), "--device", args.device,
               "--batch_windows", str(args.batch_windows)]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT / "runtime/backbone", check=True)
        got = hashlib.sha256(base_out.read_bytes()).hexdigest()
        want = prov.get("base_result_sha256_h20")
        if want and got != want and not args.skip_pinned_check:
            raise SystemExit(f"backbone result sha256 {got} != pinned {want} "
                             "(cross-GPU bf16 noise; rerun with --skip_pinned_check)")
        if want and got != want:
            print("WARN backbone result SHA differs from pinned (float noise)", flush=True)
        base = json.loads(base_out.read_text(encoding="utf-8"))

    # ---- 2. test features ----
    files = sorted((args.data_dir / "test").glob("*.txt")) if (args.data_dir / "test").is_dir() \
        else sorted(args.data_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no test curve files under {args.data_dir}")
    builder = FeatureBuilder.from_metadata(
        json.loads((ROOT / "feature/meta.json").read_text(encoding="utf-8")))

    gate_models = []
    for p in sorted((ROOT / "sw_head").glob("SW_gate_seed*.cbm")):
        m = CatBoostClassifier()
        m.load_model(str(p))
        gate_models.append(m)
    reg_models = []
    for p in sorted((ROOT / "sw_head").glob("SW_reg_seed*.cbm")):
        m = CatBoostRegressor()
        m.load_model(str(p))
        reg_models.append(m)

    result = {"resultData": []}
    stats = {"rows": 0, "sw_changed": 0, "to_default": 0, "to_cont_head": 0}
    base_wells = [w["logId"] for w in base["resultData"]]
    file_stems = [f.stem for f in files]
    if base_wells != file_stems:
        raise SystemExit("backbone well order does not match data dir order")

    for wi, (f, wb) in enumerate(zip(files, base["resultData"])):
        frame = read_curve(f)
        enhanced = builder.transform(frame).enhanced.astype(np.float32)
        assert len(enhanced) == len(wb["predictions"]), \
            f"row count mismatch in {f.stem}: {len(enhanced)} vs {len(wb['predictions'])}"
        gate = np.zeros(len(enhanced))
        reg = np.zeros(len(enhanced))
        for m in gate_models:
            gate += m.predict_proba(enhanced)[:, 1]
        gate /= len(gate_models)
        for m in reg_models:
            reg += m.predict(enhanced)
        reg /= len(reg_models)
        our_def = gate >= dec["threshold"]
        our_cont = np.clip(reg * dec["scale"] + dec["offset"], *BOUNDS_SW)

        rows = []
        for j, p in enumerate(wb["predictions"]):
            base_def = abs(p["SW"] - DEFAULT_SW) < 1e-9
            if our_def[j]:
                new_sw = DEFAULT_SW
                if not base_def:
                    stats["to_default"] += 1
            elif base_def:
                new_sw = round(float(our_cont[j]), 6)
                stats["to_cont_head"] += 1
            else:
                new_sw = p["SW"]
            if abs(new_sw - p["SW"]) > 1e-9:
                stats["sw_changed"] += 1
            rows.append({"depth": p["depth"], "POR": p["POR"], "PERM": p["PERM"], "SW": new_sw})
            stats["rows"] += 1
        result["resultData"].append({"logId": wb["logId"], "predictions": rows})
        print(f"DONE {wb['logId']} rows={len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with zipfile.ZipFile(args.output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(args.output, "result.json")
    report = {
        "method": "Drift-Augmented Gate Transplant (B-prime): backbone POR/PERM + backbone SW values, SW default decisions from drift-augmented gate",
        "decode": dec,
        "rows": stats["rows"], "wells": len(result["resultData"]),
        "stats": stats,
        "final_result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "seconds": round(time.time() - t0, 1),
    }
    args.output.with_suffix(".inference.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print("INFERENCE_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
