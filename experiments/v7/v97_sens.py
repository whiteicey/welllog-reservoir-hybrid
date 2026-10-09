"""Measure v97's SW drift sensitivity on the train wells.

Writes the SAME eval-perturbed wells used by drift_sw_train.py (tag 'eval',
identical seeds/sigma/bounds) plus the originals into two temp data dirs,
runs the v97 runtime on both, and scores SW with true labels.

v97's predictions on train wells are fold-contaminated (4/5 bundles saw each
well), but the DELTA between original and perturbed passes isolates the
drift sensitivity fairly (same contamination in both passes).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import score
from src.data import TARGET_COLUMNS, load_split
from drift_sw_train import CURVES, perturb_frame

MISSING = -99999.0
ROOT = Path("/home/ubuntu/kunlun_welllog_v3_20260919")
V97_RT = ROOT / "v97_verify" / "runtime" / "v97"
PY = str(ROOT / ".venv/bin/python")


def write_data_dir(out: Path, frames: dict[str, pd.DataFrame], raw_root: Path) -> None:
    """Write wells as official-format txt: keep the original 2 header lines
    (names + units) so read_curve's skiprows=[1] eats the units row, not data."""
    (out / "test").mkdir(parents=True, exist_ok=True)
    for w, df in frames.items():
        lines = (raw_root / f"{w}.txt").read_text(encoding="utf-8-sig").splitlines()
        header, units = lines[0], lines[1]
        data = df.to_csv(header=False, index=False, float_format="%.3f")
        (out / "test" / f"{w}.txt").write_text("\n".join([header, units, data]) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.time()
    records = dict(load_split(ROOT / "data", "train"))
    wells = sorted(records)

    # pooled IQR + bounds (identical logic to drift_sw_train)
    iqr, pools = {}, {}
    for c in CURVES:
        allv = np.concatenate([records[w][c].to_numpy(np.float64) for w in wells])
        allv = allv[allv > MISSING / 2]
        pools[c] = allv
        q75, q25 = np.percentile(allv, [75, 25])
        iqr[c] = max(float(q75 - q25), 1e-6)
    FLOOR = {"GR": 1.0, "RT": 0.05, "RXO": 0.05, "DEN": 1.6}
    bounds = {}
    for c in CURVES:
        q_lo, q_hi = np.percentile(pools[c], [0.1, 99.9])
        bounds[c] = (max(FLOOR.get(c, -np.inf), float(q_lo)), float(q_hi) + 0.5 * iqr[c])

    orig_frames, drift_frames = {}, {}
    for w in wells:
        frame = records[w]
        orig_frames[w] = frame
        drift_frames[w] = perturb_frame(frame, w, "eval", iqr, bounds, 0.03, 0.20)

    d1, d2 = Path("/tmp/v97_sens/orig"), Path("/tmp/v97_sens/drift")
    raw_root = ROOT / "data" / "train"
    write_data_dir(d1, orig_frames, raw_root)
    write_data_dir(d2, drift_frames, raw_root)
    # stale results from a bad file layout must not be reused
    for tag in ("orig", "drift"):
        p = Path(f"/tmp/v97_sens/result_{tag}.json")
        if p.exists():
            p.unlink()
    print(f"[prep] data dirs written ({time.time()-t0:.0f}s)", flush=True)

    # run v97 runtime on both dirs (skip if results already exist)
    for tag, dd in (("orig", d1), ("drift", d2)):
        if Path(f"/tmp/v97_sens/result_{tag}.json").exists():
            print(f"[v97-{tag}] reusing existing result", flush=True)
            continue
        r = subprocess.run(
            [PY, str(V97_RT / "predict.py"), "--data_dir", str(dd),
             "--output", f"/tmp/v97_sens/result_{tag}.json",
             "--device", "cuda", "--batch_windows", "32"],
            cwd=V97_RT, capture_output=True, text=True)
        print(f"[v97-{tag}] exit={r.returncode} ({time.time()-t0:.0f}s)", flush=True)
        if r.returncode != 0:
            print((r.stdout + r.stderr)[-400:])
            return

    # score SW on valid rows
    out = {}
    for tag in ("orig", "drift"):
        res = json.load(open(f"/tmp/v97_sens/result_{tag}.json"))
        pred = {w["logId"]: {round(p["depth"], 2): p["SW"] for p in w["predictions"]}
                for w in res["resultData"]}
        ys, ps = [], []
        for w in wells:
            frame = records[w]
            valid = frame[TARGET_COLUMNS].notna().all(axis=1)
            for depth, sw in zip(frame.loc[valid, "DEPTH"], frame.loc[valid, "SW"]):
                ys.append(float(sw))
                ps.append(float(pred[w][round(float(depth), 2)]))
        out[tag] = round(score(np.array(ps), np.array(ys), "SW") * 100, 4)
        print(f"[score] v97 SW on {tag} train wells: {out[tag]}", flush=True)
    report = {"v97_sw_orig_train": out["orig"], "v97_sw_drift_train": out["drift"],
              "drift_delta": round(out["drift"] - out["orig"], 4),
              "note": "predictions fold-contaminated; delta isolates sensitivity",
              "seconds": round(time.time() - t0, 1)}
    (ROOT / "outputs/drift_sw/v97_sensitivity.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
