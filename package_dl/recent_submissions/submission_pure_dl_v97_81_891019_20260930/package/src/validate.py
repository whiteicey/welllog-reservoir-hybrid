"""Validate a result.json against the official template and the test wells.

python -m src.validate --data_dir ./data --result ./result.json [--reference other_result.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .data import load_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", "--data-dir", dest="data_dir", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--reference", type=Path, default=None, help="optional other result.json to diff against")
    args = ap.parse_args()
    r = json.loads(args.result.read_text(encoding="utf-8"))
    for k in ("modelId", "modelName", "version", "resultData"):
        assert k in r, f"missing top-level key {k}"
    wells = {w: f for w, f in load_split(args.data_dir, "test")}
    got = {e["logId"]: e for e in r["resultData"]}
    assert set(got) == set(wells), f"well id mismatch: missing {set(wells) - set(got)}, extra {set(got) - set(wells)}"
    total = 0
    stats = {"POR_default": 0, "PERM_default": 0, "SW_default": 0}
    for wid, frame in wells.items():
        preds = got[wid]["predictions"]
        assert len(preds) == len(frame), f"{wid}: {len(preds)} rows vs {len(frame)} in file"
        depth = frame["DEPTH"].to_numpy(np.float64)
        for i, p in enumerate(preds):
            assert set(p) == {"depth", "POR", "PERM", "SW"}, f"{wid} row {i} keys {set(p)}"
            assert abs(p["depth"] - depth[i]) < 1e-6, f"{wid} row {i} depth {p['depth']} vs {depth[i]}"
            for k, (lo, hi) in (("POR", (0, 50)), ("PERM", (0.01, 10000)), ("SW", (8.3, 99.9))):
                v = p[k]
                assert isinstance(v, (int, float)) and math.isfinite(v) and lo - 1e-9 <= v <= hi + 1e-9, f"{wid} row {i} {k}={v}"
            stats["POR_default"] += p["POR"] <= 0.1; stats["PERM_default"] += p["PERM"] <= 0.01; stats["SW_default"] += p["SW"] >= 99.9
        total += len(preds)
    print("OK wells", len(wells), "rows", total, {k: round(v / total, 4) for k, v in stats.items()})
    if args.reference and args.reference.exists():
        ref = {e["logId"]: e for e in json.loads(args.reference.read_text(encoding="utf-8"))["resultData"]}
        diff = {"POR": 0, "PERM": 0, "SW": 0}; n = 0
        for wid in wells:
            for a, b in zip(got[wid]["predictions"], ref[wid]["predictions"]):
                n += 1
                for k in diff:
                    diff[k] += abs(a[k] - b[k]) > 1e-9
        print("rows differing from reference:", {k: (v, round(v / n, 4)) for k, v in diff.items()})


if __name__ == "__main__":
    main()
