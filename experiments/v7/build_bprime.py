"""Build the v7 candidate as B-prime: drift-augmented SW GATE + v97's continuous values.

Uses the final head models trained by final_head.py (all 80 wells + K=2 aug copies,
seeds 42/2026 averaged) and the pinned v97 test result.

Rule per row:
  - our gate (>= 0.09) says default  -> SW = 99.9
  - our gate says continuous:
      v97's SW != 99.9 (v97 continuous) -> SW = v97's value (unchanged)
      v97's SW == 99.9 (v97 default)    -> SW = our regressor value (decoded, clipped)
POR / PERM / depth: v97's values untouched.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BOUNDS, DEFAULTS, load_or_build_cache

ROOT = Path(__file__).resolve().parent
FIN = ROOT / "outputs/drift_sw/final"
DEC = {"threshold": 0.09, "scale": 1.05, "offset": -3.0}


def main() -> None:
    C = load_or_build_cache(Path("data"), Path("cache"))
    Xe_test, te_well = C["te_enh"], C["te_well"]
    meta = C["meta"]
    test_wells = meta["test_wells"]

    gate = np.zeros(len(Xe_test))
    reg = np.zeros(len(Xe_test))
    for seed in (42, 2026):
        g = CatBoostClassifier()
        g.load_model(str(FIN / f"SW_gate_seed{seed}.cbm"))
        gate += g.predict_proba(Xe_test)[:, 1]
        r = CatBoostRegressor()
        r.load_model(str(FIN / f"SW_reg_seed{seed}.cbm"))
        reg += r.predict(Xe_test)
    gate /= 2
    reg /= 2
    np.save(FIN / "test_gate.npy", gate)
    np.save(FIN / "test_reg.npy", reg)

    our_def = gate >= DEC["threshold"]
    our_cont = np.clip(reg * DEC["scale"] + DEC["offset"], *BOUNDS["SW"])

    with zipfile.ZipFile(ROOT / "dl_method/v97_result.zip") as z:
        v97 = json.loads(z.read("result.json"))
    assert [w["logId"] for w in v97["resultData"]] == test_wells
    starts = np.concatenate([[0], np.cumsum([int((te_well == i).sum()) for i in range(len(test_wells))])])

    out = {"resultData": []}
    stats = {"rows": 0, "changed": 0, "to_default": 0, "to_cont_v97": 0, "to_cont_ours": 0}
    for wi, w97 in enumerate(v97["resultData"]):
        rows = []
        s0, s1 = int(starts[wi]), int(starts[wi + 1])
        for j, p in enumerate(w97["predictions"]):
            v97_def = abs(p["SW"] - DEFAULTS["SW"]) < 1e-9
            if our_def[s0 + j]:
                new_sw = DEFAULTS["SW"]
                if not v97_def:
                    stats["to_default"] += 1
            else:
                if v97_def:
                    new_sw = round(float(our_cont[s0 + j]), 6)
                    stats["to_cont_ours"] += 1
                else:
                    new_sw = p["SW"]
            if abs(new_sw - p["SW"]) > 1e-9:
                stats["changed"] += 1
            rows.append({"depth": p["depth"], "POR": p["POR"], "PERM": p["PERM"], "SW": new_sw})
            stats["rows"] += 1
        out["resultData"].append({"logId": w97["logId"], "predictions": rows})
    (FIN / "candidate_v7_bprime.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with zipfile.ZipFile(FIN / "candidate_v7_bprime_result.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(FIN / "candidate_v7_bprime.json", "result.json")
    (FIN / "bprime_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
