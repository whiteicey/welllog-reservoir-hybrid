"""Decompose the drift-augmented SW head's advantage on OOF: gate vs continuous values.

Variants (all decoded with our tuned constants, scored with the official kernel):
  A  full swap      : our gate + our continuous            (measured: 74.7392 / 74.7058)
  B  values-only    : v97's gate (decoded==99.9) + our continuous where v97 continuous
  B' gate-only-ish  : our gate + v97's continuous where available
                      (v97's value is 99.9 where it gated default, so this is
                       our gate with v97 values = hybrid; computable)
  v97 reference    : its own decoded OOF SW (clean only; drift needs driver numbers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BOUNDS, DEFAULTS, score

ROOT = Path(__file__).resolve().parent
DEC = {"threshold": 0.09, "scale": 1.05, "offset": -3.0}


def main() -> None:
    oof = np.load(ROOT / "outputs/drift_sw/oof_aug.npz")
    z = np.load(ROOT / "dl_artifacts/runtime__v97__models__oof_predictions.npz")
    v97_sw = z["pred"][:, 2]
    y = np.load(ROOT / "cache/tr_y.npy")

    v97_def = np.abs(v97_sw - DEFAULTS["SW"]) < 1e-9
    our_cont = lambda reg: np.clip(reg * DEC["scale"] + DEC["offset"], *BOUNDS["SW"])

    variants = {}
    # A: full swap (clean + drift)
    for tag, gate, reg in (("clean", oof["gate_oof"], oof["reg_oof"]),
                           ("drift", oof["gate_drift"], oof["reg_drift"])):
        p = np.where(gate >= DEC["threshold"], DEFAULTS["SW"], our_cont(reg))
        variants[f"A_full_{tag}"] = round(score(p, y[:, 2], "SW") * 100, 4)
    # B: v97 gate + our continuous (clean + drift)
    for tag, reg in (("clean", oof["reg_oof"]), ("drift", oof["reg_drift"])):
        p = np.where(v97_def, DEFAULTS["SW"], our_cont(reg))
        variants[f"B_values_{tag}"] = round(score(p, y[:, 2], "SW") * 100, 4)
    # B': our gate + v97 continuous values where v97 is continuous (else our value if our gate says cont)
    for tag, gate, reg in (("clean", oof["gate_oof"], oof["reg_oof"]),
                           ("drift", oof["gate_drift"], oof["reg_drift"])):
        cont_v97 = np.where(v97_def, our_cont(reg), v97_sw)
        p = np.where(gate >= DEC["threshold"], DEFAULTS["SW"], cont_v97)
        variants[f"Bp_gate_{tag}"] = round(score(p, y[:, 2], "SW") * 100, 4)
    # reference: v97 clean
    variants["v97_clean"] = round(score(v97_sw, y[:, 2], "SW") * 100, 4)

    print(json.dumps(variants, indent=1))
    (ROOT / "outputs/drift_sw/variant_decomposition.json").write_text(json.dumps(variants, indent=1))


if __name__ == "__main__":
    main()
