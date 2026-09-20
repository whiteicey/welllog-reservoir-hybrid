"""Build ablation result files for the submission matrix by mixing two result.json files row by row.

  gate_from=A, cont_from=B : take A's default/non-default decision; where non-default use B's continuous value
                             if B also predicted non-default, else fall back to A's continuous value.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEF = {"POR": 0.1, "PERM": 0.01, "SW": 99.9}


def is_default(v, t):
    return v >= DEF[t] if t == "SW" else v <= DEF[t]


def load(p):
    r = json.loads(Path(p).read_text(encoding="utf-8"))
    return r, {e["logId"]: e["predictions"] for e in r["resultData"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-from", required=True)
    ap.add_argument("--cont-from", required=True)
    ap.add_argument("--targets", default="POR,PERM,SW", help="targets to mix; others copied from --gate-from")
    ap.add_argument("--output", required=True)
    ap.add_argument("--name", default="ablation")
    args = ap.parse_args()
    ra, A = load(args.gate_from); rb, B = load(args.cont_from)
    targets = args.targets.split(",")
    changed = {t: 0 for t in targets}; fallback = {t: 0 for t in targets}
    out = []
    for wid, pa in A.items():
        pb = B[wid]; assert len(pa) == len(pb)
        rows = []
        for a, b in zip(pa, pb):
            assert abs(a["depth"] - b["depth"]) < 1e-6
            r = dict(a)
            for t in targets:
                if not is_default(a[t], t):
                    if not is_default(b[t], t):
                        if abs(b[t] - a[t]) > 1e-9:
                            changed[t] += 1
                        r[t] = b[t]
                    else:
                        fallback[t] += 1
            rows.append(r)
        out.append({"logId": wid, "predictions": rows})
    res = {"modelId": "", "modelName": args.name, "version": ra.get("version", "3.0"), "resultData": out}
    Path(args.output).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", args.output, "changed", changed, "fallback_to_gate_source_cont", fallback)


if __name__ == "__main__":
    main()
