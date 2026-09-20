"""Collect every experiment report on the server into one machine-readable summary + markdown table."""
from __future__ import annotations
import glob
import json
from pathlib import Path

rows = []
summary = {}
for p in sorted(glob.glob("outputs/*/report.json")):
    name = p.split("/")[1]
    r = json.load(open(p))
    entry = {"name": name}
    if "nondefault_only" in r:  # exp_variant bins/regression
        t = r["args"]["target"]
        entry.update({"kind": "catboost_variant", "target": t, "features": r["args"]["features"], "model": r["args"]["model"], "bins": r["args"].get("bins"),
                      "seeds": r["args"]["seeds"], "nondef_v2": r["nondefault_only"]["v2"], "nondef_utility": r["nondefault_only"].get("utility", r["nondefault_only"].get("point")),
                      "full_v2": r["full_v2gate"]["v2"]["micro"], "full_utility_v2gate": r["full_v2gate"].get("utility", r["full_v2gate"].get("point"))["micro"],
                      "joint_full": r.get("joint_full", {}).get("micro"), "joint_ci95": r.get("joint_full", {}).get("paired_vs_v2", {}).get("ci95"),
                      "joint_well_wins": r.get("joint_full", {}).get("paired_vs_v2", {}).get("well_wins"),
                      "fold_seconds": [round(f["seconds"]) for f in r["folds"]]})
    elif "gate" in r and "args" in r:
        entry.update({"kind": "gate_variant", "target": r["args"]["target"], "features": r["args"]["features"], **r["gate"]})
    elif "TOTAL" in r and "nn" in r["TOTAL"]:
        entry.update({"kind": "sequence_net", "TOTAL_nn": r["TOTAL"]["nn"], "TOTAL_v2": r["TOTAL"]["v2"], "args": r["args"]})
        for t in ("POR", "PERM", "SW"):
            entry[f"{t}_nondef_nn"] = r[t]["nondefault_only"]["nn"]; entry[f"{t}_nondef_v2"] = r[t]["nondefault_only"]["v2"]
            entry[f"{t}_gate_nn"] = r[t]["gate_acc"]["nn@0.5"]; entry[f"{t}_gate_v2"] = r[t]["gate_acc"]["v2"]
            entry[f"{t}_full_nn"] = r[t]["full"]["nn"]; entry[f"{t}_full_joint"] = r[t]["full"].get("nn_joint")
        entry["fold_seconds"] = [round(f["seconds"]) for f in r["log"]]
    elif "state_accuracy_argmax" in r:
        entry.update({"kind": "state_model", "state_acc": r["state_accuracy_argmax"]})
        for v in ("raw",):
            for k, val in r[v].items():
                if isinstance(val, dict):
                    entry[f"{k}_nested"] = val["nested_score"]; entry[f"{k}_v2"] = val["v2_score"]; entry[f"{k}_ci95"] = val["paired_vs_v2"]["ci95"]
    elif "TOTAL_nested" in r:
        entry.update({"kind": "assembly", "TOTAL_nested": r["TOTAL_nested"], "TOTAL_v2": r["TOTAL_v2"], "TOTAL_optimistic": r["TOTAL_global_selected_optimistic"]})
        for t in ("POR", "PERM", "SW"):
            entry[f"{t}_nested"] = r[t]["nested"]["micro"]; entry[f"{t}_v2"] = r[t]["v2"]["micro"]; entry[f"{t}_choice"] = r[t]["global_choice"]
            entry[f"{t}_ci95"] = r[t]["paired_nested_vs_v2"]["ci95"]; entry[f"{t}_fold_scores"] = r[t]["nested"]["fold_scores"]
    else:
        entry["kind"] = "other"
    rows.append(entry)
    summary[name] = entry
Path("outputs/summary_all.json").write_text(json.dumps(summary, indent=1, default=str))
lines = ["| name | kind | target | features | nondef v2 | nondef new | full v2 | full new(v2 gate) | joint | joint CI95 | wins |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for e in rows:
    if e["kind"] == "catboost_variant":
        f = lambda v: "" if v is None else f"{v:.5f}"
        lines.append(f"| {e['name']} | {e['model']} | {e['target']} | {e['features']} | {f(e['nondef_v2'])} | {f(e['nondef_utility'])} | {f(e['full_v2'])} | {f(e['full_utility_v2gate'])} | {f(e['joint_full'])} | {e['joint_ci95']} | {e['joint_well_wins']} |")
Path("outputs/summary_variants.md").write_text("\n".join(lines))
print("\n".join(lines))
for e in rows:
    if e["kind"] != "catboost_variant":
        print(json.dumps(e, default=str)[:1500])
