"""Generate the manuscript's Volve tables directly from verified experiment metrics."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
M = json.loads((ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(x: float) -> str:
    return f"{x:.5f}"


assert sha(ROOT / "metrics.json") == M["outputs"]["metrics.json"]["sha256"]
assert sha(ROOT / "predictions.npz") == M["outputs"]["predictions.npz"]["sha256"]
assert sha(ROOT / "reproduce_public_validation.py") == M["script_sha256"]
assert R["dataset"]["test_rows"] == 11275

variants = [
    ("rf_five_log", "RF, five logs"),
    ("tree_regression", "Tree regression"),
    ("distribution_mean", "Distribution, mean"),
    ("distribution_tolerance_bayes", "Distribution, tolerance action"),
    ("tree_regression_joint_sw_gate", "Tree regression + joint SW gate"),
]
lines = [
    "\\begin{table}[t]",
    "\\centering",
    "\\caption{Public Volve reproducibility check. RMSE is pooled over VSH, PHIF and SW. Auxiliary tolerance is $\\max(0,1-|a-y|/[c(|y|+10^{-7})])$, with $c=0.08$ for PHIF and $c=0.05$ for SW; it is not the contest's primary RMSE metric. Tolerance actions use 600 grid points between each target's bin representatives.}",
    "\\label{tab:public-metrics}",
    "\\small",
    "\\resizebox{\\textwidth}{!}{%",
    "\\begin{tabular}{l ccccc}",
    "\\toprule",
    "Variant & RMSE & PHIF RMSE & SW RMSE & PHIF tol. & SW tol. \\\\",
    "\\midrule",
]
for key, label in variants:
    s = R["variants"][key]
    lines.append(f"{label} & {f(s['rmse_all3'])} & {f(s['PHIF'])} & {f(s['SW'])} & {f(s['tol_PHIF'])} & {f(s['tol_SW'])} \\\\")
lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", "",
          "\\begin{table}[t]", "\\centering",
          "\\caption{Per-well RMSE for the public baseline, target-specific regression and the joint SW gate. A pooled improvement need not occur in every well.}",
          "\\label{tab:public-wells}", "\\small",
          "\\resizebox{\\textwidth}{!}{%",
          "\\begin{tabular}{l cccc}", "\\toprule",
          "Variant & Well 100 & Well 101 & Well 103 & Well 104 \\\\",
          "\\midrule"]
for key, label in [("rf_five_log", "RF, five logs"), ("tree_regression", "Tree regression"), ("tree_regression_joint_sw_gate", "Tree + joint SW gate")]:
    vals = [R["per_well"][w][key]["rmse_all3"] for w in ["100", "101", "103", "104"]]
    lines.append(label + " & " + " & ".join(f(v) for v in vals) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
(ROOT / "public_table.tex").write_text("\n".join(lines), encoding="utf-8")

v = R["variants"]
summary = {
    "source": "metrics.json and experiment_manifest.json from the SSH public reproduction",
    "rf_to_tree_relative_rmse_change": v["tree_regression"]["rmse_all3"] / v["rf_five_log"]["rmse_all3"] - 1,
    "rf_to_gated_relative_rmse_change": v["tree_regression_joint_sw_gate"]["rmse_all3"] / v["rf_five_log"]["rmse_all3"] - 1,
    "tree_to_gated_relative_rmse_change": v["tree_regression_joint_sw_gate"]["rmse_all3"] / v["tree_regression"]["rmse_all3"] - 1,
    "mean_to_tolerance_phif_gain": v["distribution_tolerance_bayes"]["tol_PHIF"] - v["tree_regression"]["tol_PHIF"],
    "tree_to_gated_sw_tolerance_gain": v["tree_regression_joint_sw_gate"]["tol_SW"] - v["tree_regression"]["tol_SW"],
    "distribution_mean_rmse_higher_than_regression": v["distribution_mean"]["rmse_all3"] > v["tree_regression"]["rmse_all3"],
    "distribution_tolerance_rmse_higher_than_regression": v["distribution_tolerance_bayes"]["rmse_all3"] > v["tree_regression"]["rmse_all3"],
    "latex_table_sha256": sha(ROOT / "public_table.tex"),
    "scope": "Public reproducibility check, not fresh blind test, no PERM, no universal transfer claim",
}
(ROOT / "public_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
