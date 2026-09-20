"""Build the transductive final config from final_cfg_v3.json + an assembly decision, if the assembly beats D."""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembly", default="outputs/assembly_H")
    ap.add_argument("--reference", default="outputs/assembly_D")
    ap.add_argument("--min-gain", type=float, default=0.05)
    ap.add_argument("--out-cfg", default="final_cfg_v3_pl.json")
    ap.add_argument("--final-dir", default="outputs/final_v3_pl")
    args = ap.parse_args()
    H = json.load(open(Path(args.assembly) / "report.json")); D = json.load(open(Path(args.reference) / "report.json"))
    gain = H["TOTAL_nested"] - D["TOTAL_nested"]
    print(f"nested total: candidate {H['TOTAL_nested']:.4f} vs reference {D['TOTAL_nested']:.4f} -> gain {gain:+.4f}")
    if gain < args.min_gain:
        print("GAIN_BELOW_THRESHOLD"); return
    cfg = json.load(open("final_cfg_v3.json"))
    dec = json.load(open(Path(args.assembly) / "decision.json"))
    cfg["decision"] = dec
    pl_nets = []
    for n in cfg["nns"]:
        m = json.loads(json.dumps(n)); m["name"] = n["name"] + "_pl"; m["pseudo_test"] = "outputs/final_v3/result.json"
        m["train_args"]["pseudo_weight"] = 0.5
        pl_nets.append(m)
    cfg["nns"] = pl_nets
    used_gates = {s for t in dec for s in dec[t]["gate_sources"]}
    needs_stack = any(w > 0 for t in dec for s, w in zip(dec[t]["gate_sources"], dec[t]["gate_w"]) if s == "stack_states")
    cfg["stack"] = {"iterations": 800, "nn_order": ["nn_gate_b", "nn_base_c"],
                    "tree_dist": {"POR": "por_bins80_base_extra", "PERM": "perm_bins60_base", "SW": "sw_bins60_enh"}} if needs_stack else None
    json.dump(cfg, open(args.out_cfg, "w"), indent=1)
    fd = Path(args.final_dir); fd.mkdir(parents=True, exist_ok=True)
    if not (fd / "model_meta.json").exists():
        shutil.copytree("outputs/final_v3/models", fd / "models", dirs_exist_ok=True)
        shutil.copy("outputs/final_v3/model_meta.json", fd / "model_meta.json")
    print("PROCEED", "needs_stack" if needs_stack else "no_stack")


if __name__ == "__main__":
    main()
