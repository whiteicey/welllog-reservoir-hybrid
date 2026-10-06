"""Bounded status/result report; never starts, stops, or changes training."""
from pathlib import Path
from datetime import datetime
import argparse
import json
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/deep_learning/latent_gate_v17_20260928"
MODES = ("shared_global", "local_gate")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def trim(metrics):
    return {k: v for k, v in metrics.items() if k != "per_well"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--wait-seconds", type=int, default=0)
    args = ap.parse_args()
    if args.wait_seconds:
        jobs = []
        for mode in MODES:
            status = load(ROOT / "outputs" / f"pure_dl_latent_gate_v17_{mode}_fold5/runtime_status.json")
            if status and status.get("status") == "running":
                try:
                    proc = psutil.Process(status["pid"])
                    if any("train_latent_gate_v17.py" in a for a in proc.cmdline()):
                        jobs.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        psutil.wait_procs(jobs, timeout=min(max(args.wait_seconds, 0), 40))
    rows = []
    for mode in MODES:
        out = ROOT / "outputs" / f"pure_dl_latent_gate_v17_{mode}_fold5"
        status = load(out / "runtime_status.json") or {"status": "not_started"}
        cfg = load(out / "config.json")
        final = load(out / "fold_5_report.json")
        hist = load(out / "validation_history.json") or []
        brief = {k: v for k, v in final.items() if k not in ("fit", "validation")} if final else None
        if brief:
            brief["validation"] = trim(final["validation"])
            brief["fit"] = trim(final["fit"])
        latest = {k: v for k, v in hist[-1].items() if k not in ("fit", "validation")} if hist else None
        if latest:
            latest["validation"] = trim(hist[-1]["validation"])
        failure = load(out / "failure.json")
        row = dict(mode=mode, id=out.name, status=status, pid_alive=psutil.pid_exists(status.get("pid", -1)),
            initial_hash=cfg.get("initial_state_sha256") if cfg else None,
            trainable_parameters=cfg.get("trainable_parameters") if cfg else None,
            gradient_scope=load(out / "gradient_scope.json"), final=brief, latest=latest, failure=failure)
        rows.append(row)
    result = dict(observed_at=datetime.now().astimezone().isoformat(), rows=rows,
        all_complete=all(r["final"] is not None for r in rows), assembly_complete=(STUDY / "assembled_report.json").exists(),
        complete_five_folds=False, remaining_folds_started=False)
    if all(r["initial_hash"] for r in rows):
        result["identical_initial_state"] = rows[0]["initial_hash"] == rows[1]["initial_hash"]
        result["identical_parameter_count"] = rows[0]["trainable_parameters"] == rows[1]["trainable_parameters"]
        assert result["identical_initial_state"] and result["identical_parameter_count"]
    if result["all_complete"]:
        a, b = rows
        result["local_minus_shared_SW"] = b["final"]["target_score"]-a["final"]["target_score"]
        result["local_minus_shared_TOTAL"] = 0.35*result["local_minus_shared_SW"]
    (STUDY / "progress.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.record:
        graph_path = ROOT / "docs/deep_learning/atlas/curation.json"
        graph = load(graph_path)
        if graph is not None:
            for row in rows:
                text = "运行中，未完成最终验证" if not row["final"] else f"完成：SW={row['final']['target_score']:.9f}，最佳{row['final']['best_epoch']}轮；第五折开发结果，实际三通道重载见研究目录。"
                graph.setdefault("nodes", {})[row["id"]] = dict(title="v17 " + ("局部门控/全局密度" if row["mode"] == "local_gate" else "全局读出对照"),
                    stage="architecture", summary="174849参数、同初始状态，仅门控的局部/全局特征路由不同。", conclusion=text,
                    parents=["pure_dl_latent_v16_latent_fold5"], evidence=["outputs/"+row["id"]+"/README.md"],
                    status="recorded" if row["final"] else row["status"]["status"],
                    primary=dict(value=row["final"]["target_score"], scope="原第五折 · SW单通道", kind="diagnostic", source="outputs/"+row["id"]+"/fold_5_report.json") if row["final"] else None)
                edge = dict(source="pure_dl_latent_v16_latent_fold5", target=row["id"], type="evolution", note="参数匹配的门控上下文路由试验")
                if not any(x.get("source")==edge["source"] and x.get("target")==edge["target"] for x in graph.setdefault("edges", [])):
                    graph["edges"].append(edge)
            graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
