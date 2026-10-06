"""Complete v17 documentation from actual saved reports; no model mutation."""
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
from launch_latent_gate_v17 import ROOT, STUDY, active_jobs, append_once


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    subprocess.run([sys.executable, "-X", "utf8", str(Path(__file__).with_name("report_latent_gate_v17.py")), "--record"], check=True, stdout=subprocess.DEVNULL)
    progress = read_json(STUDY / "progress.json")
    assembled = read_json(STUDY / "assembled_report.json")
    assert progress["all_complete"] and assembled["all_new_SW_and_fixed_POR_PERM_reloaded"]
    rows = {r["mode"]: r for r in progress["rows"]}
    candidates = {r["mode"]: r for r in assembled["candidates"]}
    a, b = rows["shared_global"]["final"], rows["local_gate"]["final"]
    diff = assembled["comparison"]
    launches = [read_json(ROOT / "outputs" / r["id"] / "launch.json") for r in progress["rows"]]
    start = min(datetime.fromisoformat(r["started_at"]) for r in launches)
    end = max(datetime.fromisoformat(r["final"]["finished_at"]) for r in progress["rows"])
    timing = dict(explicit_parallel_jobs=2, launch_to_both_complete_seconds=(end-start).total_seconds(),
        per_run_seconds={m: r["final"]["seconds"] for m, r in rows.items()},
        per_run_peak_allocated_MiB={m: r["final"]["peak_allocated_MiB"] for m, r in rows.items()},
        final_replay_setup_seconds=assembled["setup_and_replay_seconds"],
        scope="Training wall includes both explicit process starts/import/setup/train/eval/restore, excludes implementation and later final disk replay. No speedup claim.")
    (STUDY / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    text = f"""# v17 局部门控 / 全局密度：最终结果

观察时间：{datetime.now().astimezone().isoformat()}。

## 已完成与保留结果
两个预先定义的SW网络都从随机骨干训练完成，均174849可训练参数，初始state_dict完全一致。沿用原64/16井、原九曲线派生206特征、Q=1、80/18预算、相同损失与单覆盖解码。

|方案|SW|三通道TOTAL|最佳轮次|停止轮次|
|---|---:|---:|---:|---:|
|保留v5/v8|{assembled['reference_scores']['SW']:.9f}|{assembled['reference_scores']['TOTAL']:.9f}|历史固定|已验证|
|共享全局门控对照|{a['target_score']:.9f}|{candidates['shared_global']['scores']['TOTAL']:.9f}|{a['best_epoch']}|{a['stopped_epoch']}|
|局部门控/全局密度|{b['target_score']:.9f}|{candidates['local_gate']['scores']['TOTAL']:.9f}|{b['best_epoch']}|{b['stopped_epoch']}|

本批无新增收益，当前配方继续保留原v5/v8，未替换交付。局部组相对匹配对照SW差{diff['local_minus_shared_SW']:+.9f}、TOTAL差{diff['local_minus_shared_TOTAL']:+.9f}。

## 结构与因果边界
hL=LocalConv2(LocalConv1(stem(X)))；hG=LocalConv2(Memory(LocalConv1(stem(X))))，特征形状B×256×64。两个卷积块及最终LayerNorm在SW内部共享；不同通道之间没有共享参数。
两个模式的连续密度都使用hG。shared_global门控使用hG；local_gate门控使用hL。没有加参数或增加模型来源。两模式都算两条前向路径，但反向梯度依赖不同，不能声称实际FLOPs完全一样。
门控读取上下文与门控梯度能否进入记忆同时改变，因此结果不能唯一归因于“全局特征是否有用”或“梯度是否冲突”。

## 实测错误分解
|真实SW子集|共享全局门控|局部门控|局部减全局|
|---|---:|---:|---:|
|默认段，100786点|{a['validation']['default_SW']:.9f}|{b['validation']['default_SW']:.9f}|{diff['default_SW_delta']:+.9f}|
|非默认段，43879点|{a['validation']['numeric_SW']:.9f}|{b['validation']['numeric_SW']:.9f}|{diff['nondefault_SW_delta']:+.9f}|

两类均下降：本配置不支持为门控完全绕过全局记忆。16井中{diff['well_wins']}井提高、{diff['well_losses']}井下降。整井重采样的选择未校正TOTAL区间为{diff['exploratory_selection_unadjusted_TOTAL_95']}；原第五折参与早停和多轮开发，不作独立泛化结论。
全局对照与v16同架构历史结果略有差异；当前采用独立同实现对照，不把该差异记为机制收益。使用warn_only确定性模式，不承诺跨运行逐位训练复现。

## 实际验证
两份新SW与固定POR/PERM都从磁盘重新构建。144665点预测逐元素复现，原行序一致，POR/PERM保持不变，三目标参数存储无共享。总分来自实际合并数组，不只由单通道分数公式估算。
局部组在训练输入窗口上绕过记忆后，门控logit最大变化为0；连续分布仍变化，证实路径分工生效。填充值替换和全空窗口检查通过。这是推理依赖诊断，不是额外候选或重训效果。
首次诊断探针因float16填充值溢出失败，修改仅限探针转float32；两模型权重/训练代码未改。错误及修复留存在verification_attempts.json，随后整体重跑成功。

## 不继续的方向与下一步
不继续对本次硬局部门控做宽度/层数扫描，不从单次负结果推出全局信息或局部建模普遍无效。未启动第三模型或其他原始折。
下一候选（尚未实现）：允许门控根据输入学习局部/全局特征的组合，并以不含可学习组合的同容量对照检验；或者把已显示数值段收益的潜变量主干用于POR的私有模型，避免只围绕SW继续同质微调。应先固定一个假设再启动，不自动扫参。
第五折严格>80.5与完整原OOF>82均未达成，旧交付不变。

## 时间与证据
本批两个任务并发，显式启动至均完成墙钟{timing['launch_to_both_complete_seconds']:.3f}秒；单任务报告秒数{timing['per_run_seconds']}，不能相加当作并发墙钟。另有完整磁盘重载/数据准备{timing['final_replay_setup_seconds']:.3f}秒。详见timing.json。
源：各实验fold_5_report.json、config.json、gradient_scope.json，以及本目录progress.json、assembled_report.json、CURRENT_RECIPE.json。
"""
    (STUDY / "FINAL.md").write_text(text, encoding="utf-8")
    marker = "## v17最终完成"
    entry = marker + f"\n{datetime.now().astimezone().isoformat()}。两组训练/原144665点磁盘重载与三模型实际合并已完成。shared_globalTOTAL={candidates['shared_global']['scores']['TOTAL']:.9f}，local_gateTOTAL={candidates['local_gate']['scores']['TOTAL']:.9f}；无新增收益，保留80.274135。未完成：新假设/其他通道结构/完整OOF>82。无第三候选、补折或交付变更。证据results/deep_learning/latent_gate_v17_20260928/FINAL.md。"
    for p in (ROOT.parent / "PROJECTS.md", ROOT / "PROJECTS.md", STUDY / "PROJECTS.md", ROOT / "docs/deep_learning/atlas/PROJECTS.md"):
        append_once(p, marker, entry)
    for row in progress["rows"]:
        out = ROOT / "outputs" / row["id"]
        end_note = marker + f"\n最佳SW={row['final']['target_score']:.9f}@{row['final']['best_epoch']}，自然停止{row['final']['stopped_epoch']}；最终三个私有模型磁盘重载与合并评分完成。没有超过旧SW，不替换交付。范围与未做项见研究FINAL.md。"
        append_once(out / "README.md", marker, end_note)
        append_once(out / "PROJECTS.md", marker, end_note)
    attempts = read_json(STUDY / "verification_attempts.json")
    attempts["successful_rerun"] = dict(completed_at=assembled["finished_at"], all_replays_passed=True, trained_weights_unchanged=True)
    attempts["attempts"][-1]["rerun_required"] = False
    (STUDY / "verification_attempts.json").write_text(json.dumps(attempts, ensure_ascii=False, indent=2), encoding="utf-8")
    # Finish the missing v16 narrative, without rewriting its experiment results.
    v16 = ROOT / "results/deep_learning/latent_fold5_v16_20260928"
    old = read_json(v16 / "assembled_report.json")
    if not (v16 / "FINAL.md").exists():
        lines = ["# v16 原第五折：补齐归档", "", "依据已落盘assembled_report.json补齐断线时未写入的总结；本次没有重训或改变v16结果。", ""]
        for c in old["candidates"]:
            lines.append(f"{c['mode']}: SW={c['scores']['SW']:.9f}, TOTAL={c['scores']['TOTAL']:.9f};三模型实际磁盘重载已完成。")
        lines.append("保留原v5/v8 TOTAL80.274135337791，无其他折或交付替换。progress.json早于组合验证，应以assembled_report.json为最终组合证据。")
        (v16 / "FINAL.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    graph_path = ROOT / "docs/deep_learning/atlas/curation.json"
    graph = read_json(graph_path)
    edge = dict(source="pure_dl_latent_gate_v17_shared_global_fold5", target="pure_dl_latent_gate_v17_local_gate_fold5", type="ablation", note="同参数同起点，改变门控特征路径")
    if not any(e.get("source")==edge["source"] and e.get("target")==edge["target"] for e in graph.setdefault("edges", [])):
        graph["edges"].append(edge)
    dead_id = "dead:latent_gate_v17_hard_bypass"
    dead = dict(id=dead_id, title="v17硬局部门控：本设置未获增益", stage="architecture",
        parents=[r["id"] for r in progress["rows"]], summary="同174849参数、相同初始化，局部门控比全局对照SW低0.325453；两类标签段均下降。",
        reason="关闭本设置下完全绕过记忆的门控路径；不否定局部/全局自适应组合或其他模型。")
    graph["deadends"] = [d for d in graph.get("deadends", []) if d.get("id") != dead_id] + [dead]
    graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL_REPORT", STUDY / "FINAL.md")
    print("SELECTED", assembled["selected_scores"])
    print("CURRENT_TRAINERS", active_jobs())
    print("CURATE_TAIL", repr((ROOT / "docs/deep_learning/atlas/curate.py").read_text(encoding="utf-8-sig")[-1100:]))


if __name__ == "__main__":
    main()
