"""Explicit launcher and documentation for two bounded v17 trials; no queue."""
from pathlib import Path
from datetime import datetime
import argparse
import ast
import json
import subprocess
import sys
import psutil

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/deep_learning/latent_gate_v17_20260928"
MODES = ("shared_global", "local_gate")

RULES = """# v17 潜变量网络门控路径：目标与边界

目标：在原第五折比较共享全局读出与局部门控/全局密度读出，实际计算与v5总分80.274135的差异。
两组174849参数，完整初始state_dict一致；固定原64/16划分、原九曲线206特征+相对深度、v16仅由64训练井拟合的缓存、Q=1、原256格/800动作/单覆盖评分。
两个模型都从头训练，不使用v16或内层checkpoint初始化。两个局部卷积在SW内部共享，两候选的前向都计算局部和全局路径；不同的是门控输入及相应梯度路径。POR/PERM各自保留私有模型。
max80/patience18/min10、batch64、每轮64步、AdamW6e-4、cosine至1e-5、clip2；normal CUDA最多2任务，显式启动/恢复，无自动队列，不补其他四折或覆盖交付。
维护PROJECTS.md：目标、已做、计划、未做和完成后选项；每批结束更新。每个实验README写清探索重点、验证证据、收益或未证实、停止探索的边界。记录失败，不覆盖历史源码/权重。
"""
METHOD = """# v17：局部门控与全局密度的匹配结构对照

## 探索重点
v16潜变量SW73.476515，三通道80.223081，未超过v5的80.274135。全局记忆相对TCN改善非默认段，却降低默认段。本轮检验门控是否应绕过全局记忆。

## 结构
输入X为B×256×207。h0=LocalConv1(GELU(stem(X)))；hL=LocalConv2(h0)；hG=LocalConv2(Memory(h0))。LocalConv2在两条路径内共享参数。
shared_global：门控和256格密度均读取hG。
local_gate：门控读取hL，密度读取hG。均使用原norm和同一257行输出权重的对应行，无新增参数；两组均174849参数、初始状态完全相同。
两组都计算两条前向路径，但反向依赖不同；不声称计算量或梯度传播完全相同。局部路径相对输入特征的感受野13点，但206维输入已有统计上下文，不称原始曲线仅13点。

## 训练及验证
复用v16缓存及其训练侧元数据，原第五折只作早停/选模；每轮原SW分及固定POR/PERM替换总分仅为诊断。最终加载新SW权重、验证行序后实际合并评分；固定POR/PERM使用其已有预测与磁盘重载证据，并明确本轮是否再次重载。
训练损失、初始化先验、窗口随机数、学习率、80/18预算完全固定。两组从随机骨干开始；不把新实验称旧训练续跑。完整当前模型/优化器/调度器/RNG原子保存。

## 当前收益与边界
尚未证实。仅这两个预定义候选；不追加第三臂、网格或原其他折。局部/全局路由是本轮唯一结构差异，但第五折已反复开发，差值不是独立泛化结论。
若局部路由无收益，不自动增宽或宣称全局信息有害。若有收益，先检查默认与非默认人口、逐井表现，再决定候选保留；严格80.5/完整OOF82目标不变。

## 已知工具问题
首次只读Python -c命令因shell引号解析失败，未触及数据或训练。后续使用独立脚本；模型训练失败另外记录failure.json。
"""


def active_jobs():
    jobs = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            item = proc.info
            if "python" not in (item.get("name") or "").lower():
                continue
            for arg in item.get("cmdline") or []:
                value = arg.replace("\\", "/")
                if "pure_dl/" in value and Path(value).name.startswith(("train_", "benchmark_")):
                    jobs.append(item)
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return jobs


def append_once(path, marker, text):
    current = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if marker not in current:
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n\n" + text + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", choices=MODES, nargs="+", required=True)
    args = ap.parse_args()
    if len(set(args.launch)) != len(args.launch):
        raise ValueError("Duplicate mode")
    jobs = active_jobs()
    if len(jobs) + len(args.launch) > 2:
        raise RuntimeError(f"Insufficient slots; existing jobs={jobs}")
    for name in ("latent_gate_routing_v17.py", "train_latent_gate_v17.py"):
        ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"))
    for mode in args.launch:
        out = ROOT / "outputs" / f"pure_dl_latent_gate_v17_{mode}_fold5"
        if (out / "launch.json").exists() or (out / "config.json").exists():
            raise RuntimeError(f"Refusing duplicate launch: {out}")
    STUDY.mkdir(parents=True, exist_ok=True)
    for name, text in [("AGENTS.md", RULES), ("METHOD.md", METHOD)]:
        if not (STUDY / name).exists():
            (STUDY / name).write_text(text, encoding="utf-8")
    if not (STUDY / "PROJECTS.md").exists():
        (STUDY / "PROJECTS.md").write_text("# v17进度\n\n目标：结构对照及原第五折总分。\n\n已做：v16实证与状态核对、两个同参数模型/训练器实现。\n\n计划：显式启动两组、自然早停、重载并合并评分。\n\n未做：真实训练成绩、最终收益与跨折验证。\n\n完成后：保留有收益的固定配方；无收益则保留v5，不自动补折或覆盖交付。\n", encoding="utf-8")
    marker = "## v17本轮局部门控结构对照"
    instruction = marker + "\n继续优化授权下预定义shared_global/local_gate两项，174849参数一致；仅门控读取局部或全局特征不同，密度都使用潜变量记忆。复用v16原64训练井拟合的缓存，从头训练，最终实际三通道合并评价。旧任务/交付/其他折不改；目标与未完成项记录latent_gate_v17_20260928/PROJECTS.md。"
    append_once(ROOT.parent / "AGENTS.md", marker, instruction)
    run_records = []
    for mode in args.launch:
        out = ROOT / "outputs" / f"pure_dl_latent_gate_v17_{mode}_fold5"
        out.mkdir(exist_ok=True)
        (out / "AGENTS.md").write_text(RULES, encoding="utf-8")
        (out / "README.md").write_text(METHOD + "\n本目录模式：" + mode + "。初步结果以runtime_status和日志为准。\n", encoding="utf-8")
        (out / "PROJECTS.md").write_text("# 实验进度\n\n目标：" + mode + "结构的原第五折验证。\n已做：代码、协议和来源核对。\n计划：训练并恢复最佳。\n未做：训练收益与最终重载。\n完成后：与另一臂/v5比较，不自动补折。\n", encoding="utf-8")
        command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).with_name("train_latent_gate_v17.py")), "--mode", mode, "--out", str(out)]
        with (out / "train.log").open("wb") as log, (out / "train.err").open("wb") as err:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=err,
                close_fds=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
        item = dict(mode=mode, pid=process.pid, command=command, started_at=datetime.now().astimezone().isoformat())
        (out / "launch.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        run_records.append(item)
        print("LAUNCHED", json.dumps(item, ensure_ascii=False))
        append_once(ROOT / "EXPLORATIONS.md", out.name,
            f"- [v17 {mode}：局部/全局门控路径](outputs/{out.name}/README.md)：同参数结构对照；启动时未证实收益。")
    entry = "## " + datetime.now().astimezone().isoformat() + " v17显式启动\n" + json.dumps(run_records, ensure_ascii=False) + "\n仅启动两个预定义SW候选，原其他折不变；成绩与最终组合尚待运行结果。"
    for path in (ROOT.parent / "PROJECTS.md", ROOT / "PROJECTS.md", STUDY / "PROJECTS.md"):
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n\n" + entry + "\n")


if __name__ == "__main__":
    main()
