"""Assemble, verify end-to-end, and zip the v7 submission package on the server.

Package: v97 base (A-board 83.9329 anchor) + drift-augmented SW gate (B-prime).
The package's predict.py regenerates everything from raw test data; its own
output IS the submission result (H20-reproducible byte-exact).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/home/ubuntu/kunlun_welllog_v3_20260919")
PY = str(ROOT / ".venv/bin/python")
PACK = ROOT / "v7_pack"

README = """# 基于测井曲线的储层参数预测 · 提交代码包（v7：漂移增强门控移植）

## 1. 算法名称

漂移增强门控移植（十八级联骨干网基座 / Drift-Augmented Gate Transplant on an Eighteen-Stage Cascaded Backbone）

## 2. 方法简介

对每口井的 13 条输入曲线（不使用 CASE）逐深度点预测 POR、PERM、SW。

**基座**：十八级联骨干网（18 阶段级联链 · 双 U-Net 骨干 · 5 折集成，A 榜 83.9329，OOF 81.8910），
本包 `runtime/backbone/` 原样承载，负责 POR / PERM 全部预测与 SW 的连续值。

**漂移增强 SW 门控（本包新增，`sw_head/`）**：CatBoost 二分类（Logloss，950 轮 × 双种子平均），
在 80 口训练井 + 每井 2 份漂移增强副本（标签保持的曲线平移/尺度扰动，
KS 校准至测试井实测漂移区间）上训练；门控判定替换基座的 SW 默认/非默认判定。

**组合规则（逐行）**：
- 漂移增强门控判默认 → SW = 99.9
- 门控判连续且基座也判连续 → SW = 基座的连续值
- 门控判连续而基座判默认 → SW = 门控头回归值（×1.05 − 3.0，裁剪到 [8.305, 99.9]）
- POR / PERM / 深度：基座原封不动

**证据（按井 5 折 OOF，官方核，同一套伪漂移评估）**：基座的 SW 头漂移敏感度 −1.61
（75.54 → 73.93）；漂移增强门控 + 基座连续值（本组合）漂移下 75.29，漂移敏感度 −0.03，
漂移条件下 SW 净赢基座 +1.36。改动集中在 5,882 / 95,948 行（6.1%）。

## 3. 运行环境

- Python 3.10（3.9~3.12 均可）；依赖见 `requirements.txt`
  （numpy、pandas、torch ≥ 2.4、catboost ≥ 1.2.5、psutil）
- GPU 或 CPU（`--device auto|cuda|cpu`）。H20 实测全程约 80 秒
  （基座推理 ~30 s + 特征/门控 ~50 s CPU）。
- 基座跨 GPU 存在 bf16 自动混合精度浮点噪声（H20 与原始 Windows GPU 相比
  287,844 值中约 1,071 值微差、默认翻转 19 处）；同 GPU 重复运行字节级一致。
  跨 GPU 复现可用 `--skip_pinned_check` 跳过中间结果 SHA 校验。

## 4. 运行命令

```bash
pip install -r requirements.txt
python predict.py --data_dir ./data --output ./result.json
```

## 5. 文件结构

```text
submission_code.zip
├── README.md
├── predict.py               # 主入口：基座推理 → 特征 → 漂移增强 SW 门控 → 组合
├── requirements.txt
├── SOURCE_PROVENANCE.json
├── feature/                 # 树-序列混合特征栈（FeatureBuilder，237 维增强特征）
│   ├── meta.json            # 中位数 + 特征表（推理不读训练数据）
│   └── src/
├── sw_head/                 # 漂移增强 SW 门控 + 回归（CatBoost × 双种子）
│   ├── SW_gate_seed{42,2026}.cbm
│   ├── SW_reg_seed{42,2026}.cbm
│   └── meta.json            # 解码常数 + 训练来源
└── runtime/backbone/        # 十八级联骨干网完整运行时（权重 5 折）
```

## 6. 交叉验证证据（80 口训练井，官方核）

| SW 头 | 干净 OOF | 伪漂移 OOF | 漂移敏感度 |
|---|---|---|---|
| 十八级联骨干网（原 SW 头） | 75.5364 | 73.9284 | −1.61 |
| 漂移增强门控 + 骨干连续值（本包） | 75.3101 | **75.2853** | **−0.03** |

A 榜：十八级联骨干网单独 83.9329；本组合待测（预期 +0.2 ~ +0.5）。
"""


def main() -> None:
    if PACK.exists():
        shutil.rmtree(PACK)
    (PACK / "runtime").mkdir(parents=True)
    (PACK / "feature").mkdir()
    (PACK / "sw_head").mkdir()

    # runtime/v97 (from the extracted verify tree)
    shutil.copytree(ROOT / "v97_verify/runtime/v97", PACK / "runtime/backbone")
    for d in (PACK / "runtime/backbone").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)

    # feature stack: v3 package src + slim meta
    shutil.copytree(ROOT / "hybrid_build/runtime/v3/src", PACK / "feature/src")
    for d in (PACK / "feature/src").rglob("__pycache__"):
        shutil.rmtree(d, ignore_errors=True)
    meta = json.load(open(ROOT / "cache/meta.json"))
    slim = {"medians": meta["medians"],
            "base_feature_names": meta["base_feature_names"],
            "enhanced_feature_names": meta["enhanced_feature_names"]}
    (PACK / "feature/meta.json").write_text(json.dumps(slim, indent=1))

    # sw head models + meta
    fin = ROOT / "outputs/drift_sw/final"
    for f in ("SW_gate_seed42.cbm", "SW_gate_seed2026.cbm", "SW_reg_seed42.cbm", "SW_reg_seed2026.cbm"):
        shutil.copy(fin / f, PACK / "sw_head" / f)
    shutil.copy(fin / "meta.json", PACK / "sw_head/meta.json")

    (PACK / "requirements.txt").write_text(
        "numpy>=1.26\npandas>=2.0\ntorch>=2.4\ncatboost>=1.2.5\npsutil>=5.9\n")
    (PACK / "README.md").write_text(README)
    prov = {
        "base_result_sha256_h20": "798f3e0c131f8c1a5043a034f701a388087a1087fae0a4287adb5b84050a6a90",
        "base_result_sha256_pinned_windows": "06e2a5baff6243111718401e9d3043e70e519d80b3ccc60f3377975e4ddb5760",
        "base_submission_a_board": 83.9329,
        "sw_head": "outputs/drift_sw/final (trained on all 80 wells + K=2 drift-augmented copies, seeds 42/2026)",
        "note": "package predict.py output is the submission; H20 byte-reproducible",
    }
    (PACK / "SOURCE_PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    print("[pack] tree assembled", flush=True)

    # predict.py comes from the local file uploaded next to this script
    shutil.copy(ROOT / "v7_predict.py", PACK / "predict.py")

    # end-to-end run
    r = subprocess.run(
        [PY, str(PACK / "predict.py"), "--data_dir", str(ROOT / "data"),
         "--output", str(PACK / "result.json"), "--device", "cuda"],
        cwd=PACK, capture_output=True, text=True)
    print(f"[e2e] exit={r.returncode}")
    if r.returncode != 0:
        print((r.stdout + r.stderr)[-800:])
        return
    print((r.stdout + r.stderr).strip().splitlines()[-1][:400], flush=True)

    # zip
    files = sorted(p for p in PACK.rglob("*")
                   if p.is_file() and "__pycache__" not in str(p)
                   and p.name not in ("submission_code.zip", "result.zip", "result.inference.json"))
    with zipfile.ZipFile(PACK / "submission_code.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(PACK).as_posix())
    zf = zipfile.ZipFile(PACK / "submission_code.zip")
    print(f"[zip] entries={len(zf.namelist())} integrity={'OK' if zf.testzip() is None else 'BAD'}", flush=True)


if __name__ == "__main__":
    main()
