# experiments/v7 — 漂移增强门控移植实验（2026-10-09）

配套文档：[docs/drift_augmented_gate_transplant_20261009.md](../../docs/drift_augmented_gate_transplant_20261009.md) · 数据：[results/drift_sw/](../../results/drift_sw/) · 提交：[submissions/v7/](../../submissions/v7/)

动机：悖论诊断（[docs/paradox_dl_fusion_20261008.md](../../docs/paradox_dl_fusion_20261008.md)）定位十八级联骨干网（v97）的 SW 门控是漂移失败主犯。本实验不拼装现有预测，而是**从头训练一个漂移增强的 SW 门控**（CatBoost + 标签保持曲线扰动增强），以 B′（换门不换值）方式移植到骨干基座上。

## 运行顺序（服务器 /home/ubuntu/kunlun_welllog_v3_20260919/）

1. `drift_sw_train.py --arm both --copies 2` —— 两臂消融（增强/无增强），干净 + 伪漂移 OOF，产出 `outputs/drift_sw/report.json`
2. `v97_sens.py` —— v97 SW 头折内污染版漂移敏感度（80.21→76.40）
3. `v97_sens_driver.py` —— v97 SW 头诚实 OOF（单折 bundle/井）漂移敏感度（75.54→73.93，Δ−1.61）
4. `drift_sw_bins.py --arm both --copies 2` —— 60 档分位（bins）变种双臂，收敛验证
5. `final_head.py` —— 全 80 井 + 增强副本上训最终头（4 个 .cbm），产出全换候选
6. `decompose.py` —— A/B/B′ 分解：**门控贡献全在门控本身，连续值质量骨干更优** → 选择 B′
7. `build_bprime.py` —— 构建 B′ 候选（骨干连续值 + 漂移增强门控）
8. `assemble_v7.py` + `v7_predict.py` —— 组装提交包（runtime/backbone + feature + sw_head），端到端验证 + zip

## 环境

Python 3.10 venv（catboost 1.2.10、torch 2.14、numpy、pandas、psutil），H20 GPU（只用于 v97 推理与训练推理均可总线）。

## 关键结果（A 榜 83.8506，低于基座 −0.082）

OOF/伪漂移证据全部正（漂移敏感度 −0.69→−0.03，B′ 漂移下 +1.36 SW），但 A 榜未传导——第 7 次观测到“OOF 侧增益在 5 口隐藏井上蒸发”。完整终审见 docs/drift_augmented_gate_transplant_20261009.md §元规律。
