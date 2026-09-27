# 后验路由集成增益计划(第二轮)

> 基线:后验路由三种子投票,OOF 81.9842,A 榜 83.6709(历史最佳 83.6915,差 0.02)
> 目标:把 A 榜推过 83.69,确认 test 增益真实存在
> 编写时间:2026-09-27,仅规划,未执行

## 现状事实

| 模型 | 第五折 | 完整 OOF | A 榜 |
|---|---|---|---|
| 校准双骨干(75 轮,v3 交付) | — | 81.7132 | **83.6915** |
| 后验路由单种子(seed0/1/2) | 80.14/80.02/80.30 | 81.64/81.70/81.64 | — |
| 后验路由三种子等权投票 | **80.45** | **81.98** | 83.6709 |

观察:
- 三种子投票在 OOF 上比单种子高 0.29~0.34,比 81.71 高 0.27
- 但 A 榜几乎没动(-0.02),说明 OOF 增益在 test 上被噪声稀释,或者 fold5 筛出来的 80.4 阈值本身对 test 不敏感
- 要继续涨 test,需要真分歧模型(异构),而不是同种子堆叠

## 计划(按优先级)

### Phase 1: 异构集成(最高优先)

把 local-router(第五折 80.2219)和 disagreement-router(第五折 80.2133)也跑 5 折,和 posterior 三种子混投。

- 模型文件:已有 `paired_local_context_router_sequence.py`、`paired_disagreement_router_sequence.py`
- 训练脚本:已有 `train_paired_local_context_router80_sequence.py`、`train_paired_disagreement_router80_sequence.py`
- 需要:各跑 5 折(每个模型 5 个训练,2 个模型 = 10 个训练,2 并发 ≈ 2 小时)
- 融合方式:全部 15 个模型(posterior 3 种子 + local 5 折 + disagreement 5 折 = 3+5+5=13 个,实际 13 个 checkpoint)在期望效用空间等权平均
- 预期:OOF 可能到 82.1~82.3,test 增益未知但方向正确

### Phase 2: 加权投票(Phase 1 完成后)

如果异构集成有效但增益有限,尝试按单折 OOF 分数给每种每折一个 softmax 权重:

- 权重 = softmax(分数 / 温度),温度调 0.1~1.0
- 只调 13 个权重,自由度低,不易过拟合
- 不上 stacking(第二层学习器),风险太大

### Phase 3: 种子扩展(可选)

如果 Phase 1+2 后仍有时间且 GPU 空闲:
- posterior-router 加 seed3/seed4(再补 8 折)
- 或 local-router 加种子(如果 Phase 1 显示 local 有效)

## 执行顺序

1. 先跑 local-router 5 折(2 并发,~1 小时)
2. 同时跑 disagreement-router 5 折(2 并发,~1 小时)
3. 全部齐后跑 13 模型投票 OOF
4. 如果 OOF > 82.0,打包推理出提交
5. 如果 OOF < 82.0,说明异构无效,回退到 posterior 三种子 + 加权

## 文件改动

- 新增:`package_dl/src/neural/paired_local_context_router_sequence.py`(相对导入版)
- 新增:`package_dl/src/neural/paired_disagreement_router_sequence.py`(相对导入版)
- 新增:`package_dl/predict_ensemble.py`(支持 13 模型投票)
- 修改:`tools/deep_learning/train.py` 注册 local-router、disagreement-router 的 fold-count 5 支持

## 验证标准

| 阶段 | 通过条件 |
|---|---|
| Phase 1 OOF | 13 模型投票 OOF > 82.0 |
| Phase 1 提交 | A 榜 > 83.6915 |
| Phase 2 加权 | OOF 加权后 > 等权,且 test 不崩 |

## 风险

- local-router 和 disagreement-router 的单折分数低于 posterior,五折后可能拉低整体
- 13 模型推理时间变长(约 5 分钟 vs 95 秒),但可接受
- 如果异构集成 OOF 反而低于 posterior 三种子,说明 local/disagreement 是负贡献,立即停止
