# 后验路由三种子投票提交（2026-09-27）

## 结果

| 项 | 值 |
|---|---|
| 完整 OOF（三种子等权投票） | **81.98422196426628** |
| 单种子 OOF | seed0 81.641955 / seed1 81.698839 / seed2 81.644394 |
| A 榜（用户上报） | 83.6709 |
| 历史交付对照 | OOF 81.713233 / A 榜 83.6915 |

OOF 比历史交付高 0.271，但 A 榜低 0.0206。集成是安全的（未崩），但 test 增益未兑现。

## 方法

- 基模：`paired_posterior_router_sequence`（68 维输入 + 24 维 detach 后验描述量的路由，路由损失只反传路由，专家各自 NLL）
- 种子：seed 20261006 + 偏移 0/1/2（每折另加 fold 索引）
- 融合：每个（种子，折）内部两骨干完整后验等权融合 → 期望效用；跨五折与三种子等权平均期望效用，再取 Bayes 最优动作
- 预算：80 轮，patience 18，lr 0.0006 cosine

## 目录

- `code.zip`：推理代码 + 三种子五折权重（15 个），解压后根为 `package_dl/`
- `result.zip`：仅根目录 `result.json`（10 口井，95,948 行）
- `result.json` / `result.inference.json`：未压缩结果与推理元信息

## 复现

```bash
python package_dl/predict_posterior_router.py \
  --data_dir /path/to/data \
  --model_root package_dl/models_posterior \
  --seeds seed0 seed1 seed2 --folds 1 2 3 4 5 \
  --output result.json --device cuda
```

## 完整 OOF 复算

```bash
python tools/deep_learning/vote_full_oof.py \
  outputs/full_seed0 outputs/full_seed1 outputs/full_seed2
```

注：需要 GPU 训练产物 `outputs/full_seed{0,1,2}/fold_{1..5}_distributions.npz`（未入库，见 `.gitignore`）。
