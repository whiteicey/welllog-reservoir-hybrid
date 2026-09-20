# 测井预测 v3 候选实现

独立优化代码；v2 模型和原提交不变。**尚无 v3 五折或榜分增益结论。**

实现内容：POR 官方效用决策（`utility`）、默认质量点联合决策（`joint`）、兼容 v2 的 `argmax`、按井五折 POR 消融、OOF 概率缓存、训练预处理中位数持久化、历史 CV 分数与本次训练状态分离。

## 环境与检查

在本目录运行，使用 Python 3.10+ 和 `requirements.txt` 中的依赖。

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python audit_data.py --project-dir .. --output reports/data_audit.json
```

## 先验证，再训练

```bash
python validate_por.py --data-dir ../data --output-dir outputs/por_cv --seeds 42,2026
python train.py --data-dir ../data --output-dir outputs/full --seeds 42,2026 --por-decision utility
python predict.py --data-dir ../data --model-dir outputs/full/models --metadata outputs/full/model_meta.json --output outputs/full/result.json
python validate_result.py --data-dir ../data --result outputs/full/result.json
python package_submission.py --solution-dir . --outputs-dir outputs/full
```

`validate_por.py` 默认训练 5 折，每折 POR 默认分类 500 轮、分箱分类 1200 轮，和全量训练一致；没有 early stopping。CPU 可加 `--cpu`。短轮数参数只用于工程冒烟测试，不能据此验收效果。

三个固定方案共享同折同模型概率：`argmax` 和 `utility` 都固定门控 0.5、不缩放；`joint` 在完整分布中直接选择官方效用最高值。这个新基线不是历史调过阈值的 v2，不能将其绝对分数与历史 0.779717 相减。新 v3 POR 全量训练同样使用 0.5 门控；PERM/SW 保留 v2 历史校准并在元数据明确标记。

输出 `fold_*.npz`、`oof.npz`、`manifest.json`、`scores.json`。报告包含 POR micro/macro、逐井/逐折、最差井、默认/非默认和标签区间切片、井级配对 bootstrap 区间。总分变化仅为 `30 × POR 增量` 的条件换算，不是三个目标完整评估。没有 OOF 参数搜索；如果看分后选择最优模式，必须在额外 grouped holdout 上复核。bootstrap 不能替代重复训练验证。

## 使用既有 v2 模型做推理消融

把原提交代码包解压到一个新目录，例如 `/tmp/welllog-v2-reference`；模型在其 `models/`。为保留历史行为，不指定选项且元数据是 v2 时仍使用 argmax。

```bash
python predict.py --data-dir ../data --model-dir /tmp/welllog-v2-reference/models --metadata ../submission_v2/model_meta.json --por-decision utility --output outputs/por_utility_candidate.json
```

该命令沿用 v2 历史门控 0.261378；只改变 POR 连续值决策。它与新 v3 固定门控 CV 的绝对口径不同，必须单独获得同口径验证后才能据此替换提交。对 v2 的兼容路径需要训练输入来重建中位数；v3 自己训练的模型将中位数写入元数据，推理只需测试输入。

`utility` 的最优性仅对“类别中位数上的离散概率分布”成立，并不保证真实标签得分改善。它没有消除分箱量化误差。`joint` 对概率校准更敏感，属于探索性选项。

输出目录或结果文件已存在时拒绝覆盖。不要把试验结果写入 `submission_v2`。原依赖版本、固定轮数和 seed 保持；GPU 训练仍可能有微小非确定性。
