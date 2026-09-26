# 冻结实验源码目录

该目录保留探索期间的原始模块、训练器、核验器和运行脚本，以便追溯结果。文件较多是实验历史，不表示应该批量运行。

**用户已要求最后一项训练结束后停止。不要运行队列脚本自动继续训练。**

## 推荐阅读顺序

1. [方法](../../docs/deep_learning/METHOD.md) 与 [实验路径](../../docs/deep_learning/EXPERIMENTS.md)。
2. `paired_calibrated_view_sequence.py`：当前完整五折最佳结构。
3. `train_paired_calibrated_view_sequence.py`：历史 75 轮最佳完整方案的训练、验证选模和导出。
4. `paired_detached_router_sequence.py`、`paired_local_context_router_sequence.py`、`paired_posterior_router_sequence.py`：有明确对照的路由扩展。
5. [可配置数据路径的入口](../../tools/deep_learning/README.md)：在另一个工作区复现固定第五折。

历史训练器保留原有默认路径、预算、筛选门槛和哈希依赖，不能把文件名中的 802、旧首折门槛或 `screening_passed` 当作当前晋级授权。部分历史诊断脚本依赖原本地输出或旁边的旧方法目录，不能据源码存在就声称所有旧脚本可直接移植运行。

可移植入口目前仅支持第五折筛选；五折已完成版本的原源码和结果均保留，完整历史核验流程需对应的本地产物。推理使用 [package_dl](../../package_dl/README.md)，避免导入实验训练依赖。

数据、checkpoint、逐行预测、完整训练日志和大型降权探测缓存不进入 Git。精简报告集中于 [results/deep_learning](../../results/deep_learning/README.md)；无收益分支的大文件可能已按用户要求删除。
