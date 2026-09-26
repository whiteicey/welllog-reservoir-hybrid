# 纯 DL 精简结果

- `leaderboard_result.json`：用户反馈的提交后 A 榜分数 **83.6915**，对应校准双骨干五折集成；未独立读取榜单，与 OOF 分开记录。

- `full_oof_original_verification.json`：当前最佳完整五折，80 口井、723,556 行，原评分器 TOTAL=81.71323318479078；其中 POR/PERM/SW 为 0–1，TOTAL 为百分制。
- `calibrated_continuation_verification.json`：冻结配置、第五折权重保护及五折全训练点降权表回放结果。
- `delivery_validation.json`：最终代码/结果 ZIP 哈希、五权重、严格 JSON 与真实井号/深度行覆盖检查。
- `standalone_execution_verification.json`：两次实际独立 GPU CLI 执行，10 口井、95,948 行，结果字节一致。没有测试精度声明。
- `experiment_index.json`：历史实验的精简折级报告，去掉训练日志、逐行预测、标签和权重。保留不同折/不同预算的原始报告，不应混作统一排名；有报告不等于每项均独立审计。

有效路线与代表性负结果见 [EXPERIMENTS.md](../../docs/deep_learning/EXPERIMENTS.md)。所有分数必须区分完整 OOF、开发用单折和未知标签测试预测。权重清单见 [model_meta.json](../../package_dl/models/model_meta.json)。

只发布摘要不意味着可以重放已清理的历史 checkpoint；源码和实验结论保留，失败分支的大文件按用户要求删除。当前最佳和关键里程碑模型保存在本地。
