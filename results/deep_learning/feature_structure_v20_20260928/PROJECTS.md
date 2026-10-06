# v20进度

目标：将特征改进与融合结构改进分开验证，以原第五折总分衡量。

已完成：核对v19记录及无运行训练，检查原206维特征定义；已含井排名、残差与斜率，尚无逐曲线的多尺度局部方差标准化侧视图。

计划：训练侧准备两种63维视图；实现同参数早融合/晚融合网络；显式分批训练3项；重载新POR并与固定PERM/SW数组实际合并评分。

未完成：特征准备、训练、收益和最终重载。当前保留v5/v8 TOTAL80.274135337791。

达成后：仅保留有证据的固定方案，再决定其他通道或更多验证；不自动扩大网格、补折或替换旧交付。


## 2026-09-28T20:49:20.172055+08:00 v20显式启动
[{"mode": "amplitude_early", "pid": 5840, "started_at": "2026-09-28T20:49:20.161054+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_feature_structure_v20.py", "--mode", "amplitude_early", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_feature_structure_v20_amplitude_early_fold5"]}, {"mode": "shape_early", "pid": 36524, "started_at": "2026-09-28T20:49:20.171054+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_feature_structure_v20.py", "--mode", "shape_early", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_feature_structure_v20_shape_early_fold5"]}]
预定义三项之一或两项，最多2GPU；未启动的项目不在自动队列，需本次显式调用。


## 2026-09-28T20:53:03.503313+08:00 v20显式启动
[{"mode": "shape_late", "pid": 11960, "started_at": "2026-09-28T20:53:03.499797+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_feature_structure_v20.py", "--mode", "shape_late", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_feature_structure_v20_shape_late_fold5"]}]
预定义三项之一或两项，最多2GPU；未启动的项目不在自动队列，需本次显式调用。


## v20特征与融合实验完成
2026-09-28T20:56:42.218786+08:00。三项全部完成并重载，特征TOTAL差+0.152891789，晚融合TOTAL差-0.080073697；保留TOTAL=80.274135338。仅本批新POR，其他目标/折/交付不变；强骨干特征接入与完整OOF尚未做。证据feature_structure_v20_20260928/FINAL.md。
