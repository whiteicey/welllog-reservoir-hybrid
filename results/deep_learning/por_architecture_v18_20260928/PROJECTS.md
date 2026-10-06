# v18进度

目标：探索潜变量全局上下文对POR的价值，得到同第五折实际三通道总分。

已做：核对v17无新增收益、当前保留80.274135；确认无运行训练；确定TCN104和latent64两项及共同8分量Student-t3输出，新增模型和POR数据接口。

计划：准备仅POR训练标签；两项从随机网络训练，恢复最佳并重载新POR；与固定PERM/SW预测按原行序合并评分；维护图谱与总结。

未做：真实训练、结构收益及完整五折验证。当前没有已确认的新成绩。

达成后：保留有真实收益的固定开发配方；第五折严格80.5、完整OOF82仍是未完成目标。不自动补折或替换旧交付。


## 2026-09-28T20:09:47.896873+08:00 v18显式启动
[{"mode": "tcn104", "pid": 9468, "started_at": "2026-09-28T20:09:47.882493+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_architecture_v18.py", "--mode", "tcn104", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_architecture_v18_tcn104_fold5"]}, {"mode": "latent64", "pid": 23700, "started_at": "2026-09-28T20:09:47.894874+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_architecture_v18.py", "--mode", "latent64", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_architecture_v18_latent64_fold5"]}]
仅两项预定义POR训练；未获得新的最终分数，未改旧模型/交付/其他折。


## v18 POR结构对照完成
2026-09-28T20:16:03.903565+08:00。TCN TOTAL=79.550729668，latent TOTAL=79.147915298，保留TOTAL=80.274135338。两个新POR磁盘重载精确，固定PERM/SW数组合并实测；原模型与其他折未改。后续结构/完整五折未运行，当前无新增队列。记录results/deep_learning/por_architecture_v18_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
