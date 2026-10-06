# v17进度

目标：结构对照及原第五折总分。

已做：v16实证与状态核对、两个同参数模型/训练器实现。

计划：显式启动两组、自然早停、重载并合并评分。

未做：真实训练成绩、最终收益与跨折验证。

完成后：保留有收益的固定配方；无收益则保留v5，不自动补折或覆盖交付。


## 2026-09-28T19:28:42.582911+08:00 v17显式启动
[{"mode": "shared_global", "pid": 39272, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_latent_gate_v17.py", "--mode", "shared_global", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_latent_gate_v17_shared_global_fold5"], "started_at": "2026-09-28T19:28:42.569549+08:00"}, {"mode": "local_gate", "pid": 6468, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_latent_gate_v17.py", "--mode", "local_gate", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_latent_gate_v17_local_gate_fold5"], "started_at": "2026-09-28T19:28:42.580396+08:00"}]
仅启动两个预定义SW候选，原其他折不变；成绩与最终组合尚待运行结果。


## v17最终完成
2026-09-28T19:37:40.442844+08:00。两组训练/原144665点磁盘重载与三模型实际合并已完成。shared_globalTOTAL=80.219534179，local_gateTOTAL=80.105625735；无新增收益，保留80.274135。未完成：新假设/其他通道结构/完整OOF>82。无第三候选、补折或交付变更。证据results/deep_learning/latent_gate_v17_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
