# v19进度

目标：多尺度POR表征及全局辅助路径对照。
已做：规则/进程/原分数核对，新结构实现。
计划：启动两组，恢复最佳、实际组合评分。
未做：真实训练收益、最终重载、其他目标/折验证。
完成后：保留有收益的开发配方；不自动补折，不覆盖交付。


## 2026-09-28T20:27:53.845030+08:00 v19显式启动
[{"mode": "regional", "pid": 43316, "started_at": "2026-09-28T20:27:53.835030+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_pyramid_v19.py", "--mode", "regional", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_pyramid_v19_regional_fold5"]}, {"mode": "global", "pid": 46940, "started_at": "2026-09-28T20:27:53.843030+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_pyramid_v19.py", "--mode", "global", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_pyramid_v19_global_fold5"]}]
仅两项新POR训练，其他通道/折/交付保持不变；实际成绩待运行结果。


## v19多尺度POR对照已完成
2026-09-28T20:35:12.522433+08:00。regional TOTAL=79.551149548，global TOTAL=79.555720337，当前保留80.274135338。新POR磁盘复现及固定其余两通道数组合并完成。尚未运行新对照/其他目标/原其他折，完整OOF>82未完成。记录por_pyramid_v19_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
