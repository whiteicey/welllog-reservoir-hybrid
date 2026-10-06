# v23进度

目标：检验相同POR双专家从头学习时内部容量扩展是否带来收益。

已完成：确认v22两项已结束、当前无训练；基线80.274135不变。核对原80.22训练含多任务和动态Q，明确本轮采用单目标私有双专家同协议容量对照。

计划：实现无旧权重加载的base1/ffn2；同输入、Q50和损失从头训练；核算最佳POR与固定PERM/SW的实际总分，记录容量增益和训练—验证差。

未完成：v23训练、最终结果与采用判定，其他目标、其他折及完整OOF>82。

达成后：只保留实测有收益的固定配置；不自动追加实验、补折或更改旧交付。


## 2026-09-28T22:08:53.041374+08:00 v23显式启动
[{"mode": "base1", "pid": 52504, "started_at": "2026-09-28T22:08:53.026053+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scratch_scaling_v23.py", "--mode", "base1", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_scratch_scaling_v23_base1_fold5"]}, {"mode": "ffn2", "pid": 52524, "started_at": "2026-09-28T22:08:53.039378+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scratch_scaling_v23.py", "--mode", "ffn2", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_scratch_scaling_v23_ffn2_fold5"]}]
仅这两个预定义任务，无其他折或自动追加。


## v23 从头双专家扩容对照已完成
2026-09-28T22:25:40.881346+08:00。base1 TOTAL=79.548859162；ffn2 TOTAL=79.541070769；容量差-0.007788393，当前保留80.274135338。两模型从随机权重训练/最佳磁盘重载/训练与验证诊断均完成。其他目标/原其他折及原多任务完整配方未执行；无自动后续。记录scratch_scaling_v23_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
