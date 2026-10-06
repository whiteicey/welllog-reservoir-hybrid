# v37 Q50 与等权训练对照

目标：以第五折80.336513436816为固定基线，检验旧Q50降低权重的样本是否仍含可学习信号。
本批仅POR/SW各q50、uniform两个候选。PERM固定；每个目标两头同11857参数、同初始化、同训练窗口与新优化器，仅质量权重不同。新319→32→49逐点头同时修正门控、混合权重、位置和尺度；旧模型全冻结。门控/权重/rawscale增量界.25，位置界为父标准差的.25。零输出初始化保留原函数。
原9曲线/206特征、已有63幅值侧特征、原64训练井/16验证井与144665点评分、单覆盖256格800动作解码不变。原Q50是训练侧历史质量权重，不使用验证/测试标签构造权重。uniform不把困难点预先认定为噪声。
max80/patience18/min10/batch64/64步，AdamW lr3e-4/weight_decay.01，cosine80*64至1e-5，clip2。两目标独立进程、至多2GPU任务；同目标两候选共享冻结前向以节省时间，参数/优化器/早停独立。新头NLL/3，不增加辅助标签或外部模型。
已完成：实现模块、复用原训练侧数据和评分；启动后成绩见输出状态。
待完成：真实训练、最佳恢复、固定三列组装验证、记录消融差与基线差；当前尚未证实收益。
不继续的点：v32至v36输入侧统计/微型协同/检索未证实新增收益；不扩展检索库扫描。本实验不代表Q1从头重训，也不能把单折正差宣称完整OOF。
最终目标：严格第五>80.5且明显收益后才考虑其余折；完整OOF>82未达成。无旧队列恢复、其他折或交付覆盖。完成后根据实测决定是否有必要扩大当前有效方向。
每批更新PROJECTS.md、README、EXPLORATIONS与图谱；原子last_state保存，恢复须显式授权。


## v37 SW 显式启动
{"pid": 64628, "target": "SW", "started_at": "2026-09-29T14:32:10.484297+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quality_readout_v37.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quality_readout_v37_SW_fold5"], "automatic_queue": false}


## v37 POR 显式启动
{"pid": 62660, "target": "POR", "started_at": "2026-09-29T14:32:32.394994+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quality_readout_v37.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quality_readout_v37_POR_fold5"], "automatic_queue": false}


## v37质量权重对照完成
四项完成，TOTAL=80.336513436816，相对80.336513增加+0.000000000000；采用[]。完整记录quality_readout_v37_20260929/FINAL.md；无其他折或交付改动。


## v37 closure and storage
The four candidates finished and restored successfully; no candidate was adopted. Keep the v31 recipe at TOTAL80.336513436816. Uniform weighting has only small post-update relative gains over Q50, not an improvement over the parent. Do not expand this bounded all-factor head sweep. Next independent hypothesis to assess: train a small direct action-value readout against the original scoring kernel, with the original density decoder retained as control; this is not implemented or launched yet. Full OOF>82 remains unachieved; no other folds or old queues started.
Removed 22 non-adopted checkpoints/cache files totaling104118150 bytes across v35-v37, with SHA256/size manifests. Sources, configurations, compact predictions, reports and all baseline/deployment dependencies remain.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
