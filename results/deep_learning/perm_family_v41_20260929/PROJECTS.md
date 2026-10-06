# v41 PERM条件分布族

## 探索重点
固定原第五折TOTAL80.336513436816，PERM89.396724049366，比较Gaussian与Student-t3条件混合分布。历史POR用t3、SW有t3对照；历史PERM有CRPS/MAE损失尝试。本次是当前独立PERM强预测器的同参数分布族对照，不恢复旧训练。
两头各11824参数，319→32→48，仅修正最终16分量的权重、位置、尺度；默认概率及所有旧模型固定。两候选同初始化、同训练窗、同优化器，唯一不同是高斯还是df3 Student-t密度及其对应CDF。Student scale=标准差/sqrt(3)，两候选初始分量均值/标准差相同，分布形状不同。

## 初始效果与训练收益分离
高斯第0轮逐点复现父预测。t3第0轮是改变分布族的实测结果，不冒称函数相同。始终单独记录原父模型回退、候选初始分、候选最佳和训练后最高；不把第0轮差异算成训练收益。
训练在原标准化log10 PERM空间做对应NLL/3，Q50与默认/缺失掩码保持。复用已验证高斯CDF与df3 Student CDF，不改原256格支持、原800动作加默认动作及官方评分。默认概率固定不意味着最终默认动作选择完全不变。
新权重/rawscale增量界.25，位置增量界为父分量标准差的.25；两者相同。原9曲线206主特征+既有63幅值侧视图不变。

## 训练预算与进度
80上限/pat18/min10/b64/64步，lr3e-4/AdamWwd.01/clip2/cosine80*64至1e-5。一个GPU进程共享冻结前向，两个头独立参数/优化/早停；总GPU任务最多2，不自动排队。使用实际最佳checkpoint，保存原子last_state供显式恢复。
已实现模块/训练器；实际收益待训练。完成后恢复候选、固定POR/SW实际三列评分；采用新PERM才另做磁盘重建。所有拟合只用原64训练井，第五折144665点用于原选模评分。
v37-v40无收益设置不继续扫描。本次若t3未显示收益，不扩大自由度/混合系数/解码网格搜索。完整OOF>82仍未完成；第五严格>80.5且明显收益才考虑其余折，本批不补折/重启旧任务/覆盖交付。
维护PROJECTS.md、独立README、EXPLORATIONS与图谱；结束后记录差值/负证据并清理无用权重。


## v41显式启动
{"pid": 53764, "started_at": "2026-09-29T15:44:58.654667+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_family_v41.py", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_family_v41_fold5"], "automatic_queue": false}


## v41 PERM分布族完成
Gaussian与Student-t3均18轮结束，均回退原PERM。训练后最佳89.367194812859/89.372730778423，父模型89.396724049366。实际三列TOTAL=80.336513436816不变，无其他折/旧队列；详细证据见results/deep_learning/perm_family_v41_20260929/FINAL.md。目标仍未完成，不延伸本设置分布族扫描。

## v41负权重清理
已完成恢复和实际三列评分，删除本实验负候选checkpoint共5个、7224100字节；源/配置/训练日志/紧凑预测和原冠军依赖保留。清单见perm_family_v41_20260929/deletion_manifest.json。无活动训练或自动队列。



## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
