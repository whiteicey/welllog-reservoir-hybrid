# v90 SW全井直接读出结果

当前独立路线TOTAL 80.382093291214 → 80.382093291214。

|读出|SW|固定POR/PERM TOTAL|最佳/停止|
|---|---:|---:|---:|
|reference|73.713486630808|80.354154778626|14/32|
|well|73.718401716608|80.355875058656|14/32|

探索与验证：两项同841906参数、原v56骨干/FiLM和SW先验，同初始基本state/旧头hash与144665点初始动作；新增2425参数零输出。旧FiLM两组用本井，仅新支路reference/well不同，两code都算。原输入/分折/seed/Q50/评分/Gaussian/效用1/DropKey.4/80-18保持。真实骨干/FiLM/旧头/新头及效用梯度、hash、行序、选模恢复和实际三列核验通过；没有独立预检或重复Q核验。原始输入磁盘重建=[{'mode': 'reference', 'rows': 144665, 'fresh_disk_replay_exact': True, 'score': 73.71348663080758, 'sha256': '6053132fabc6774b465c5015bccc4593162f2632a22d77b6109f5bdb02c8cc70'}, {'mode': 'well', 'rows': 144665, 'fresh_disk_replay_exact': True, 'score': 73.71840171660811, 'sha256': '5a66c8ec89d5afbebbd0b1f8b745f8452b41ddb8b99219d2f9ece94e7700763b'}]
reference对Gaussian SW-0.079824321680/TOTAL-0.027938512588，默认+0.056505769310/数值-0.392961917125，井升/降10/6。
well对Gaussian SW-0.074909235880/TOTAL-0.026218232558，默认+0.182056108342/数值-0.665134837710，井升/降7/9。

收益：well-reference SW+0.004915085801，完整训练后最佳匹配差+0.004915085801，采用=None；绝对和公平局部收益分开，局部小正不等同SOTA或独立泛化。
不继续：若两项无绝对、well-reference无正收益，关闭本共享96编码/零线性25输出/来源/预算组合，不临时扫宽度LR强度；证据不泛化所有全井机制。

成本：{"candidates": [{"mode": "reference", "start": "2026-09-30T14:11:50.204084+08:00", "end": "2026-09-30T14:17:02.526921+08:00", "seconds": 307.98883449999266, "peak_allocated_MiB": 1870.6533203125, "peak_reserved_MiB": 2040.0, "launch_to_completion_seconds": 312.322837}, {"mode": "well", "start": "2026-09-30T14:14:00.748921+08:00", "end": "2026-09-30T14:19:09.824847+08:00", "seconds": 303.84612190001644, "peak_allocated_MiB": 1870.5380859375, "peak_reserved_MiB": 2042.0, "launch_to_completion_seconds": 309.075926}], "first_launch_to_last_completion_seconds": 439.620763, "first_launch_to_verified_seconds": 502.629085, "peak_process_concurrency": 2, "overlap_seconds": 181.778, "sum_individual_launch_to_completion_seconds": 621.3987629999999, "finalization_seconds": 5.112849299999652, "scope": "data loading/warmup/train/eval/save and explicit launch gap included; before-launch authoring/atlas separate; measured concurrency, no capacity claim"}
负项清理0个权重，共0字节；完整删除计划先落盘，正源/比较依赖/旧交付保留。
已做：真实两项、必要重载、实际组合、记录清理。未做：其他折和独立泛化；第五严格>80.5且明确收益才考虑其余四折，原完整OOF>82仍未达成。目标后独立井和新目录交付。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
