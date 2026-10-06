# v156 进展

目标：扩大PERM已证实联合CDF监督收益，最终完整OOF>82。

已完成：同容量/初始化和预算，只有系数1->2改变；PLAN及原正控制固定。

计划/未完成：显式启动、真实训练观察；最佳重载/实际三通道合并/通道对照/耗时显存；结论、负权重清理和图谱更新。

可选：有收益保留各通道独立设置，负则不自动扫系数或补折。

RUNTIME FAILURE: CUDA illegalmemory access duringbackward after completeepoch8; notOOM, not a mechanism negative/convergence result. PID72544 absent, no autoresume. Preserve atomiclast/source/config/log/error for diagnosis. Earlier launchsuccess records remain historical.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
