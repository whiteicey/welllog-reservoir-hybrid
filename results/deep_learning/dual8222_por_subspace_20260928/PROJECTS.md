# 实验进度
目标：在80.257638固定配方上获得可复现的POR读出增益。
已做：来源核对，三个有界匹配对照已实现。
计划：实际训练并最终共同恢复，核对冻结因子和PERM/SW，再作配对比较。
未做：当前尚无训练分数、无恢复等价性验证、无完整OOF提升。
达成后：固定有益更新子空间，进一步评估泛化；当前不自动补折，完整OOF>82仍未完成。


## POR子空间实际启动
本次明确启动 [('mean_only', 62404), ('mean_scale', 93744)]；共3个预先限定对照，无自动队列。初始复现、训练结果与最终验证仍待实际日志。


## 2026-09-28T09:49:26+08:00 POR子空间结果核对
[('mean_only', 'running', 4), ('mean_scale', 'running', 4), ('density_all', 'planned', 0)]
已完成最佳={'mode': 'previous_fixed_recipe', 'TOTAL': 80.25763862460869, 'delta': 0.0}；全部完成=False。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## POR子空间实际启动
本次明确启动 [('density_all', 16688)]；共3个预先限定对照，无自动队列。初始复现、训练结果与最终验证仍待实际日志。


## 2026-09-28T09:52:33+08:00 POR子空间结果核对
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'running', 0)]
已完成最佳={'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}；全部完成=False。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## 2026-09-28T09:54:12+08:00 POR子空间结果核对
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'running', 13)]
已完成最佳={'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}；全部完成=False。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## 2026-09-28T09:56:37+08:00 POR子空间结果核对
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'complete', 19)]
已完成最佳={'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}；全部完成=True。目标完整OOF>82尚未达成，补折暂停，旧交付不改。详见results/deep_learning/dual8222_por_subspace_20260928。


## Final evidence and next boundary
Completed best: mean_only, TOTAL=80.268734508852, delta=+0.011095884243. POR=77.430912944379; PERM/SW unchanged. All3 runs completed; no additional training launched.
Exploratory 10000 whole-well resamples: [-0.012002,+0.034743], selection-unadjusted; not confirmatory independent validation.
Next unexecuted hypothesis: keep mixture weights/scales fixed and compare a constrained location readout against this linear mean-only control. Do not automatically open width/LR sweeps or fullCV. Full OOF>82 remains unmet. Source and prior submission unchanged.


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
