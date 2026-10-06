# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v177 井均衡私有PERM完整网络SAM

## 探索重点
保留175已实测井均衡收益，只加SAM0.05；对照175PERM88.893539547789，同685873参数/初始化/采样/loss/RPS1/80-14。假设读出SAM局部收益可以转移到完整网络优化。

## 已探索与验证
已实现，未训练无分数。NLL+BCE+jointRPS1、lr6e-4、50%行+50%井、原Q50、seed20261010/80-14/min10/b64/64步/cos80*64；不加载175训练权重。全模型梯度L2扰动半径0.05，两次前后向、精确回滚、RNG复用及clip2。最终恢复ownPERM最佳并实际组合144665点。

## 收益
尚未证实。单头SAM已证明微小增益，但完整骨干更新范围与梯度尺度不同，不能预先声称收益可迁移。等优化步数但SAM两次前后向，不是等算力对照。全方法来源计入仍5阶段。

## 不继续的点
不同时加入效用/改CDF/扫SAM半径或学习率；无益则关闭本次完整模型SAM。未超过绝对v15489.405564856919，不称SOTA；严格TOTAL>80.8有收益才补折。

耗时待实测：与174或176错峰，max2GPU、峰值/数据/预热记录，单项不冒称全五阶段成本。

v177已显式启动PID14800，normal GPU/临时SYSTEM_REQUIRED，active685873。当前训练快照{'status': 'running', 'pid': 14800, 'stage': 'perm_conditional', 'epoch': 12, 'best': 88.34151401360728, 'best_epoch': 6, 'bad': 6, 'peak_MiB': 800.61279296875}，非最终收益；最多2GPU、完整方法5阶段、其余四折未启动。
原私有初始化函数与175一致，完整网络SAM梯度范围与旧独立头不同；CUDA插值/attention反向warn_only非确定性警告如对照，未出现运行错误，最终实际恢复待做。

V177 final PERM=88.970890837572, compactcontrolgain=-0.407503373572; actualTOTAL=80.460339511223, fivephases inclsource;strict808=False, no otherfolds. best/stop=24/38, seconds=551.783, peakMiB=800.613; all144665replay/frozen exact.

最终完整骨干SAM局部正收益：PERM88.970890837572，相对175 +0.077351289783；同模型/初始化/采样/loss/Q/预算/优化步数核验通过；SAM2遍vsAdamW1遍，不等算力。实际TOTAL80.460339511223，best/stop24/38，144665磁盘恢复/冻结一致；保留正结果best/last。仍低于有效154，不替换配方/补折。




