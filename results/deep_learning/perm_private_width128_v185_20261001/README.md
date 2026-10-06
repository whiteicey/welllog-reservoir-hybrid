# v185 私有PERM骨干96到128

## 探索重点
仅扩大177已获井均衡/SAM局部收益的TCN和token attention编码宽度96→128；最终128维读出、16Gaussian、局部块/注意力层数/头数/采样/RPS1/SAM0.05/LR6e-4/预算80-14/Q/评分保持。对照177PERM88.970890837572，绝对15489.405564856919。

## 已探索与验证
已实现，未训练无成绩。新的随机宽骨干，原NN预测头初始化为训练64井固定prior，初始输出分布保持；hidden权重维度/随机初始化不同，不是同容量消融。无teacher/整井条件化/旧177权重。实际训练后验证峰值显存/时间，不启数值预检或独立审查。

## 收益
尚未证实。报告对各自初始和177最终的差值及实际144665三头总分。来源仅计一次，总5阶段；更多参数/全局SAM范数/算力变化如实区分。

## 不继续的点
本批只固定128，不展开宽度/学习率/层数网格；無收益就关闭这次容量增加。未严格TOTAL>80.8有收益不补折，原OOF>82未完成。

耗时待实测：单项/总工作墙钟/错峰并发/峰值显存、data与warmup区分，不能冒称完整5阶段总时间。


Live launch confirmed: {"process_verified_live": true, "pid": 90200, "state": {"status": "running", "pid": 90200, "stage": "perm_conditional", "epoch": 17, "best": 88.98853363832089, "best_epoch": 16, "bad": 1, "peak_MiB": 833.32861328125}, "normal_GPU": true, "budget": "80/14/min10/b64/64steps", "source_counted_total_stages": 5, "complete_five_folds": false, "remaining_folds_started": false, "result_scope": "live training snapshot, not final gain"}. v185 repairs missing functional import only; random width128 differs in hidden initialization/capacity from width96 control, zero-output train-prior initial predictive function is shared. v184 freezes original head/router parameters but allows body gradients; fixed POR/SW remain unchanged.

V185 final PERM=88.988533638321, compactcontrolgain=-0.389860572823; actualTOTAL=80.466514491485, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=467.565, peakMiB=833.329; all144665replay/frozen exact.


Final width128 PERM=88.988533638321, versus96 control +0.017642800749, best/stop16/30. Actual144665-point assemblyTOTAL80.466514491485; exact best replay/frozen verified. Positive best/last retained, no SOTA or fullOOF claim. Capacity and random hidden initialization differ, initial predictive function and other controls match; not equal compute. Close failed183 entry separately, no score. Single training467.564975s, peak833.329MiB, overlapped184; not full five-phase cost. Gain supports this capacity direction but no automatic width queue.
