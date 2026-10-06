

## 2026-10-01 v105 POR末端投影
已完成：原源同函数重建mean_scope+16640末端投影梯度链实现，own POR；现在：直接训练，与SW spectral最多双任务。待做：与v101最终更新范围差值和实际三通道；不将部分分数当收益。目标：含源≤5阶段、第五>80.8、原完整OOF>82；达成后新交付。

### 最终更新范围消融
POR=77.477365174678，对固定投影v101=-0.023003212967；best9/stop27，159.577秒，峰值786.5MiB。新增投影实际梯度0.372072，其他源参数冻结且实际最佳预测重现；同初始官方分。负结果，继续采用v101，不扫该3e-6范围或扩大解冻。仅第五折，未验证完整收益。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
