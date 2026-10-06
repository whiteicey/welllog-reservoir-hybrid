# 项目进度
目标：检验复杂预测头的真实结构增量。
已做：核对80.274135来源；预定义三组结构、训练合同和记录入口。
计划：正式训练、最佳恢复、按井分解和胜出权重重载；核对attention有效性。
未做：尚无本批训练收益、完整OOF提升或中断续跑等价性证据。
达成后：固定结构与通道策略评估泛化；当前无自动补折。


## 2026-09-28T10:32:50.058856+08:00 复杂预测头v6
预定义linear/deep_mlp/attention三项；本次明确启动[('linear', 76148), ('attention', 93284)]。同80.274135、原双专家和评分；真实初始/训练/收益待日志。无旧路线或补折，目标仍未完成。详见results/deep_learning/dual8222_attention_v6_20260928。

## 本批实际核对
观察时间：2026-09-28T10:41:20+08:00；全部完成=True；当前保留={'mode': 'previous', 'TOTAL': 80.27413533779057}。
已做：结构与正式状态记录。剩余：未完成训练/重载检查及独立泛化验证。目标>80.5和完整OOF>82尚未实现；无自动补折。


## 2026-09-28T10:42:57+08:00 复杂预测头v6已完成
三项均best_epoch0、stop18，原80.274135保持。实际更新后最高linear80.272137140362/deepMLP80.270535660860/attention80.272549291930；没有新增收益。attention相对偏置真实更新，mask四项检查通过。当前无训练进程，无补折或交付变更；分量query/cross-attention尚未实现或启动。重载不重复，保留原v5证明。详见results/deep_learning/dual8222_attention_v6_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
