# 项目进度

目标：验证分量查询读出在原双专家上的增量。

已完成：两项正式训练、同参数/初始化确认、最终恢复、旧参数与其他通道冻结验证、查询更新与局部遮挡诊断；均未超越80.274135，原模型保持。

未完成：独立泛化、完整OOF>82、中断续跑等价性；完整分布query/监督对齐尚未实现。

后续计划：不追加本批候选；下一次选择明确机制并固定对照，避免同时改变输出范围与损失。

达成目标后：固定配方验证原分组泛化；当前不补其他折，不改旧交付。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
