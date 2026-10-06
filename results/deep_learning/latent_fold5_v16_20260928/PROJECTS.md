# 目标与进展

目标：实际验证v15 latent的原第五折总分。
已做：原64井重新拟合预处理，原验证144665行核对，两个训练入口实现。
计划：两项从头训练，恢复最佳SW，再重载固定POR/PERM共同评分。
未做：最终训练成绩、磁盘组合验证、相同容量消融及完整OOF。
达成后：按实际收益保留新候选，原交付和其他四折保持不动。


## 2026-09-28T18:03:35.019301+08:00 v16原第五折启动
实际显式启动[('tcn', 46268), ('latent', 10204)]。不加载内层权重，原64/16井，固定其余通道，最终总分待核对。原旧任务、交付、其他四折不改。研究记录results/deep_learning/latent_fold5_v16_20260928。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
