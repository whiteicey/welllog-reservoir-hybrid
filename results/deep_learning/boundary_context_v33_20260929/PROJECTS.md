# v33 已完成

## 目标
基于80.336513当前预测器，诊断输入边界并验证边界截断统计是否提供增量。

## 已完成
九曲线输入边界阈值只由64训练井拟合，原行序和评分已复现；默认/非默认、逐井、总失分预算均已计算。POR与SW分别完成fixed/segmented两项，共四项；每项11344参数，全部第18轮自然早停、选第0轮。冻结状态和训练内最佳恢复通过。最终三列实际评分80.336513436816，与原预测完全相同，未采用新头，未重复未变化模型的全量磁盘推理。

## 结论与停止方向
边界附近的非默认点难度高，但±5点仅占总失分2.08%，>25点占87.92%。截断51点统计未取得收益，不追加阈值、窗口和宽度搜索。保持原v31配方和交付。

## 未完成与后续
未进行不同种子、新井、标签错位诊断或数值群组诊断；均未启动。其他原折仍暂停，完整OOF>82未完成。后续应先检查当前主要失分群的特征和分布，按证据决定新试验。

## 完成后维护
记录FINAL.md、loss_budget.json、assembled_report.json、CURRENT_RECIPE.json、训练配置/最佳更新后分数和状态；增量图谱脚本curate_boundary_v33.py。原训练来源、权重、标签、交付均未修改。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
