# 项目进度

## 目标
检验从头学习的SW专用时序模型能否超过旧SW73.622384767129，同时保持三个目标独立。

## 已完成
v8全部六项及三模型恢复已经核实，v9两项结束但未增益。本批两种结构和统一训练入口已实现。

## 计划
显式运行两个候选、完成独立最佳模型恢复、分井/默认状态误差比较；仅有实际改善才固定新SW并合并验证。

## 未完成
本批训练成绩、独立磁盘重载、异构组合收益、完整OOF>82均尚未完成。

## 达成后
固定胜出配方评估泛化与真实推理成本；补折仍需明确授权，旧交付保持原样。


## 2026-09-28T13:17:58.091719+08:00 SW独立冷启动v10完成
固定配方TOTAL=80.274135337791，相对旧+0.000000000000；采用新SW=False。
两候选真实训练/恢复及磁盘核验完成；无新补折或交付变更。结果：results/deep_learning/independent_sw_scratch_v10_20260928/FINAL.md。尚未证明独立新井增益。


## 实测误差分解与下一步边界
两个新模型从磁盘重载后，全部144665点评分预测逐元素复现；训练后填充不变性与全空窗口检查通过。
tcn: 真默认段通道分变化+0.600186，对TOTAL贡献+0.146349；真非默认段变化-3.094669，对TOTAL贡献-0.328530。4/16井提高，12井下降。
bigru: 真默认段通道分变化+0.704171，对TOTAL贡献+0.171705；真非默认段变化-3.439228，对TOTAL贡献-0.365109。4/16井提高，12井下降。
这说明本次最终动作在默认段更准，但连续标签段损失更多；不能据此独立归因于门控分类正确率，因为密度和效用解码同时影响最终动作。
下一步尚未启动：先在训练侧诊断SW的默认状态与非默认数值任务是否相互牵制，再以门控固定/可训练和独立数值表征作有限匹配对照；不直接扩大GRU/TCN层数或推断模型家族上限。
模型是随机权重从头学习，但固定Q来自先前训练井诊断，预处理/支持格也沿用训练侧元数据；不称整条pipeline完全不依赖此前训练产物。
当前保留原三私有模型80.274135337791；无新SW替换、无补折、无旧交付变更。
记录工具问题：最终报告首次因numpy.bool_ JSON序列化失败，修复仅作用于报告脚本并明确重跑；训练权重未改。curate.py源文件编辑被内容保护拒绝，原文件未改；本轮说明由report_independent_sw_scratch_v10.py --record可复现维护到curation.json，再build_atlas.py生成页面。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
