# 实验进度
目标：验证SW独立TCN分布读出与训练容差代理的增量。
已完成：v8全部六项和实际三模型恢复；v9两项实现及预定义。
计划：实际训练、梯度与最佳恢复、匹配目标比较、固定三模型评价。
未完成：v9成绩、独立泛化、完整OOF>82。
达成后：固定配方与成本评价；补折仍暂缓，旧交付保持。


## 2026-09-28T11:54:31.739535+08:00 SW独立TCN分布v9启动
已显式启动[('ce', 58148), ('ce_utility', 90832)]。只比较两个预定义目标，未有新收益证据；旧POR/PERM、交付和原数据不变，无补折。记录见results/deep_learning/independent_sw_v9_20260928。


## 2026-09-28T12:02:17.041218+08:00 独立SW分布v9收尾
实际三模型TOTAL=80.274135337791，delta=+0.000000000000；使用新SW=False。POR/PERM逐元素不变，三模型无共享存储且全部144665点重载复现。
已完成v8遗留项、v8组合以及v9两项匹配训练与组合；未补折，旧交付不替换。目标完整OOF>82仍未完成。下一步按实际结果确定，不自动启动更多训练。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
