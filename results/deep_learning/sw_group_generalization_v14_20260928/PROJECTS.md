# 目标与进展

目标：定位未见井泛化瓶颈并检验井均衡监督。
已完成：固定内层48/16切分与严格重拟合预处理；未用旧模型/Q/目标统计。
计划：本批两组训练及训练/监控曲线比较。
未完成：实测增益、原第五折评价、完整OOF>82。
达成后：固定机制再依授权做原第五折比较；不自动补其他折或替换交付。


## 2026-09-28T16:44:49.116604+08:00 v14跨井诊断启动
在继续优化授权下预设point/well两组，实际启动[('point', 22756), ('well', 100016)]。当前阶段inner；内层48训练/16监控，全部预处理重新拟合、Q=1、无旧权重。原第五折未用于当前内层训练或选模。真实收益待核实；不自动补折、旧交付不变。记录results/deep_learning/sw_group_generalization_v14_20260928。


## 2026-09-28T16:47:32.678918+08:00 v14内层诊断完成
两项均已自然结束。pointSW=76.23493007968595，wellSW=76.165444754071；差-0.06948532561494469。这是48→16内层SW，不是原第五折；原第五折未评估，旧80.274135保持。训练与监控轨迹及诊断已保存，无自动补折。报告results/deep_learning/sw_group_generalization_v14_20260928/FINAL_inner.md。


## 2026-09-28T16:50:26.967310+08:00 v14最终收尾
完成严格48/16内层实验，原第五折未评估。按井均衡未通过预设内部门槛，不启动额外原64井训练。发现默认率差可掩盖总体泛化差，后期训练分升而监控分降；具体见results/deep_learning/sw_group_generalization_v14_20260928/FINAL.md。后续跨井局部参考未实现，旧三私有模型80.274135保持，原其他折与交付不改。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
