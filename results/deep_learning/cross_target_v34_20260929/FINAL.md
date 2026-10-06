# v34 跨目标预测协同完成结果

完成时间：2026-09-29T13:31:08.938650+08:00
原TOTAL=80.336513436816；实际组合TOTAL=80.336513436816；增量=+0.000000000000。

## 探索重点与真实完成范围

原模型保持，12个新辅助MLP（4组×3目标）完成，随后训练3个SW权重修正头。原POR/PERM输出未变化。所有辅助模型不使用当前强模型权重；这是微型辅助预测试验，不是当前冠军POR/PERM的完整交叉拟合重训。
每组40井拟合，8井NLL早停，16井接收OOF预测；查询井同时排除于拟合和早停，特征统计只拟合40井。原第五折不进入辅助拟合、统计或停止选择。

## 原第五折SW对照

|方案|参数|更新后最高SW|最终SW|最佳/停止轮次|
|---|---:|---:|---:|---|
|self|11152|73.658076471|73.663082797|0/18|
|cross|11152|73.656148110|73.663082797|0/18|
|shuffled|11152|73.656649624|73.663082797|0/18|

第0轮回退不记作训练收益。shuffled只作负对照，不允许部署。
机制对照：{"cross_minus_self": 0.0, "cross_minus_shuffled": 0.0, "post_update_cross_minus_self": -0.0019283615195888615, "post_update_cross_minus_shuffled": -0.0005015141030924042}
跨目标机制是否通过预设比较=False；新采用=无，保留原模型。

## 来源、信息范围与限制

每个辅助目标输出六维描述，self重复SW自身描述，cross拼接POR/PERM，shuffled在同一OOF组内打乱cross。三个头均331→32→16、11152参数、同初始化、lr3e-4和80/18；只改SW16分量权重。
辅助输入来自预测，没有读取真实其他目标值；打乱也不跨留出组，避免间接换入见过查询井的模型输出。原第五折使用四模型描述均值，与训练单模型OOF存在分布差异；平均分位数不等于混合分位数。
当前SW主模型曾在原64井训练，原第五折长期参与选模，因此本批只有辅助预测来源是井外OOF，整个系统并非独立OOF验证。
辅助质量诊断：[{"target": "POR", "OOF_rows": 578891, "numeric_n": 185691, "gate_brier": 0.05591554567217827, "numeric_mean_spearman": 0.9182675867606863, "predictive_interval_coverage": 0.7322164240593244, "scope": "Original64 auxiliary OOF descriptive diagnostics; not a new total score; not used to select feature variants"}, {"target": "PERM", "OOF_rows": 578891, "numeric_n": 179827, "gate_brier": 0.05570795387029648, "numeric_mean_spearman": 0.9357647299672738, "predictive_interval_coverage": 0.7412235092616792, "scope": "Original64 auxiliary OOF descriptive diagnostics; not a new total score; not used to select feature variants"}, {"target": "SW", "OOF_rows": 578891, "numeric_n": 161423, "gate_brier": 0.06129360571503639, "numeric_mean_spearman": 0.6427152066548336, "predictive_interval_coverage": 0.739894562732696, "scope": "Original64 auxiliary OOF descriptive diagnostics; not a new total score; not used to select feature variants"}]

## 验证与保留

三个头训练内最佳恢复、冻结状态及原评分行序均通过。新采用模型额外磁盘重建：[]；无采用则不重复原模型推理。三列按原行序合并实测。
组合0井升/0井降；探索性未校正区间[0.0, 0.0]。预测未变时[0,0]不构成新增证据。

## 不继续和未完成

如没有同时超过原模型和两个对照，不扩展本批MLP宽度、描述项或学习率网格；结果只约束本次微型辅助来源与固定权重头，不否定全部跨目标方法。
当前强POR/PERM完整骨干的嵌套交叉拟合、跨井检索、状态序列模型、新井确认均未实施。原其他四折继续暂停，旧交付未改。

## 证据文件

METHOD.md、auxiliary_manifest.json、4份aux_config、feature_manifest.json、assembled_report.json、CURRENT_RECIPE.json、timing.json；完整运行与最佳参数位于outputs/pure_dl_cross_target_v34_*。
