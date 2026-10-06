# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## FINAL.md

# SW从头训练：结果

|结构|参数|初始SW|最好SW|最佳/停止轮次|替换诊断TOTAL|
|---|---:|---:|---:|---|---:|
|tcn|82113|71.197115|73.101868|13/31|80.091954|
|bigru|88321|71.197115|73.069803|13/31|80.080732|

当前固定配方TOTAL=80.274135337791，相对旧配方+0.000000000000；使用新SW=False。
两个新网络均从磁盘重载，所有144665点精确复现。训练后有效段/填充检查通过。
从训练集先验初始化的提升不能等同于超过旧SW；旧模型使用预训练，不构成随机起点匹配对照。
两结构的参数、精度和感受范围存在差别，不把分差全归因于某一算子。每井/默认段分解与未校正选择的探索性区间见assembled_report.json。
不追加本批搜索或恢复旧路线。完整OOF>82与独立新井验证仍未完成。
后续应先依据错误分解决定目标专用表示或状态/连续头分离，不能默认更宽或更多训练一定有效。
图谱说明由本report脚本的record维护；若重建主curate.py，须再运行本脚本--record再build_atlas.py。

## 实测误差分解与下一步边界
两个新模型从磁盘重载后，全部144665点评分预测逐元素复现；训练后填充不变性与全空窗口检查通过。
tcn: 真默认段通道分变化+0.600186，对TOTAL贡献+0.146349；真非默认段变化-3.094669，对TOTAL贡献-0.328530。4/16井提高，12井下降。
bigru: 真默认段通道分变化+0.704171，对TOTAL贡献+0.171705；真非默认段变化-3.439228，对TOTAL贡献-0.365109。4/16井提高，12井下降。
这说明本次最终动作在默认段更准，但连续标签段损失更多；不能据此独立归因于门控分类正确率，因为密度和效用解码同时影响最终动作。
下一步尚未启动：先在训练侧诊断SW的默认状态与非默认数值任务是否相互牵制，再以门控固定/可训练和独立数值表征作有限匹配对照；不直接扩大GRU/TCN层数或推断模型家族上限。
模型是随机权重从头学习，但固定Q来自先前训练井诊断，预处理/支持格也沿用训练侧元数据；不称整条pipeline完全不依赖此前训练产物。
当前保留原三私有模型80.274135337791；无新SW替换、无补折、无旧交付变更。
记录工具问题：最终报告首次因numpy.bool_ JSON序列化失败，修复仅作用于报告脚本并明确重跑；训练权重未改。curate.py源文件编辑被内容保护拒绝，原文件未改；本轮说明由report_independent_sw_scratch_v10.py --record可复现维护到curation.json，再build_atlas.py生成页面。




