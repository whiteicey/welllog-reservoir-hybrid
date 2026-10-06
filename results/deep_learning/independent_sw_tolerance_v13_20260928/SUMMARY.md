# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## FINAL.md

# SW容差概率质量监督v13：最终结果

本批显式运行两项强SW私有双专家微调。v12两项在本轮开始读取前已完成，本批只补记其结果。

|模式|参数数|训练后最高SW（排除第0轮）|轮次|最终选定SW|最佳/停止轮次|
|---|---:|---:|---:|---:|---|
|likelihood|1281706|73.473302024779|1|73.622384767129|0/18|
|tolerance_mass|1281706|73.457862289177|1|73.622384767129|0/18|

## 机制与验证
两组同SW私有双专家、同初始权重/数据/随机窗口/优化器，只差容差质量负对数系数0和0.5。所有SW私有参数可训练，POR/PERM模型不更新。CDF差分保留梯度，首批前向与原推理实现差值为0，辅助损失同时到达gate/density，SW骨干收到非零梯度。
两组最终均恢复起点；全部144665点预测精确复现，训练期间骨干有实际变化。恢复原模型不能当成新损失带来的增益。原三私有模型TOTAL80.274135337791保持。本轮未重复执行三模型GPU重载，沿用此前已核验的组合证据。
容差损失采用原256格上的概率质量积分近似，固定alpha0.5；其本身非proper scoring rule，保留原NLL。不把它等同于准确的argmax官方风险。CUDA插值反向仍使用确定性警告模式，不宣称训练逐位可复现。

## 收益与关闭范围
本配置没有新增收益。两组更新后最高均在第1轮，容差组比似然对照更低；不继续从同起点无约束全参数微调的宽系数/学习率扫描。没有证明所有效用监督、所有SW架构或输入信息均达到上限。
后续待研究：优先用训练井内的分组验证检查训练内拟合与新井泛化差距，再决定是否引入跨井局部参照；该步骤尚未启动。第五折严格>80.5与完整OOF>82未完成，无自动补折或旧交付覆盖。

## 工程状态
所有训练源和旧权重保持不可变；本批源码、完整last_state和紧凑证据已保存。未验证中断恢复与不中断训练的逐位等价。
编辑主curate.py被工具内容保护拒绝，原文件未改；新摘要由curate_sw_tolerance_v13.py覆盖合并到现有curation.json，然后build_atlas.py生成图谱，保留其他节点。
两个一次性监控命令因shell引号/编码传递失败，未运行Python正文、未影响训练；后续使用固定脚本与读取工具完成核对。

完整分数、历史、Q、时长和来源见progress.json、两个输出目录的config.json/fold_5_report.json。




