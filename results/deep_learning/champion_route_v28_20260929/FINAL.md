# v28：内部专家路由更新范围对照

完成：2026-09-29T09:35:05.077458+08:00

原配方TOTAL=80.300730612039；实际组合TOTAL=80.324397278674；增量=+0.023666666635。

## 六项实测结果

|目标|模式|新增参数|训练后最高通道分|最终通道分|最佳/停止轮次|
|---|---|---:|---:|---:|---|
|POR|joint|10273|77.522023947|77.522023947|10/28|
|POR|conditional|10273|77.535990268|77.535990268|8/26|
|PERM|joint|10273|89.382439190|89.382439190|21/39|
|PERM|conditional|10273|89.370385497|89.370385497|30/48|
|SW|joint|10273|73.651700983|73.651700983|3/21|
|SW|conditional|10273|73.656418520|73.656418520|4/22|

第0轮回退只表示保留当前源模型；不计新增训练收益。各模式同结构同初始权重，但默认分布是否参与更新会改变梯度，比较的是这项范围选择。

## 实际采用与收益分解

POR：采用conditional，通道77.535990267625，新增推理参数10273。
默认/非默认贡献：{"default": {"n": 94357, "reference": 97.24980658562693, "candidate": 97.24768697605901, "channel_delta": -0.0021196095679130167, "TOTAL_contribution": -0.0004147513220189441}, "numeric": {"n": 50308, "reference": 40.516197715561695, "candidate": 40.56500023984175, "channel_delta": 0.04880252428005605, "TOTAL_contribution": 0.005091398869417744}}
PERM：采用joint，通道89.382439190068，新增推理参数10273。
默认/非默认贡献：{"default": {"n": 95533, "reference": 97.13456081203742, "candidate": 97.22740216894043, "channel_delta": 0.09284135690300843, "TOTAL_contribution": 0.021458505320259127}, "numeric": {"n": 49132, "reference": 74.21183923771699, "candidate": 74.12857514503447, "channel_delta": -0.08326409268251211, "TOTAL_contribution": -0.009897528708305497}}
SW：采用conditional，通道73.656418519609，新增推理参数10273。
默认/非默认贡献：{"default": {"n": 100786, "reference": 98.30562115037564, "candidate": 98.30064597136351, "channel_delta": -0.004975179012134845, "TOTAL_contribution": -0.0012131471826008907}, "numeric": {"n": 43879, "reference": 16.969502868432727, "candidate": 17.050910008646554, "channel_delta": 0.08140714021382678, "TOTAL_contribution": 0.008642189658209495}}

组合14井升、2井降；未校正选择的探索性区间[0.011283674717372546, 0.039267992224015745]。
同一第五折已反复开发，六项选模，无独立种子或新划分确认；微小分数变化不自动触发补折。

## 结构与验证

每目标只增加319→32→1的10273参数小头，读取两个128维隐藏向量和63幅值。零输出起点，log(a1/a2)增量限制±0.25。joint同时改变最终默认概率与连续条件专家占比；conditional只改变后者。各专家输出参数、旧修正和原路由全部固定。
六项初始全144665点精确，实际训练有新头梯度且旧状态固定。每项选定预测训练内全量恢复；六份更新后检查点另做少量训练窗口范围检查。conditional最终门控精确固定，混合分量位置/尺度不变。
新采纳目标独立磁盘重建并全量推理，未变目标复用已验证预测，实际三列合并调用原metric；新重载目标：['POR', 'PERM', 'SW']。
前向调用复用v27的原v26加载器，不执行v27负候选头；外部B/D投票、测试标签和后处理阈值拟合均未使用。

## 不继续与未执行

冷启动、扩容、重复后验条件头、其他折和旧任务均未启动。负路由候选不继续扫幅度、宽度或种子；结论仅限当前头与训练协议。当前原五折>82仍未完成，旧交付保持不变。
下一步只依据采用方案和逐子集结果讨论，不自动叠加同通道多个路由。

## 记录

METHOD.md、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json；源码champion_route_v28.py和train_champion_route_v28.py。


## Final closeout
All six trials completed; all three adopted targets independently reloaded and scored together. Atlas updated through curate_champion_route_v28.py; original curate.py unchanged after editing protection. No new training launched in closeout. Current jobs: []. See execution_notes.json.
