# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# PERM v162：私有局部表示适配

## 探索重点

在原两专家128隐藏表示和原预测头之间，加入各自128→32→depthwise7→128适配器，零初始化输出。新残差有界于逐点隐藏RMS的0.1；输入仍原九曲线。适配器改变真实预测及新128位置/权重头的输入，检验表示层限制，而非再加尺度/宽度。

## 已探索内容与验证

实现完成，未启动无分数。原骨干所有参数冻结，只有新适配器1e-4和读出3e-4训练；原gate参数冻结但其函数可随适配表示改变。hook在冻结body的no_grad内部仅开启新模块梯度，capture保留其梯度，Evaluator评估保持no_grad。零初始与原SOURCE函数一致，不加载已训154。新增17216参数，活跃99936，容量与历史控制不同，80/14对历史80/18也非完全匹配。结束后实际固定POR151/SW104合并。

## 收益

尚未证实；预定义Gaussian/RPS正控制89.405564856919，同时报告实际TOTAL，五阶段含来源。

## 不值得继续的点及证据

尺度159与Laplace158未改善初始，关闭具体配置；gate161仍训练不能下结论。此项不无证据扩大适配器或恢复历史骨干队列。


## 实际启动与首轮验证

PID59780已核验存活，normal GPU、临时SYSTEM_REQUIRED、与POR160双并发；初始PERM89.344760174753与原函数一致，99936活跃参数。已完成第2轮，最佳仍初始，第2轮89.313387736773，peak786.465MiB，stderr无错误。训练的两组正梯度和冻结签名检查已通过，80/14配置确认。尚未证实收益，待最佳磁盘恢复/实际三通道组合；图谱已更新。


## 早停预算适用范围澄清

历史正控制154前15轮未超过初始，第16轮首次改善；pat14会截断这条已知有效轨迹。14轮停于初始只支持当前配置/预算未取得收益，不是所有同类机制无效证据。此诊断不放宽用户pat14，不自动恢复。

V162 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=212.983, peakMiB=786.465; all144665replay/frozen exact.


## 最终结论与关闭

80/14预算下PERM89.344760174753，无初始改善；相对历史有效154−0.060804682166，best0/stop14。实际TOTAL80.591193779236，144665行最佳重载与冻结精确，两梯度组为正。当前配置/预算无收益，历史154需16轮首改善，不能泛化全部适配器无效。关闭本配置，负权重SHA清单清理，其他证据保留。训练212.983秒，启动至报告墙钟219.845秒，数据/初始/最终评估计入，与POR160并行；不是完整五阶段耗时。




