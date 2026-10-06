# PERM v166：私有骨干与有效RPS读出协同

## 探索重点

冻结预测错误诊断显示默认/连续动作错配占45.2%失分，连续低估为主；小头/门控/局部适配多项未改善。保持有效Gaussian128/NLL/BCE/RPS1，让原两专家私有编码表示LR1e-5参与训练，而不继续仅改输出参数。

## 已探索内容与验证

实现完成，准备未训练、无分数。移除冻结body的外层no_grad，capture保留梯度；原main/aux输出参数和路由参数冻结，仅原body（包括训练侧输入校准）及新头更新。骨干维持eval关闭dropout但允许梯度，固定原推理函数起点；旧gate/scale参数冻结，其函数可以随表示改变。源SOURCE来自原64训练井，不加载已训154头，私有克隆仅预测PERM，不更新独立POR/SW。训练只用自身标签和原Q50；80/14、head3e-4/body1e-5、原解码不改；五阶段含来源。活跃范围/历史18轮预算不同，不能称严格匹配消融；报告实际绝对分与初始增益。入场至少6GiB空显存，不启动第三GPU任务。

## 收益

尚未证实；预定义PERM正控制89.405564856919，最终固定POR151/SW104实际组合验证，不用其他通道真值填预测。

## 不值得继续的点及证据

尺度/门控/小适配/低LR14预算配置均未超过初始；关闭具体配置，不泛化网络类别。此项无收益不继续全骨干LR网格，不恢复旧模型队列。

V166 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=249.026, peakMiB=1050.815; all144665replay/frozen exact.


Completed perm_conditional: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.591193779236; negative best/last deleted with SHA manifest after exact replay. Training=249.026s, launch/report wall=255.981s, peak=1050.815MiB; includes data/initial/final eval, other jobs overlap, not fullfivephase wall. Comparisons retain immutable configs/results/predictions.
