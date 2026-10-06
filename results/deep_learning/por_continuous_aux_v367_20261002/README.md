# POR连续值辅助读出：实现准备

当前状态：训练入口与小模块已准备，语法检查通过；尚未训练、没有成绩，收益尚未证实。当前五种子完整八阶段批次继续，本实验不排队、不自动启动。

## 探索重点

让已有效的341新增编码器在学习完整分布的同时，获得真实连续值的位置监督。保留原151预测函数、341曲线/深度交互与主分布损失，在同一个POR追加阶段内增加49参数的训练期标量读出。整体仍八阶段，三个头各至多新增一次；不从已经训练的341再微调。

## 已探索的内容与验证

前置证据：[阶段分析](../stage_gain_analysis_v366_20261002/STAGE_REMOVAL_DECISION.md)。原及前两个新模型POR341通道分提高而连续值分下降；第三个连续值也提高，说明该现象不是每个种子都必然出现。三个新SW104的通道分提高而连续值分下降。既有POR151连续值收益保留。

实现读取已有341结构代码，从原151父权重初始化新的编码器/主读出，不加载341/338/339已训练权重。aux0和aux01各有同样的Linear(48,1)辅助读出，共49参数，辅助初始化位于独立CPU RNG作用域，不改变原结构初始化或训练随机流。辅助读取主分布读出前的48维融合特征，只在训练时计算；推理仍使用原完整分布与原官方效用解码。

辅助目标使用原训练侧变换后的POR z（labels[...,0]），仅非默认、有效训练点参与。定义

`L_cont = sum(Q * 1_nondefault * smooth_l1(aux(h), z; beta=1)) / clamp(sum(Q * 1_nondefault), 1e-12)`。

主损失完全复用341。总损失为`L_341 + lambda * L_cont * mean(Q) / 3`，lambda预定义0/0.1；此缩放匹配已有直接CE辅助项的全局Q与通道归一化，没有用验证标签拟合系数。没有连续点时辅助项为零，不把默认值当连续标签。标量不是Cauchy分布的期望，也不参与预测、融合或选动作。

两组新增编码器和主分布读出均训练；辅助参数与主读出置于同一优化器组。aux0辅助梯度为零，只是同结构的无辅助监督对照；aux01辅助梯度用于训练。原151整个函数固定。训练中记录主损失、归一化辅助损失、系数与各参数组梯度，不能把辅助loss下降直接算成官方收益。

## 实际收益

尚未证实。尚无训练、初始整折模型评价或成绩。本批未来比较同容量aux01−aux0、相对原POR78.04643071793232及整体80.70612972323806差值，同时报告默认/连续分。其余PERM336、SW351按原八阶段固定，不做跨种子择头。

## 不值得继续的点与待做

不增加新修补链路，不把辅助标量用作推理捷径，不删除有效151，不根据第五折逐点选择预测，不做外部权重/教师/伪标签/测试标签。若这个固定0.1方案无收益，只关闭该配置，不宣称所有多任务监督无效。

待完成当前五种子比较后，再决定本实现是否启动；准备阶段仅语法检查，不运行GPU预检或forward。每候选独立优化器与完整原子last-state，保存RNG/调度/最佳状态，停止后显式恢复。只允许固定第五折，未超过80.8且有收益不得补折。最终目标原完整五折OOF>82尚未达成。

V367 terminal CPU recorder prepared/syntax compiled, not executed: fixed original336/351 with checksum/report/prediction checks, all144665 actual columns, DEFAULT/CONT, same-capacity configuration match after excluding only auxiliary coefficient/stage, cached immutable outputs, no independent model disk reload or complete-recipe speed claim. Unlike original341 recorder, this uses8stage fixed heads, not6stage154/104. No GPU preflight or training launch.

One-shot manual post-seed launch entry prepared/syntax checked, not run. Initial two routes aux01/local, same-capacity aux0/point controls still held. Guard requires all5complete/current-controller exit/no existing GPUjobs/no stop marker; if an8stage seed passes80.8, promotion takes priority. No watcher, background queue, automatic retry/resume. Current launch_enabled remainsfalse and seed training unchanged.

Latest explicit plan: route held, launch entry retired unexecuted; V362 completes, then V370 original+seed13 remaining-fold CV. No new optimization launch.
