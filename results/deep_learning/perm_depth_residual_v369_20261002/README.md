# PERM骨干局部深度残差

## 探索重点
原336新增可训练分支是逐点稀疏重采样、分组有理响应及低秩交互；考虑是否缺少显式邻点信息。保留这些有效结构，在stem交互后、两次原曲线selector之前，新增48→12→48的有界残差。沿12维潜变量读取[-8,-4,0,4,8]点，按有效掩码归一化，不跨井/无效窗口。up投影零初始化，初始预测函数与原154一致；新增支路与原新编码器、主分布读出联合训练，绝非只训练头。
point与local共享1260新参数、初始化和训练条件，仅邻点连通不同。point将5个tap均映射当前点，核权重功能冗余，因此这是连通消融，不是同等有效容量证明。局部跨度是行位置，原相对深度输入保留，不宣称固定物理距离。

## 已探索内容与验证
模型、有限训练入口和终点CPU三头组装将准备并做语法检查；尚未执行forward、数值预检或训练，没有分数。历史336权重不载入，只复用原154正收益父模型及有效336结构，新优化器。固定POR341/SW351，替换PERM最后追加阶段，总8阶段、各头最多1追加阶段。当前5种子批次原样运行，入口保持关闭，不排队。
旧全双专家adaptive dilation第五折最佳79.997751、比当时原双专家低0.253143，已关闭；旧SW116膨胀局部滤波亦未胜原控制。这些负结果不能证明本次PERM有效逐点分支的邻点残差无效，但不做大网络/多dilation扫描。

## 实际收益
尚未证实。主要比较local-point实际全PERM官方分；同时对原336 PERM89.518389398791和原组合TOTAL80.706129723238报告绝对差值。共同初始化即原154 PERM89.405564856919，初始无收益。CONT/DEFAULT使用同一个全头选出的模型，不按真实子集选checkpoint。只有实际三头组合严格>80.8且有明确收益后才考虑补其他四折；完整OOF>82未完成。

## 不值得继续的点及证据
尚无本试验负结果；只预定义point/local两项，不堆叠历史失败模块、不扫kernel/latent宽度、不加载训练后336再多算一阶段。若local对point无增益，关闭这组偏移/支路/loss条件，不能外推所有局部上下文无效。准备状态不当作已训练收益。

## 耗时与未完成
尚未训练，显存和单阶段/同等工作总墙钟未测。将记录串行/并发、预处理与预热；阶段日志不能冒充完整8阶段速度。待最后种子完整8阶段结束后显式选择两路线启动安排；本文件不启动队列。

V369 complete model/trainer/terminal CPU assembly entries prepared and syntax checked; no forward/numeric preflight/training/score. Same1260new/38646active parameters point/local connectivity pair, fresh336design/old154only, jointly train new backbone and categorical readout; fixed originalPOR341/SW351, one replacement PERM phase/8total. Point effective kernel redundant, not equal-function-capacity claim. Launch disabled/no queue until finite five-seed batch completes.

One-shot manual post-seed launch entry prepared/syntax checked, not run. Initial two routes aux01/local, same-capacity aux0/point controls still held. Guard requires all5complete/current-controller exit/no existing GPUjobs/no stop marker; if an8stage seed passes80.8, promotion takes priority. No watcher, background queue, automatic retry/resume. Current launch_enabled remainsfalse and seed training unchanged.

Latest explicit plan: route held, launch entry retired unexecuted; V362 completes, then V370 original+seed13 remaining-fold CV. No new optimization launch.
