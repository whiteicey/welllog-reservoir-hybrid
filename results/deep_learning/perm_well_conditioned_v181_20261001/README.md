# v181 输入整井条件化私有PERM

## 探索重点
保留177井均衡+SAM+TCN/token attention，增加可训练整井输入编码器，读取128个深度分层原206特征点，mean/std/max汇聚后FiLM调节stem、四token块、fuse。假设点/256窗上下文不足以区分井间输入分布偏移。
对照177PERM88.970890837572；输出零初始化FiLM保持初始预测函数和原骨干参数初始化，额外编码器参数增加，不冒称等活动容量。

## 已探索与验证
已实现、未训练无成绩。只复用46的类定义，不加载旧SW权重。输入bank来自既有固定深度采样，不读取标签；heldout井自己的输入bank仅该井推理用，不拟合验证总体参数。原数据归一化/Q/网格仍64训练井拟合。随机PERM、ownNLL/BCE/jointRPS1、SAM0.05、0.5行/0.5井、6e-4/80-14/min10/b64/64步/cos80*64保持；不加载177/179训练权重，不用teacher。

## 收益
尚未证实。必须最终恢复实际三头组合144665点；报告绝对v15489.405564856919和局部差值。额外结构及SAM全局梯度范数影响需如实说明，全方法仍5阶段。

## 不继续的点
不同时改teacher/容量宽度/loss，不扫FiLM幅度或输入采样点数。无增益关闭本配置，不泛化整井条件化；严格TOTAL>80.8有收益才补折，OOF>82未完成。

耗时待实测：单项/总工作墙钟、实际并发显存、data/预热区分，不计为完整5阶段成本。

v181已显式启动PID63344、normal GPU、临时SYSTEM_REQUIRED。当前训练快照{'status': 'running', 'pid': 63344, 'stage': 'perm_conditional', 'epoch': 33, 'best': 88.85628937090793, 'best_epoch': 27, 'bad': 6, 'peak_MiB': 826.8720703125}，active839761，非最终收益；max2任务/完整方法5阶段/未补折。
井bank只含原206输入，128深度采样、无label/井ID；额外条件编码器容量改变SAM全局梯度范数，不是等容量消融。初始函数保持，插值/attention反向warn_only警告如177，不声称位级训练复现。

V181 final PERM=88.856289370908, compactcontrolgain=-0.522104840236; actualTOTAL=80.420228997890, fivephases inclsource;strict808=False, no otherfolds. best/stop=27/41, seconds=639.703, peakMiB=826.872; all144665replay/frozen exact.

最终整井条件化无益：PERM88.856289370908，相对177 -0.114601466664；best/stop27/41，actualTOTAL80.420228997890。初始函数/预算/loss/SAM/采样/Q/解码一致，额外条件编码器容量改变SAM梯度范围、不是等容量消融；144665恢复/冻结验证通过。清理负权重留SHA；关闭本128点0.5FiLM设置，不扫点数/界限，不泛化全部整井上下文。原177保留，不补折。
