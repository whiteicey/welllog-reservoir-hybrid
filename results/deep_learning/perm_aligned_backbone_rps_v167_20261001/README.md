# PERM v167：原训练折隐藏表示约束

## 探索重点

166私有body更新明显退步、14轮仍初始最佳。保持同活跃容量/初始化/80-14/两组LR，只加0.1原SOURCE隐藏表示归一化MSE，检验保留训练侧原表示是否减少灾难性偏移。既有132r1研究曾探索train-only hidden alignment，但不同架构不直接转移成绩或权重。

## 已探索内容与验证

实现完成，未训练无分数。teacher是初始化SOURCE骨干的冻结复制，明确清除复制的学生capture闭包再装独立teacher hook。只在model.training且原训练井窗口调用teacher，忽略其输出，仅隐藏128×2用于自身训练核心mask/Q加权。验证/推理不调用teacher；原SOURCE已计入phase1，不增加训练阶段，不加载166已训权重。student原输出/router参数冻结；自身body1e-5/head3e-4可训练。两项活跃容量相同1355414，teacher增加冻结参数/训练计算，不是同计算预算，记录单项与总墙钟、显存、预热和数据。固定POR151/SW104实际合并，五阶段。

## 收益

尚未证实；预定义对166最佳PERM89.344760174753的matched差值、对有效154的89.405564856919绝对差值，以及实际TOTAL。

## 不值得继续的点及证据

166已在14轮停于初始，不能继续无限无约束body LR网格；小适配/门控/低LR已有负配置。本项无收益不扫anchor系数，不泛化所有表示约束无效。


## 实际启动与训练验证

normalGPU/临时SYSTEM_REQUIRED，已核验进程存活；原SOURCE初始PERM89.344760174753，活跃参数1355414，配置80/14确认。实际快照：{"pid": 25140, "status": "complete", "epoch": null, "best": null, "best_epoch": 0, "bad": null}；峰值1072.6279296875MiB，训练梯度和冻结签名检查已通过。尚未证实最终增益。骨干路径有PyTorch warn_only线性插值反传非确定性提示，不是运行失败；保持真实记录，不声称训练轨迹严格确定。

V167 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=235.218, peakMiB=1072.628; all144665replay/frozen exact.

最终结论：v167 PERM=89.344760174753，相对v166 +0.000000000000；实际三头TOTAL=80.591193779236。best/stop=0/14，80/14；144665点磁盘恢复预测及冻结参数核验通过。未超过有效v154，不替换配方、不补折。不继续本轮0.1隐藏锚定系数扫描；结论仅限该设置/14轮早停。
