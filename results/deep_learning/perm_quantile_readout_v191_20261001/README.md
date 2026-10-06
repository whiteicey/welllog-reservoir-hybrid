# v191 SOURCE参考单调IQF32读出

## 探索重点
借鉴概率时序预测I(S)QF单段特例，将Gaussian16的位置/权重读出替换为32递增分位节点与指数尾。冻结SOURCE默认门，保留128维trunk、82720活动参数、NLL+BCE、conditionalRPS1、效用0.1、SAM0.05。主对照174 Gaussian PERM89.364985921273。表示转换可能改变初始官方分，必须独立报告。

## 已探索与验证
已实现，未训练、无分数。参考分位只由同输入窗口的冻结SOURCE条件Gaussian分布求得；原255有限边缘加动态8sigma端点夹逼、4次局部二分，tau从0.02到0.98共32点。参考增量至少0.002，新首点增加tanh修正，31正增量为0.001+(参考增量-0.001)*exp(tanh修正)。尾斜率绑定首末内段，CDF/PDF闭式；内部PDF可以跳变，尾接点连续。只保留本次有界方案，不扩大节点/epsilon网格。

NLL沿用Gaussian对照的标准化log10目标坐标；物理PDF还需除以scale*y*ln10，静态Jacobian不影响参数梯度。不得直接裁PDF破坏归一。原255边缘RPS、256格概率质量和800动作完全保留。最终必须恢复最佳、覆盖144665点并真实三列合并。

同SAM步缓存冻结源hidden/gate/参考节点，第二遍只重算新头；下一步和train/eval切换清空缓存。源参数不更新、源eval；参考是模型内的确定性函数，不是训练伪标签/teacher监督，不新增预训练阶段。缓存不进last；从轮边界恢复时按原输入重建。反解和首次/最终预测成本必须计入，不声称与174等计算或更快。

## 收益
尚未证实。与Gaussian仅同容量、来源、优化、数据和损失系数；有限反解、分段线性和指数尾改变初始CDF/PDF，不能称同初始函数消融。报告SOURCE89.344760174753到本候选初始的变化、自身初始到最终的训练变化、相对174最终差，以及154绝对89.405564856919对照。进一步匹配IQF初始化的损失消融，只有本表示值得继续才做。

## 不继续的点
不同时更换CRPS、默认门、骨干或动作分辨率，不扫尾族、节点数和epsilon；无益则关闭这次有界32分位替换，不泛化全部分位模型。原九曲线/206特征、64训练井/Q50、原第五折/seed、80最大轮次/14早停/min10/bs64/64步不变，最多2GPU。只替换PERM训练阶段，全方法5阶段含SOURCE；严格TOTAL>80.8且有益后才补其他折，完整OOF>82未完成。

[原论文](https://arxiv.org/abs/2111.06581)、[官方ISQF源码](https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/distributions/isqf.py)。尾斜率绑定、有界SOURCE参考和缓存是本任务改造；不使用外部权重，论文收益不是当前实测收益。

耗时待实测：分别记录单项及同工作量总墙钟、并发/显存、数据与首末预测是否计入；无独立预热，不当完整五阶段成本。


v191 iqf explicitly launchedPID67908, normalGPU/guard max2/temporary SYSTEM_REQUIREDTrue verified, active82720, initialPERM89.345246688523; current{'status': 'running', 'pid': 67908, 'stage': 'iqf', 'epoch': 5, 'best': 89.34524668852337, 'best_epoch': 0, 'bad': 5, 'peak_MiB': 786.46533203125} is training snapshot, not final mechanism gain. SOURCE original89.344760174753 initial representation delta+0.000486513770, not learned improvement. Inherited Gaussian diagnostic keys describe old SOURCE/old parameterization; actual new output isIQF32, no new component logweight correction. Target-space exponential tails have finite z moments, physical PERM moments not asserted. Same-step cache reset before each optimizer step and eval.

V191 iqf final PERM=89.345246688523, compactcontrolgain=-0.033147522620; actualTOTAL=80.591364059055, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=495.978, peakMiB=786.465; all144665replay/frozen exact.

FINAL IQF32 PERM89.345246688523, best0/stop14, learned gain0; minus Gaussian174 -0.019739232749. Initial representation delta +0.000486513770 is not learning. Actual TOTAL80.591364059055,144665 exact reload/frozen. Close this 32-node piecewise-linear/exponential-tail SOURCE-conditioned representation with existing NLL/RPS/utility/SAM only; not all quantile functions. Training495.978274s/peak786.465332MiB, no full5phase time.

Finalized negatives v190 both arms/v191 IQF: six best/last checkpoints deleted with SHA manifests after exact replay and actual assembly. Preserve positive185/174/current154 and all active192 outputs. v190 paired total launch-to-report wall1330.530043s includes inter-arm gap and overlap other studies, not dedicated benchmark/full5phase time.
