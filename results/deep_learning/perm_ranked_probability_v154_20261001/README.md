# v154 向PERM有效读出迁移联合分布监督

探索重点：将POR上同容量实证+0.014277872142的jointRPS监督迁移PERM，不预设跨通道收益。保留v136有效128位置/权重头和82720参数、同初始化、原Gaussian族；仅损失NLL+1*jointRPS改变。

已经探索与验证：实现完成，尚未启动、无成绩。原PERM255有限边界上联合CDF=g*I(default_z<=edge)+(1-g)*Gaussian mixtureCDF，与真实自身标签CDF均方误差，Q50加权、原/3归一化。默认物理值直接读取src.utils.DEFAULTS['PERM']，按原log10/训练median-scale标准化；不是套用POR的log1p/默认值或Student/Cauchy族。1024行gradient checkpoint分块控制显存，不改动作、边界、分辨率或阈值。

收益：尚未证实。比较v136PERM89.387936095784；严格同容量/初始化/种子/预算/学习率，最终恢复后才判断增益。固定POR15177.926380760724与SW10474.121752828158真实预测合并，完整三通道TOTAL才用于80.8门槛，不以其他目标真值填充。暂不加载v136训练头，替换PERM阶段，保持source+POR113+POR151+PERM154+SW104共5阶段。

不值得继续：PERM256扩容负，SW相同条件头直接迁移负，因此不假定所有头都跟POR需要同宽度/族；POR自由区间/CDF形状/分量通信/专家交互负配置不叠加。POR系数2尚在训练，不能称最终负结果。本批仅预定义PERM系数1，不扩自动目标/系数网格。

约束：原9曲线/206+63幅值/64训练井统计/Q50/原第五折/seed20261010；原Gaussian/冻结旧骨干、门控、尺度，只有PERM新头自身损失训练；normal/max2、80/18/min10/b64/64步/lr3e-4/cos80*64、完整原子last与显式恢复/进程级防睡眠。第五严格TOTAL>80.8且有收益后补折，完整OOF>82未完成，旧交付不改。

待完成：显式启动，记录实际训练/单项与并发墙钟/显存/预热；最佳磁盘恢复/144665覆盖/冻结完整性/实际合并、通道与TOTAL增益，负权重留SHA清理并更新PROJECTS/图谱。

Verified explicit launch PID79880 and actual epoch2; source initial PERM89.344760174753,82720parameters, matched effective128head initialization. Partialbest89.344760174753, peak786.465MiB, noOOM/stderr. Two GPUjobs withPOR153; finalgain/replay/actual fixedPOR151+SW104 assembly pending. No otherfolds or more candidates.

Verified PERMlive snapshot epoch26, partialbest89.348440734238; originalv13689.387936 not exceeded, final conclusion pending. OverlapSW155, no morethan2GPUjobs.

V154 final PERM=89.405564856919, compactcontrolgain=+0.027170645775; actualTOTAL=80.612475417994, fivephases inclsource;strict808=False, no otherfolds. best/stop=31/49, seconds=561.870, peakMiB=786.465; all144665replay/frozen exact.

V154 final positive adopted in NEW CURRENT_RECIPE.json: PERM89.405564856919, delta samecapacityv136+0.017628761135; actual TOTAL80.612475417994 (+0.006170066397). Best31/stop49, exact144665/frozen replay, training561.870s/peak786.465MiB, wall/overlap timing.json. Defaultdelta-0.001962329657/numeric+0.055721972822, positive7/16wells: modest singlefold gain, not universal/default improvement. Preserve positivebest/last and oldcontrols. Source-counted5phases, strict808false/nootherfolds, completeOOF82unmet. ActiveSW155 original plannedPERM136 assembly remains unchanged; any latestPOR151/PERM154/SW155 assembly must be separately evaluated afterSWfinal.


## 冻结PERM失分诊断

[诊断](ERROR_SOURCES.md)基于已验证固定预测，不拟合/改动预测或输入。默认真值被预测成连续值贡献失分1.797072，连续真值预测成default贡献2.988269；合计4.785341/总失分10.594435约45.2%。连续真值低估贡献6.414631、高估2.382732；符号组与默认/连续组重叠，不相加。此为第五折描述性诊断，默认动作不等同单独gate分类错误，不用于后处理/行选择。下一步机制需真实训练验证，不宣称诊断本身有增益。
