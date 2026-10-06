# v154 进展

目标：验证POR正收益监督能否改善PERM有效头，最终完整OOF>82。

已完成：同容量/初始化PERMGaussian联合CDF监督实现、PLAN及固定POR151/SW104真实合并约定。

计划/未完成：显式启动并记录实际成绩、时间显存；恢复最佳/原第五144665覆盖/冻结完整性/实际三通道合并；结论与清理、图谱更新。

可选：有收益保留独立PERM设置，再研究SW；无收益不机械迁移或自动扫系数，不自动补折。

Verified explicit launch PID79880 and actual epoch2; source initial PERM89.344760174753,82720parameters, matched effective128head initialization. Partialbest89.344760174753, peak786.465MiB, noOOM/stderr. Two GPUjobs withPOR153; finalgain/replay/actual fixedPOR151+SW104 assembly pending. No otherfolds or more candidates.

Verified PERMlive snapshot epoch26, partialbest89.348440734238; originalv13689.387936 not exceeded, final conclusion pending. OverlapSW155, no morethan2GPUjobs.

V154 final PERM=89.405564856919, compactcontrolgain=+0.027170645775; actualTOTAL=80.612475417994, fivephases inclsource;strict808=False, no otherfolds. best/stop=31/49, seconds=561.870, peakMiB=786.465; all144665replay/frozen exact.

V154 final positive adopted in NEW CURRENT_RECIPE.json: PERM89.405564856919, delta samecapacityv136+0.017628761135; actual TOTAL80.612475417994 (+0.006170066397). Best31/stop49, exact144665/frozen replay, training561.870s/peak786.465MiB, wall/overlap timing.json. Defaultdelta-0.001962329657/numeric+0.055721972822, positive7/16wells: modest singlefold gain, not universal/default improvement. Preserve positivebest/last and oldcontrols. Source-counted5phases, strict808false/nootherfolds, completeOOF82unmet. ActiveSW155 original plannedPERM136 assembly remains unchanged; any latestPOR151/PERM154/SW155 assembly must be separately evaluated afterSWfinal.


## 冻结PERM失分诊断

[诊断](ERROR_SOURCES.md)基于已验证固定预测，不拟合/改动预测或输入。默认真值被预测成连续值贡献失分1.797072，连续真值预测成default贡献2.988269；合计4.785341/总失分10.594435约45.2%。连续真值低估贡献6.414631、高估2.382732；符号组与默认/连续组重叠，不相加。此为第五折描述性诊断，默认动作不等同单独gate分类错误，不用于后处理/行选择。下一步机制需真实训练验证，不宣称诊断本身有增益。
