# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v155 在有效SW模型上增补联合CDF监督

探索重点：完成三个通道的独立监督迁移，保留SW104有效local模型的845913参数和Gaussian似然+效用1/SAM0.1，仅新增jointRPS1，不直接套用POR/PERM头结构。

已经探索与验证：实现完成，尚未启动、无成绩。同v104随机初始化完整私有SW网络，原96维TCN/局部65tap滤波/注意力/DropKey.4/输入井上下文保留，无旧SW训练权重加载。自己的原Gaussian混合CDF加默认atom，在255有限边界上与真实自身SW标签CDF作均方监督；DEFAULTS['SW']通过原log/训练median-scale标准化。仅共享Gaussian RPS计算函数，不加载PERM网络、标签或权重。

对照与收益：尚未证实。比较SW10474.121752828158，保持同容量/初始化/种子/SAM/预算/学习率/utility1，只增补RPS1。默认与连续标签都监督，仍用原默认/800动作/256区间argmax评分，不作后验拟合或阈值搜索。固定POR151/PERM136真实预测合并，最终才能报告TOTAL。source+POR113+POR151+PERM136+SW155共5阶段；不追加训练SW104后第六阶段。

不值得继续：以前SWStudent3、冻结source条件头迁移均负，不改变此处已有效的Gaussian/TCN/utility配方；POR自由区间/平滑CDF/通信/专家交互负，不叠加。POR jointRPS1正收益较小、PERM迁移尚在训练，SW迁移不能预先宣称有效。本批只有local/RPS1单项，无自动系数/结构网格。

约束：原9曲线/206+相对深度207输入/训练侧统计/Q50/原第五折/seed20261010；自身SW损失，全SW私有骨干可训练，其他目标不动。normal/max2且至多1个SW、80/18/min10/b64/64步/lr3e-4/cos80*64，1024窗口/原128监督核心、1024行RPS梯度checkpoint；原子完整last/显式恢复/进程级防睡眠。严格第五TOTAL>80.8且收益后补折，完整OOF>82未完成，旧交付/正控制保留。

待完成：POR153真实终止后利用空位显式启动，实测显存/训练与总墙钟/预热/并发；最佳磁盘恢复、144665覆盖/冻结完整性及真实三通道合并；结论、负权重SHA清理和PROJECTS/图谱更新。

Explicitlaunch PID53596 verified actual epoch2 and commandline. InitialSW71.197115081472/845913parameters same randomlocal104 start, partialbest71.197115081472, peak1728.449MiB, noOOM. First epoch32.045s with PERM154 overlap, not standalone equal-work speed evidence. Existing CUDAwarn_only attention/interpolation warnings observed; no 

监督迁移，保留SW104有效local模型的845913参数和Gaussian似然+效用1/SAM0.1，仅新增jointRPS1，不直接套用POR/PERM头结构。

已经探索与验证：实现完成，尚未启动、无成绩。同v104随机初始化完整私有SW网络，原96维TCN/局部65tap滤波/注意力/DropKey.4/输入井上下文保留，无旧SW训练权重加载。自己的原Gaussian混合CDF加默认atom，在255有限边界上与真实自身SW标签CDF作均方监督；DEFAULTS['SW']通过原log/训练median-scale标准化。仅共享Gaussian RPS计算函数，不加载PERM网络、标签或权重。

对照与收益：尚未证实。比较SW10474.121752828158，保持同容量/初始化/种子/SAM/预算/学习率/utility1，只增补RPS1。默认与连续标签都监督，仍用原默认/800动作/256区间argmax评分，不作后验拟合或阈值搜索。固定POR151/PERM136真实预测合并，最终才能报告TOTAL。source+POR113+POR151+PERM136+SW155共5阶段；不追加训练SW104后第六阶段。

不值得继续：以前SWStudent3、冻结source条件头迁移均负，不改变此处已有效的Gaussian/TCN/utility配方；POR自由区间/平滑CDF/通信/专家交互负，不叠加。POR jointRPS1正收益较小、PERM迁移尚在训练，SW迁移不能预先宣称有效。本批只有local/RPS1单项，无自动系数/结构网格。

约束：原9曲线/206+相对深度207输入/训练侧统计/Q50/原第五折/seed20261010；自身SW损失，全SW私有骨干可训练，其他目标不动。normal/max2且至多1个SW、80/18/min10/b64/64步/lr3e-4/cos80*64，1024窗口/原128监督核心、1024行RPS梯度checkpoint；原子完整last/显式恢复/进程级防睡眠。严格第五TOTAL>80.8且收益后补折，完整OOF>82未完成，旧交付/正控制保留。

待完成：POR153真实终止后利用空位显式启动，实测显存/训练与总墙钟/预热/并发；最佳磁盘恢复、144665覆盖/冻结完整性及真实三通道合并；结论、负权重SHA清理和PROJECTS/图谱更新。

Explicitlaunch PID53596 verified actual epoch2 and commandline. InitialSW71.197115081472/845913parameters same randomlocal104 start, partialbest71.197115081472, peak1728.449MiB, noOOM. First epoch32.045s with PERM154 overlap, not standalone equal-work speed evidence. Existing CUDAwarn_only attention/interpolation warnings observed; no bitwise training reproducibility claim. Initiallearning is not gain vs SW10474.121753; final replay/assembly pending. Onlytwo jobs, no otherfolds.

SWverifiedlive PID53596 snapshot epoch13, partialbest73.396265624043; initiallearning not gainvsSW10474.121753. PERM154positive finished, now oneGPUprocess; original fixedPERM136 assembly plan unchanged, no source/config mutation. Current combined positivePOR151/PERM154/SW10480.612475 recipe recorded separately.

V155 restored SW=73.670351522101, delta v104=-0.451401306057; actual TOTAL=80.448314894477, five phases/source included, strict808=False; no otherfolds. best/stop=20/21; exact144665/frozen replay.

V155 priority-stop final (INCOMPLETE): saved completeepoch21/best20 restoredSW73.670351522101, below retainedSW104by0.451401306057. User latest focusesPORaugmentation/MAINPERM, not full-budget convergence negative. Exact144665/frozen replay/actual original-planned assembly verified. Best/last cleanedwith SHAmanifest, no auto-resume. Timing records stop/replay/overlap; frozen copiedv104 trainingentry lacked TrainingAwake call, finalizer used it; no retrospective source mutation. POR157 explicitnewlaunch and PERM156 now2jobs; latest positive5phase recipe80.612475 untouched.


