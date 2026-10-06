# Experiment rules

## 2026-09-28 latest scoped resumption
User explicitly requests further optimization of independent POR/PERM/SW output-head structures on the approximately80.22 dual-expert model. This authorizes the new dual8222 head study only, superseding older stop directions for these new runs. Follow current root/pure_dl constraints; do not restart dormant ABD tasks or fullCV. Preserve original sources and maintain PROJECTS.md with completed/planned/uncompleted work and post-goal options.

Follow root AGENTS.md. Keep PROJECTS.md current. Historical files preserve original methods/configuration; do not launch old queues or batch scripts automatically. Current user directive: finish last posterior-router trial then stop training. Use documented entrypoints and read status before any execution.


## Explicit user resumption
The user explicitly requests continuing the80.2508937748 posterior-router direction. This supersedes the previous stop-after-last-run instruction for newly authorized exploration. Preserve fixedfifth strict>80.4, max80/patience18/min10/bs64/64steps,normalGPU,max2jobs,originalsplits/seeds/scoring,no testlabels. Currenttrial local+posterior router only; deliveries immutable.


## 2026-09-28 最新授权：通道独立模型，无共享骨干
用户允许不同模型分别预测与训练POR/PERM/SW，无须共享backbone。覆盖此前仅POR/骨干冻结限制。本批六项：各目标single完整视图和dual双视图私有模型；全骨干、当前目标门控/密度和私有路由可训练，各自优化器/早停/完整checkpoint，最后固定三模型共同验证。初始化来自原训练折预训练，不声称从零。原9曲线/分折/评分/80-18/max2保持，新低学习率全骨干微调配置须记录。无额外数据/自动补折/旧任务或交付覆盖，维护PROJECTS与图谱。


## 2026-09-28 继续推进：独立SW分布模型v9
v8六项与三模型恢复完成后，本批预定义ce/ce_utility两项：SW私有冻结双专家先验+直接读取原206特征的可训练TCN概率格修正；只差训练容差效用项0/0.1，其余原数据/分折/评分/Q/80-18/max2保持。仅新TCN训练，POR/PERM独立且不更新，不把当前设计扩成自动搜索或补折。维护PROJECTS/README/EXPLORATIONS/图谱与真实失败记录。


## 2026-09-28 本轮继续优化：独立SW从头训练v10
本轮预定义tcn/bigru两项，均从原207维输入直接学习SW门控与256格连续分布；不加载旧网络或预测，复用原训练侧预处理/网格/Q以保持评分边界。POR/PERM保留独立模型。原第五折、seed20261010、80/18/min10、batch64/64步、最多2GPU；无旧任务恢复、无自动搜索/补折。每批维护PROJECTS/README/EXPLORATIONS/图谱，严格区分冷启动改善与超越旧SW。


## 2026-09-28 当前继续：SW状态/数值表征解耦v11
本批预定义shared/split两个匹配候选，各有两套独立48维TCN、同初始化和92273参数；仅改变两个头读共享均值还是各自编码器。训练与解码相同，按整个SW官方分选checkpoint。原输入/分折/Q/80-18/max2保持，POR/PERM独立且不变；训练侧梯度诊断不读验证标签，无自动搜索/旧任务/补折。维护PROJECTS、README与图谱。


## 2026-09-28 当前继续：SW联合动作效用v12
本批likelihood/joint_utility两个匹配实验，v11shared第13轮同起点，92273参数同结构；全SW参数训练，损失分别L_like与L_like+0.5*L_joint，softaction温度0.05。原动作/分折/Q/80-18/max2不改，新lr1e-4优化器重置；POR/PERM独立不变，无旧任务、补折或交付覆盖。明确默认与连续动作都参与联合项；维护PROJECTS/README/图谱，代理损失不等同精确官方argmax。
