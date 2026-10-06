# v83 SW效用监督人口平衡

## 探索重点与同条件对照
v82系数1当前原第五TOTAL80.382093291214/SW73.793310952488，优于.5仅TOTAL+0.001956689615；默认SW+0.050181618258，数值-0.096831032031，6井胜/10负。本批检验联合效用的人口分配，不宣称已证实默认占比是退步原因。
预定义balanced/numeric75两项，默认/数值份额.5/.5和.25/.75；只改变联合效用点权重。先用原64训练井SW原点标签计数，权重=份额/训练人口比例，使训练人口平均权重1。没有第五验证人口拟合，原Q50另乘且不改，BCE/NLL不加人口权重，总辅助系数1、温度.05固定。比较已完成全人口系数1；既有系数0和.5仅历史机制证据。
同839481参数、v56第11轮训练折内骨干/FiLM加原64井输出先验重置，全SW网络训练；不加载v82训练后权重。原9曲线206/输入井bank/分折/seed20261010/1024上下文128中心/40%DropKey/原256支持800动作+默认官方argmax保持。固定POR/PERM并实际三列评分；没有按人口/行挑选模型。

## 已探索与验证
已实现，启动无新分数。直接训练，第二batch记录门控/密度/骨干真实梯度与训练人口配置，不做另一次数值预检或独立审查。实际初始状态/全点预测须与全人口1对照一致；新训练AdamW1e-4/wd.01/clip2/cos80*64至1e-5，max80/pat18/min10/b64/64步。原子last含优化器/scheduler/RNG/停止状态，显式恢复。原训练侧logSW反变换及有限CDF尾代码不改。
最终报告对全人口1的总体/默认/数值及各井差，以及对旧无效用的历史机制差；正候选144665点磁盘重建后实际三列评价。只有第五开发折，不把初步/消融写成完整五折收益。

## 实际收益、不继续与未完成
收益尚未证实；若两项都无匹配收益，结束此份额网格并换结构，不继续临时调份额。即使数值点改善也必须报告默认和总体代价，固定一个完整SW模型。正模型保留，无新增亮点checkpoint完成后按清单删除，源/config/日志/紧凑证据保留；旧.5/1及交付不改。
最多2GPU且资源足够才显式双并发，不开队列或旧任务/其他折。单项耗时和同工作量总墙钟/并发/显存分开，加载预热训练验证保存与启动间隔计入，准备和图谱收尾另计。维护PROJECTS目标/已做/待做/未做/目标后工作、独立README/EXPLORATIONS和图谱。
严格原第五>80.5且明确收益再考虑其余四折；完整原分组OOF>82仍未完成。目标后独立井确认及新目录交付。


## v83 balanced显式启动
{"pid": 47720, "mode": "balanced", "started_at": "2026-09-30T10:58:58.125342+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_population_utility_v83.py", "--mode", "balanced", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_population_utility_v83_SW_balanced_fold5"], "automatic_queue": false}


## v83 numeric75显式启动
{"pid": 7752, "mode": "numeric75", "started_at": "2026-09-30T10:59:52.736332+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_population_utility_v83.py", "--mode", "numeric75", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_population_utility_v83_SW_numeric75_fold5"], "automatic_queue": false}


## v83最终结果
两项完成，当前独立通道原第五TOTAL=80.382093291214，相对.5组合+0.000000000000；系数网格见sw_population_utility_v83_20260930/FINAL.md，采用=None，保留=[]。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。

## v83收尾、边界与下一步
两项均无整体收益，当前独立路线第五TOTAL80.382093291214不变。balanced最佳37/停止38，数值SW+0.487570但默认-0.468565；numeric75最佳14/停止32，数值+0.141598但默认-0.749867。均衡组在早先第31轮观察后仍提高到37轮，停止理由引用旧快照；实际请求状态和最终恢复为73.614756，第38轮提前结束，不能称80轮完整负结果。numeric75自然pat18结束，拟停止时PID已不存在，没有写请求或重启。四份负权重已删除38096908字节，旧正模型/交付保留。下一批计划固定原默认网络、训练私有数值分布，两个似然/均衡效用对照；见NEXT.md，尚未实现或启动。GPU实际峰并发2、各1870.504MiB；首启动到验证502.257秒，到文档/图谱及子群解释收尾1060.358秒，验证后558.1秒，范围见artifact_finalize.json。图谱622实验/83关闭路线，分数读回一致；无活跃v83进程/其他折。第五严格>80.5及完整OOF>82仍未完成。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
