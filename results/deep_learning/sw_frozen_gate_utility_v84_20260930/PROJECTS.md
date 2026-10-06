# v84 固定默认网络与私有SW数值分布

## 探索重点与固定对照
当前保留全人口系数1第五TOTAL80.382093291214/SW73.793310952488。v83均衡效用数值SW+0.487570但默认-0.468565、总体负；这可能含门控偏移，不证明密度独立改善。本批固定原默认预测网络，检验私有密度似然 versus 加均衡效用，两项同初始函数/容量/数据/预算。
两份网络均加载保留v82第25轮原64训练井模型，不重置输出。固定网络eval/no_grad，只有私有密度的骨干/FiLM/输出进优化器；其129个gate输出参数未用于损失或推理。单个SW默认原子/8Gaussian预测分布，联合原效用argmax；没有模型投票或依真实人口选择预测器。固定门控不保证默认点评分固定，条件密度仍参与动作选择，需实际完整SW评价。
旧v62已试过此前v56来源的固定默认/私有密度原似然且无整体收益。本批新条件为当前联合效用来源与明确人口tradeoff，不把结构当首次新发现，不恢复旧队列。

## 已探索、训练与验证
实现完成，启动无新分数；直接训练。初始全部144665点须与当前SW函数相同，训练实际第二batch记录私有body/output/FiLM梯度、辅助密度梯度及冻结gate无梯度；记录固定网络当前/最佳hash不变。两项都计算同一均衡效用图，唯一损失差是系数0/1；原BCE固定无可更新gate，原数值NLL/Q50不改。人口份额.5/.5，比例只用原64训练井计数，人口平均权重1，验证人口不参与拟合。
原9曲线206/输入井bank/分折/seed20261010/1024上下文128中心、私有分支40%DropKey、256支持800动作+默认99.9/评分不改，固定POR/PERM。新AdamW3e-5/wd.01/clip2、cos80*64至1e-5/max80/pat18/min10/b64/64steps；是同checkpoint新优化，不是旧轨迹续跑。原子last含当前/最佳、优化器/scheduler/RNG/停止状态，仅显式恢复。

## 收益、关闭条件与未完成
收益尚未证实。按完整SW官方分选checkpoint，报告默认/数值/各井/真实三列和同损失对照；正候选原始数据磁盘重建。若两项都恢复起点或无有效匹配增益，停止此结构/损失组合，不继续扫LR或人口份额。数值点改善必须同时报告默认和整体代价；不以局部子群正差当总体收益。未完成80轮/其他四折不能写成完整验证。
无亮点权重完成后先清单再删除，原正模型/来源/交付保留。最多2GPU，观察实际显存后第二项显式启动；不足则串行，无自动队列/旧任务/其他折。单项耗时和同工作量总墙钟/并发/显存分开，加载/预热/训练/验证/保存/启动间隔计入，准备和图谱收尾另计。维护PROJECTS目标/已做/待做/未做/目标后工作及README/EXPLORATIONS/图谱。
严格原第五>80.5且明确收益后考虑其余四折，最终完整原分组OOF>82仍未完成；目标后独立井确认和新目录交付。


## v84 likelihood显式启动
{"pid": 70312, "mode": "likelihood", "started_at": "2026-09-30T11:25:00.808507+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_frozen_gate_utility_v84.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_frozen_gate_utility_v84_SW_likelihood_fold5"], "automatic_queue": false}

## 初始化失败与新目录修复
原v84缺评估接口length，未完成初始评分或训练，不是算法负结果。源码/配置/日志保持。v84r1另目录补length1024/c2后完成两项训练，均18轮恢复起点，真实结果见../sw_frozen_gate_utility_v84r1_20260930/FINAL.md；不恢复本失败实例。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
