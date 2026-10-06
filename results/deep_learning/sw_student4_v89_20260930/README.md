# v89 SW Student-t4条件密度

探索重点：Gaussian薄尾假设可能限制当前私有SW密度；测试固定df4，scale=std/sqrt2，NLL和概率积分同族。网络839481参数和原组件mean/std初始化不变。t4 CDF=1/2+3w/4-w³/4，w=(edge-mean)/sqrt((edge-mean)²+2std²)；无穷尾在算术前替换，再恢复常数，保持可导有限梯度。

固定对照：历史v82 a100 Gaussian SW73.793310952488/当前固定POR-PERM组合80.382093291214。同原v56第11轮训练折内骨干/FiLM、训练64井输出先验重置、raw参数状态/1e-4/80-18/效用1/40%DropKey；不加载Gaussian已训练权重。原支持256/动作800+默认99.9/官方核/exp逆变换/Q50/原9曲线206与输入bank/seed20261010保持。分布族导致初始概率分布不同，不强制初始动作相同或不同。POR/PERM不训练，最后实际三列评价。

历史负证据：旧共享三通道global-cosine+t3，SW73.120707746865对Gaussian73.207488167083为-0.086780420218、TOTAL79.606136235786对79.989866973822负。该源码/权重/队列不复活；新假设限当前私有SW/已有效效用与DropKey配方的df4，不声称重尾已有效。

已探索与验证：记录建立时未训练，收益尚未证实。直接训练，无独立预检/审查。实际第二batch记录全骨干/FiLM/输出及效用gate/density梯度；raw初始hash、源/评分/预处理、原144665行序与最佳恢复核对。正候选原始输入磁盘重建。另保存排除epoch0的训练后最佳完整state/pred，不能按选定轮数重训补证明。

收益和不继续：完整结果分开报告对历史Gaussian绝对/默认/数值/各井差；不能把t4从自身较差先验训练提高当作超越Gaussian。无正绝对或固定同容量对照收益则关闭本df4/来源/预算，不继续扫df或scale。只覆盖本原始初始化配方，不泛化全部重尾模型。

已做：新分布、训练与评估入口。计划：仅此项实际训练、组合、正重载、负清单与图谱。未做：真实收益/其他折/独立泛化。新AdamW1e-4/wd.01/clip2/max80/pat18/min10/bs64/64步，cos80*64到1e-5；原子全状态/优化器/scheduler/RNG/停止，恢复显式。一个GPU进程，最多2，记录加载预热/训练验证保存/总墙钟和allocated/reserved，准备与收尾另计。目标后：第五严格>80.5且明确收益再考虑其余四折，原完整OOF>82后新目录交付与独立井确认；旧交付保持。


## 最终结果
# v89 SW Student-t4结果

候选SW 73.779643062735，固定POR/PERM TOTAL 80.377309529801；对Gaussian SW -0.013667889753、TOTAL -0.004783761414。最佳/停止14/32，patience18。采用=False，当前独立路线TOTAL仍为80.382093291214。

探索与验证：同839481参数、raw全模型/输出初始hash、原训练侧先验/输入bank/预算/效用1/40%DropKey对齐历史Gaussian。t4标准差参数化与NLL/CDF同步；初始概率分布不同，但动作选择可能相同，不强行要求动作差异。实际梯度/源/预处理/评分hash、最佳144665恢复、原行序和固定三列评价通过。正候选重载=[]；负项不重复原Q或独立预检。

默认/数值/井诊断：{"default": {"n": 100786, "reference": 97.48140224897922, "candidate": 97.57638028978397, "channel_delta": 0.09497804080474737, "TOTAL_contribution": 0.02315943654091552}, "numeric": {"n": 43879, "reference": 19.383958656215817, "candidate": 19.120741124100434, "channel_delta": -0.2632175321153838, "TOTAL_contribution": -0.02794319795452821}}；9井升/7井降。

收益与不继续：未超同容量Gaussian则关闭当前df4/迁移骨干/输出先验重置/效用1/DropKey0.4组合，不扫df或scale；不把从自身先验提高当超越对照，不泛化所有重尾网络。旧共享t3负证据单独保留，不重启。

成本：{"single_candidate_seconds": 222.34578519998468, "launch_to_completion_seconds": 226.76713, "launch_to_verified_seconds": 826.911832, "peak_training_processes": 1, "peak_allocated_MiB": 1814.59765625, "peak_reserved_MiB": 1918.0, "finalization_seconds": 0.2179382999893278, "gaussian_control_retrained": false, "scope": "candidate data/warmup/train/eval/save included; historical control reused, authoring/atlas separate"}。
清理3个冗余权重；删除前计算的大小/哈希未能成功落盘，无法恢复，清单已明确标为未知，未补造数值；原正源/旧交付不改。

已完成：真实训练、匹配历史比较、诊断、最佳恢复、实际组合、必要重载/清理。未做：其他折和独立泛化；原第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82仍未完成。目标后：独立井确认和新目录交付。

