# v92 POR完整私有模型动作效用结果

当前独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|POR|固定PERM/SW TOTAL|最佳/停止轮|训练后最高POR|
|---|---:|---:|---:|---:|
|likelihood|77.551936801885|80.382093291214|0/18|77.204330766346|
|joint|77.551936801885|80.382093291214|0/18|77.175227305241|

探索与验证：完整v31 anchored第7轮/v30实际bundle/v25-v8私有POR同起点，共1334125学习参数。解除隐藏/输入路由梯度冻结及旧v25 train override，两骨干/实际输出头/v28-v29更新，未使用源输出副本/route anchor/其他目标packed行固定。原9曲线206+已有幅值63/分折/Q50/t3/log1p-expm1/官方.08/default.1/256支持800均匀动作不改；仅系数0/1，新body1e-5/head3e-5/温度.05/80上限pat18。真实梯度、初始全预测、source/预处理/评分hash、frozen权重/行不变、最佳144665恢复及三列评价通过。原始输入磁盘重载=[]；CUDAwarn_only不声称逐位训练复现。

收益：含起点joint-likelihood POR+0.000000000000，完整排除起点的训练后最佳差-0.029103461105。采用=None。epoch0不是新收益；同容量局部对照和绝对采用分开。
同轮轨迹：{"common_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "delta_joint_minus_likelihood": [0.02396718750594573, -0.054862612888911144, -0.08130939606429877, -0.06067208430047799, -0.03394448739429379, -0.07905650599043668, -0.06340824048081117, -0.08590868535316076, -0.029646296151057072, 0.023456607866464196, -0.03625673141200991, -0.024129890646108265, -0.01314190436906415, -0.08045443730770785, -0.02586527779922676, -0.023392760911775667, -0.02723206288985125, -0.055271393814962266], "positive_epochs": 2, "negative_epochs": 16, "mean_delta": -0.040396054022319085, "last_delta": -0.055271393814962266}
likelihood 更新后默认/数值/各井：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 96.76229638500588, "channel_delta": -0.46525430015790903, "TOTAL_contribution": -0.09103791518335427}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.52167667793252, "channel_delta": -0.12694655186662374, "TOTAL_contribution": -0.013243895478462876}}；井升/降6/10。
joint 更新后默认/数值/各井：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 96.91808768824782, "channel_delta": -0.30946299691596835, "TOTAL_contribution": -0.060553693014896534}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.14578711363368, "channel_delta": -0.5028361161654615, "TOTAL_contribution": -0.052459155978402595}}；井升/降4/12。
不继续：无绝对或完整匹配收益则关闭本完整POR/v31来源/t3/低LR/系数1/温度.05/预算组合，不临时扫参数，不泛化所有POR效用方法。开发第五折不是完整OOF或独立泛化证明。

成本：{"candidates": [{"mode": "likelihood", "start": "2026-09-30T14:45:02.182137+08:00", "end": "2026-09-30T14:51:26.799452+08:00", "seconds": 378.90524160000496, "peak_allocated_MiB": 934.5498046875, "peak_reserved_MiB": 986.0, "launch_to_completion_seconds": 384.617315}, {"mode": "joint", "start": "2026-09-30T14:45:43.510778+08:00", "end": "2026-09-30T14:52:06.716301+08:00", "seconds": 377.0130022999947, "peak_allocated_MiB": 934.5498046875, "peak_reserved_MiB": 986.0, "launch_to_completion_seconds": 383.205523}], "first_launch_to_last_completion_seconds": 424.534164, "first_launch_to_verified_seconds": 490.723593, "process_overlap_seconds": 343.288674, "peak_training_processes": 2, "sum_individual_launch_seconds": 767.822838, "finalize_seconds": 0.6222320000233594, "scope": "loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring/atlas separate; not capacity evidence"}
清理6个冗余状态72280274字节，清理计划先落盘；正源/当前/匹配依赖和旧交付保留。
已做：完整两项实测/最佳恢复/实际组合/必要重载/清理/记录。未做：其他折、独立泛化。第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。
