# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# v92 POR完整私有模型与动作效用

探索重点：v86只更新30819路由、专家固定，两项无新增收益，不足以否定完整POR效用训练。SW效用已有完整同容量收益，但跨目标不先验成立。本批仅likelihood/joint两个POR完整热启动，从当前v31 anchored第7轮/v30 bundle/v25幅值修正/v8私有双专家全部函数开始，原参数不新增、原输出先验不重置。解除真实隐藏/输入路由梯度detach/no_grad，实际bundle读出和已有效位置修正可训练；未使用source读出副本、路由anchor参考副本、其他目标packed行冻结。明确绕过旧v25 frozen-body train override，训练时骨干原dropout恢复、eval关闭。两组同容量、同初始函数/参数/seed/预算，仅效用系数0/1不同，同损失计算图。

对照：当前POR77.551936801885，固定PERM89.396724049366/SW73.793310952488组合TOTAL80.382093291214。原9曲线206/已有训练侧幅值63、原64/16训练578891/验证144665、seed20261010/Q50/官方评分/t3分布/log1p-expm1/256支持800均匀动作+default0.1保持。POR官方相对容差.08，主Q50 BCE+t3NLL/3与效用1/3，softaction温度.05、512点梯度checkpoint，代理损失不等于精确官方argmax。新AdamWbody1e-5/head3e-5/wd.01/分组clip2/cos80*64到1e-5/80上限pat18min10batch64steps64，body等于eta_min固定。不是旧轨迹续跑，不宣称CUDA逐位复现。

已探索与验证：建立时尚未训练，收益未证实。直接训练，初始144665完整当前函数一致，实际第二batch的两骨干/输入路由/实际输出读出/v28/v29与效用gate/density梯度，冻结无梯度/终态权重不变，source/预处理/评分hash、最佳恢复及原三列实际评价。没有独立预检/审查或重复Q检查。保存排除epoch0的训练后最佳完整state/pred；正绝对或完整匹配增益两侧磁盘重载，不能选定轮数重训补证据。

收益与不继续：分开报告绝对对当前、含起点joint-likelihood、完整训练后最佳同容量差、同轮轨迹、默认/数值/各井。epoch0恢复不算收益，开发折非完整OOF/独立泛化。没有绝对或完整公平局部增益则关闭本全网络/来源/低LR/系数1/温度.05预算配置，不临时扫参数或扩容；不泛化所有POR效用学习。

已做：两有限入口。计划：显式分批训练、实际组合/必要重载/清理/图谱。未做：真实收益、其他折、独立泛化。normal最多2GPU，以实测显存启动第二项，无自动队列。记录单项加载预热/训练验证保存、同工作量两项总墙钟/重叠、allocated/reserved，准备收尾单列。原子完整last含current/best/更新后最佳/优化器/scheduler/RNG/停止，恢复显式。负项删除计划含hash/bytes先落盘，正源/当前/对照依赖/旧交付保留。目标后：原第五严格>80.5且明

的训练后最佳差-0.029103461105。采用=None。epoch0不是新收益；同容量局部对照和绝对采用分开。
同轮轨迹：{"common_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "delta_joint_minus_likelihood": [0.02396718750594573, -0.054862612888911144, -0.08130939606429877, -0.06067208430047799, -0.03394448739429379, -0.07905650599043668, -0.06340824048081117, -0.08590868535316076, -0.029646296151057072, 0.023456607866464196, -0.03625673141200991, -0.024129890646108265, -0.01314190436906415, -0.08045443730770785, -0.02586527779922676, -0.023392760911775667, -0.02723206288985125, -0.055271393814962266], "positive_epochs": 2, "negative_epochs": 16, "mean_delta": -0.040396054022319085, "last_delta": -0.055271393814962266}
likelihood 更新后默认/数值/各井：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 96.76229638500588, "channel_delta": -0.46525430015790903, "TOTAL_contribution": -0.09103791518335427}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.52167667793252, "channel_delta": -0.12694655186662374, "TOTAL_contribution": -0.013243895478462876}}；井升/降6/10。
joint 更新后默认/数值/各井：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 96.91808768824782, "channel_delta": -0.30946299691596835, "TOTAL_contribution": -0.060553693014896534}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.14578711363368, "channel_delta": -0.5028361161654615, "TOTAL_contribution": -0.052459155978402595}}；井升/降4/12。
不继续：无绝对或完整匹配收益则关闭本完整POR/v31来源/t3/低LR/系数1/温度.05/预算组合，不临时扫参数，不泛化所有POR效用方法。开发第五折不是完整OOF或独立泛化证明。

成本：{"candidates": [{"mode": "likelihood", "start": "2026-09-30T14:45:02.182137+08:00", "end": "2026-09-30T14:51:26.799452+08:00", "seconds": 378.90524160000496, "peak_allocated_MiB": 934.5498046875, "peak_reserved_MiB": 986.0, "launch_to_completion_seconds": 384.617315}, {"mode": "joint", "start": "2026-09-30T14:45:43.510778+08:00", "end": "2026-09-30T14:52:06.716301+08:00", "seconds": 377.0130022999947, "peak_allocated_MiB": 934.5498046875, "peak_reserved_MiB": 986.0, "launch_to_completion_seconds": 383.205523}], "first_launch_to_last_completion_seconds": 424.534164, "first_launch_to_verified_seconds": 490.723593, "process_overlap_seconds": 343.288674, "peak_training_processes": 2, "sum_individual_launch_seconds": 767.822838, "finalize_seconds": 0.6222320000233594, "scope": "loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring/atlas separate; not capacity evidence"}
清理6个冗余状态72280274字节，清理计划先落盘；正源/当前/匹配依赖和旧交付保留。
已做：完整两项实测/最佳恢复/实际组合/必要重载/清理/记录。未做：其他折、独立泛化。第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。



## FINAL.md

# v92 POR完整私有模型动作效用结果

当前独立路线TOTAL 80.382093291214 → 80.382093291214。

|方案|POR|固定PERM/SW TOTAL|最佳/停止轮|训练后最高POR|
|---|---:|---:|---:|---:|
|likelihood|77.551936801885|80.382093291214|0/18|77.204330766346|
|joint|77.551936801885|80.382093291214|0/18|77.175227305241|

探索与验证：完整v31 anchored第7轮/v30实际bundle/v25-v8私有POR同起点，共1334125学习参数。解除隐藏/输入路由梯度冻结及旧v25 train override，两骨干/实际输出头/v28-v29更新，未使用源输出副本/route anchor/其他目标packed行固定。原9曲线206+已有幅值63/分折/Q50/t3/log1p-expm1/官方.08/default.1/256支持800均匀动作不改；仅系数0/1，新body1e-5/head3e-5/温度.05/80上限pat18。真实梯度、初始全预测、source/预处理/评分hash、frozen权重/行不变、最佳144665恢复及三列评价通过。原始输入磁盘重载=[]；CUDAwarn_only不声称逐位训练复现。

收益：含起点joint-likelihood POR+0.000000000000，完整排除起点的训练后最佳差-0.029103461105。采用=None。epoch0不是新收益；同容量局部对照和绝对采用分开。
同轮轨迹：{"common_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "delta_joint_minus_likelihood": [0.02396718750594573, -0.054862612888911144, -0.08130939606429877, -0.06067208430047799, -0.03394448739429379, -0.07905650599043668, -0.06340824048081117, -0.08590868535316076, -0.029646296151057072, 0.023456607866464196, -0.03625673141200991, -0.024129890646108265, -0.01314190436906415, -0.08045443730770785, -0.02586527779922676, -0.023392760911775667, -0.02723206288985125, -0.055271393814962266], "positive_epochs": 2, "negative_epochs": 16, "mean_delta": -0.040396054022319085, "last_delta": -0.055271393814962266}
likelihood 更新后默认/数值/各井：{"

点的训练后最佳差-0.029103461105。采用=None。epoch0不是新收益；同容量局部对照和绝对采用分开。
同轮轨迹：{"common_epochs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "delta_joint_minus_likelihood": [0.02396718750594573, -0.054862612888911144, -0.08130939606429877, -0.06067208430047799, -0.03394448739429379, -0.07905650599043668, -0.06340824048081117, -0.08590868535316076, -0.029646296151057072, 0.023456607866464196, -0.03625673141200991, -0.024129890646108265, -0.01314190436906415, -0.08045443730770785, -0.02586527779922676, -0.023392760911775667, -0.02723206288985125, -0.055271393814962266], "positive_epochs": 2, "negative_epochs": 16, "mean_delta": -0.040396054022319085, "last_delta": -0.055271393814962266}
likelihood 更新后默认/数值/各井：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 96.76229638500588, "channel_delta": -0.46525430015790903, "TOTAL_contribution": -0.09103791518335427}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.52167667793252, "channel_delta": -0.12694655186662374, "TOTAL_contribution": -0.013243895478462876}}；井升/降6/10。
joint 更新后默认/数值/各井：{"default": {"n": 94357, "reference": 97.22755068516379, "candidate": 96.91808768824782, "channel_delta": -0.30946299691596835, "TOTAL_contribution": -0.060553693014896534}, "numeric": {"n": 50308, "reference": 40.64862322979914, "candidate": 40.14578711363368, "channel_delta": -0.5028361161654615, "TOTAL_contribution": -0.052459155978402595}}；井升/降4/12。
不继续：无绝对或完整匹配收益则关闭本完整POR/v31来源/t3/低LR/系数1/温度.05/预算组合，不临时扫参数，不泛化所有POR效用方法。开发第五折不是完整OOF或独立泛化证明。

成本：{"candidates": [{"mode": "likelihood", "start": "2026-09-30T14:45:02.182137+08:00", "end": "2026-09-30T14:51:26.799452+08:00", "seconds": 378.90524160000496, "peak_allocated_MiB": 934.5498046875, "peak_reserved_MiB": 986.0, "launch_to_completion_seconds": 384.617315}, {"mode": "joint", "start": "2026-09-30T14:45:43.510778+08:00", "end": "2026-09-30T14:52:06.716301+08:00", "seconds": 377.0130022999947, "peak_allocated_MiB": 934.5498046875, "peak_reserved_MiB": 986.0, "launch_to_completion_seconds": 383.205523}], "first_launch_to_last_completion_seconds": 424.534164, "first_launch_to_verified_seconds": 490.723593, "process_overlap_seconds": 343.288674, "peak_training_processes": 2, "sum_individual_launch_seconds": 767.822838, "finalize_seconds": 0.6222320000233594, "scope": "loading/warmup/train/eval/save and explicit launch gap included; prelaunch authoring/atlas separate; not capacity evidence"}
清理6个冗余状态72280274字节，清理计划先落盘；正源/当前/匹配依赖和旧交付保留。
已做：完整两项实测/最佳恢复/实际组合/必要重载/清理/记录。未做：其他折、独立泛化。第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82未完成；目标后独立井确认与新目录交付。


