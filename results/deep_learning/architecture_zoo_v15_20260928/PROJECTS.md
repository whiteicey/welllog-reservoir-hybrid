# 目标与进展

目标：比较三种不同骨干机制，拓宽网络结构设计。
已完成：历史与资源核对，三种结构和统一训练接口。
计划：仅三项预定义内层SW实验，显式分批最多两任务。
未完成：实测训练收益、机制消融、三目标扩展和原第五折验证。
目标完成后：选择有证据的结构再进行匹配消融及其他独立通道实验，不自动补折或改交付。


## 2026-09-28T17:00:30.734246+08:00 v15结构候选启动
显式启动[('adaptive', 70224), ('latent', 96352)]；只做既定三骨干内层SW比较，初始与最终分数待实际记录。无原第五折或其他通道模型更新，无自动队列。


## 2026-09-28T17:06:20.393426+08:00 v15结构候选启动
显式启动[('state_space', 37424)]；只做既定三骨干内层SW比较，初始与最终分数待实际记录。无原第五折或其他通道模型更新，无自动队列。


## v15三种完整骨干筛查完成
2026-09-28T17:11:23.595216+08:00。{'adaptive': 76.43706340163116, 'latent': 76.59073085729038, 'state_space': 76.38645544042299}；全部为48/16内层SW。是否结构信号=True。未训练原第五折，旧三私有模型80.274135保持。详细研究记录results/deep_learning/architecture_zoo_v15_20260928/FINAL.md。参数匹配消融/其他通道/完整OOF>82均未完成。


## v15完成后的机制验证
2026-09-28T17:14:37.682304+08:00。潜变量/SSM完整144621点磁盘复现；自适应采样完成有限批验证。潜变量为当前内层结构候选，未改原第五折模型；匹配容量/其他目标/跨折结果待做。详见architecture_zoo_v15_20260928/FINAL.md。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
