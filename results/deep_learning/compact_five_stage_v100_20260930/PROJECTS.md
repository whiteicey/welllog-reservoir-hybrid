# v100进度

目标：最多5个总训练阶段达到更好的固定第五折成绩，并在严格>80.8且有收益后补全原OOF>82。

已完成：原五折81.891019提交包冻结；定义compact五阶段结构和直接训练入口，禁止旧后续权重。

计划：准备幅值缓存与原阶段真实收益诊断；显式双并发POR/PERM，之后运行SW joint及density；最终组合实际三目标并核验。

未完成：任何紧凑方案最终成绩、源成本与同工作量时间比较、其他四折/完整OOF>82。

目标达成后的可选工作：新目录交付紧凑模型，保留历史交付和关键正控制。


v100当前实际快照：{"por": {"status": "complete", "score": 77.40726398783815, "live": false}, "perm": {"status": "complete", "score": 89.37839421114361, "live": false}, "sw_joint": {"status": "running", "score": 72.88882977698545, "live": true}, "sw_density": {"status": "prepared", "score": null, "live": false}}。完整紧凑三通道收益尚未证实；无其他折。


v100当前实际快照：{"por": {"status": "complete", "score": 77.40726398783815, "live": false}, "perm": {"status": "complete", "score": 89.37839421114361, "live": false}, "sw_joint": {"status": "running", "score": 73.80277957680566, "live": true}, "sw_density": {"status": "prepared", "score": null, "live": false}}。完整紧凑三通道收益尚未证实；无其他折。


v100紧凑五总阶段已完成原第五实际组合：TOTAL=80.428679759145，相对旧配方-0.075517665332，严格>80.8=False；无其他折，完整OOF>82未完成。证据results/deep_learning/compact_five_stage_v100_20260930/FINAL.md。
