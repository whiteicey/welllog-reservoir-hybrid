# v91目标与约束
仅当前v82 Gaussian SW冻结全模型，参考/本井两独立2425参数零线性头。原FiLM用本井，两个code均计算，共享冻结推理；无DropKey训练随机化。原9曲线206/输入bank/分折/Q50/官方评分/Gaussian效用1/80-18/min10/bs64/64步/1e-4cos到1e-5不变。normal/max2GPU，本批一个进程，两头各自优化/早停/选模。原子last含current/best/优化器/scheduler/RNG/停止，恢复显式。维护PROJECTS/独立记录/图谱，清理计划先落盘。无其他折/旧队列/交付覆盖，第五严格>80.5且明确收益，完整OOF>82未完成。
