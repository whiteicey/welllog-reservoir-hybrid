# 目标与约束

- 固定原第五折，检验共享分量读出与跨分量通信，预定义shared/communicate两项。
- 原数据、train-only统计、Q50、评分与解码不变；只POR新头训练，旧预测器及PERM/SW冻结。
- normal GPU最多2任务；80/18/min10/b64/64步。原子完整恢复状态，显式恢复，无自动队列。
- 每配方含来源最多5阶段；不加载已训练v140头叠加新阶段。严格TOTAL>80.8并有收益后补折，完整OOF>82未完成。
- 维护PROJECTS.md、README四项证据、EXPLORATIONS和DL图谱；负权重清理保留SHA及紧凑结果，历史交付不改。
