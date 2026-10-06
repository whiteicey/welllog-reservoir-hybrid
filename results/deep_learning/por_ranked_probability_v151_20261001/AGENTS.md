# 目标与约束

- 在有效256位置/权重结构上增补联合RPS，预定义单项joint_rps，不加入负模块。
- 原数据/统计/Q/分折/评分/解码不变，只自身POR真实标签；其他目标固定。
- 含来源最多5阶段，不加载已训练v140头添加第六阶段；samecapacity重训对照如实记录。
- normal最多2GPU，80/18，完整原子last及显式恢复。维护PROJECTS、README四项证据、EXPLORATIONS和DL图谱。
- 第五严格TOTAL>80.8且有收益后补折；完整OOF>82未完成。清理负权重留SHA，交付不改。
