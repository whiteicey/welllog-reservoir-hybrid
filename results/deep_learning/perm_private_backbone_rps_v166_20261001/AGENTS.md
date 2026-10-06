# 目标与约束

PERM私有双专家骨干LR1e-5与有效128/RPS1读出共同训练。只用自身标签；原输出参数/输入路由/其他通道模型冻结，原九曲线/分折/训练统计/Q50/评分。原SOURCE训练折初始化，无已训154头，五阶段替换，不加载其他预测或外部预训练。max80/pat14/min10/b64/64步/normalGPU/max2，必要显存入场6GiB，不做数值预飞。原子last/显式恢复，维护PROJECTS/README/EXPLORATIONS/atlas，无自动队列/补折或覆盖交付。
