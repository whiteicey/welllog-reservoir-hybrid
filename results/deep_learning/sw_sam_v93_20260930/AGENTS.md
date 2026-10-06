# v93目标与约束
仅SW同839481参数/原v56训练折内骨干FiLM+训练先验，两次forward/backward rho0/SAM.05同计算控制，原Gaussian/效用1/DropKey.4/训练窗口和回放mask完全相同。首未clip全局L2扰动、次梯度更新，clone精确恢复，RNG每更新消费一个forward；原9曲线206/输入bank/64-16/Q50/评分/1e-4/80-18/min10/b64/64steps/cos不改。只SW标签，无released权重/其他预测；normal/max2GPU，实测分批，原子完整last/显式恢复。维护PROJECTS/独立记录/EXPLORATIONS/图谱；删除计划先落盘，正源/旧交付保留。第五严格>80.5且明确收益后才考虑其他折，原OOF>82未完成。
