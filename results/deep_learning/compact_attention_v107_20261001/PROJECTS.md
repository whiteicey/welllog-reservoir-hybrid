

## 2026-10-01 v107 注意力参数化
已完成：local SW74.121753保留；频域、源隐藏特征、POR末端投影与local密度无新增益，停止各负设置。新dot/cosine同845929参数实现，local结构从随机开始。现在：直接串行dot训练。待做：cosine、最终消融和实际三通道，必要时第5细化。目标：原第五>80.8、原OOF>82、含源≤5阶段；达成后新交付。

### dot最终实际组合
SW=73.996259635993，TOTAL=80.431239362791，对旧最好-0.072958061686；strict808=False。本项best/stop=34/52，713.209秒，峰值1725.2MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

2026-10-01 v107 dot最终SW=73.996259635993，实际固定三通道TOTAL=80.431239362791（4依赖阶段含源）；best34/stop52，713.209秒，未过80.8。cosine PID42588已核实活跃：epoch23，最佳SW=73.820020434479，峰值1743.0MiB，尚无最终参数化消融。当前仍保留v104 local实际紧凑TOTAL80.475161980049，不补其他折；完整原OOF>82未完成。

### cosine最终实际组合
SW=74.038234103145，TOTAL=80.445930426295，对旧最好-0.058266998182；strict808=False。本项best/stop=34/52，691.858秒，峰值1743.0MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

### 注意力参数化最终消融
cosine-dot SW=+0.041974467153；same845929容量与其他参数初始化/原数据/Q50/预算，两项均实际恢复最佳。只有本QK归一化/温度初始化设置的单折证据，非完整OOF收益。

2026-10-01 v107已结束：cosine SW74.038234103145、dot73.996259635993，同容量差值+0.041974467153（QK归一化和初始温度联合变化）。均低于local74.121752828158，当前精简4阶段TOTAL80.475161980049不变，未过80.8。保留cosine局部正对照，不扫温度。v108已准备730268行POR语义统计与4曲线对照，训练侧拟合，共4依赖阶段，尚无训练增益。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
