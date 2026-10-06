

## 2026-10-01 v102 紧凑SW源特征
已完成：v100五阶段80.428680，v101均值范围80.456611；均未过80.8，提交包不改。现在：实现冻结随机/原训练源隐藏特征同容量SW对照，串行直接训练。待做：最终SW公平差值、必要时第五阶段密度、实际固定三通道验证。目标：包含源预训练≤5阶段且完整原OOF>82；过筛后才补折。达到后可做交付与速度对比。

2026-10-01 v102启动修正：源专家1024窗口初始评估失败，无训练轮；保留旧源码/config/log。新匹配两项使用四个256点源块，SW仍1024点。random_block256 PID72544已启动，pretrained_block256尚未启动；无自动队列，待真实完成再显式启动。

### 训练中速度证据（17轮，非最终收益）
随机源隐藏特征最佳SW=73.573365207933；平均30.930秒/轮，同轮次原输入平均15.442秒/轮。额外输入容量/初始化不同，不把与原输入差值当同容量消融。预训练对照尚未启动，暂不下最终结论；若来源表征有实测收益，后续优先测试固定输入块离线隐藏缓存来降低计算成本，须另立配方记录其上下文差异。当前不启动第三项、其他折或旧路线。

### random_block256实际完成
SW=74.025418470081；固定三通道TOTAL=80.441444954722，对旧最好-0.062752469755，严格80.8=False。best/stop=24/42，本项1331.302秒，峰值1754.2MiB。仅4个依赖训练阶段，拟第五密度阶段尚未执行；完整OOF未完成。

2026-10-01 v102随机源对照已最终完成：SW=74.025418470081，实际固定三通道TOTAL=80.441444954722，仅4依赖阶段；未过80.8。预训练源同容量候选PID54004已核实活跃，第2轮，仍无最终来源消融收益。固定当前POR/PERM时，SW需严格>75.049861456590才过门槛；继续评估而不补折。v103仅准备，无缓存或训练。

### pretrained_block256实际完成
SW=73.649125550614；固定三通道TOTAL=80.309742432908，对旧最好-0.194454991568，严格80.8=False。best/stop=15/33，本项1026.458秒，峰值1754.2MiB。仅4个依赖训练阶段，拟第五密度阶段尚未执行；完整OOF未完成。

### 预定义编码器来源消融
pretrained-random SW=-0.376292919467。同容量、私有骨干/新增stem初始化、源预处理buffers、损失/Q50/原第五折/80-18；两项实际最佳恢复完毕。该差值只证明本设置的源隐藏特征收益；不能当完整OOF或总体架构收益。
本来源特征方案不继续扫描编码器种子、LR或层数；负结论只适用于本256源块/1024新SW骨干设置。

v102来源特征消融为负，清理4个完成模型/恢复状态，共68.00MiB；原源与v103活跃产物、最终预测、源码/配置/日志/报告和删除清单保留。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
