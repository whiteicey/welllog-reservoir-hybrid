

## 2026-10-01 v103缓存方案仅准备
已完成：原源256窗口/128核心隐藏缓存生成器、私有SW和训练入口，禁止源预测/标签缓存。当前v102对照仍活跃，无新训练/生成队列。计划：先看来源特征最终消融，若值得保留再测试缓存定义与速度。未完成：任何缓存、训练、增益和加速验证。目标：纯DL原OOF>82、第五>80.8、含源≤5训练阶段；达成后可做新交付。

2026-10-01显式开始pretrained输入缓存准备：源256窗口/128核心，不读标签或源预测；与v102训练共享最多2个GPU名额。此为一次输入预处理，不启动v103训练。实际准备时间/显存/覆盖和上下文定义写缓存manifest，等待v102最终消融再决定新训练。

### v102结论与本轮启动理由
v102最终来源消融pretrained-random SW=-0.376292919467，无增益；不扫描旧移动SW锚点配置。v103改变源上下文：原256源窗口自己的relative锚点、只保留中心128，以避免旧SW核心落在切分源块边缘。这是根据结构定义提出的新假设，尚未证实是失败原因，不声称修正必然有收益。预训练输入缓存生成完成：730268×256 float16，12.793秒，峰值412.7MiB，源预测和标签均未使用。显式继续同缓存定义random/pretrained匹配实验；不补其他折。

2026-10-01 v103双方已核实活跃：random PID30732，pretrained PID86228；各自峰值约2074MiB，双进程合计显存约10104MiB，未OOM。缓存单项此前约13-14秒/轮，双并发后约28秒/轮；暂未证明吞吐优势，最终记录同工作量总墙钟，后续新SW优先串行。原源成本1289.152秒及输入缓存10.114/12.793秒仍计入，不称免费或与v102数学等价。第五>80.8/完整OOF>82未完成；无其他折。

### random实际完成
SW=73.989575926996；固定三通道TOTAL=80.428900064642，对旧最好-0.075297359835，严格80.8=False。best/stop=34/52，本项1191.204秒，峰值2074.2MiB。仅4个依赖训练阶段，拟第五密度阶段尚未执行；完整OOF未完成。

### pretrained实际完成
SW=73.823152241908；固定三通道TOTAL=80.370651774862，对旧最好-0.133545649615，严格80.8=False。best/stop=15/33，本项892.476秒，峰值2074.2MiB。仅4个依赖训练阶段，拟第五密度阶段尚未执行；完整OOF未完成。

### 预定义编码器来源消融
pretrained-random SW=-0.166423685087。同容量、私有骨干/新增stem初始化、源预处理buffers、损失/Q50/原第五折/80-18；两项实际最佳恢复完毕。该差值只证明本设置的源隐藏特征收益；不能当完整OOF或总体架构收益。
本来源特征方案不继续扫描编码器种子、LR或层数；负结论只适用于本256源块/1024新SW骨干设置。

v103两项最终无来源增益（SW差-0.166423685087），清理负模型、恢复状态和2份输入隐藏缓存，释放750.25MiB；配置/源码/日志/最终预测/manifest/图谱保留，v104不依赖这些权重或缓存。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
