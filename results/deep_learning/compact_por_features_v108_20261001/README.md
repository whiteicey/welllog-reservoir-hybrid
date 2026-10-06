# v108 精简POR语义特征与私有SW

## 探索重点
假设：冻结精简POR的4项后验统计比4项原始曲线冗余输入更有助于SW。curve_control/por_posterior同容量、其他初始化、损失和预算；全配方源预训练/v101 POR/v100 PERM/新SW共4阶段，老师计入第2阶段。不使用旧18阶段最终权重。

## 已探索内容与验证
730268行输入统计已生成，native256/stride128中央128点覆盖一次。POR模型固定，读取默认概率、变换空间条件均值/方差及分量熵；Student-t3使用等方差sd。控制为GR/DEN/CNL/RT。两套median/IQR只拟合578891训练行；无SW标签或验证标签参与准备，输入依赖不声称完全无共享骨干。缓存准备9.691秒、GPU峰值501.36MiB，含加载/推断/归一化/写盘，不含历史训练。
新SW保留原206输入和local65，从随机网络联合训练，原own SW NLL+utility1、SAM.1、DropKey.4、80/18和原解码保持。显式串行，无自动队列。

## 收益
尚未训练、尚未证实。最终报告posterior-control SW差值、与local SW74.121752828158的差值、固定三通道实际TOTAL。目标原第五折严格>80.8、完整原OOF>82未完成。

## 不值得继续探索的点及证据
v102/v103 256维源隐藏输入均负；旧v27冻结头后验增量无收益，不能将本次全SW训练预先宣称有效。v107 cosine-dot局部+0.041974但两者低于local，停止温度扫描。本次若无收益，关闭此4统计设置，不自动扫描更多老师/特征。旧交付保留。

Provenance note: teacher features on training wells are in-sample predictions of the training-only POR teacher; heldout inputs use the same frozen teacher. No cross-fitted teacher training is hidden in the stage count. This may cause train/heldout representation mismatch and is an empirical limitation, not established gain. Initial copied plan metadata is archived; active source/config hashes unchanged.

### curve_control最终实际组合
SW=73.843376994495，TOTAL=80.377730438267，对旧最好-0.126466986210；strict808=False。本项best/stop=34/52，642.404秒，峰值1726.5MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

### por_posterior最终实际组合
SW=73.683727261052，TOTAL=80.321853031562，对旧最好-0.182344392915；strict808=False。本项best/stop=20/38，472.643秒，峰值1726.5MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。

### POR语义输入最终消融
posterior-control SW=-0.159649733443；same846297容量与其他参数初始化/原数据/Q50/预算，两项均实际恢复最佳。只有本4维POR语义输入设置的单折证据，非完整OOF收益。
关闭本4维POR语义统计设置；不自动扫描更多老师或统计，不泛化为所有语义特征无效。

2026-10-01 v108结束：posterior SW73.683727261052，对control73.843376994495为-0.159649733443，均低于local，关闭4统计设置；负检查点/完整状态及特征缓存清理47.57MiB并留清单。两项串行墙钟1180.004秒，含启动间隔/加载/预热/评价/IO，缓存此前9.691秒另计，不与历史来源成本混同。v109换为逐点读取全局SW记忆，同容量geometry/content两项，新增12672参数零输出、残差±0.25，原pool/window/scoring保持，含源4依赖训练阶段；尚未训练，不补其他折。
