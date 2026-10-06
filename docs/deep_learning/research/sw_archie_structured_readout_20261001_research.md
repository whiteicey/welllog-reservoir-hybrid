# SW：Archie 结构化神经分布是否值得一试

2026-10-01。独立只读研究；未训练、未证明收益。只新增本研究文件，不修改训练源码、不占 GPU、不建立自动队列。结论：**可以做一轮软物理分布／同起点自由系数的有限对照；不推荐硬替换104、不推荐仅添加Archie特征，也不把它当作大幅突破的可靠预期。**

## 探索重点

主Agent提供：SW275直接组件mix为74.137236，仍低于SW252的74.149228；104默认分约98.716，连续约17.632，连续约66.6%得零分。最后三项诊断未在本研究重新计算，不能当作独立验证。希望新编码器和分布读出联合学习，直接约束连续分布的形状，而非再次修补输出头／扫TCN。

Archie关系在自然对数下为 log(Sw)=(log(a Rw)-log(Rt)-m log(phi))/n。原论文给出F与孔隙度及电阻率指数关系，讨论的是特定砂岩、盐水条件；公式是经验近似，井径、侵入、层厚、岩性和水电导等影响不能省略。[Archie原文扫描](https://personal.ems.psu.edu/~radovic/EME590_Archie_1942.pdf)，正式DOI [10.2118/942054-G](https://doi.org/10.2118/942054-G)。扫描正文已读取，出版商DOI页面403。

泥质导电会破坏clean-sand简单关系，Waxman–Smits模型需要孔隙单位体积阳离子交换量等未观测量。[Waxman–Smits原文](https://blasingame.engr.tamu.edu/z_zCourse_Archive/P663_03C/P663_03C_TAB_Ref_FormEval/SPE_01863_Waxman_Smits_Shaly_Sand_Sw_Relation.pdf)，DOI [10.2118/1863-A](https://doi.org/10.2118/1863-A)。本次打开了原文扫描/第一页；不声称已逐页深读。没有这些测量，不应把GR近似成真实Qv后宣称已实现Waxman–Smits物理反演。

## 已探索的内容与来源核对

实际读取package/src/feature.py、src/ref/solution_v2/src/features.py、train_invariant.py筛选规则、SW275/SW252 README及此前SW直接组件研究。

- package/src/feature.py:187–201已经构造DENSITY_POROSITY=(2.65-DEN)/1.65*100、SONIC_POROSITY=(AC-182)/438*100、CNL/密度均值与差、SHALE_INDEX_GR、INVASION_LOG_DIFF。
- 其中phi_proxy=clip((clip(CNL,.1)+clip(density_porosity,.1))/200,.001,.8)，ARCHIE_PROXY=log10(max(RT,.01))+2log10(phi_proxy)。206维筛选仅排除轨迹/钻具等规定项，未排除此物理代理。因此新算一个logRt+2logphi或把三种porosity proxy拼进MLP，不是新机制。
- src/nn_seq.py旧模型注释也提过全井校准Rw/matrix density；它包含旧DEVI及树融合，不是可直接复用的合规新训练入口，不加载其权重，也不把“well context Rw”说成首次提出。
- SW275已联合训练新raw-point encoder和direct256读出，辅助CE1、固定mix .1正于tilt，但仍未胜保留252。不能用相同网络再换物理名称冒称新表示。
- package/src/utils.py原SW默认99.9、范围8.305–99.9，SW单位是百分数。Archie中的Sw必须是0–1比例；在ln(SW_percent)目标下需要加ln100。不可直接把99.9当物理饱和比例。

新增机制只能是：**输入通路限制+受控负Rt响应+分布参数共享的输出结构**。让Rw/phi/m/n均由含Rt的任意大网络生成会取消该结构约束，退化成MLP重新参数化；同时phi、Rw、m、n不具备仅靠SW监督的唯一辨识性。

## 可实行的有限对照：弱Archie四分量组件

这是一项新SW阶段的两个平行候选，二者都从104原始prior开始，不加载252/275修补权重；若采用其中一项，总链路只增加一个SW阶段，其他头采用链路仍须总≤8。

1. 原104完整prior/gate冻结。新小raw-point encoder与新四分量读出同时训练，约48维、2–3个残差MLP块，不用教师/POR真标签或POR预测。学习目标及固定mix日程复制275有效的辅助CE1/半hard半物理5%soft-target/conditionalRPS1，alpha从0五轮到.1，之后固定；不同时扫损失或alpha。
2. 新组件的phi候选来自已有DEN/CNL/AC代理，用可训练softmax权重形成正数加权phi。先把百分数除100，再clip [.001,.8]。代理不是实测孔隙度；单位常数与原预处理保持并记录，缺失使用原训练侧插补。
3. encoder及全井/区域输入context仅读取不依赖RT/RXO的既有特征；必须按依赖剔除RT_LOG、ARCHIE_PROXY、INVASION、RT/RXO比值、GR_RTV_PRODUCT及相关滚动统计，而非只去掉两列。原104仍可读取全部原206。通过确定性静态依赖清单定义这组输入，不按验证成绩挑选。此限制作用是防止context把logRt抵消掉，不是恢复旧曲线。
4. context输出每位置phi混合权重，低自由度m/n及四个c_k、scale_k、mixture weights。m=1.3+.7*sigmoid(u_m)，n=1.5+sigmoid(u_n)；m范围参考Archie砂岩资料，n范围是本研究预设容错范围，不是被本数据证实。c_k=训练侧固定c_k0+.5*tanh(delta_c_k)，不是实测log(aRw)。不用井ID embedding，不拟合验证/测试井专属标签参数。context可读本井输入，但其统计/神经参数拟合仅用训练64井。
5. 两组均计算mu_k=ln100+(c_k-m*ln(phi)+beta*ln(max(RT,.01)))/n。**结构组beta=-exp(.5*tanh(r))**；**控制beta=-1+2*tanh(r)**。r末层零初始化，二者初始beta=-1，所有其他初始参数/函数一致，参数数目一致；控制允许正/零Rt响应，结构组始终负。二者函数边界不同正是被测机制，不能声称相同函数容量。这个形式是校准的Archie启发，不是严格原式beta=-1。
6. scale设正下限，四个lnSW Gaussian分量计算原256支持边缘CDF质量，按现有截断/质量归一化投影到同一bin。这四分量只作为10%的连续组件，与104概率混合；默认atom维持原gate。不得更换原800动作、网格、阈值/评分。每个Gaussian的均值无需硬裁成合法输出，连续分布投影保持与原支持相同；不能裁剪均值后继续假称原密度归一化。
7. 初始c_k0仅从训练连续SW标签与训练输入计算r= n0*ln(SW/100)+lnRT+m0*lnphi的四分位位置，m0=1.65、n0=2（对应上述m/n映射的零logit），phi权重初始相等；是一次统计初始化，不是额外训练/伪标签。两个候选共享该统计。新分量scale用训练侧r离散度；原104评估在alpha0保持，若共同初始clamp/归一化产生细差单列记录。非零新读出和独立组件CE使新encoder首步有梯度。

条件m/n/c/logits不读Rt相关特征，使“固定其他原曲线、仅改变Rt”时新组件各log均值负响应可定义；这不保证最终混合预测严格单调（旧104权重/形状仍可能变），也不保证上下文岩性转变符合Archie。四分量c不同可表达异质水/岩性，不可把其分量称作真实水层。

显存风险：约数万参数，单新增encoder；仅中央有效监督锚点计算辅助分量/bin质量，避免B×1024×4×256全激活。四分量CDF比275全256自由logits更便宜是设计预期，未实测。训练80/14/min10/b64/64steps、normal GPU、最多2任务；两个候选源/初始化/采样/Q/LR/预算一致，记录单项耗时与实际同工作量墙钟，不宣称容量严格匹配275。

## 收益与适用假设

收益尚未证实，没有新预测/检查点。期待有益的条件是：RT仍携带真实电阻率信号、代理phi至少保持相对顺序、训练/未知井水电导差异可由非Rt输入context粗略识别，且近clean-sand区域不少。泥岩、裂缝/双孔隙、导电矿物、侵入、盐度差异以及错误代理可使约束系统性错误。

值得试的是结构组对自由控制的官方SW差值，不是c/m/n看起来有物理意义。原ARCHIE_PROXY已包含主要算式，加上未观测Rw/岩性，**没有证据期待它单次产生足够TOTAL突破**。只作一轮有限比较，不硬接“物理loss”强迫模型满足不能核验的真值关系。

## 不值得继续与停止条件

- 不新增相同Archie代理；不拟合不可辨识的真实Rw/POR；不输入POR真标签、教师预测、外部预训练或标签侧层位。
- 恢复最佳后结构组既不胜自由控制、也不胜SW252，关闭本beta/c范围/四分量设置，不扫m/n/Rw、组件数或phi混合网格。
- 默认99.9主群体官方分下降抵消连续收益时不采用；gate冻结不等于默认动作/成绩冻结。若训练CE改善但整体SW无益，不能用拟合物理残差更小宣称成功。
- 若持续大量代理在clamp边界、c_k达到±.5而仍低于控制，视为模型依赖补偿或适用范围不足，不放松到任意c来保住Archie名称。诊断只训练侧一次汇总，非冗余数值预检。
- 原第五折全144665实际三头组合后才报告TOTAL；分井/泥质子群只解释，不做逐井挑模型、oracle融合。严格TOTAL>80.8且清楚收益后才补其他折；完整OOF>82未完成。

## 不同方向备选：曲线分组加性神经分布

如果物理适用性不可信，选择NAM启发的**分组加性分布logits**，而不是另换TCN/GRU。对9条原曲线各自的一阶值/该曲线局部统计/relative输入建小私有MLP，输出256维logit贡献，求和softmax形成新组件，所有小encoder与读出联合训练，仍10%混104/同275监督。跨曲线特征另设一个固定小interaction组，不能散放到单曲线组后假称严格加性。配对控制用相同总隐藏宽度的concat+小交互混合，报告参数数目和功能容量差。它测试“减弱跨曲线快捷相关、保留各曲线非线性主效应”，与Archie物理约束和过去时序骨干不同；也可能丢掉真实地质交互，未证实收益。其训练/阶段/止损边界同上，不自动启动。

借鉴来源：[Neural Additive Models，NeurIPS2021原论文](https://arxiv.org/abs/2004.13912)、[作者Google Research实现](https://github.com/google-research/google-research/tree/master/neural_additive_models)、[已读取官方README](https://raw.githubusercontent.com/google-research/google-research/master/neural_additive_models/README.md)。论文研究的是一般回归/分类主效应；本任务分组条件概率logits是适配推论，不是原论文已证明测井SW提升。

## 已做／未做

已做：原始来源查询、现有代理/白名单规则及SW252/275机制对照核对、有限候选与失败条件记录。未做：独立统计重算、实现、训练、GPU测量、全折。仅写本文件；PROJECTS/EXPLORATIONS/图谱登记由主Agent汇总批次负责，本文本身不是新训练节点。
