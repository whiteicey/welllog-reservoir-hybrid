# PROJECTS
?????????????????????OOF>82?
???????????????????????
????????/????/????/????
????????????????????2GPU???
?????????????????


?????????????/???????13????????????[LITERATURE_IDEAS.md](LITERATURE_IDEAS.md)?188???????????187??????????????????????????????WTConv/MixStyle/????????????

IQF191??????????89.34524668852337???????SOURCE+0.000486513770????????MixStyle189??45?????/???WTConv190??????????????

Sequence expansion complete: [SOFTS/Sonnet/CATS/xLSTM source-backed ideas](results/deep_learning/cross_domain_research_20261001/SEQUENCE_IDEAS.md). Proposed migrations and cost estimates unverified. v192 add/multiply explicitly launched; no automatic research queue.

## 2026-10-02 SW 新 raw 分支的全井条件研究

- 已完成：[历史边界与固定 well/reference 方案](SW_RAW_WELL_CONDITIONING_RESEARCH_20261002.md)。275/285/289/294/296 的新 raw 编码器未显式读取 bank；46与277/278已有全井条件，不能称首次使用。
- 计划：仅作为待实施构想，9曲线128点→32→32，mean/std/max→32→96的小FiLM；旧104两臂均保留本井bank，只有新条件输入区分本井与64训练井参考。
- 未做：未写训练代码、未占GPU、未训练或验证新成绩。数值收益、实际耗时与显存尚未证实；不自动启动。
- 目标与边界：原104初始化、同参数/初始函数/80-14预算，最多2GPU、每头最多一个新增阶段、总阶段不超过8。完整OOF>82仍未完成；后续若有明确收益，再按用户授权决定下一步。

## 2026-10-02 SW连续值层级表征研究

- 已完成：[coarse/fine连续分布研究](SW_HIERARCHICAL_CONTINUOUS_RESEARCH_20261002.md)。原始HCA论文/官方代码与已有RNC/ConR/TabM/DeepONet/SW95重复范围已核对；仅推荐一个归一化配对，未训练、无收益证据。
- 计划：如主Agent采纳，从原104初始化的新raw encoder+8值区间affine+coarse/fine读出同阶段联合训练；H组条件分组softmax versus同容量F组全局softmax。只有归一化变化，不增加teacher/蒸馏/外部权重，不改256bins/800actions。
- 未做：代码、GPU训练、成绩、实测显存/耗时、其他折。现有297继续其既有授权，本研究不建立自动队列。
- 目标与边界：连续增益须由恢复整体SW最佳后的真实人口分解验证，不将默认增益冒称连续突破。80/14、最多2GPU、每头最多一个新增阶段、总≤8；原第五严格TOTAL>80.8且有明显收益后才补其他折，完整OOF>82仍未完成。

## 2026-10-02 POR九曲线token历史与原文核对

- 已完成：[曲线token历史研究](POR_CURVE_TOKEN_HISTORY_RESEARCH_20261002.md)。已确认旧逐点九曲线attention、零残差和v135曲线族Transformer；本次不能称首次机制。FT-Transformer数值token与TabTransformer类别token的区别已按原论文说明。
- 计划：仅保留151固定prior/fresh290式raw分支上的full-versus-diagonal有限问题；未在所查来源发现同mask配对。须披露diagonal的Q/K无效，注册参数相同不等于有效容量相同。
- 未做：未修改训练代码、未启动GPU、无新增成绩/显存/耗时或补折；不恢复旧曲线attention负路线，不自动搜索或建队列。
- 目标与边界：研究确认重复范围，不声称模型获益。原数据/评分/80-14/max2与单新增阶段/总≤8保持；完整OOF>82仍未完成。后续只按主Agent及用户既有授权决定是否实施。

## 2026-10-02 POR302数值嵌入的单项研究

- 已完成：[共享8段PLE候选研究](POR_CURVE_TOKEN_PLE_RESEARCH_20261002.md)。核对276局部正差+0.0106716842及旧PLE/periodic/输入样条/RBF深度bias，引用原论文和作者实现；当前302投影确认为九曲线共享。
- 推荐：只做一个从151 prior+fresh302 full联合训练的新候选，训练输入8段PLE连续外推→共享Linear(8,32)，保留零bridge；与已完成302 full相比多224参数，不能称精确同容量。无302训练权重、无新增阶段堆叠、不补重复baseline。
- 未做：没有改训练代码或启动GPU，没有新收益/实际成本；未建自动队列。失败则关闭此8段配置，不扩成频率/RBF/bins搜索。
- 目标与边界：原输入/分折/评分/80-14/max2、单新增阶段/组合总≤8保持；真实三头第五严格>80.8且明显收益前不补其他折，完整OOF>82仍未完成。
- 后续采纳：主Agent另行启动v305并报告76816参数、初始POR77.926380760724；训练仍在进行，不是最终收益。本研究记录其edge最小宽度抬高策略与论文去零宽bins的区别，未改活跃实验。

## 2026-10-02 PERM291曲线交互迁移研究

- 已完成：[PERM sparse曲线交互研究](PERM_SPARSE_CURVE_INTERACTION_RESEARCH_20261002.md)。所查历史未发现154 prior+fresh291上的同等迁移；194是patch16/STAR-style/128私有骨干且HOLD未训练，不能冒称等价历史成绩。
- 配对边界：仅把fresh302 grouped/full模块加到fresh291 stem输出、两selector之前；point选择、原154固定、所有新组共同训练、每候选单新增阶段。两臂55490参数相同，但相对无模块291增加19024，须分别报告匹配差和历史父分差。
- 未完成：本研究不改训练代码/不占GPU。主Agent已准备307但尚未启动；无本次迁移成绩、成本或最终收益，不建立自动队列。
- 目标与约束：原九曲线/数据/Q/评分/80-14/max2/总阶段≤8保持；原第五严格TOTAL>80.8且明显收益后才考虑其他折，完整OOF>82仍未完成。

## 2026-10-02 SW非对称共享梯度研究

- 已完成：[SW单向PCGrad迁移研究](SW_ASYMMETRIC_PCGRAD_RESEARCH_20261002.md)。确认旧三目标专家PCGrad与PERM v68双向投影已经存在，v68含起点最终零收益；当前SW275→282→285→289合并backward、301 detach未发现等价单向规则。原论文/作者源码及原289损失精确分解已核对。
- 计划：主Agent采纳后只做原104起点、原289网络的一项候选，共享新body负内积时投影state梯度，density原梯度/私有head保持；复用历史289完整control，披露非同期，不增加参数/推断/训练阶段。
- 未做：未写训练代码、未启动GPU/Agent、无新增成绩与实测显存/耗时。当前302/PRUNE300由主Agent原授权管理，不形成研究队列。
- 目标与边界：研究机制未证实；保留密度梯度不保证Adam或官方连续分单调。80/14/max2、每头单新增阶段/总≤8、原训练标签隔离；原第五严格TOTAL>80.8且明显收益才补折，完整OOF>82仍未完成。

## 2026-10-02 POR低秩anchored raw骨干研究

- 已完成：[有限DCN-V2迁移研究](ANCHORED_LOWRANK_RAW_RESEARCH.md)。查出旧四member raw anchored cross第五TOTAL79.448021/POR75.442844，v152 product局部负与v192 multiply局部正，明确不是首次尝试乘法/DCN。
- 保留候选：fresh207->48、两rank8 anchored cross替换rawbody，original151固定；与同容量additive对照同初始body和directdistribution头共同训练。研究文档披露有效函数容量差、替换父三分支的整体容量差及初始混合函数依赖父alpha/readout。
- 未做：未写训练源码、未占GPU、未启动训练/子Agent，无新成绩。主流程当前先优先PLE token迁移，DCN不进入自动队列。
- 目标与后续：保留default5和历史交付，不隐藏热启阶段；只有配对实测正才考虑后续，严格第五TOTAL>80.8+明显收益才补折，完整OOF>82未完成。关闭条件与成本见独立研究文件。

## 2026-10-02 PERM DEFAULT锚点监督研究

- 已完成：[DEFAULT-anchor CE历史与语义](PERM_DEFAULT_ANCHOR_CE_RESEARCH_20261002.md)。确认291 directCE为CONT-only，但继承268 jointRPS已经给DEFAULT行numeric mass梯度，不能称默认完全无监督；核对PERM/SW直接CE族及旧默认utility/RPS重复边界。
- 计划：主Agent最新固定fresh307full/原154起点、final-q真实DEFAULT z CE固定.05 versus历史307full的0，一次有偏正则化候选，291仍绝对门槛；保原continuous损失、全部new encoder/readout同stage训练，不用307已训修补/教师/decoder更改，不扫系数。原center不含精确DEFAULT，实际z和canonical应都落PERM尾bin0；只能推动边缘质量，训练侧日志确认，不扩grid。无自动POR/SW迁移。
- 未做：训练代码、GPU、子Agent、训练成绩/实测显存时间/其他折。当前任务由主Agent管理，本文不建立队列。
- 目标与边界：DEFAULT CE不能称严格conditional-density似然或表达能力提升；默认收益须和CONT损伤共同核验，整体不胜即停。每头单新增阶段/总≤8、80/14/max2和训练标签隔离保持；严格第五TOTAL>80.8+明显收益后才补折，完整OOF>82未完成。

## 2026-10-02 SW DEFAULT锚点迁移研究

- 已完成：[SW297 DEFAULT CE迁移边界](SW_DEFAULT_ANCHOR_TRANSFER_RESEARCH_20261002.md)。确认297 balanced路径numeric mixedCE/directCE/conditionalRPS均只CONT，DEFAULT通过BCE监督共享body而非numeric readout；真实99.9/100按原网格应落末bin255，不能代表独立默认center。
- 计划：仅留待309最终局部结果后的一个低优先级hardCE.05可行性，original104/fresh297well全新模块同一stage joint，不加载297权重；保原CONT半hardsoft目标，不准备DEFAULT softCE或系数扫描。没有自动训练队列。
- 未做：训练代码、GPU、子Agent、迁移成绩与实测时间显存、其他折。309增益未证实，本文不安排实施。
- 目标与边界：若实施只评价默认保护及真实人口权衡，不冒称纯条件校准/更大CONT收益；原预算/数据/评分/max2、单新增阶段/总≤8保持，严格第五TOTAL>80.8+明显收益后才补折，完整OOF>82未完成。

Rootdecision: 309FINAL localwhole+.017859 butCONT-.069533 andbelow retained89.515555 control; SWdefaultanchorproposal not adopted/notqueued, noGPU. Continuous rawdepth encoder research nowrequested.

## 2026-10-02 SW局部depth gated raw骨干有限研究

- 已完成：[SW短深度残差配对](SW_LOCAL_DEPTH_GATED_RESEARCH_20261002.md)。核对275/297逐点raw、262/263新TCN/GRU联合头失败、257/273 PERM局部负证据及292/295非depth边界；明确不是首次depthcontext。
- 唯一候选：fresh297well三支分别加stride1 DW3/7/15 gated零输出残差，配对同tap参数当前点latent通道循环位移，全tap有效；保original104固定、输入condition/body/state/readout一次joint，不载297权重。新增6432参，约71697 active；披露与父容量差及两臂有效函数类差。
- 未做：训练代码、GPU、Agent、训练成绩、实测显存/时间、其他折。研究不自动加入队列。
- 目标/止损：要求更大CONT证据，不围绕微小DEFAULT收益。CONT不胜point、或overall收益靠DEFAULT吃掉CONT即关该短核设置；两臂不胜252只算局部诊断。保持default5、原window1024/中心128监督、80/14/max2、strict第五TOTAL>80.8且明显收益才补折，完整OOF>82未完成。

v310实施规格更新：主Agent采用零DW + sigmoid Linear48gate直接残差，三核共8688新参/73953 active，无rank16投影；研究文件补充明确取代初始6432参建议，首步DW有效梯度与后续gate梯度、invalid中心mask/禁止depth roll环接。无额外研究GPU/源码。

## 2026-10-02 SW条件序数概率链研究

- 已完成：[单一CORN-inspired配对](SW_CONDITIONAL_ORDINAL_RESEARCH_20261002.md)。核对CONT占约30.31%的默认人口权衡、现directCE已连续归一化、TCN/GRU/body微调/TabM/8组层级/RNC/CDF校准历史边界，避免将低CONT归咎一个未经证实原因。
- 唯一保留机制：fresh297well body/FiLM/state +48→255条件survival头，一阶段joint；ordered vs固定seedshuffled chain再逆映射原bin。两臂uniform新q初始化；保原CE/RPS，不加ordinal loss，original104固定、不加载297/310权重。主源CORN原论文/作者代码；只是迁移而非完整复现。
- 未做：训练代码/GPU/Agent/成绩/显存时间/其他折，v310活动源码不改，不建立自动队列。
- 目标/后续：要求配对CONT正且绝对超过252、whole不降；默认微增吃掉CONT即关闭。明确256条件链尾人口/误差累积风险，无链长/反向/阈值loss网格；default5与strict第五TOTAL>80.8明显收益后补折/fullOOF>82未完成保持。

## 2026-10-02 PERM分组有理特征与读出研究

- 已完成：[原291固定宽度group-rational配对](PERM_GROUP_RATIONAL_RESEARCH_20261002.md)。KAT主论文/作者源码与历史关键词核对；区别既有PLE/RBF输入、目标CDF/RQS、CORN、DCN、curveattention及defaultCE。309默认小增益损CONT，不迁移/扫描。
- 首选待定：fresh291point两GLU价值响应与final categorical读出，anchored bounded rational versus同104有效shape标量的polynomial，同初始function/原48宽/2块/256格、original154only，所有newbody/readout一阶段joint。额外104标量约.285%，不是零参数/同表达族；不扩层/宽/branch。
- 未做：训练代码/GPU/子Agent/训练/成绩/实测时间显存或其他折。现有310不改，研究不进自动队列。
- 目标/关闭：要求CONT配对正、overall和绝对291不退；无收益不扫degree/group/bounds/LR，不泛化所有KAN。原数据/80-14/max2/每头单新增阶段/总≤8与strict第五>80.8+明显收益后补折/fullOOF>82目标保持。

## 2026-10-02 prior固定mix纠错限制核查

- 已完成：[固定prior纠错限制有限研究](SW_PRIOR_CORRECTION_LIMIT_RESEARCH_20261002.md)。核对POR290/292、PERM281/283/298、SW285/294/262/263/275真实alpha和连续人口；294是固定share的sparsebody，不是allocation。推导mass floor与TV≤alpha，同时确认directCE独立alpha、不等同body梯度饥饿。
- 唯一保留：PERM298 conditional allocation的局部CONT+.053668机制，低优先级迁移到fresh297well已有state/body/density joint；conditional vs scalar，nominal.25/实际[0,.5]固定原机制，original104 only，不加载旧repair。披露48 active参数差，state补回默认分尚未证实。
- 未做：训练代码/GPU/Agent/成绩/计时/其他折；311和312保持主Agent管理，没有自动队列。
- 目标/止损：关闭单纯fixedalpha扫参；配对CONT不正/整体仅default正/仍不胜252即关，不追加state阶段或界值grid。default5、strict第五TOTAL>80.8明显收益后补折/fullOOF>82未完成不变。

## 2026-10-02 SW joint-proxy background research (research_generalization)
- 已完成：[SW_JOINT_PROXY_BACKGROUND_RESEARCH_20261002.md](SW_JOINT_PROXY_BACKGROUND_RESEARCH_20261002.md)。核实206已有Archie/DEN-CNL/井内归一化；旧GR条件残差第五TOTAL79.992319负，297 well/reference CONT17.502753/17.463982局部微正但低于25217.884868。
- 首选待主Agent决定：仅既有ARCHIE_PROXY的同井128输入bank固定q20背景差，对照64训练井等权固定参考背景；同初始化/容量/实际bank FiLM/Q50/损失/预算，仅一个追加坐标，同fresh phase，不加载297训练权重。
- 收益尚未证实；物理论文正文访问受限，文献事实仅引用可核实摘要，不声称估计Rw/水线/SW真值。无代码/GPU/子Agent/审查/成绩/训练计时，311/312未触碰。
- 未做/止损：训练与全144665组合归主Agent；q20一次无CONT配对正收益即关，不扫描分位/斜率/物理参数，不增检索/教师/预训练/阶段。严格第五TOTAL>80.8后才补折，fullOOF>82未完成。

## 2026-10-02 SW point-query well-bank history (research_generalization)
- 已完成：[SW_POINT_BANK_ATTENTION_HISTORY_20261002.md](SW_POINT_BANK_ATTENTION_HISTORY_20261002.md)。发现v47直接同型整井128 input bank learnedK/V，section query对window-mean query，全骨干/FiLM共同训练；两组best0，真正训练后SW73.476844/73.473301低父约0.204/0.208，section对shared负0.003543，所有memory参数实际更新。
- 结论：不建议把raw9/stem48改动当新bank-query路线继续优先开pair。fresh连续监督/更早更细query并非数值等价，文档保留迁移边界；不能泛化所有attention失败。v109内容query对几何query负0.087760为辅助证据，277/278的FiLM、295曲线选择和v16/17窗口latent不误判等价。
- 未做：代码/GPU/子Agent/审查/新成绩/benchmark；313/311/312未动。后续若明确重开仅固定32宽2头早层query一次，不叠313、不扫bank/头数/宽度，仍同fresh phase与原Q/预算/解码，CONT不正即关。

## 2026-10-02 SW输入bank条件矩阵核研究

- 已完成：[单一CondConv低秩MLP核迁移](SW_BANK_CONDITIONAL_KERNEL_RESEARCH_20261002.md)。核对47银行crossattention、296四member、310depth、311chain、313proxy负边界；明确非对角矩阵核与297对角FiLM/旧输入低秩校准/预测专家router不同，不称首次conditioning。
- 唯一候选：fresh297三MLP的48→96矩阵加4rank8 kernel bases，共享inputbank router，actual vs训练inputreference kernel routing；两臂原FiLM仍actualbank，原104only，body/router/kernel/state/density单新增阶段joint。新增15444参/估计80709，比父+23.66%，披露reference有效容量差和零U首步梯度。
- 未做：训练代码/GPU/Agent/成绩/实际耗时显存/其他折；314/315活动源码不动，不建立自动训练队列。
- 目标/止损：寻找实质CONT提升，不能承诺单头+.3～.35或80.8；actual CONT不胜reference或收益靠DEFAULT即关闭，不扫rank/experts/temperature。315allocation未证瓶颈，不作为新阶段前置；default5/max8/max2/strict第五TOTAL>80.8明显收益后补折/fullOOF>82未完成保持。

## 2026-10-02 SW原九曲线学习阈值的L/LR研究（residual_neural）

- 已完成：[SW_INPUT_LINEAR_RELU_RESEARCH_20261002.md](SW_INPUT_LINEAR_RELU_RESEARCH_20261002.md)。核实PERM276 PLE仅局部+0.010672、POR305 PLE迁移-0.008901、旧周期编码完整OOF负证据；区分152专家product、192 MogaNet、302 token通信、314隐藏有理响应与输入数值编码。原论文及作者L/LR源码已查阅，不称首次数值嵌入或保证大幅提升。
- 唯一推荐：主Agent拟定v317 fresh297well、original104固定，九原曲线各8维仿射，LR接ReLU而L省略；72→48无偏置零出口加在stem Linear之后、原GELU/FiLM之前。两臂各加3600参数、总68865，原新encoder/state/FiLM/readout共同训练。L可折叠回原仿射stem，注册/优化器参数匹配不等于相同函数容量。
- 未做：本研究未写训练源码、未运行GPU/预检/审查、未派Agent，无新成绩/显存/训练耗时或其他折。v316活动代码不动，无自动队列；研究记录不是实验完成。
- 收益/止损：目前尚未证实。按保留80.695707958008，仅SW需whole增量>0.297977才过80.8；默认不变约需CONT+0.9824。若最终LR不同时胜L的whole/CONT，则关闭本部署，不扫维度/频率/阈值/激活/LR。微正但低297 whole或252 CONT仅记局部证据，不夸大。
- 约束与后续：原9曲线/分折/Q50/评分/80-14/max2、单头仅一次新增阶段/总≤8保持；无repair权重串联、teacher/priorKL/检索/新增预测voting。实际三头严格第五>80.8且有明确收益后才讨论补折，完整OOF>82仍未完成。主Agent实施后维护实验记录/EXPLORATIONS/图谱，单项与同工作量总墙钟分别报告。

## 2026-10-02 SW共享冲突与先验相对辅助读出研究

- 已完成：[same-link aux prior-offset](SW_PRIOR_OFFSET_AUX_RESEARCH_20261002.md)。精确v11/301/304/v300/utility/采样排重；304实际负内积均值38.519%但投影无收益，不能定位普遍有害共享。263/275正确mainposterior CE、aux仍standalone；219/269/242主CE不是同raw四branch aux配对。
- 首选待定：actual/aux均同delta=gamma*3tanh(raw/3)，only aux offset0 versuslogp104；original104/fresh297全部newbody/FiLM/state/readout joint一phase，65265同参数。primary Menon ICLR2021+Google utils加logitoffset，不套balanced理论。
- 317边界：两臂均terminal（LR31/L40），LR CONT17.469774、L best26CONT17.477643，最终CONT配对−.007868，关闭旧LR设置；318两臂不复用317LR，退fresh297，无其权重/新增repair阶段。
- 未做：源码/GPU/子Agent/新成绩/实测时间显存/其他折，现有任务不改，无自动队列。原单phase/max8/80-14/max2、训练隔离与strict第五>80.8+明显收益后补折/fullOOF>82保持。

## 2026-10-02 original104原生joint adapter历史核查

- 已完成：[104 warm-start联合adapter核查](SW_ORIGINAL104_JOINT_ADAPTER_AUDIT_20261002.md)。更正104 SW为8Gaussian非StudentT；识别真实LongSection/FiLM/65tap/attention/128输出、原NLL/BCE+utility1/SAM.1/DropKey.4。251/260已同源full continuousbody微调失败，277/278同源feature+新categorical失败；225已6272函数保持FFadapter小正但原body冻结，242 lastFF+masspatch失败。
- 唯一建议：完整original104全nativeSW参数3e-5 control versus加v225同位置零FFadapter3e-4，两臂保原Gaussian gate/density直接输出/原loss/SAM/DropKey，不冻结prior混合、不载297/317/318repair、不reset原output。未覆盖的只是全state+density原loss联合交叉，不称首次全骨干/adapter。
- 未做：源码/GPU/Agent/训练成绩/实测时间显存/其他折，没有自动队列；容量845913/852185与SAM扰动耦合明确披露。
- 目标/止损：实用CONT和whole收益；best0或CONT不胜control/默认微增吃掉CONT即关，不扫width/layer/LR/T3。原数据/oneextra phase/max8/max2/default5/strict第五TOTAL>80.8明显收益后补折/fullOOF>82未完成保持。

v319事件更新：主Agent实现并启动native104 joint control PID440；研究仅补状态，未查GPU/改训练代码。对照fresh6272 code225 adapter、原参数3e-5/新增3e-4、原loss utility1/SAM.1/DropKey.4；不同容量/SAM norm耦合与251/260历史负结果不变。启动不算收益，等待主流程终端实测。

## 2026-10-02 native104 masked GRN encoder有界研究

- 已完成：[GRN与静态affine同容量对照研究](SW_MASKED_GRN_ENCODER_RESEARCH_20261002.md)。原104 frequency、190wavelet、137pyramid、127ConvNeXt-GLU+GRU和310短depth均有负/弱证据，未发现可称更强完整骨干的替换。限定搜索未见GRN实现，不能称穷尽历史。
- 唯一低成本候选：original104最后globalFF LN256后/output前maskedGRN vs同512参affine，两臂gamma/beta0、nativebody/gate/Gaussian readout joint；原3e-5/新3e-4、原loss/SAM.1/DropKey.4、oneextra阶段。参数846425，不扩宽/改头/MAE或releasedweights；官方ConvNeXt V2论文与GRN/Block代码已核查。
- 未做：源码/GPU/Agent/训练成绩/实测成本/其他折；319活动源码不动，先待319终端决定，无自动队列。
- 目标/边界：GRN仅跨token channel-energy竞争假设，不已证feature collapse、不能称强骨干或承诺+.3。CONT不胜affine/收益靠DEFAULT即关，不扫多层/norm/window；default5/max8/max2/strict第五TOTAL>80.8明确收益后补折/fullOOF>82未完成保持。

## 2026-10-02 PERM目标数值格邻接读出研究

- 已完成：[目标bin-axis残差公平配对](PERM_TARGET_BIN_NEIGHBOR_READOUT_RESEARCH_20261002.md)。核对291/293/312/307/255/268/191，未发现检索范围内等价bin-logit卷积；精确区别input-depth pooling、hidden rational、coordinate scorer和IQF，不宣称全历史首次。
- 唯一有限候选：fresh291point/original154only，全newbody/readout joint一个新增phase；共享Conv1→8→1/k5/GELU/reflect，zeroout、残差.5tanh(D/.5)，仅ordered versusseed20261320端点固定内部乱序。两臂89新参/36555同初始化，main与finalcomponent aux读同修正logit，不叠312权重/代码。
- primary原2016 CDE-TF与2017 SDP/作者tansey源码支持目标拓扑启发；新Conv不复现其惩罚/树方法、不加loss、不套论文收益。bound、第二层bias后续有效性/首步延迟、full隐藏512MiB估算及chunk不自动消saved-memory均披露。
- 未做：研究未改源码/GPU/子Agent/训练/新成绩/实测成本/其他折；主Agent准备320而319两任务未结束，不自动排队。收益尚未证实，CONT不胜乱序/仅default增益/bothbest0关闭一次k5设置，不扫width/kernel/bound/置换。
- 目标/后续：局部配对与绝对291/312及实际TOTAL分开；oneaddedphase/max8/max2/原数据和80-14，strict第五TOTAL>80.8明确收益后补折/fullOOF>82未完成保持。

## 2026-10-02 POR151末encoder/native joint历史核查

- 已完成：[151 late/native +fresh302近重复核查](POR151_LATE_NATIVE_JOINT_HISTORY_20261002.md)。发现266已exact151 copied两expert dec0/combine34688参+已有151context_head joint265632active，control/V-REx都best0/natural14；226/228 context-only也best0。149是frozen113+freshcomponentreadout，不能当151warmjoint控制。
- 结论/计划：不推荐将“解冻151末encoder+读出”包装高幅新路线。精确fresh302full+可适配151prior交叉未见，但证据仅支持低优先级有限scope，不足更大收益推断；如主Agent仍选，仅同fresh302冻结对joint copiedlate/context一phase，不载302weights、不扩原scale/gate/全SOURCE。
- 未做：代码/GPU/Agent/成绩/实测成本/其他折；320活动不改，没有自动队列。
- 风险/目标：top302 no_grad、140/100 no_grad/capture.detach多重阻断需显式copyforward，requires_grad alone不够；freeze gateparams也不固定state预测，SOURCE.main_head非真正Packed目标读出。联合不胜frozen302即关，不延伸全源/LR/scales。oneadded/max8/max2/default5/strict第五TOTAL>80.8明显收益后补折/fullOOF>82未完成保持。

## 2026-10-02 SW私有跨目标latent联合骨干研究

- 已完成：[private151/154→freshraw joint历史核查](SW_PRIVATE_CROSS_LATENT_JOINT_RESEARCH_20261002.md)。更正230是own154hidden而非cross；215/216已做sharedSOURCE→SW冻结patch且source−repeat−.001143；151/154底层main_head前hidden同SOURCE，直接借会重复。34预测描述/143SOURCE头/277278own encoder/266319 joint负证据分开记录。
- 唯一有限候选：original151私有expand前256+154私有expand前128 versus104.output前128 repeat3；donor冻结、fresh297body/FiLM/state/readout+单LN384→48GELU→zero48 bounded.25fusion共同一phase。affineLN版本21600新增/86865active，同初始化；有效feature rank不同明确披露，不声称纯同容量/新teacher。
- 来源/约束：仅原151/154/104折内权重；63幅值side原9曲线input-only/v20训练64统计，amplitude与manifest SHA已CPU核验写入文档。donor原256窗口约束有215失败证据，必须中央256与SW1024中心128精确对齐，不能只把1024称domainshift而忽略执行风险。
- primary：Progressive Neural Networks lateral feature迁移与Cross-stitch activation sharing；新模块非完整复现，不使用外部预训练/预测/KL/其他目标真标签/repair链。未查到可靠Cross-stitch author代码，不拿第三方或404冒称官方。
- 未做：源码/GPU/Agent/新成绩/实测成本/其他折，无自动队列；cross不同时胜ownCONT/whole或收益只DEFAULT即关闭，不扫source/layer/width。source5+单SWextra，配其他每head各一extra总≤8，max2/80-14/strict第五>80.8明确收益后补折/fullOOF>82未完成保持。

## 2026-10-02 POR窗口统计可学习归一化核查

- 已完成：[DAIN-inspired有限window/reference配对](POR_ADAPTIVE_WINDOW_NORMALIZATION_RESEARCH_20261002.md)。v20原206raw+63shape已有，matchedCONT+2.532936但绝对负；更早adaptive_center已206可学窗口mean移除、第五TOTAL79.391379/POR75.806728。不把rawbypass/中心化/18RevIN affine包装首次。
- 唯一候选：fresh302full/original151only，九curve-token前9×9 shift+9×9 bounded-positive scale，当前valid256统计vs64训练input reference；同162参/76754，同zero302bridge初始实际函数。原raw206+relative分支与原151/63side不改，全newencoder/token/readout同onephase joint。
- primary RevIN/DAIN论文与官方代码已读；九inputs→POR1不做outputdenorm，单gamma/beta可折叠，DAIN工程正exp/去gate改造明确，不完整复现。270是latebody不是embedding、282是SWjoint不是PORmulti，避免历史名称误引。
- 收益尚未证实，不能承诺高幅；仅default改善/CONT或whole不胜reference即关，不扫window/floor/gate/LR。有效容量/reference矩阵弱梯度、窗口anchor变化与旧206形态冗余风险披露。
- 未做：训练源码/GPU/子Agent/新成绩/实测成本/其他折，无自动队列。原数据/80-14/max2/每head+1/总≤8、strict第五TOTAL>80.8明确收益后补折与fullOOF>82未完成不变。

## 2026-10-02 POR private latent迁移可行性核查

- 已完成：[fresh302 own151256 vs PERM154128+SW104128](POR_PRIVATE_LATENT_TRANSFER_FEASIBILITY_20261002.md)。准确捕获expand pre256/128和SWoutput pre128；151/154保持256source，SW256对称384:640 zero/validfalse padded1024/crop合法但有mask/position/context漂移，不能声称等价完整1024或无全井bank输入。
- 条件性计划：仅321最终有效后考虑一个同freshfusion15200参/总91792新参的PORpair，原151完整prior固定、新body/curveattention/fusion/readout同阶段joint；old来源151/154/104 samefold，无302/321repair/预测特征/新label输入。SOURCE provenance与五阶段+POR1=6/整体≤8记清；未来每fold需ownsource。
- 未做：代码/GPU/Agent/成绩/实测耗时显存/其他折，没有自动队列；321早期74.143435 vs74.124119仅live未结论。
- 边界：计算上安全依赖至少1validkey、mask/crop/alignment和independenthook，不保证更大CONT收益；cross不胜own或绝对不胜302不继续维度/层位扫参，default5/max8/max2/strict第五TOTAL>80.8明显收益后补折/fullOOF>82未完成保持。

## 2026-10-02 321 own连续收益来源与latefusion有限研究

- 已完成：[321 authoritative收益与ownlatefusion对照](SW321_OWN_LATENT_GAIN_AND_LATE_FUSION_RESEARCH_20261002.md)。own74.166030/CONT17.911355，vs297CONT+.408603、vs252+.026487；crosswhole微+.000561但CONT-.280225。fusion各层与末层确更新，不支持失活说；own128repeat3信息rank≤128，task/context/capacity混杂不能因果归单个来源。
- 唯一建议：fresh321 own earlystem versus同86865参/初始化latent模块移至finalfuse，原104固定、新rawbody/FiLM/state/readout/fusionjoint一phase，不载321repair、不加315alpha/gate。保证state capture也读注入后h；原4aux计数不改，披露位置导致辅助梯度路径变化。
- 未做：代码/GPU/Agent/新成绩/实测耗时显存/其他折，没有自动队列。250nativehiddenadapter+masspatch等历史不同但负证据保留，不宣称首次latefusion。
- 目标/止损：lateCONT不胜early或whole靠DEFAULT即关位置设置，不扫width/LR/LN/bound；保留own321实际optional8TOTAL80.697807943123，严格80.8/fullOOF82仍未完成。oneextra/max8/max2/default5/原数据隔离保持。

## 2026-10-02 POR raw/归一化位移同token有条件增补

- 已完成：[单2scalar tokenizer历史与初始化](POR_RAW_NORMALIZATION_DELTA_TOKEN_RESEARCH_20261002.md)。读取实际322纠正研究初稿：shift/scale均0、r0=u，故δ=r−u初始0，无需A=I/fixedanchor。v20已有raw206+shape63但融合在view hidden；25是frozenlocation、305是PLE，限定搜索未见同curve token独立raw/δ双投影。
- 仅若322终端fairwhole/CONT正仍成立，保留一项futurefresh双tokenadaptive/referencepair；Linear2→32两列都复制原w、bias原b，同旧302初始数学function，+32总76786。两非零列避免normalizer零输出×零secondweight死锁；bridge/δ造成真实梯度延迟披露。初期代数同w*r，后续权重分离才成为独立幅值通路。
- 当前322+.007907whole/+.036652CONT是未结束快照、绝对低302，不写最终机制正。新pair可验证其统计来源，但历史single322少32param，不可只凭历史差因果归tokenizer，禁止另填0死列控制/自动增加更多arm。
- 未做：源码/GPU/Agent/训练成绩/新成本/其他折，无自动队列。原151only/freshallnew joint一phase/max8/max2/80-14和strict第五>80.8明确收益后补折/fullOOF>82未完成保持。

## 2026-10-02 SW nonlinear morphology backbone有限研究

- 已完成：[SW latent max-plus/min-plus单模块候选](SW_LATENT_MORPHOLOGY_BACKBONE_RESEARCH_20261002.md)。原始MorphoN论文/作者官方层源码已读；只一组fresh321own earlyfusion第一ScaleMLP后的k7残差，depth邻点 versus同点channel循环邻居，同LN/structuring/P/zero末投影5424参，预计active92289。root独占训练代码与任何后续实验目录（包括326），本研究不建立实验目录。
- 准确历史边界：310已有depth vs channel但线性DW且CONT负；15adaptive/latent/SSM、18–24 coldbody/scaling/shape、127实际POR、137joint含teacher、190实际PERM、277/278copySW、266PORlate、305/322数值编码均排除近重复。原206已有median/range，不称首次形态信息；提案新意限定可训练latent峰/谷运算，未证明高幅收益。
- 参数初始化/风险：Pzero保留fresh321初始函数，step1P/step2选中structuring才可能更新；两臂winner梯度都稀疏，alltaps有作用但不保证每步grad。invalid按max−inf/min+inf后中心安全清零；1024采样行序不补偿不等距。相同参数数目≠严格有效函数容量；硬max噪声放大/薄层抹平/既有range冗余披露，预估buffer与计算并非实测显存/耗时。
- 唯一计划：仅父Agent在PERM325与POR324终端仍未突破后决定是否保留为SW下一候选，无自动launch。两臂104only、不载321/297/310repair、保留ownpositive代码/损失/readout/bodyjoint一phase；CONT及whole均需胜配对并绝对胜321own，defaulttradeoff或微正不算高幅目标完成，不扫k/位置/温度/LR。
- 未做：源码/GPU/实验目录/Agent/新成绩/实测成本/其他折；default5/max8/max2/80-14/原9与206/strict第五>80.8/fullOOF>82保持。研究结束，有实测收益后再最小最终核验，不追加独立复查。

## 2026-10-02 PERM输入选择记忆联合骨干研究

- 已完成：[单双向选择记忆与window条件transition配对](PERM_INPUT_SELECTIVE_MEMORY_RESEARCH_20261002.md)。读291/312实际stem、selector、GLU与rational，核对271/272/273、293/298/300及SW294/296和v15固定tau状态空间；320目标轴拓扑微改已关闭。限定历史未见本族point-vs-window input-dependent transition同型配对，非宣称从未做过SSM。
- 唯一候选：original154only、fresh312rational全新body/readout共同onephase，在首GLU后和第二selector前加共享48维双向记忆；逐点LN输入gate vs validwindow均值gate，其余同7056新增／43626active、同function初始g=1/16且zero输出、原Q/loss/alpha.3/80-14。不是head-only、完整Mamba、teacher或追加旧repair权重。
- 已给精确8阶段FP32 affine-prefixscan：组合(R∘L)=(a_Ra_L,b_R+a_Rb_L)，同步旧A/B、invalid=(0,0)、反向flip；不需作者CUDA、不用256Python递归/cumprod除法、不宣称O(L)或实测速度。首步u/g梯度延迟、19.29%新支路参数增长及100MiB以上可能额外autograd内存明确披露。
- primary论文Mamba与作者selective_scan源码已浏览；只迁移输入相关传播/遗忘，连续收益未证实。若selective不同时胜window whole/CONT或只DEFAULT好，即关闭；局部正与绝对312/TOTAL分开。未做训练代码/GPU/Agent/实测/其他折，不安排队列；原数据、每head+1/max8/max2及严格第五>80.8明确收益后补折/fullOOF>82未完成保持。

## 2026-10-02 SW joint train-input reconstruction有限研究

- 已完成：[joint-vs-detached masked auxiliary representation](SW_JOINT_MASKED_RECONSTRUCTION_RESEARCH_20261002.md)。SupMAE原论文/官方joint源码及Ti-MAE/SimMTM主来源已读；旧600step masked pretrain首折81.754非当前第五对照，v45已joint重建且POR73.2697/负4.2822，明确否认重建首次/确定大收益。contrastive_global是attentionmap相减非InfoNCE；v40井对抗及style负结果排除。
- 唯一低优先级候选：fresh321own cleanmain保持、第二maskedview共享fresh297body/FiLM/fuse，不运行original104/ownfusion/cleanhidden；Linear48→32→9 decoder1865参，两臂88730active。joint vs h.detach辅助梯度控制，aux同mask/target/loss值/decoder训练，新增真实encoder梯度而不加推理架构，不包装成已证强骨干。
- 原206+relative=207；按当前metadata遮所选curve全部派生/alias、unknowncross隐藏，auxbank同curve遮挡；主bank和104保持clean。训练井已知输入core128 targets、不用labeled/default/Q人口筛选，不存densecache；缺失target与imputed解释披露。整curve恰好一个均匀mask、loss固定.1/3 SmoothL1，无mask/weight/width grid。
- 已明确：v45强负证据使低优先级，reconstruction可学无关因子/损伤thinlayers；初始clean函数相同但梯度路径和实际计算成本不同。任何收益需joint比detach及绝对321CONT更好，不能仅inputloss下降/default微升。原104only/fresh全部joint一phase/default5|max8/80-14/max2/strict第五>80.8/fullOOF82仍未完成。
- 未做：源码/GPU/Agent/新实验目录/实测成本/成绩/其他折；父Agent等待326terminal决定，无自动launch，不追加预训练或独立验证流程。研究结束。

## 2026-10-02 SW shared morphology first→all coverage条件研究

- 已完成：[同5424形态模块三parallel branch覆盖](SW_SHARED_MORPHOLOGY_COVERAGE_RESEARCH_20261002.md)。实际326纯函数模块/LN无runningstats可安全复用，3ScaleMLPs是并行同stem而非串行stack；同Morphology单注册/单group/branch0只hook一次、再branch1/2。first/all均92289active，zeroP同fresh321初始数学function，研究未数值验证。
- 仅若326terminal仍CONT/whole胜321保留一个futuredepth固定pair：freshfirst vs freshall，共享同SE/LN/P、同种子/损失/原104only/own早融合；无326repairweights、新stage、axis/kernel/LR/aux网格。326live两臂CONT约+.052/+.051而paired仅+.001047，不能把公共非线性收益归depth。
- 310是三个独立linearDW不同k，57独立FF扩容，75同模型跨pass/dropout，302token通信scope；限定源码未见准确形态first/all重复，不称首次共享思想。MorphoN和ALBERT主来源说明借鉴范围，ALBERT串行depth与本parallel明显不同，不搬weights。
- 明确风险：共享梯度sum/clip增加、独立branch通道语义冲突/多样性下降，4aux数量同但模块从1→3条direct梯度路径，参数同不等于有效容量/计算/优化同；不除3/改LR。算子调用与activation约3倍，whole模型不一定3倍，显存时间未测。
- 优先同批freshpair；若使用historical326first，预先等epochceilE、原80cosine保留、到E记bounded预算结束非natural14，早停提前则披露不等工作量，不能宣称paired因果。allCONT/whole不胜first或绝对不胜terminalbaseline则关闭覆盖，不扩网格。
- 未做：代码/GPU/新目录/Agent/实测/初始预测核验/其他折；研究结束，无自动队列。maskedjoint研究不实施；default5/head+1/max8/80-14/max2/strict第五>80.8/fullOOF82不变。

## 2026-10-02 POR latent形态机制迁移核查

- 已完成：[original151only／fresh302full迁移326形态运算](POR_LATENT_MORPHOLOGY_TRANSFER_RESEARCH_20261002.md)。读326实际源码和终端JSON、302/287forward及终端分层；326depth-channel CONT仅+.0010474，两臂均vs321约+.052，不能把公共包络非线性收益归depth，POR收益尚未证实。
- 唯一有限筛选：76592+5424=82016，firstScaleMLP完整输出hook后由原mask/features/fuse/aux读取，原9/206/63依赖和151prior固定，freshbody/curveattention/readout/morph全部onephase joint；同原326 .25*tanh(P)、SEstd.01/seed20261326/zeroP。无326或302 trainedweights、SW预测/teacher/state/FiLM迁移，不叠SW327三branch覆盖。
- 历史206range/median是固定输入统计，127wellinputs.amax是井统计，v20shape是输入视图；限定POR源码未见同型learnedlatentSE残差，不宣称max/min首次。MorphoN原论文/作者代码已浏览；1D残差改写不称复现或前沿新算法。
- 先单depthvs历史302full（容量不同/非因果matched），只有whole78.030552794145及CONT41.675079907071同时突破才考虑freshchannel82016匹配对照；不phase串联/扫轴宽度。保留mask±inf、invalid中心安全0、原256行邻域/gap假设、winner梯度稀疏与首步SE延迟；约75497472projectionMAC/forward，真实peak/耗时未测。
- 未做：代码/GPU/Agent/训练/性能/其他折；不排队、不修改活动源码。默认5+POR1，所有head各一最多8；80-14/max2/严格第五TOTAL>80.8明确收益后授权补折/fullOOF>82未完成保持。

## 2026-10-02 SW positive morphology完整96body容量候选研究

- 已完成：[SW positive326 morphology width96](SW_POSITIVE_MORPHOLOGY_WIDTH96_RESEARCH_20261002.md)。限定历史未见同SW fresh326 fullrawbody/ownfusion/FiLM/state/readout/morph48→96准确重复；最接近PERM288 rawwidth96是负，明确降低预期。SW222仅frozen104 masspatch宽128负，57旧nativeFF并行grow负，123名scaled实际same845913 RMS-SAM无扩宽，277/278copyencoder与POR21–24不当同控制。
- 单一2倍候选算术262017active/原92289的2.839×；冻结104845913不扩，总1107930，未实例化但root329prepared读回同count。stem19968/3MLPs111456/fuse27744/readout24832/gate97/condition10816/ownfusion47040/morph20064。bankpoint32不变，code192输出，oldnorm128repeat3→LN38496→96，firstdepthk7和earlystemhooks保留。
- root负责V329源码/目录，研究不改code/GPU；两处zeroFiLM/ownfusion、zeroMorph/gate、非零categorical保持，alpha0初始原prior，不加载326repair。相同seed但形状/fan-in/梯度clip几何不同、参数/计算不同，不能纯机制因果proof，SAM0/DropKey0保持326，不能套104历史SAM.1/dropkey.4。
- 预定义≤27 wholebest diagnostic与正常80/pat14 wholebest分开；同80cosine不压27，若早自然stop披露完成量不同。historical48不算同批paired，full80超27收益不充等epoch结论。activation2x/多denseMAC4x、frozen104不变，12GB可行仅估算，未测显存/时间/收益，不写OOM或成功。
- 未做：代码/GPU/新目录/独立审查/numericalpreflight/测量/新成绩/其他折；root329prepared不等训练gain。只一个capacity候选，不沿128/192/256或LR/alpha/kernel网格；default5/head+1/max8/max2/80-14/原9与206/strict第五80.8/fullOOF82未完成。研究结束。

## 2026-10-02 SW固定half-mixture历史及梯度核查

- 已完成：[fresh326固定alpha.5覆盖核查](SW_FIXED_HALF_MIXTURE_AUDIT_20261002.md)。实际315范围0..0.5而非.125..375；scalar bestalpha≈.045009106，whole74.142090/CONT17.657627。允许.5不等固定.5训练，限定SW源码/结果搜索未见fresh326同型fixed.5 exactnegative。v83/84numeric_share.5是utility采样，v60/80/82是loss系数，非posterior份额。
- 数学已按实际269balanced/275aux推导：main hardCE responsibility=a*q_i/m_i；soft目标是各格responsibility加权和；conditionalRPS通过a*softmaxJacobian，残差也随a变化。aux四CE不乘alpha、RPS系数仍1，DEFAULT上densityCE/RPS清0。冻结prior不意味defaultaction冻结，也不能从.25推出成绩提升上界或已定位瓶颈。
- 最多一个有限fixed.5候选：original104only/fresh326depth92289body/state/FiLM/ownlatent/morph/readout alljoint一phase，仅5epoch ramp终值.25→.5；同初始化/Q/loss/9/206/80-14/budget，无329width或旧326weights。≤27wholebest历史参考与full80best分别报告，historical326非同批因果pair；whole/CONT任一不胜326即关、不扩alpha/bound/temp/decoder网格。
- 收益未证实，固定份额是训练-推理一致的有限假说，不是新骨干机制。未做代码/GPU/Agent/review/梯度测量/耗时/其他折，不授权launch或排队；原max8/max2/严格第五>80.8明确收益后授权补折/fullOOF>82未完成保持。

## 2026-10-02 SW latent state-density task router研究

- 已完成：[SW latent task routing](SW_LATENT_TASK_ROUTING_RESEARCH_20261002.md)。MMoE Google Research作者primary/DOI支持同expert任务gate，非标准完整复现/PLE，不冒称第三方代码官方。3freshScaleMLP latent concat144，2Linear144→3 zero870参数，softmax×3 weightedconcat，共享原fuse分别产density/state，93159active；shared两logits平均、task各自路由。只original104+fresh326代码joint一phase，alpha.25保持。
- 精确近重复：11独立TCN固定平均/硬分开，17local/global fixed层读出，29coupled/split冻结v28预测routing且SWsplit微负，299coarse/fine8bin归一化不是latent专家，301detach及304priorityPCGrad均负，275aux/315allocation/325memory/318offset不复用。不能称首次state/density分路；新意在输入自适应latent专家joint学习而非冻结预测mixture。
- 共享控制实际函数容量弱/差logit方向冗余，即使两router参数都获grad也不严格容量等效；taskstate首步zeroGate阻断、第二步有条件学习，shared可由density更新。四aux仍3expert+h_d、allbody/fuse共享仍有冲突；softgate可塌缩/放噪，不加balance/temp/aux/LR网格。
- root331已普通训练initial两组74.12175268976877、93159、groupgrad>0、peak1822.855MiB/e1-e2；returnNone和父capture后state覆盖正确。文档仅记父反馈，不GPU/preflight/review，新终端收益未证。root代码/目录/atlas独占；研究完成不继续观察训练。
- 未做：源码/GPU/新study/Agent/独立readback/其他折。sameonephase/head+1/default5|max8/80-14/max2/原9与206/strict第五>80.8/fullOOF82未完成保持。task需胜shared及绝对326whole/CONT；共同正不能归任务分离，无增益关闭单模块。

## 2026-10-02 POR residual RBF value-token有限研究

- 已完成：[POR302rawlinear+RBFtoken](POR_NUMERIC_BASIS_RESEARCH_20261002.md)。FastKAN2024原论文/作者官方Gaussian+base源码、NeurIPS2022 numericembedding主来源已读；只curveattention前numericLinear1→32保留，增加zero shared8→32无bias256参，new76848。RBF局部bumps vs同center/σ的smooth cumulative tanh对照，两armfreshjointbody/readout/原151only/onephase，不载任何repair。
- 原305同位置PLE替换负−.008901，314是latent48body/readout rational/poly非scalar，324rawdelta归一化/328morph/303corr/266V-REx不叠加；317SW ReLU CONT负、periodic已有、SW71Gaussian是physicaldepthattentionbias，不能称首次numeric/RBF大类。限定搜索未见准确rawlinear residual RBF numeric tokenizer，收益未证。
- traininputs-only九curveq05/q95均匀8grid；robustrange近常量fallbackfulltrainrange，fullrange近constant关闭basis，不epsilon窄峰。不label/default/Q拟合center；missing flags仍原206，填补点可能编码缺失而非地质信息披露。rawlinear保留外推，scalar不做LayerNorm；buffers/hashes固化。
- 0U同初始302，母bridge0让numeric梯度延迟需bridge先更新；samephysicalparams不等于basis函数类/梯度/有效容量同，对照不是重复scalar伪容量。RBF需胜same-runCDFwhole/CONT及绝对302，否则关闭8grid，不扫σ/K/LR/width/window。估计basis4.5MiB/37.7mMAC非测量，未preflight/独立review。
- 未做：代码/GPU/newstudy/Agent/参数实例化/成绩/其他折。root决定实施，研究结束；original9/206、original151only/default5/head+1|max8/max2/80-14/strict第五>80.8/fullOOF82仍未完成。

## 2026-10-02 PERM local edge-message骨干有限研究

- 已完成：[fixed7-neighbor EdgeConv式pair-message](PERM_EDGE_MESSAGES_RESEARCH_20261002.md)。DGCNN原论文与作者get_graph_feature/非线性/max源码已浏览；迁移center+neighbor差的消息代数，无dynamicKNN/retrieval/外部权重，不称完整DGCNN或2026新算法。
- original154only/fresh312rational stem输出、首selector前：无affineLN48，96→24GELU(center,neighbor-center)，valid邻域max vs mean，zero24→48后.25tanh；3528新增、40098active，同初始化数学function，全部newbody/rational/readout/message joint一phase。不要选diff-vs-absolute伪表达能力对照：第一线性层可互换；匹配仅检验max/mean聚合，不单独证明depth贡献。
- 限定历史未见exactedge消息；325递归、190wavelet、192fixedDW/addmul、307同点curveattention、320targetbin、旧v7冻体位置query均不同。326/328逐通道SEmaxmin同属极值大类但未跨48维编码center-neighbor pair；328CONT负是直接风险，原206range/median仍有冗余。旧DCN乘法已研究，非本轮推荐。
- 7tap runningreduce/noL×Lgraph，±inf/mean0mask、有效self、invalid中心安全0；48维pair6MiB/tap/message1.5MiB，约283115520MAC/forward非测量。zero输出首步edgeW延迟、maxwinner稀疏梯度/outlier/gap/上下方向对称限制披露，不许小参数当免费。
- 收益未证：max须胜mean whole/CONT，绝对31289.515804404879/74.002888404013及actualTOTAL单列；两臂绝对均负关闭，无k/width/LR/loss grid。未代码/GPU/Agent/review/预检/训练/成本实测/其他fold、不排队；原max8/max2/80-14/strict第五>80.8明确收益后授权补折/fullOOF>82未完成保持。

## 2026-10-02 SW variance-only DSU (research_generalization)
- 已完成：[SW_VARIANCE_DSU_RESEARCH_20261002.md](SW_VARIANCE_DSU_RESEARCH_20261002.md)。MixStyle三项真实历史核实：51最终回退/零，189对185负0.523100，267弱配对零；不重复其命名，未找到DSU batch统计不确定性随机尺度实现。
- 首选334已由主Agent明确采纳准备：原104only+fresh326 joint92289，无新增参数；stem FiLM及own-latent fusion之后，valid-window std的batch不确定性驱动B×1×48 noise，per-window p.5，relative amplitude .1*tanh，有界std-only保留mean/旧prior。identity同抽noise/masks/stat，仅返回原h；两臂所有new body/readout继续训练，一个fresh phase。
- Primary paper/official DistributionUncertainty源码已浏览；本案不是全DSU复现，保留均值/限正尺度/valid mask/perwindow trigger是迁移变体。标签语义仍可能受地层异质性影响，未证明增益或真实domain uncertainty。
- 未做代码/GPU/审查/子Agent/新成绩/benchmark，主Agent管理334训练；仅identity/noise两项，不扫幅度/p/位置或全mean+variance。CONT无配对正收益即关，第五严格>80.8后才补折/fullOOF>82未完成。

## 2026-10-02 POR302 curve token单次width64容量研究

- 已完成：[POR curve token width64](POR_CURVE_TOKEN_WIDTH64_RESEARCH_20261002.md)。原302full natural43/best29确认；76592→128336，curve19024→70768（numeric128/id576/2block66944/bridge3120），原raw57568与48readout不变。2blocks/4heads/FF128/numeric1→64/bridge64→48zero，original151only/fresh全部joint/一phase，不载302repair或332basis等叠加。
- 精确历史：21旧POR双expert内部FF函数保持增长；329/288是rawbody宽度；45coldcurvepatch64轴向+重建负；旧curveTransformer64是PLE10tokenFF192多目标且POR75.580负；135curvefamily32完整alignedbackbone负；6为旧PORlocation depthatt64。准确intersection限定搜索未见，不称64curveattention首创，负证据降低预期；原302微正/nondeterminism不足容量瓶颈证明。
- 预定义bestthrough43 whole-selected与nominal80/14终端分开，保持cos80；earlystop早于43披露不等量，不强force；historicalinit/params/计算/backend不等，不纯同容量因果或同轨迹。未numericalpreflight/review/实例化count，父trainer自然readback。
- B64/L256=16384 pointrows×9tokens，非256²depth；scoreFP32每block20.25MiB不变，tokenBF16 9→18MiB；估MAC2.616G→10.063G(3.85×)非零成本，raw/source不变。12GB/peak/wall仅估算，302历史peak1813.38MiB/full811.18s有并发非speedbenchmark；须实际记录。
- 未做：训练源码/GPU/newstudy/Agent/独立review/成绩/其他折；root决定单64实现，无width/LR/head/depth网格。保留default5/head+1/max8/max2/80-14/原9/206/strict第五>80.8/fullOOF82未完成；CONT退或full不胜关闭单候选。研究结束。

## 2026-10-02 PERM低秩featurecross有限迁移核查

- 已完成：[816参数stemcross/additive exactdup研究](PERM_LOW_RANK_FEATURE_CROSS_RESEARCH_20261002.md)。DCNv2主论文/Google官方Cross源码已读；POR306实际cross-additive+.004795594278但绝对低302，不是matchednegative。旧BatchEnsemble/rawinteraction、192局部mul+.0071565/绝对负、POR152product负均保留边界，不称新DCN大类。
- PERM255是154frozenhidden×目标coordinate只训头，307是numericcurve attention而非variational，v32/303固定输入corr非learnedcross；限定历史未见original154only/fresh312单stem816cross-vs-additive exact终端负。333neighbor、315allocation、312一元rational不同，不把absence当gain证明。
- 精确有限pair：stemmask内affinefreeLN48→无bias48→8→含bias8→48，cross .25tanh(z*r) vs additive .25tanh(r)，up0/down非零，同initfunction/816新增37386active；所有freshbody/shape/readoutjointonephase。首stepdown梯度延迟、二阶仅LN坐标/tanh前、函数capacity/梯度family不同、非完整DCNv2/非pretrainedtransfer明确。
- 相同原Q/loss/alpha.3/ramp5/seed/9/206/80-14，cross须同时胜additive whole/CONT，绝对312/TOTAL分别报告；负关闭rank8/singlelayer/.25，不width/LR/blockgrid。约12.58mMAC/forward，z/r各3MiB与rank.5MiB非peak实测，成本不免费。
- 父336准备接口仅报告状态，无研究Agent源审查/预检/训练/显存测量/GPU/新Agent/其他fold，不观察后续、不排队。原source154only/head+1/max8/max2/strict第五>80.8明确收益后补折/fullOOF>82未完成保持。

## 2026-10-02 POR curve token128条件容量研究

- 已完成：[POR curve token width128](POR_CURVE_TOKEN_WIDTH128_RESEARCH_20261002.md)。仅335最终whole/CONT正时保留单候选；livebest28不是终端收益，不自动launch/排队。original151only/fresh302raw48+readout joint一phase；numeric128/2blocks4heads FF256/mean/zero128→48，330128active、272560curve，较64+201792。
- 64→48桥接结构maxrank已48，128不再提高输出maxrank；只增加内部非线性交互，不承诺增益。定向检索未见同intersection128；v45的128是merge输出不是curve token，旧curve64/FFN扩容/rawwidth负证据明确保留。
- B64/L256估计39.454G curve forwardMAC（64的3.92倍），BF16 token36/QKV108/FF72MiB，非peak实测；12GB/max2非OOM或速度保证。终端335实际E预定bestthroughE与full80/14分列，非同容量配对。两指标任一不胜64关闭，不扫width/LR/head。
- 未做：源码/GPU/newstudy/数值preflight/独立review/Agent/benchmark/补折。保留default5/head+1/max8/original9/206/strict第五>80.8/fullOOF82未完成；root决定下一步，研究结束。

## 2026-10-02 POR depth-curve耦合有限研究

- 已完成：[POR局部低秩depth/curve mixer](POR_DEPTH_CURVE_COUPLING_RESEARCH_20261002.md)。首选仅fresh335两个64curveblocks之间加共享LN64/down8/17邻点+valid153→32→8/zeroUp64，6416新增、134752active；depth vs同点8latentcyclicshift单pair，原151only/freshbody-readoutjoint一phase，不加载335/337repair。
- 主源TSMixer论文及Google官方temporal→feature residual源码；本地局部低秩设计为迁移推断非官方复现。已明确v45完整coldcurve/depthaxial+reconstruction强负，v6冻结headlocalatt负、v18/19cold替换负、TimeMixer多尺度pool差异、310pooled48 gatedDWCONT负、325SSM负；不称首次depthcontext。
- 335whole/CONT对302递进正但CONT仍低151-.22945022；DEFAULT贡献为主，+.92555CONT目标只按默认不变人口推算，不能承诺。depth须终端whole/CONT胜channel并CONT胜335；两组共同正不归因depth，负关闭17/8/32单候选不扫范围/宽度/LR。
- B64/L256估新增.911GforwardMAC，down8后邻域38.25MiB避免17×64物化306MiB；非peak/速度测量。valid显式mask/井隔离/core标签 unchanged，17sample非等距物理范围有边界风险，同参数control有效函数类不同。未代码/GPU/newstudy/Agent/preflight/review/queue，不触337；default5/head+1/≤8/max2/strict第五>80.8/fullOOF>82未完成保持。
- 后续主 Agent规划v338：两臂预定bestthrough42/full80-14分列；torchroll禁止depth边界wrap、Network显式valid、curve整体optimizer组，不把bridge/up双零早期延迟误判永久无梯度。这里只记规划，未创建v338源码/目录或GPU。

## 2026-10-02 POR conditional有限RPS替换核查

- 已完成：[fresh335 joint→conditional RPS唯一候选](POR_CONDITIONAL_RPS_RESEARCH_20261002.md)。读287/335实际objective与151/153/150/228、genericcrps和SW252；限定未见当前151prior+fresh335同型replacement exactnegative。旧generic三目标analyticGaussianCRPS.2含POR的CONTmean额外项已存在，不能称第一次conditional；150是CDFwarp，153joint×2，228仍joint全样本，不误引为conditional负。
- 根据父最新澄清，仅一项CONT conditionalCDF RPS DEFAULT清0后(q*row).mean()/3，保持全人口归一；不加CONT分母/系数。仍同时去defaultgradient、CONT atom偏移和gate衰减：oldCONTgrad2(1-g)[(1-g)(F-T)+g(A-T)]、oldDEFAULT2(1-g)^2(F-A)，newCONT2(F-T)/default0。samecoef/mean非等梯度，defaultaction不保证稳定。
- 原151only/fresh335128336body/64curveattention/raw48/readout共同onephase，原CE/aux/alpha.25/Q/gate/decoder等不动、无repairchain。335whole78.042212635002/CONT41.724510830138为历史control、sourceCONT41.95396105另列；predeclared≤42prefix和full80/14wholebest分开，historical非同批因果pair。
- primary Gneiting/Raftery作者PDF有限RPS/CRPS与zero-adjusted BS原论文two-part语义已浏览；finite255threshold严格只辨格mass非全连续CRPS，Q/softCE/fixedgate/prior受限不称整体proper/calibration保证。候选whole/CONT须同时胜335，否则关闭，不cont归一/coeff/gate网格。
- 收益未证、未源码/GPU/新Agent/review/preflight/训练/gradient测量/其他fold、不授权launch或队列；原max8/max2/80-14/strict第五>80.8明确收益后授权补折/fullOOF>82未完成保持。

## 2026-10-02 POR fresh gate joint 历史辨别

- 已完成：[POR fresh gate joint](POR_FRESH_GATE_JOINT_RESEARCH_20261002.md)。当前335复用151frozengate，DEFAULT仍能因mass改变argmax；BCE原为freshbody常量，新gate会引入状态梯度/竞争。只保留一项fused48→zero1、±.5 logodds49参数候选，active128385，原151only/fresh33564body-density joint一次phase，不载335/337或338/339机制。
- 定向por源码未见精确samefreshintersection；133±1/best0、157±.125 vs151-.163102、26旧gatebest0是真实近负；150±.5是CDFwarp不是gate。SW269旧128head、275/289已有freshgatejoint、301detachCONT负、331routingwhole负，不能称首次或保证缓解冲突。
- 对照固定gate fresh335，同期优先否则明确historical；多49容量不装伪同参。维持已有普通BCE/mixedCE/jointRPS/directCE，不balanced/BCEweight/bound/LR网格；whole/default/CONT同checkpoint与335/151两层比较。whole或CONT退关闭单候选，零init不保证默认后续不掉。未源码/GPU/study/Agent/review/preflight/queue，不动338等活跃训练；default5/max8/onePORphase/max2/80-14/strict第五>80.8/fullOOF82未完成保持。

## 2026-10-02 POR continuous-support核查与关闭建议

- 已完成：[bin内offset与概率/loss/decoder同步边界](POR_CONTINUOUS_SUPPORT_RESEARCH_20261002.md)。287/335 actualcategoricalmass→fixedcenter kernel，原151早有continuouslearnedlocations；148/150/249propertilt/CDFwarp/value transport负结果末端仍CDFdiff→fixedcenter，不能称已训真正continuousutility exactnegative。316/317是inputbasis/embedding，非outputsupport。
- CPU只load originalfold5 savedcenters/edges/scale，无rowtruth/validation/oracle：POR256centers.205..33.177，log1pstep.013117935；首末±∞。254finitebin半width/8%容差grid节点分位数.08449/.08552/.09712/.19889/.45310；不是行人口/误差原因或oracle收益上限，未发现普遍粗于容差证据。
- 推荐本轮关闭简单offsethead：binCE不含位置，固定edgeRPS对留bin位置梯度0；动态Dirac不是continuousdensity。真正方案需normalizeddensity+matchingNLL/CDF和∫utility同步，sourcecontinuousprior保留，不能mean替积分或eval-only移动centers；已经非一个小模块，缺更大收益证据。
- 动态utility常规POR4096core×801×256FP32单tensor3.129GiB，32768点才25.031GiB；512chunk约400MiB，可降低peak，未OOM/速度实测。full144665约29.66Bpair；256smoothcomponent×255edges×16384train点单tensor3.984GiB+backward，成本边界明确，不能小参数当免费。
- BishopMDN与Gneitingproper主论文已浏览，只支持normalizeddensity/一致预测对象，不证明本项目瓶颈或效用gain。未code/GPU/Agent/review/preflight/rowlabel/oracle/训练/其他fold，不提候选/不排队、不观察339/340；原151only/freshcode/onePORphase/max8/max2/80-14/strict第五>80.8/fullOOF82未完成保持。

## 2026-10-02 POR train-only auxstate/frozen maingate

- 已完成：[POR auxstate frozen gate](POR_AUX_STATE_FROZEN_GATE_RESEARCH_20261002.md)。same340fresh128385+49ga仅训练BCE，主mixedCE/jointRPS/所有forward推断用original151g0；原BCE常量B0替换同whole/q/3N ordinary Ba，不doubleBCE/balanced/auxcoef。fresh33564 allbody/readoutjoint，original151only，不载340weights、不eval挑gate或roworacle。
- 340savedbest20/到27 whole77.977630697、CONT42.059591810/default97.127929036非终端，不早停；连续趋势不能归因auxBCE，候选改变RPS梯度与gate角色非纯因果。零initloss同但梯度不同，主RPS固定g0不等于339conditionalRPS。新density仍能影响default动作，出口冻结无分数保持保证。
- exact搜索未见当前POR同intersection；301detach去state-body梯度且新gate推断、304prioritygrad且新gate推断均近负；157旧hiddengate负、331routingtradeoff，309实际PERMdefaultdensityCE整体微正CONT负，不能误记SW。主源为历史实际源码/终端记录，无无界论文/分支网格。
- 只单candidate，完成后340 historical同128385对照+335/151报告；whole最佳固定、whole/default/CONT三层差，不CONT挑epoch。若whole/CONT不胜335关闭auxstate setting，不扫BCE/bound/detach/LR。未source/GPU/study/Agent/review/preflight/queue，不动340/341；original9/206/onePORphase/default5/max8/max2/80-14/strict第五>80.8/fullOOF82未完成。

### 2026-10-02 auxstate研究依据的340终端更正

- 仅更新事实，无新研究/源码/GPU/Agent。340natural43/wholebest29，POR77.992995342042、DEFAULT97.227550685164、CONT41.916925164121；vs335 whole-.049217292960/default-.178047203705/CONT+.192414333982；vs原151 CONT-.037035890784。此前中间best20 CONT42.05959被best29替代，不再作为finalnativeCONT突破，不回选CONTepoch。
- [auxstate研究](POR_AUX_STATE_FROZEN_GATE_RESEARCH_20261002.md)已改为最终连续相对335小正/whole负、仍低native151；一个可辨别候选而非已证恢复。341组合主Agent报告PID82236单独运行，auxstate仍无代码/启动；研究结束。

## 2026-10-02 SW326 central256 curveinteraction迁移研究

- 已完成：[SW curve transfer](SW_CURVE_INTERACTION_TRANSFER_RESEARCH_20261002.md)。单32full候选：numeric64/id288/2blocks17088/zeroBridge1584=19024，fresh92289→111313。original104only/fresh326 allbody/state/condition/latent/morph/readoutjoint；stem父FiLM→ownlatent→新curve，中心384:640计算、mask再padzero，保留prior完整1024与实际bank/128core标签。
- full vs同参302physicalgrouped连通mask控制，若只有historical326则明确+19024/非配对。lookup旧SW16headcorrection实际整个双expertjoint且SW负；135整体负但SW+.060665不误记；45curveaxial是POR非SWlongsection；32固定corr负；190Haar是PERM而非SWexact。同intersection未见，不称首次curveattention。
- 机会未证，CONT17.96低不证明信息不足；full须whole/CONT胜grouped及绝对326，否则关闭32/centralstem单位置不扫width/head/LR。中心2.616GMAC/token9/QKV27/FF18MiB估计非peak；全部1024会4×，不篡改prior。original9名字精确映射<206，无DEVI等；一次SWphase/default5/max8/max2/80-14保持。
- 未source/GPU/study/Agent/review/preflight/queue，不动341PID82236/342prepared；研究结束，strict第五>80.8/fullOOF82未完成。

## 2026-10-02 两项输入表征有界候选

- 已完成：[adaptive depth / ordered numeric](ADAPTIVE_DEPTH_AND_ORDERED_NUMERIC_RESEARCH_20261002.md)。A优先：original151+fresh341保留17点正模块，在curveblock1→temporal之间加5tap[-8,-2,0,2,8]contentoffset±2/weights/低秩8/zeroUp64，1874参数/136626fresh；fixedreference控制两套logits平均供weights避免dummydead，但effectivecapacity不等。原blurdeform首折负、15内层整体正CONT弱、45coldaxial负/71bias负明确；不称采样未试或物理depth不变。
- B低置信：original154+fresh336rational/cross保留，九curve8uniform PLE orderedtrainableincrements+zeroS→48 preGELU，最多3528新增/40914active，对照fixedsameuniformknots少≤72可训练参数明确。276局部+.010672但不胜274、305PLE替换负、332RBF/cumulative已学仍负、旧fixedhatinputcalibration负；不同点是当前位置/freshjoint/学习有序区间，不重开center/bandwidth网格。
- 浏览DeformTime2025/DeformableTST2024原论文官方仓库和2026orderedknots论文/作者仓库；纸面事实与本地迁移推断分开，均不承诺大幅收益。A估.245GforwardMAC/低秩5tap11.25MiB BF16，B56.6mMAC/basis4.5MiB FP32，非实测peak/时间。当前nativeCONT缺口保持；whole/default/CONT同checkpoint配对和native两层报告，不挑CONTepoch。
- 两项各onephase/head/原151或154only/freshcode不载repair/default5/max8/原9-206/80-14/max2；任whole或CONT无配对且绝对正关闭，不扩K/范围/width/LR。研究只写新MD与本目录PROJECTS，无源码/GPU/study/Agent/review/preflight/launch/queue；342/344由root调度，strict第五>80.8/fullOOF82仍未完成。

## 2026-10-02 PERM frozen-mixture SPO+ 有界研究

- 已完成：[PERM SPO+ mixture](PERM_SPO_PLUS_MIXTURE_RESEARCH_20261002.md)。只一个固定0.1候选：fresh336cross/rational原154only37386，同完整P@原K得s，max(2s-u)-2s_a*+u_a*，all-valid/Q全mean/3；原CE/RPS/directCE、decoder、alpha.3ramp、原输入/seed/80-14全部保留。同容量fresh336系数0对照，onephase/max8，无新增推理参数。
- 实际责任梯度与directCE不同，但没证明冲突：mainCE r_y=alpha*q_y/m_y，aux独立q；DEFAULT原jointRPS已监督，不能称无监督。SPO+最坏动作是argmax(2s-u)，不同243当前winner regret-hinge；243终端SW+.00817而非负。87softaction两best0、186加强utility-.01334、188kernel-.01500、318offsetbest0、309CONT损失等负边界披露，不重开温度/系数网格。
- 原论文1710.08005/作者sgd.jl及2021calibration primary均已浏览；非proper、neural非convex、低维labelutility+DEFAULT+frozenprior不满足可直接套校准条件。零新param非免费，1024chunk两score矩阵约6.26MiB，16384点P@K约3.37G MAC估算非实测。whole与CONT不能同时胜固定control则关闭，不迁移SW/POR、不扫参；不承诺大幅增益。
- 本批仅研究新MD/本目录PROJECTS，未源码/GPU/训练/Agent/review/preflight/queue，不观察346/345、不影响root调度。当前actual341 TOTAL80.704705566183和8phase仅第五；strict第五>80.8/fullOOF82尚未完成。
- SPO+文档最终语义修正：PERM kernel/metric无DEFAULT特殊分支，默认atom.01保留而u_y必须来自实际own train truth（不因dmask强制canonical）；复用336massEvaluator.actions/utilities与原objective wrapper，不另造decoder。训练真标签a*仅cost监督，禁止heldout oracle/上限/输入使用。历史336best5/stop19仅work参考，新SPO+0/0.1同37386cross控制。16384整window预算s50.06MiB/3.37G MAC，当前128core最多8192/s25.03MiB/1.686G MAC；可1024chunk但非peak/OOM实测。未实施/启动。

## 2026-10-02 original154 native encoder LoRA排重研究

- 已完成：[PERM154 native encoder rank4](PERM_NATIVE154_ENCODER_LORA_RESEARCH_20261002.md)。完整载原154warmcontext，两个真正expert enc2.expand128→256各rank4 zeroB LoRA1536，共3072；原weightsfreeze/context82720joint，85792active，同warm headonly82720控制，active容量差明确。Gaussian16/native解码loss原样，不randomraw70%prior/repairweights/teacher。
- 核心近重复270确实原154warm+dec0/combine实际latejoint，双best0/natural14且gradient非零；162末端adapter/166SOURCE全bone+freshcontext负，272fullcopiedexpert+随机directcategory不胜271，22POR冻结增长best0。不能称未做过bone或adapter，只enc2rank限制+完整warm/nativeGaussian表示共同变化交集未见exactterminal。
- 梯度边界实际154双层no_grad+capture.detach必须显式nativeforward贯通，参考166公式但载154trainedcontext；freezeparam不阻止后续ops保存反向。LoRA影响nativegate/route/sigma函数虽其weights冻，DEFAULT/CONT风险明确。rank4随机A/zeroB初始function数学保持但真实inittrainerreadback，firststepA梯度延迟。
- 浏览LoRA原论文/Microsoft官方Conv/Linear实现，仅借代码机制无releasedweights；语言证据不保证测井收益。估12.58mforwardMAC/参数Adam48KiB，主成本enc2后完整backward，非peak/速度/OOM实测。若接口无法正确梯度或whole/CONT不胜headonly及当前336关闭，不rank/位置/LRgrid/预算延长。
- 仅新MD+本目录PROJECTS，无source/GPU/train/review/Agent/模型实例化/study/queue，保留original154only/原9-206-Q50/80-14/onePERMphase/default5/max8/max2/strict第五>80.8/fullOOF82未完成；347solo不触碰。

## 2026-10-02 SW动态跨曲线频谱读取（仅研究）
- 完成 [SW_DYNAMIC_CROSS_SPECTRUM_RESEARCH_20261002.md](SW_DYNAMIC_CROSS_SPECTRUM_RESEARCH_20261002.md)：original104only+fresh326 joint，中心256九曲线FP32FFT512，其他8曲线同频log幅度门控目标曲线相位保留频谱，zero9→48 stem桥接；72offdiag mix+480bridge=552，估active92841。唯一同容量参考只置换donor非DC频率，保留目标/输出/输入行与loss；有效函数平滑性差异披露。
- 精确排重：104静态spectral-local−.147272193、316bank矩阵CONT−.001590055、旧静态spectral/FFT状态卷积、v32统计和curve attention均非此cross-spectrum输入条件模块；同类失败降低信心，不称首次频域/条件化。不提334DSU为未训练方向、不触碰345/350。
- 更正此前LoRA研究generic层估计：实际154enc2.expand128→128而非128→256；348shape启动失败后349修正，natural14/best0，encoder/head梯度非零/Bnorm增长仍无收益。旧3072/85792估计不得作为349真实容量，不重开LoRA。
- 已浏览FilterTS AAAI25原文与作者官方models，支持动态跨变量频谱过滤机制而非本bounded实幅度迁移/测井收益。512零扩展不保证数据依赖滤波无周期伪影；行索引频率非物理depth，不等距/mask/裁窗谱泄漏风险明确。首步零bridge延迟mix梯度；全freshencoder/readoutonephase、orig104frozen、原Q50/score/split/80-14/max2/总≤8保留。
- 仅研究MD/本目录PROJECTS，无训练源码、GPU、CPU模型实例化、Agent/review/preflight/study/queue。收益未证实；root自行决定是否实现。aligned与打乱参考及绝对326 whole/CONT不能同时胜则关闭，不扫频数/窗口/温度/宽度/LR；严格第五>80.8/fullOOF>82仍未完成。

## 2026-10-02 SW physical-depth/multiscale exact duplication (research_generalization)
- 已完成：[SW_PHYSICAL_DEPTH_MULTISCALE_DUPCHECK_20261002.md](SW_PHYSICAL_DEPTH_MULTISCALE_DUPCHECK_20261002.md)。64训练井全部depth step min/max≈0.1、nonpositive0；uniform row→meter重采样缺少新几何瓶颈。
- 发现关键直接历史：timemixer_well_sequence已有trend/detail分解、detail fine-to-coarse/trend coarse-to-fine三级mix，完整5折OOF80.303113 vs旧baseline80.950671负0.647557，第五78.612636；v137普通pyramid SW负0.041047。v71真physical attention及v15 row adaptive明确区分，不将bank FiLM或v295曲线选择误认同机制。
- 本轮关闭irregular resampling/普通pyramid/TimeMixer改名迁移，无新候选/增益；原论文/官方TimeMixer、NeuralCDE代码浏览，仅作适用边界，不否定所有多尺度方法。
- 未做代码/GPU/审查/Agent/队列/新计时，350/351及其它活跃不改。目标仍ownSW/104only+fresh joint onephase/max8、固定第五808/fullOOF82；后续由主Agent更换真正不同机制。

## 2026-10-02 SW351后续 phase coherency 条件研究

- 已完成：[SW phase coherency](SW_PHASE_COHERENCY_RESEARCH_20261002.md)。先等351permutedterminal，若同频pairedwhole/CONT无正关闭后续。aligned74.173612052233/CONT17.979399150047与326局部正，但实际80.706129723238未adopt，不冒称机制已证。
- 唯一条件pair：保留351全部552params/92841freshbody，5tap邻频crosspower/power形成rho，amp+.25Re(rho) vs amp+.25abs(rho)进入原gate；仅用训练输入、保留targetphase/zeroBridge/原104only全jointonephase。单频magnitude恒1不可用；fixedcomplexmix可能ordinarylinear不推荐。magnitude参考并非amplitude-only也非纯因果隔离，不承诺大增益。
- 查104/spectral32固定corr/v27预测descriptor/316bank/321323latent等negative，未见局部频率coherency同型已训。FilterTS原文/作者代码、SciPy官方coherence/源码已浏览；Welch DOI全文入口受robotchallenge未读，候选非Welch估计/新论文复现。跨频9MiB复pair/rho各9MiB，avoid45MiBunfold，非实测peak/耗时。
- whole/CONT均须胜magnitude并绝对351/326有用，否则关闭固定5tap/.25，不扫window/rank/LR。original9/206/Q50/80pat14/default5/max8/max2/fullOOF82未达、strict第五>80.8不变。仅MD/PROJECTS，无源码/GPU/review/Agent/queue/launch，不观察livecontrol。

## 2026-10-02 SW prediction-confusion监督表征有限研究
- 完成 [SW_CONFUSING_CONTINUOUS_REPRESENTATION_RESEARCH_20261002.md](SW_CONFUSING_CONTINUOUS_REPRESENTATION_RESEARCH_20261002.md)。精确近负260 wholebody+mass RNC双best0/early12；261adapter+mass RNC−CE−.006208653/early14，不能重新包装泛RNC/SupCon。metric_aux是输出utility、contrastive_global是attention相减，语义辨清；ConR原论文/作者官方函数、RNC官方loss及ACCon原文已浏览。
- 只保留一个中低优先未证候选：fresh351positive代码/original104only，fuse48真CONT labels近positive，真值远但ownconditionalq动作近的confusing negatives（动作detach只筛pair，不teacher/伪标签/检索），训练联合表征aux .1/3 vs auxfeature.detach control，同参数/初始函数；主encoder/readout/state/FiLM/Morph/spectral/原loss仍joint一阶段。温度.1/阈值log1.05/metadata row32去重8perwell256固定，不扫参。
- 256² pair张量约.25MiB、相似度3.15M MAC，新参数0/推理0；positive/negative空anchor安全跳过，若几乎无eligible监督则关闭而非扩阈值。Ownpred反馈使paired包含梯度+采样闭环非纯因果。共享state梯度竞争、噪声/多模态/井过拟合、近pair稀缺与earlyq不稳定披露；utility0不证明表征collapse，不承诺大增益。可由root直接研究关闭不实施。
- 351根确认SW+CONT正/optional8TOTAL80.706129723238；352/353由root管理不触碰。仅新MD/研究PROJECTS，无source/GPU/train/model实例化/review/Agent/study/queue。whole/CONT不同时胜paired及绝对351则关闭，不迁其他头、不重开RNC网格；original9/206/Q50/80-14/max2/onephase/max8/strict808/fullOOF82未完成保留。
- 最终物理语义修正：已查train_stage_fullcv_v97 Evaluator→package/src/utils.py kernel_matrix，SW真实容差.05*(abs(truth)+1e-5)、bounds8.305..99.9/default99.9。ConR pair最终用actualphysical trainingtruth及有方向容差，不用log1.05近似；若从275normalizedlabels取则exp(z*scale+median)仅一次。q256动作乘utilities[1:,:]排除atom行，保留原合法action99.9；只训练mask/Q/rawrow去重CONT，label/predpair均与anchor物理容差一致。

## 2026-10-02 PERM局部路径面积备用研究
- 完成 [PERM_LOCAL_PATH_AREA_RESEARCH_20261002.md](PERM_LOCAL_PATH_AREA_RESEARCH_20261002.md)：original154only+fresh336代码allencoder/readoutjoint，336stemcross后/firstselector前并行zero36→48/.25tanh残差1776，总39162估。9原curves原标准化输入、17点−8..8、16有效adjacentedges，gap resetprefix；area=.5Σ(prefix_i Δ_j−prefix_j Δ_i)，control=.5ΣΔ_iΔ_j，两臂同TVpair denominator max(T_i*T_j,.01)、featureabs≤.5，center valid/≥2edge共同eligibility，无labels/statfit/新增训练阶段。
- 排重291本就是TabNet式sparsemax重读GLU；293selectorlocal17−.001334881，325selectiveEMA−.002441064且低312，333max−mean whole−.006122156/CONT−.00936392，268prototype memory−MLP−.000562813。336同点bilinear、旧correlation/latent/pyramid/SSM不是本orderedantisymmetric累积，但同类depth近负保留。精确logsignature/signatory/iterated-integral/antisym无实现；inspect.signature误匹配排除。
- Signatory ICLR21原论文/作者实现、NeuralRDE ICML21原论文/作者实现已浏览，仅借order-area输入表征，不安装库/解ODE/外部weights；17点二阶不套universal/不等距robustness保证。不是首次depthcontext，不使用reverse-sign伪对照（自由桥可吸收）。Sameparam不等effectivefunction/featuredistribution；imputed curve伪面积和仪器滞后噪声风险明确。
- 最新356nativepredictionresidualbest0/14、357nativefeaturepatch+warmheadbest0/13已闭；355finalCONT-.26544虽TOTAL+.001216未采纳，detach未训无causalpair。新预算max80/min10/pat10until strictcurrentSOTA then14 latch已写，不自行改SOTA。仅新研究MD+PROJECTS，无code/GPU/train/review/Agent/study/queue，不加载trained336/repairs。area不同时胜paired与336whole/CONT则关闭，不阶数/window/time通道/gain/LR网格；onePERMphase/default5/≤8/max2/strict808/fullOOF82未完成保留。
- 最终路径面积单方案随root有限实现规格统一：原scaled point先tanh(x/3)，dx为相邻z差；A=.5Σ(prefixbefore_i dx_j−dx_i prefixbefore_j)，C=Σdx_i dx_j，均固定除16，不用早先TVpair/.01归一草案。Gap resetprefix/valid16edges/center≥2edges共享不变；|dx|≤2、runprefix≤2，因此phi保守abs≤4而非.5，同maxbound非同方差。只有一个normalizer无扫描，终稿MD已同步。
- Root最终358实现回报：ALL17 points valid/in_range为两臂共同eligibility，任一边界/gap descriptor与bridge残差均0，不使用研究早先多run/gap-reset累计或TV归一。39162joint/zero1776 bridge固定tanh(point/3), A=.5antisym/16,C=no.5crossvariation/16；rootcompiled/prelaunch回报非训练收益。固定strictPERM89.51838939879066对应TOTAL80.706129723238，pat10→strict突破后14 latch/min10/max80。终稿研究MD已同步；本Agent不GPU/source/review。

## 2026-10-02 true target residual supervision 排重关闭

- 已完成：[真实误差残差监督audit](TRUE_TARGET_RESIDUAL_SUPERVISION_AUDIT_20261002.md)。关闭仅改写y-frozenprediction为新机制：Huber(r-(z-f0))与Huber(f0+r-z)同梯度，合规真标签残差非pseudo，但没有新监督信息。SWmean实际beta.2/coef.2辅助NLL首折SW76.341913/TOTAL81.774688 best40/52；SWscalarlogHuber首折SW73.856836/TOTAL80.947211 best38/50，不误称当前第五/104精确负。PERM当前没有精确同型，迁移只是旧loss跨target近重复，未立项/无queue。
- 阅读356/357/85/145/train_residual/globaltrend：不同残差名字实际多为param/feature修补NLL非true-error Hub er；358快照/variation活跃不干预。Gaussianrho总和1，不可说fartruth所有gradient消失；.25SD/tanh可达域/饱和不能被更强pointloss突破。Hubertail有界/mean不是utilityBayesfunctional，风险和假说分开。
- N-BEATS原文/作者residualinput与forecast累加、NGBoost原文/作者natural-gradient拟合、ResNetprimary已浏览；不借名字重做basis/伪响应/树boosting/多stage或下载weights。用户残差方向保留，关闭范围仅代数重写/近重复loss，不泛化残差无效。
- 仅新MD/本目录PROJECTS，没训练源码/GPU/Agent/review/preflight，未观察active358；latestpat10→strictcurrentSOTA后14 latch、max80/max2/default5≤8/original151154104only/strict第五80.8/fullOOF82未达照root执行。

## 2026-10-02 fresh336 joint representation decorrelation 有限候选

- 已完成：[PERM_JOINT_DECOV_RESEARCH_20261002.md](PERM_JOINT_DECOV_RESEARCH_20261002.md)。原154冻结prior+fresh336 cross joint encoder/readout保留全部支路，readout实际48维输入协方差非对角惩罚，0新参数/0推理成本/one新增PERMphase；不是完整DeCov复现。原始ICLR2016作者论文/实验室PDF已浏览，未核验作者官方repo，后续PDF定位超时已注明。
- 精确排重DeCov/decorrel/orthogonal/VICReg/Barlow/whiten/hidden covariance源码未见；旧9曲线ridge precision派生输入不是学习隐藏协方差损失。MixStyle51/189/267及DSU334、RNC260/261、ConR355、physical-depth/TimeMixer/273/293/325、KAT312/PLE346/LoRA349/SPO347/336358区别和负证据保留，不换名重开。
- 唯一identity/coefficient.01有限对照，同37386 fresh参数/原154only/Q50/输入/seed/评分，rawrow%32去重每井8/max256 ownPERMcontinuous训练core点，C FP32/offdiagFro²除trace²+eps，分母不detach；系数.01/3在原损失外。既有body/readout均joint，无新norm/labelpair/teacher/pseudo/预训练/阶段/网格。
- 尚未证实收益或冗余瓶颈，不承诺>.25；PSD ratio有界但通道换基/真实相关因子/小batch/epsilon近常量退化风险。需报告whole+CONT+TOTAL、paired与fresh336/SOTA差，aux下降不是收益；失败关闭本有界定义。
- 本批仅研究新MD与本PROJECTS；无训练代码/GPU/review/Agent/queue，359等活跃文件不动。最新pat10→strictSOTA后14/max80/min10/b64/64steps、max2/default5≤8、strict第五>80.8及fullOOF>82照父Agent，不立项或自动补折。


## 2026-10-02 user-authorized artifact cleanup

Obsolete artifacts have been removed; SUMMARY_RESULTS.json and SUMMARY.md retain historical scores and conclusions. Detailed logs/caches/weights are unavailable for these obsolete trials. Current V370 original + seed20261013, retained-model dependency closure, reusable source, original data and deliveries are preserved. See D:/match_predict/maintenance_cleanup_20261002/VERIFICATION.json.
