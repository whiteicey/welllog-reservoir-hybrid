"""Human-readable editorial annotations. Re-run only when these annotations change."""
from pathlib import Path
import json
nodes={};edges=[];dead=[]
def node(id,title,summary,conclusion='',highlight=False,**kw):
    nodes[id]=dict(title=title,summary=summary,highlight=highlight,**kw)
    if conclusion:nodes[id]['conclusion']=conclusion
def edge(a,b,type='evolution',note='实验演进关系，见方法说明和对应实验记录。'):
    edges.append(dict(source=a,target=b,type=type,note=note,evidence='docs/deep_learning/EXPERIMENTS.md'))
def chain(*ids):
    for a,b in zip(ids,ids[1:]):edge(a,b)
def end(id,title,stage,parents,summary,reason):
    dead[:]=[entry for entry in dead if entry['id']!=id]
    dead.append(dict(id=id,title=title,stage=stage,parents=parents,summary=summary,reason=reason))

node('simple_cnn_cv','简单 CNN 基线','残差一维卷积，默认状态门控 + 连续值回归。','建立最初的完整五折基线。',True)
node('simple_cnn_anomaly_cv','CNN + 异常预处理','在相同 CNN 上评估异常处理。','完整 OOF 77.0432，低于原 CNN 77.1356；这一处理没有带来整体收益。')
node('cnn_mlp_cv','CNN + 逐点 MLP','并行逐点 MLP 与序列卷积，拼接后输出三个目标。','完整 OOF 77.1932；较简单 CNN 微增，成为后续神经网络探索起点。',True)
node('pure_dl_v1','默认原子 + 概率分布','由单点回归转向默认状态与连续分布，按官方效用解码。','完整 OOF 79.1545。',True)
node('pure_dl_rich','丰富输入视图','增补工程特征，仍使用神经网络学习。','完整 OOF 79.6011。',True)
node('pure_dl_residual','残差 MLP / MDN','用混合密度网络表达连续目标的多模态分布。','完整 OOF 79.8420；解析解码子配置另有记录。',True)
node('pure_dl_ple','分段线性特征编码','对数值特征做 PLE 编码后送入神经网络。')
node('pure_dl_invariant','206 维不变特征','约束输入特征，避免不希望模型依赖的井标识/绝对定位信息。')
node('pure_dl_equal6_specialists','六模型等权融合','常规神经网络与目标专家的完整预测分布等权组合。','完整 OOF 80.5006；属于早期融合路线。',True)
node('pure_dl_equal11','十一来源融合','继续增加纯神经网络来源，以完整后验等权融合。','完整 OOF 80.8098；没有达到 82。')
node('pure_dl_multiscale_sequence','多尺度深度网络','转入沿深度编码和解码的多尺度序列骨干。','完整 OOF 80.8989，后来作为序列机制对照。',True)
node('pure_dl_multiscale_sequence_deep','多尺度 + 辅助监督','给中、粗尺度增加辅助预测损失。','完整 OOF 80.9507，提升约 0.0518。',True)
node('pure_dl_timemixer_sequence','TimeMixer 思路','多尺度时序混合，探索不同分辨率的信息交换。','完整 OOF 80.3031，未优于当时多尺度序列基线。')
node('pure_dl_best_checkpoint_sequence','验证最佳 checkpoint','使用当前折验证最佳轮次，替代选好轮次再重训。','首折筛选结果，不是完整 OOF；这是后续保留的训练选择规则。',True)
node('pure_dl_blur_down_sequence','低通后下采样','降采样前增加固定低通滤波，减少高频混叠。','首折 81.9694；需与当时同折配置比较。',True)
node('pure_dl_blur_no_dropout_sequence','去除 Dropout','简化正则，保留卷积与密度预测结构。','首折 82.1340；常被简称“82.13”，不是完整五折分数。',True)
node('pure_dl_blur_no_point_sequence','去除 point-MLP 旁路','保留序列路径，去掉逐点 MLP 分支。','首折 82.2759；这一结果不能与后来第五折直接比较。',True)
node('pure_dl_blur_ungated_sequence','简化残差门控','把残差门控改为固定形式，简化主干。','留存完整 OOF 81.3166。',True)
node('pure_dl_spectral_sequence','频域分支','显式频域变换补充局部卷积。','历史首折 81.7757，对照 81.8357，未见收益。')
node('pure_dl_context128_sequence','缩短上下文','缩短模型可见深度范围，检验局部信息是否已足够。','历史首折 81.6352，对照 81.8357，未见收益。')
node('pure_dl_heterogeneous_moe_sequence','早期 CNN / MLP MoE','以异构专家和门控混合不同建模路径。','历史首折 80.9533，对照 81.8357，明显退步。')
for id,title,description in [('pure_dl_gen_spline','Spline flow 密度头','用条件有理二次样条流代替 MDN。'),('pure_dl_gen_flowmatch','Flow matching 密度头','学习时间条件流场，固定分位点与 Euler 积分生成。'),('pure_dl_gen_diffusion','Diffusion 密度头','训练扩散去噪头，使用固定采样设置做预测。')]:
    node(id,title,description,'仅代表该具体预算与生成头的首折结果；不能推广为所有生成模型无效。')
node('pure_dl_relative_only_sequence','仅保留相对深度','不把绝对深度通道作为输入，继续使用相对定位。','进入固定第五折阶段；当前原始九曲线约束的组成部分。',True)
node('pure_dl_global_full_sequence','覆盖全训练点的质量表','全局原点表保存训练质量证据，避免只覆盖抽到的窗口。','训练损失高本身不构成异常标签。',True)
node('pure_dl_global_recovery2_sequence','允许降权后恢复','持续条件冲突触发降权，恢复证据满足后逐步恢复。','学习优先级与疑似质量问题分开处理。',True)
node('pure_dl_por_t3_patch_sequence','POR 重尾 + patch','POR 使用 Student-t3，其余目标保留原分布；配合 patch 路线。','重尾收益不能自动推及 SW。',True)
node('pure_dl_patch_hierarchy_sequence','Patch 内 / 外分层注意力','patch 内逐点 attention，patch 外用摘要 token 交互。','区分点级细节和较粗上下文。',True)
node('pure_dl_global_cosine_sequence','Global-cosine 单骨干','局部点积 attention + 全局 cosine patch attention。','完整 OOF 81.5027；第五折 79.9899。是双专家路线的单骨干基线。',True)
node('pure_dl_paired_backbone_sequence','两个独立同结构专家','两个独立成员各自训练 NLL，再组合完整预测分布。','第五折 80.0080。',True)
node('pure_dl_paired_point_view_sequence','206 + 瞬时 41 视图','一个专家看完整特征，另一个看同深度的瞬时特征。','第五折 80.0885；改变视图比简单复制成员更有价值。',True)
node('pure_dl_paired_well_view_sequence','206 + 井内相对 68 视图','瞬时 41 维加 27 维井内 rank / ZIQR / QNORM。','第五折 80.1434。',True)
node('pure_dl_paired_calibrated_view_sequence','条件校准双骨干 · 75轮','用井内相对信息逐点校准第二专家的 31 个数值输入。','完整 OOF 81.7132，目前完整五折最佳；第五折 80.2004 按当时 >80.2 晋级。A榜 83.6915 来自用户反馈，不是 OOF。',True)
node('pure_dl_paired_calibrated_view80_sequence','同结构 · 80轮对照','保持校准结构，将预算与 cosine 周期改为 80 轮。','第五折 80.0958；75 与 80 轮 cosine 轨迹不相同。',True)
node('pure_dl_paired_detached_router80_sequence','输入条件路由','专家各自 NLL；router 用 detach 专家输出学习融合分布。','第五折 80.1712，比同预算校准对照约 +0.0754。',True)
node('pure_dl_paired_local_context_router80_sequence','路由器加入局部上下文','输入路由隐藏层加零初始化局部卷积。','第五折 80.2219；仍未达到后来 >80.4 的门槛。')
node('pure_dl_paired_posterior_router80_sequence','后验描述路由 · 80.25','68 维输入加 24 个 detach 后验描述量，分别优化专家和路由器。','第五折 80.2509；是当前大量结构/特征试验的共同基线。不是完整五折 80.25。',True)
node('pure_dl_paired_posterior_control_compact80_sequence','紧凑缓存复现对照','原结构不变，逐点探测历史改为紧凑存储。','第五折 80.2270；后续同时报告原基线和同实现对照，不能把存储变化造成的差异当新机制收益。')
node('pure_dl_paired_local_feature_gate80_sequence','逐点选择局部统计 · 373参数','共享小MLP根据每条曲线的本点输入，学习该点局部统计的0～2倍门控；初始为恒等，只作用于完整专家的90个局部统计列。','初步结果随训练快照更新。检验输入条件化的特征选择，不叠加已失败的新特征；先与80.2509原模型和80.2270紧凑对照比较。',True)
node('pure_dl_paired_context_calibration80_sequence','邻域隐藏信息校准 · 17点','第二专家校准隐藏层加入 W(17点有效均值-h)，W零初始化；新增1024参数，原输入视图、gamma/beta头和后验路由保持。','初步结果随快照更新。只在校准隐藏空间加入邻域信息，不平滑标签/最终预测，不叠加失败的特征门控；第五折严格>80.4且有收益才补折。',True)
for id,title,desc in [
 ('pure_dl_paired_directional_depth80_sequence','上下深度方向差 · 24维','1m/4m 上下文的有符号曲线差与联合变化。'),
 ('pure_dl_paired_increment_texture80_sequence','增量纹理 · 36维','方向持续性、9/51增量能量比、尖峰集中度和覆盖率。'),
 ('pure_dl_paired_conditional_residual80_sequence','跨曲线线性条件残差','井内协方差/精度矩阵提供条件残差。'),
 ('pure_dl_paired_gr_conditional80_sequence','GR 条件参照 · 16维','给定本井 GR 分位，计算其他 8 曲线的条件分位和中位数偏离。'),
 ('pure_dl_paired_patch_extrema80_sequence','Patch 极值摘要','在 patch 均值之外加入极值偏离信息。'),
 ('pure_dl_paired_point_patch_retrieval80_sequence','逐点查询 patch','每个深度点查询 patch token，替代仅共享广播信息。'),
 ('pure_dl_paired_adaptive_dilation80_sequence','共享核自适应膨胀','共享卷积核用两种 dilation，输入门控混合感受野。')]:
    node(id,title,desc,highlight=id.endswith('gr_conditional80_sequence'))
node('pure_dl_posterior_three_seed_equal80','三种子等权后验投票','原双专家模型加两次独立种子训练，完整分布等权融合。','第五折 80.3104；后续精细解码和重叠窗口基于这三个模型。',True)
node('pure_dl_posterior_fine_quadrature_fold5','后验积分细化','积分网格从 256 增至 1024；固定权重。','第五折 80.3194，小幅收益。')
node('pure_dl_posterior_fine_actions_fold5','动作网格细化','在相同后验积分上加密候选动作。','第五折 80.3221；收益递减。')
node('pure_dl_posterior_overlap_fold5','两窗口重叠预测','每点由两个窗口预测完整分布，等权合并。','第五折 80.3538，较不重叠 +0.0318。',True)
node('pure_dl_posterior_overlap4_fold5','四窗口重叠预测','stride32，每点覆盖四次，增加窗口位置多样性。','第五折 80.3627；收益较两窗口小。')
node('pure_dl_four_model_cosine_window_fold5','四模型 + cosine 窗权','加入校准75模型，使用固定 cosine 窗口权重。','第五折 80.3818，尚未过 80.4。')
node('pure_dl_five_model_cosine_window_fold5','五模型固定融合','再加入单骨干 global-cosine，固定等权模型融合。','第五折 80.3869，接近但未达到门槛。',True)
node('pure_dl_five_learned_slow_vote_fold5','五模型慢速学习权重 · 80.39','冻结五个模型，只在训练井学习各目标的 softmax 权重，验证集选择最佳更新点。','第五折 80.3868919；相对初始等权仅约 +0.0000377，没有实质收益。仍是第五折记录，不能视为完整 OOF。',True)
node('pure_dl_posterior_validation_grid_fold5','验证集权重网格 · 诊断','直接在开发验证集搜索融合权重。','通道分别搜索得到 80.3957095，但属于验证集调参的乐观诊断，不能作为独立过线证据。',False,resultText='第五折 · 验证调参 80.3957',status='diagnostic')
node('pure_dl_inner_earlystop','内部验证早停试验','在训练井内部另分验证井，以内部损失选择停止轮次。','pilot_report.json 明确没有产出外折预测；随后用户要求直接恢复当前折验证最佳 checkpoint。',status='diagnostic',resultText='内部试验：最佳19轮 / 停于24轮')
node('pure_dl_transition_loss_sequence','状态切换损失 · 用户取消','尝试学习相邻深度默认状态是否切换，只完成资源/公式原型。','用户取消，未正式训练；不能写成效果失败。后续只对默认点屏蔽连续 NLL，保留门控 BCE。',status='unscored',resultText='未训练 · 用户取消')
node('pure_dl_por_student3_sequence','POR t3 · 最初原型','将 POR 的高斯分量改为 Student-t3，匹配原方差。','本目录只保留原型记录，未进行正式GPU训练；后续训练结果见 POR t3 + patch 等独立节点。',status='unscored',resultText='原型已验 · 此目录未训练')
node('pure_dl_sequence_pair','辅助监督消融汇总','比较已独立训练的多尺度主头与深监督版本。','不是第三个新模型：对应完整 OOF 80.8989064 → 80.9506705，增益约0.0518。',status='diagnostic',resultText='OOF 对照：80.8989 → 80.9507')
node('pure_dl_dual_evidence_sequence','双证据降权 · 方案记录','探索用独立证据识别持续条件冲突。','该目录无独立评分导出，相关实测落在 dual_consensus 实验；不能把文档引用分数当作此配置独立成绩。',resultText='方案记录 · 无独立评分')
node('pure_dl_parallel_depth_sequence','并行深度卷积 · 方案记录','多感受野并行卷积的前期设计。','仅留存 README，尚无可核验的独立训练结果。',resultText='方案记录 · 无独立评分')
node('pure_dl_tabm32_aligned','TabM 编码对齐 · 已叫停','检查并修正分段编码与原始列交错顺序。','编码自检通过，但训练被用户停止；不能凭修复推断分数收益。',status='stopped',resultText='编码自检通过 · 训练叫停')
node('pure_dl_paired_curve_filter_sequence/failed_config_guard_20260926_152722','早期滤波 · 配置保护失败','一次启动失败记录。','不代表网络机制失败；后续修复后的正式实验另有节点。',status='unscored',resultText='启动失败 · 未得到评分')
node('pure_dl_paired_curve_filter_sequence/failed_dtype_20260926_152643','早期滤波 · dtype 失败','一次数据类型不匹配导致的启动失败。','不代表网络机制失败；后续修复后的正式实验另有节点。',status='unscored',resultText='启动失败 · 未得到评分')

chain('simple_cnn_cv','cnn_mlp_cv','pure_dl_v1','pure_dl_rich','pure_dl_residual')
edge('simple_cnn_cv','simple_cnn_anomaly_cv','control','相同 CNN 的异常预处理对照。')
chain('pure_dl_multiscale_sequence','pure_dl_multiscale_sequence_deep','pure_dl_best_checkpoint_sequence')
edge('pure_dl_residual','pure_dl_multiscale_sequence','inferred','从逐点概率模型走向序列模型的概念演进，不表示完全相同的代码/预算。')
chain('pure_dl_blur_down_sequence','pure_dl_blur_no_dropout_sequence','pure_dl_blur_no_point_sequence','pure_dl_blur_ungated_sequence','pure_dl_relative_only_sequence')
for x in ['pure_dl_gen_spline','pure_dl_gen_flowmatch','pure_dl_gen_diffusion']:edge('pure_dl_blur_no_dropout_sequence',x,'control','生成式路线 README 明确以首折82.134005为比较基线。')
chain('pure_dl_global_full_sequence','pure_dl_global_recovery2_sequence')
edge('pure_dl_global_recovery2_sequence','pure_dl_por_t3_patch_sequence','evolution','全点质量策略被后续重尾/patch训练保留；不是单因素消融。')
chain('pure_dl_patch_hierarchy_sequence','pure_dl_global_cosine_sequence','pure_dl_paired_backbone_sequence','pure_dl_paired_point_view_sequence','pure_dl_paired_well_view_sequence','pure_dl_paired_calibrated_view_sequence','pure_dl_paired_calibrated_view80_sequence','pure_dl_paired_detached_router80_sequence','pure_dl_paired_posterior_router80_sequence')
edge('pure_dl_paired_detached_router80_sequence','pure_dl_paired_local_context_router80_sequence')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_posterior_three_seed_equal80','fusion','原种子及另外两个独立训练种子的完整后验融合。')
for seed in [20261007,20261008]:
    id='pure_dl_posterior_three_seed_equal80/seed_'+str(seed)
    node(id,'后验双专家 · seed '+str(seed),'原后验双专家结构的独立种子重训。','单模型成绩与三种子融合成绩分别记录。',stage='vote')
    edge('pure_dl_paired_posterior_router80_sequence',id,'control','结构与协议相同，改变独立训练种子。')
    edge(id,'pure_dl_posterior_three_seed_equal80','fusion','该种子是三模型完整后验等权融合的成员。')
chain('pure_dl_posterior_three_seed_equal80','pure_dl_posterior_fine_quadrature_fold5','pure_dl_posterior_fine_actions_fold5','pure_dl_posterior_overlap_fold5','pure_dl_posterior_overlap4_fold5','pure_dl_four_model_cosine_window_fold5','pure_dl_five_model_cosine_window_fold5','pure_dl_five_learned_slow_vote_fold5')
edge('pure_dl_paired_calibrated_view_sequence','pure_dl_four_model_cosine_window_fold5','fusion','加入已训练的75轮校准双专家模型。')
edge('pure_dl_global_cosine_sequence','pure_dl_five_model_cosine_window_fold5','fusion','加入已训练的单骨干模型。')
for id in list(nodes):
    if id.startswith('pure_dl_paired_') and id.endswith('80_sequence') and not any(e['target']==id for e in edges) and id!='pure_dl_paired_posterior_router80_sequence':
        edge('pure_dl_paired_posterior_router80_sequence',id,'control','对应 README 声明以原80.2508938双专家为对照；未叠加其他失败改动。')
edge('pure_dl_posterior_three_seed_equal80','pure_dl_posterior_validation_grid_fold5','control','同一组三种子预测上的验证调参诊断。')
for x in ['pure_dl_posterior_learned_vote_fold5','pure_dl_posterior_metric_vote_fold5','pure_dl_posterior_dynamic_vote_fold5','pure_dl_posterior_dynamic_vote_slow80_fold5','pure_dl_posterior_dynamic_vote_fine80_fold5']:
    edge('pure_dl_posterior_three_seed_equal80',x,'fusion','同一组三个冻结专家；改变融合权重学习方式，具体目标和选择范围见节点记录。')
nodes['pure_dl_paired_gr_conditional80_sequence']['conclusion']='已完成第五折：79.9923191，最佳52轮、自然早停70轮，恢复最佳预测一致。较原80.2508938低0.2586，较紧凑对照低0.2347；不继续带宽/锚点扫描。'
nodes['pure_dl_paired_context_calibration80_sequence']['conclusion']='按用户要求完成原定流程，未手动停止：patience18于70轮触发，恢复52轮最佳预测一致。第五折80.1845975，比原基线低0.0663、紧凑对照低0.0424，未过80.4；不补折、不扫窗口或隐藏宽度。'
end('dead:context_calibration','DEAD END · 当前邻域校准','pair',['pure_dl_paired_context_calibration80_sequence'],'在第二专家校准隐藏空间加入17点邻域残差。','完整单折训练流程结束，80.1845975未超过原80.2508938或紧凑对照80.2270151。用户要求继续后没有手动停止，70轮自然早停并恢复最佳；本次设置无净收益，不推广为所有邻域校准均无效。')
nodes['pure_dl_paired_local_feature_gate80_sequence']['conclusion']='第58轮提前终止，观测最佳80.0311929在第48轮，连续10轮未改善。比原80.2508938低0.2197，比紧凑对照80.2270151低0.1958；未完成最终恢复核验，无checkpoint保留，不继续扫门控参数。'
end('dead:local_selection','DEAD END · 局部统计选择','pair',['pure_dl_paired_local_feature_gate80_sequence','pure_dl_paired_local_group_dropout80_sequence','pure_dl_paired_pruned_local_view80_sequence'],'比较输入条件门控、随机整组丢弃和固定删除局部特征。','门控观测最佳80.0312，随机组丢弃80.0684，固定删除79.6781，均未超过原双专家80.2509。三次均为提前结束的筛选记录，不能声称完整预算证明；关闭这些具体机制，不继续门控强度/宽度网格。')
end('dead:gr_conditional','DEAD END · GR 条件参照','pair',['pure_dl_paired_gr_conditional80_sequence'],'按相似GR水平归一化其他曲线，补充整井边际统计。','第五折79.9923191，低于原基线0.2586；最佳checkpoint恢复一致。该16维表示未带来收益，停止锚点/带宽扫描；不能据此否定所有条件特征。')
end('dead:generative','DEAD END · 当前生成式头','generation',['pure_dl_gen_spline','pure_dl_gen_flowmatch','pure_dl_gen_diffusion'],'更换分布表达能力，没有转化为官方指标收益。','首折分别约81.2902 / 80.7958 / 80.5687，均落后82.1340对照。关闭当前头、预算与采样设置的盲扫；不能据此否定所有flow/diffusion。')
end('dead:early_temporal','DEAD END · 早期时序替换','sequence',['pure_dl_spectral_sequence','pure_dl_context128_sequence','pure_dl_heterogeneous_moe_sequence'],'频域、缩短窗口和早期 MoE 的三条岔路。','同一历史首折基准81.8357下均无增益。当前设置不保留；这个结论不能直接迁移到其他折或所有attention模型。')
end('dead:capacity','DEAD END · 盲目扩大容量','context',['pure_dl_wide150_sequence','pure_dl_wide150_lr4_sequence'],'扩大CNN容量及相关学习率配合。','wide150 / lr0.0006 观测最佳79.2303，主动停止于59轮。尚无容量扩大收益证据，停止无边界宽度网格；提前终止不是充分收敛证明。')
end('dead:router','DEAD END · 路由器继续堆叠','pair',['pure_dl_paired_cdf_posterior_router80_sequence','pure_dl_paired_split_posterior_router80_sequence','pure_dl_paired_symmetric_posterior_router80_sequence','pure_dl_paired_latent_posterior_router80_sequence'],'更细后验描述、拆分默认/连续路由、对称评分、隐层表示。','这些配置均未超过原后验路由80.2509。描述更复杂不等于预测更好，暂不继续维度/容量网格。')
end('dead:local_features','DEAD END · 近期局部统计','pair',['pure_dl_paired_directional_depth80_sequence','pure_dl_paired_increment_texture80_sequence','pure_dl_paired_conditional_residual80_sequence'],'方向差、增量纹理与线性条件残差。','观测最佳分别约80.0844 / 79.9930 / 80.1572，均低于原80.2509。已提前停止；没有最终恢复核验，不扫窗口/截断强度。特征冗余是待验证解释，不是已证实原因。')
end('dead:patch_variants','DEAD END · 近期 patch 改造','pair',['pure_dl_paired_patch_extrema80_sequence','pure_dl_paired_point_patch_retrieval80_sequence','pure_dl_paired_adaptive_dilation80_sequence'],'极值摘要、逐点查询和自适应感受野。','均在后期落后原双专家并停滞；停止当前变体，保留原模型。不能由此推出所有跨patch建模无效。')
end('dead:learned_vote','DEAD END · 学习投票的收益饱和','vote',['pure_dl_five_learned_slow_vote_fold5'],'缓慢更新权重，搜索验证最佳中间点。','相比初始等权只增加约0.0000377，未达到80.4。停止继续扫描投票学习率/步长，回到特征工程和结构。')

node('pure_dl_paired_haar_detail80_sequence','首层降采样保留差分细节','完整特征专家首层增加零初始化的相邻点差分投影，与原低通降采样相加；校准专家不改。','第五折观测最佳80.1062@52；第62轮后提前停止，连续10轮未刷新。比原基线低0.1447、紧凑对照低0.1208；未验证最终最佳恢复，不补折。')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_paired_haar_detail80_sequence')
edge('pure_dl_paired_posterior_control_compact80_sequence','pure_dl_paired_haar_detail80_sequence','control','相同紧凑探测和训练配置对照。')

end('dead:haar_detail','DEAD END · 首层差分支路','pair',['pure_dl_paired_haar_detail80_sequence'],'首层保留相邻点带符号细节。','固定第五折观测最佳80.1062，低于两项对照，后期连续10轮停滞后提前结束。不扫描差分倍率或层数；只否定本次设置，不代表所有频域结构无效。')

node('pure_dl_paired_head_exchange80_sequence','全局注意力头间交换','完整特征专家的4个全局patch头，在softmax前通过零初始化4×4残差矩阵交换打分；仅新增16参数。','第五折观测最佳79.8979@44；第54轮后提前终止，连续10轮未刷新。比原基线低0.3530、紧凑对照低0.3291；最终最佳恢复未执行，不补折。')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_paired_head_exchange80_sequence')
edge('pure_dl_paired_posterior_control_compact80_sequence','pure_dl_paired_head_exchange80_sequence','control','紧凑存储训练对照，保持原始打分与划分。')

end('dead:head_exchange','DEAD END · 全局头间打分交换','pair',['pure_dl_paired_head_exchange80_sequence'],'全局注意力softmax前增加跨头残差线性混合。','第五折观测最佳79.8979，晚期连续10轮停滞且落后对照，提前结束。停止当前混合设置，不扫描头数或交换强度；不推广为所有头间通信无效。')

node('pure_dl_window_consistency80_sequence','重叠窗口分布一致性','同一训练点在错开32点窗口中的默认概率和条件CDF作对称约束；双方有梯度、无伪标签。每步额外16窗口，不增加参数。','第五折观测最佳80.1022@52，第62轮后提前停止，连续10轮停滞。比原双专家低0.1487、紧凑对照低0.1249；最终恢复未执行，不补折。')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_window_consistency80_sequence')
edge('pure_dl_paired_posterior_control_compact80_sequence','pure_dl_window_consistency80_sequence','control','相同优化步数、原网络与评分；额外窗口计算需单列。')

node('pure_dl_frozen_numeric_patch80_fold5','冻结80.39 · 数值纠偏补丁','冻结五专家及学习投票，212→32→3零初始化MLP对最终非默认数值作有界修正。只用训练井真实标签训练。','第18轮自然早停；最佳仍为初始零补丁80.3868919，收益0。训练后最佳80.3769@1，最终80.2161；恢复原预测一致，未过80.4。缓存可复用。')
edge('pure_dl_five_learned_slow_vote_fold5','pure_dl_frozen_numeric_patch80_fold5','evolution','用户要求在80.39上冻结原模型作增量训练；不继续训练投票权重。')

end('dead:frozen_numeric_smoothl1','DEAD END · SmoothL1数值纠偏','vote',['pure_dl_frozen_numeric_patch80_fold5'],'冻结80.39后的有界数值残差，训练SmoothL1损失。','训练损失下降而验证分数下滑，18轮早停后零补丁最佳。关闭当前损失/纠偏设置，不否定其他增量机制；保留约17MB冻结预测缓存。')

node('pure_dl_frozen_metric_patch80_fold5','冻结80.39 · 官方评分损失补丁','保持零初始化有界数值MLP，改用官方截断误差与通道权重，复用训练预测缓存；不改变默认判定。','18轮早停，零补丁80.3868919仍最佳；训练后最佳80.3860359@1，末轮80.3621706。比SmoothL1稳，但没有净收益，恢复原预测一致。总耗时89.46秒含预处理。')
edge('pure_dl_frozen_numeric_patch80_fold5','pure_dl_frozen_metric_patch80_fold5','evolution','同一冻结80.39补丁，仅更换数值损失，缓存复用。')

end('dead:frozen_numeric_metric','DEAD END · 逐点有界数值补丁','vote',['pure_dl_frozen_metric_patch80_fold5'],'在冻结80.39上比较SmoothL1和官方截断误差损失。','两项均以零补丁最佳；官方损失减轻退化但未提高总分。不继续扫宽度/学习率/倍率，下一步需更换补丁信息或任务。')

node('pure_dl_frozen_profile_patch80_fold5','冻结80.39 · 局部预测形态补丁','从同井9/33点预测趋势提取偏差，小MLP学习平滑或反向增强系数；原模型冻结、零修正初始化。','第五折80.3874519@1，19轮早停并恢复验证一致。比80.3868919仅+0.0005600，无实质提升且未过80.4；保留小补丁作对照，不扫描窗口/宽度。耗时80.74秒含预处理。')
edge('pure_dl_frozen_metric_patch80_fold5','pure_dl_frozen_profile_patch80_fold5','evolution','固定官方数值损失和训练设置，更换为受局部预测形态约束的补丁。')

end('dead:window_consistency','DEAD END · 重叠窗口一致性','pair',['pure_dl_window_consistency80_sequence'],'额外前向16个窗口，训练相同深度预测分布一致。','第五折观测80.1022，后期持续落后并停滞，增加计算未获净收益。不扫描位移/系数，回到冻结80.39补丁方向。')

node('pure_dl_frozen_sw_action80_fold5','冻结80.39 · SW动作价值补丁','小网络学习保留原SW预测或改用固定候选的预期官方效用；POR/PERM不变。仅训练井真实标签构造效用监督。','18轮早停，零补丁80.3868919仍最佳；训练后最佳80.3670585@2，SW73.5332低于原73.5899。恢复原预测一致，不保留负向权重；总耗时61.80秒含预处理。')
edge('pure_dl_five_learned_slow_vote_fold5','pure_dl_frozen_sw_action80_fold5','evolution','冻结原五模型，改变为SW动作价值任务，非投票权重训练。')

end('dead:sw_action_patch','DEAD END · SW动作价值补丁','vote',['pure_dl_frozen_sw_action80_fold5'],'soft-target BCE学习固定候选及原预测的预期效用。','训练后最佳80.3671，仍低于80.3869，18轮早停后恢复零补丁。不继续扫网格/初始偏置；仅否定本次设置。')

node('pure_dl_frozen_private_profile80_fold5','冻结80.39 · 分目标形态补丁','共享profile小MLP拆成POR/PERM/SW独立分支，复制同一初始隐层；其余输入、损失和修正方式不变。','第五折80.3873933@1，19轮早停恢复一致；比共享80.3874519低0.0000585，无结构增量收益。参数增加属于拆分成本，非同容量实验；不补折。')
edge('pure_dl_frozen_profile_patch80_fold5','pure_dl_frozen_private_profile80_fold5','evolution','检验不同目标共享隐层的梯度干扰，不按通道分别挑验证最佳。')

end('dead:private_profile','DEAD END · 形态补丁目标解耦','vote',['pure_dl_frozen_private_profile80_fold5'],'三个目标改用独立隐层，其他条件保持。','80.3873933低于共享版本80.3874519，未证实目标解耦收益；不继续扫分支宽度或通道学习率。')

node('pure_dl_frozen_block_patch80_fold5','冻结80.39 · 块级共享校准','同井128原始行块内平均输入与冻结预测信息，学习共享修正系数；最终输出仍逐点，标签不参与聚合。','第五折80.3884795@1，19轮早停并恢复一致；比原80.3868919高0.0015875，但后期退化且未过80.4。保留小补丁，暂不扫块长。总耗时47.49秒含预处理。')
edge('pure_dl_frozen_metric_patch80_fold5','pure_dl_frozen_block_patch80_fold5','evolution','用块级共享取代逐点纠偏，不叠加profile或目标私有分支。')

node('pure_dl_frozen_anchored_block80_fold5','冻结80.39 · 锚定块级校准','块级共享校准增加训练点输出偏离惩罚，限制补丁过度改写原预测；其余设置保持。','λ=.01第五折80.3885472@1，19轮早停恢复一致；比无惩罚仅+0.0000678，未过80.4。用户随后授权6档系数对照。')
edge('pure_dl_frozen_block_patch80_fold5','pure_dl_frozen_anchored_block80_fold5','evolution','针对最佳仅在第1轮、随后退化，检验预测空间锚定。')

node('pure_dl_frozen_anchor_sweep80','冻结80.39 · 惩罚强度6档','用户授权λ=0/.001/.003/.01/.03/.1；复用已完成0和.01，其余4档按双并发分批运行。','六档依次80.388479/80.388485/80.388502/80.388547/80.388664/80.388985。最高λ=.1，比无惩罚仅+0.000506；全部最佳1轮、19轮早停并恢复一致，无80.4过线。每项峰值约1.5GiB，无OOM。')
edge('pure_dl_frozen_anchored_block80_fold5','pure_dl_frozen_anchor_sweep80','evolution','用户明确要求并行扫描惩罚强度，覆盖此前不扫系数的计划。')

node('pure_dl_frozen_well_balanced_block80_fold5','冻结80.39 · 井级均衡块校准','λ=.1块级补丁的训练损失改为64口井等总权重，点Q保留，验证仍按原点总体评分。','第五折80.3892754@1，19轮早停恢复一致；较逐点等权80.3889853仅+0.0002901，无实质突破，未过80.4。不扫描权重指数，保留小补丁。总耗时48.94秒含预处理。')
edge('pure_dl_frozen_anchor_sweep80','pure_dl_frozen_well_balanced_block80_fold5','evolution','保留λ=.1，改变训练井损失聚合，非继续系数扫描。')

node('pure_dl_frozen_block_variation80_fold5','冻结80.39 · 块内预测变化幅度','井级均衡块校准新增3维冻结预测空间标准差，区分同均值下不同变化幅度；新增输入权重零初始化。','第五折80.3891555@1，19轮早停恢复一致；较无新增特征80.3892754低0.0001199，未证实特征收益，不补折。耗时49.81秒含预处理。')
edge('pure_dl_frozen_well_balanced_block80_fold5','pure_dl_frozen_block_variation80_fold5','evolution','保持训练设置，补充块均值缺失的变化幅度信息。')

end('dead:block_variation_patch','DEAD END · 补丁变化幅度特征','vote',['pure_dl_frozen_block_variation80_fold5'],'向块级校准增加冻结预测空间标准差。','80.3891555略低于无新增特征的80.3892754，未证实收益；停止当前特征，不堆叠高阶统计或扫描块长度。')

node('pure_dl_frozen_linear_block80_fold5','冻结80.39 · 线性块校准','井均衡块补丁从MLP减为212→3零初始化线性层+tanh，降低纠偏容量；其余训练规则保持。','第五折80.3869340@1，19轮早停恢复一致；仅比原基线高0.0000421，低于MLP80.3892754。结构RNG消耗不同，非逐更新完全对齐。无80.4过线。')
edge('pure_dl_frozen_well_balanced_block80_fold5','pure_dl_frozen_linear_block80_fold5','evolution','多个MLP后期退化，检验低容量系统校准。')

end('dead:linear_patch','DEAD END · 补丁容量缩减','vote',['pure_dl_frozen_linear_block80_fold5'],'将MLP块校准替换为639参数线性层。','80.3869340低于MLP80.3892754，容量缩减未证实收益；停止宽度/线性项调整，需改变监督或信息。')

node('pure_dl_frozen_state_repair80_fold5','冻结80.39 · 默认状态收益判别','为每目标构造一个输入来源的默认/邻域数值替代候选，神经网络学习替代相对原预测的预期官方收益，正收益才替换。','训练后最佳80.3794369@2，18轮早停后恢复原80.3868919，收益0。恢复预测一致，无权重保留。候选不读邻点标签，验证不做oracle。')
edge('pure_dl_five_learned_slow_vote_fold5','pure_dl_frozen_state_repair80_fold5','evolution','冻结原80.39，改变为默认状态修复的收益差监督。')

end('dead:state_repair','DEAD END · 默认状态收益判别','vote',['pure_dl_frozen_state_repair80_fold5'],'学习默认/邻域替代候选相对原预测的期望收益。','训练后最高80.3794，原80.3869仍最佳；18轮早停。停止该候选与监督组合，不扫描验证阈值或窗口。')

node('pure_dl_frozen_smooth_score80_fold5','冻结80.39 · 平滑评分损失','用σ=.25的高斯卷积平滑训练三角评分核，给容忍边界附近提供梯度；官方验证评分不变。','第五折80.3893267@1，19轮早停恢复一致；比硬截断对照80.3892754仅+0.0000513，无实质增益，不扫描σ。原官方评分，未过80.4；耗时66.74秒含预处理。')
edge('pure_dl_frozen_well_balanced_block80_fold5','pure_dl_frozen_smooth_score80_fold5','evolution','只改变训练数据损失的边界梯度，不改变验证评分。')

node('pure_dl_frozen_latent_patch80_fold5','冻结80.39 · 骨干内部表示补丁','将最强双专家主头前128+128维冻结表示输入纠偏MLP；仅训练井拟合表示标准化，新增投影权重从零开始。','第五折待验证；与原逐点metric补丁对照，表示只在内存中使用，不保存大缓存。')
edge('pure_dl_frozen_metric_patch80_fold5','pure_dl_frozen_latent_patch80_fold5','evolution','改变可用信息为冻结骨干学习到的表示，不叠加失败统计特征。')

# Names for the remaining records: terminology translation, not invented mechanism claims.
terms={
'absolute_depth':'绝对深度','adaptive_center':'自适应中心','adaptive_scale':'自适应尺度','annealed_aux':'退火辅助监督','annealed_por_perm_aux':'POR/PERM退火辅助监督','attention_multiscale':'多尺度注意力','aux20':'辅助头20轮','balanced_well':'井级均衡','batchensemble':'BatchEnsemble','batch_size':'batch size','beta98':'beta0.98','blur_deform':'低通+可变形卷积','blur_difference':'低通+差分','blur_direct_head':'直接预测头','blur_first':'前置低通','blur_interaction':'低通+交互特征','blur_kernel15':'卷积核15','blur_no_decay':'取消weight decay','blur_no_dropout_lr0006':'无Dropout lr0.0006','blur_refine':'低通细化','blur_steps96':'每轮96步','blur_uncentered_norm':'非中心化归一化','blur_wide_mlp':'扩大MLP','categorical':'类别分布头','channel_first':'通道优先','conflict_weighted':'冲突降权','context512':'512点上下文','continuous_difference':'连续差分','contrastive_global':'全局对比','cosine_patch_complete':'cosine patch补折','cosine_patch':'cosine patch注意力','crps':'CRPS损失','curve_dropout':'曲线Dropout','curve_token':'曲线token','curve_transformer':'曲线Transformer','decision_calibrated':'决策校准','default75':'75轮预算','dense_local':'密集局部','density_adapter':'密度适配','density_physics':'物理密度约束','depthwise_norm':'逐深度归一化','depth_norm_joint':'逐深度联合归一化','depth_patch':'深度patch','depth_trend':'深度趋势残差','dual_consensus':'双重共识','dual_evidence':'双重证据','dynamics_probe':'训练动态探测','earlystop_refit':'早停后重训（旧）','ema_context':'EMA上下文','epoch80':'80轮预算','external_context':'窗口外上下文','family_mixture':'分布族混合','fine128':'128细粒度','fine_global_tokens':'细粒度全局token','full_input':'扩展输入曲线','full_resolution':'全分辨率','gated_point':'逐点门控','gated_skip':'跳连门控','global_content':'全局内容注意力','global_entropy40':'熵P40冲突判据','global_point':'全局点表','global_probe16':'探测间隔16','global_residual':'全局残差','global_streak2':'连续2次判据','global_tail02':'尾概率0.02','heads8':'8头注意力','heterogeneous_pair':'异构双成员','hurdle_utility':'门槛分布效用','inner_earlystop':'内部验证早停','input512_core128':'输入512/输出128','input512_core256':'输入512/输出256','interaction':'交互编码','joint_default':'联合默认状态','kernel3':'卷积核3','laplace':'Laplace分布','learned_patch_pool':'可学习patch汇聚','learned_upsample':'可学习上采样','main_only':'仅主头监督','masked_pretrain':'掩码预训练','metric_auxiliary':'指标辅助损失','mid_patch_attention':'中层patch注意力','mixture16':'16分量密度','moe_head':'MoE预测头','multilag_difference':'多滞后差分','narrow75':'窄网络75轮','nearest_decoder':'最近邻解码','near_far_context':'近远上下文','neighborhood_density':'邻域密度','neighborhood_ple':'邻域PLE','nodecay_affine':'仿射参数无衰减','official_weight_loss':'官方通道权重损失','onecycle':'OneCycle调度','ordered_mixture':'有序混合分量','parameter_ema':'参数EMA','patch_dispersion':'patch离散度','patch_token_ffn':'patch token FFN','patch_width64':'patch宽度64','perm_crps':'PERM CRPS','perm_mae':'PERM MAE','perm_specialist':'PERM专家','plateau':'Plateau调度','point_patch_read':'逐点读取patch','population_batch32':'总体损失 bs32','population_likelihood':'总体似然分母','population_private':'总体损失私有头','por_log':'POR对数变换','por_specialist':'POR专家','por_student3':'POR Student-t3','por_t3_lr0004':'POR t3 lr0.0004','por_t3_lr0006':'POR t3 lr0.0006','por_t3_lr0008':'POR t3 lr0.0008','por_t5_patch':'POR t5 patch','prediction_pyramid':'预测金字塔','private_head':'目标私有头','quality_weight':'样本质量加权','recurrent':'循环网络','regularized':'正则化','restore_trajectory':'恢复训练轨迹诊断','retrieval':'检索特征','sam':'SAM优化','sequence_pair':'序列配对','shifted_patch':'平移patch','single_gaussian':'单高斯','stacked_hierarchy':'堆叠层级','staged_aux':'分阶段辅助监督','structured_conflict':'结构化冲突判据','sw_decoder':'SW专用解码','sw_huber_point':'SW Huber逐点头','sw_interval':'SW区间监督','sw_laplace':'SW Laplace','sw_logit':'SW logit坐标','sw_mean':'SW均值','sw_specialist':'SW专家','sw_t3':'SW Student-t3','sw_truncated':'SW截断分布','target_scale_readout':'按目标尺度读取','task_balance':'任务平衡','three_specialists':'三个专用专家','transition_loss':'状态切换损失','utility_finetune':'效用微调','utility_mass':'效用概率质量','utility_perm_sw':'PERM/SW效用','warmup':'学习率预热','well_adversarial':'井域对抗','well_context':'整井上下文','well_tail_risk':'井级尾部风险','window128':'128点窗口','window160':'160点窗口','window_reversal':'窗口反转增强',
'paired_aligned_side80':'双专家·对齐侧特征','paired_combined_side80':'双专家·组合侧特征','paired_complementary_blocks80':'互补块训练','paired_conditioned_mix80':'条件低秩混合','paired_cross_readout':'跨成员读出','paired_curve_attention80':'完整专家曲线注意力','paired_curve_attention_calibrated80':'校准专家曲线注意力','paired_curve_filter':'瞬时视图早期滤波','paired_curve_spline_calibrated80':'曲线样条校准','paired_density':'双专家密度','paired_dual_calibrated_view80':'两个专家均校准','paired_facies_calibrated80':'输入原型→校准','paired_facies_router80':'输入原型→路由','paired_gap_distance80':'缺失区间距离特征','paired_independent_q80':'专家独立质量表','paired_local_curve_relation80':'局部曲线关系','paired_local_detail_view':'局部细节视图','paired_local_group_dropout80':'局部特征组Dropout','paired_local_posterior_router80':'后验路由+局部卷积','paired_local_robust80':'局部稳健特征','paired_local_robust_memory80':'局部稳健记忆','paired_lookahead80':'Lookahead优化','paired_point_mlp':'CNN + 逐点MLP专家','paired_predictive':'联合预测双专家','paired_proxy_relative80':'物理代理相对特征','paired_proxy_wellmean_control80':'代理整井均值对照','paired_pruned_local_view80':'删减局部特征','paired_raw_view':'原曲线视图专家','paired_rms_full80':'完整专家RMS归一化','paired_router_utility80':'路由器效用辅助损失','paired_sampling_stride80':'随机stride2增强','paired_signed_local_contrast80':'有符号局部对比','paired_spike_gate80':'尖峰门控','paired_spike_side80':'尖峰侧特征','paired_spike_wellmean_control80':'尖峰整井均值对照','paired_sw_conditioned80':'SW条件化','paired_sw_curve_attention80':'SW曲线注意力','paired_sw_gate_prior80':'SW门控先验','paired_sw_mean_conditioned80':'SW均值条件','paired_target_specialist80':'目标专长专家','paired_target_specialist_reverse80':'反向目标专长专家','paired_tcn':'U-Net + TCN','paired_well_sw_prior80':'整井SW先验','paired_well_sw_router80':'整井SW路由','paired_well_view80':'68视图80轮','three_view_control68_80':'三视图68维对照','three_view_posterior_router80':'三视图后验路由',
'posterior_dynamic_vote_fine80_fold5':'细步长动态投票','posterior_dynamic_vote_fold5':'动态后验投票','posterior_dynamic_vote_slow80_fold5':'慢速动态投票','posterior_learned_vote_fold5':'三专家学习权重','posterior_metric_vote_fold5':'指标损失投票','posterior_logpool4_fold5':'几何窗口融合','five_model_geometric_fold5':'五模型几何融合','four_model_overlap4_fold5':'四模型四窗口融合','fixed_consensus_blend_fold5':'固定共识动作融合','fixed_distribution_blend_fold5':'固定50/50分布融合'}
node('pure_dl_frozen_latent_patch80_fold5','Frozen80.39 latent patch','Frozen dual-backbone 256 hidden features plus original212, train small correction head.','Fifth80.3868919 restored zero patch; trained best80.3759377@1, stop18. No gain, no checkpoint. Total66.72s, peak3306MiB.')
end('dead:latent_patch','DEAD END: frozen latent patch','vote',['pure_dl_frozen_latent_patch80_fold5'],'Add frozen internal features to correction head.','No trained epoch beat baseline. Stop adding hidden layers or increasing head width; training residual generalization remains unresolved.')

node('pure_dl_paired_lowrank_calibration80_sequence','Rank8 cross-feature calibration','Context-controlled multiplicative low-rank correction in calibrated second member;720 extra parameters, zero output initialization.','Running fixed fifth; no validation result yet. Compare original80.2508938 and compact control80.2270151; ensemble80.3868919 is separate reference.')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_paired_lowrank_calibration80_sequence','evolution','Keep original affine calibration; add conditional cross-feature mixing.')

node('pure_dl_well_style_plan','Weak cross-well style mixing: plan','Train-only hidden-statistics augmentation in full206 expert; calibrated68 remains unchanged.','Not implemented or trained; no score. Preserve physical signal, exclude padding and validation donors; disable for eval/Q probes.')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_well_style_plan','evolution','Return to original architecture; explore unseen-well generalization instead of calibration capacity.')

node('pure_dl_paired_weak_well_style80_sequence','Weak training-well style mixing','Train-only stem feature statistics mixing in full206 member; distinct-well donors,0..0.2 strength,p=.5,zero extra parameters.','Running fifth, no result yet. Concurrent with rank8 calibration; max80/pat18, strict>80.4 before promotion.')
edge('pure_dl_well_style_plan','pure_dl_paired_weak_well_style80_sequence','evolution','Implement predeclared domain-generalization plan.')

node('pure_dl_paired_lowrank_calibration80_sequence','Rank8 cross-feature calibration','Context-conditioned mixing of31 numeric features in calibrated member;720 added parameters.','Complete fifth79.9390794@52;natural stop70, restored predictions verified. Delta compact control -0.2879358, no checkpoint.1198.62s,peak782.76MiB,partial concurrency.')
end('dead:lowrank_calibration','DEAD END: rank8 feature calibration','pair',['pure_dl_paired_lowrank_calibration80_sequence'],'Add conditional cross-feature mixing to original affine input calibration.','All three target scores below compact baseline; total -0.2879358. Stop rank/capacity sweeps for this route; no remaining folds.')

edge('pure_dl_paired_conditioned_mix80_sequence','pure_dl_paired_lowrank_calibration80_sequence','evolution','Related rank8 family: earlier pre-affine correction80.1954 versus later bounded post-affine79.9391; both below control, stop this family.')

node('pure_dl_8039_source_ablation','80.39 source attribution','Frozen sources, matched decoding; singles/equal/learned/leave-one-out and window/decoder controls.','Diagnostic inference running; no new promoted score. Learned versus equal historical delta only0.00003773.')
edge('pure_dl_five_learned_slow_vote_fold5','pure_dl_8039_source_ablation','evolution','User-requested source attribution, not another fusion search.')
node('pure_dl_8039_training_ablation','80.39 training and loss ablations','Two separate matched80 runs: Q=1 or POR Gaussian with synchronized loss/decode/probe.','Prepared, not trained; no score. Original compact80.2270151 control; avoid75/80 budget confounding.')
edge('pure_dl_paired_posterior_router80_sequence','pure_dl_8039_training_ablation','evolution','Controlled retraining needed to attribute Q and distribution-family effects.')

node('pure_dl_paired_weak_well_style80_sequence','Weak training-well style mixing','Train-only masked stem statistics mixing from other training wells.','Complete79.6354133@48, natural stop66; -0.5916018 vs compact control. Restored predictions verified, no checkpoint.1278.49s/777.92MiB, partial concurrency.')
end('dead:weak_well_style','DEAD END: weak well-style mixing','pair',['pure_dl_paired_weak_well_style80_sequence'],'Regularize cross-well feature statistics during training.','This fixed weak augmentation lost0.5916; no strength/layer sweep. Does not establish all domain-generalization methods fail.')
node('pure_dl_8039_source_ablation','80.39 source attribution','26 predeclared frozen inference arms: singles, fusion, leave-one-out, windows and decoding.','Complete: strongest single80.331454, equal5 80.386864, learned5 80.386898. Fusion+0.055410; learned weighting negligible. Removingseed07 gives80.403220 but is diagnostic selection, no promotion.')

node('pure_dl_8039_ablation_no_quality','80.39 ablation: Q=1','Original posterior dual expert; all final loss quality weights forced1, probe retained.','Running fixed fifth80/pat18; no final result.')
edge('pure_dl_8039_training_ablation','pure_dl_8039_ablation_no_quality','evolution','One-factor training ablation.')
node('pure_dl_8039_ablation_gaussian_por','80.39 ablation: Gaussian POR','Original architecture; replace POR t3 with Gaussian consistently in loss, decode and probe.','Running fixed fifth80/pat18; no final result.')
edge('pure_dl_8039_training_ablation','pure_dl_8039_ablation_gaussian_por','evolution','One-factor distribution-family ablation.')

node('pure_dl_8040_pair_ablation','80.4032 baseline: two-source subsets','User-adopted four-source baseline0/2/3/4; evaluate all6 fixed equal pairs with unchanged decoder.','Queued, no pair results yet. Reuse checkpoints; two sources may contain4 backbones. Max2GPU tasks, CPU-only wait.')
edge('pure_dl_8039_source_ablation','pure_dl_8040_pair_ablation','evolution','User adopts dropseed07 baseline and requests removing two of its four sources.')

node('pure_dl_8040_pair_ablation','80.4032 baseline: pairs and triples','Evaluate6 pairs+4 leave-one-out triples+baseline from0/2/3/4; equal posterior weights, same decoder.','11arms queued in94542, no results yet. CPU-waiting only; two existing training jobs continue. Source predictions reused, no retraining.')

node('pure_dl_8039_ablation_no_quality','80.39 ablation: Q=1','Same posterior dual expert; final prediction loss weights fixed1.','Complete80.0925123@52,stop70; -0.1345029 vs matched80.2270151. Best restored, coverage verified, no checkpoint. Supports retainingQ in this setting.')
node('pure_dl_8039_ablation_gaussian_por','80.39 ablation: Gaussian POR','Replace PORt3 consistently in loss, decode and quality probe.','Complete80.0414425@52,stop70; -0.1855726 vs matched80.2270151. Best restored, coverage verified, no checkpoint. Supports retainingt3 in this setting.')
node('pure_dl_8040_pair_ablation','80.4032 baseline: pairs and triples','6pairs+4triples+baseline; unchanged source checkpoints and equal posterior mixtures.','Inference admitted after both training ablations completed; no subset score yet.')

node('pure_dl_8040_pair_ablation','80.4032 baseline: pairs and triples','All6pairs+4triples+baseline; fixed equal distributions and decoder, no backbone retraining.','Complete: baseline80.403220 exactly reproduced; bestpairAB80.380189; besttripleABD80.409152 (+0.005932),28.5% fewer parameters. Tiny exploratory gain, no independent superiority or fullOOF claim.')
node('pure_dl_8039_training_ablation','80.39 training and loss ablations','Separate matched80 runs: Q=1 or POR Gaussian with synchronized loss/decode/probe.','Complete: Q=1 80.092512 (-0.134503); GaussianPOR80.041443 (-0.185573), relative to dual80.227015. Bothbest52/stop70 and verified; retain originalQ/t3.')

node('pure_dl_8039_ablation_main_only','80.39 ablation: no auxiliary loss','Original posterior dual80; remove only mid/coarse auxiliary losses, retain network/init/main likelihood/Q/t3.','Training running, no result yet; matched compact80.2270151 control.')
edge('pure_dl_8039_training_ablation','pure_dl_8039_ablation_main_only','evolution','Fill auxiliary supervision attribution gap.')
node('pure_dl_internal_expert_attribution','Internal experts: A and B','Frozen seed06/08 each full206/calibrated68/equal/routed predictions under identical decoder.','Inference running, no result yet; post-training intervention not a retrained architecture ablation.')
edge('pure_dl_8040_pair_ablation','pure_dl_internal_expert_attribution','evolution','External subset study identified A/B as main complementary sources; inspect their internal branches.')

node('pure_dl_internal_expert_attribution','Internal experts: A and B','Frozen seed06/08 each full206/calibrated68/equal/routed under identical decoder.','Complete8arms, A reference exact. A:79.8868/79.5143/equal80.2382/routed80.3315. B:79.6236/79.1487/equal79.9358/routed79.9458. Strong internal complement; route gain differs by seed. Intervention only, not retraining.')

node('pure_dl_8039_ablation_no_calibration','80.39 ablation: identity calibration','Freeze original zero-output calibration initialization in posterior dual expert; input view/Q/t3/aux/router retained.','Training running, no result yet; matched compact80.2270151 control. Tests calibration conditional on learned posterior router, not another rank/width mechanism.')
edge('pure_dl_8039_training_ablation','pure_dl_8039_ablation_no_calibration','evolution','Historical calibration delta lacked current posterior-router interaction; direct one-factor retraining.')

node('pure_dl_8039_ablation_main_only','80.39 ablation: no auxiliary loss','Remove only mid/coarse auxiliary losses, retain network/init/main likelihood/Q/t3.','Complete79.8749053@52,stop70; -0.3521098 vs compact80.2270151 and all3targets worse. Restored exact/coverage verified;1235.03s/776.44MiB, no checkpoint. Retain auxiliary supervision.')

node('pure_dl_abd_fused_training80_fold5','ABD 基线·冻结 B/D 训练 A','用户采纳 ABD 第五折80.40915186；B/D固定，对三模型等权分布的最终损失反传，仅A更新。','初步结果：待训练；epoch0先复现80.40915186。小学习率增量，验证/早停/最佳checkpoint均按融合TOTAL；单模型只诊断。',stage='vote')
edge('pure_dl_8040_pair_ablation','pure_dl_abd_fused_training80_fold5','evolution','用户采用消融所得ABD为后续基线；从独立训练转为融合目标训练。')
node('pure_dl_abd_backbone_module_ablation','ABD 基线·骨干内部拆解','冻结B/D，分别关闭A两个骨干的局部/全局/整体层级、逐点MLP、瓶颈残差和U-Net跳连，共12项加基线。','初步结果待测；同时看A及ABD。推理干预只能识别已训练依赖，后续以训练消融核验收益。',stage='pair')
edge('pure_dl_8040_pair_ablation','pure_dl_abd_backbone_module_ablation','evolution','采用用户指定ABD基线，进一步追踪骨干内模块。')
edge('pure_dl_internal_expert_attribution','pure_dl_abd_backbone_module_ablation','evolution','已发现两个输入视图互补，继续拆解其内部计算。')
node('pure_dl_8039_ablation_no_calibration','输入校准关闭·训练消融','原后验双专家只将输入校准固定为identity。','完成80.04115983@52/stop70，对照-0.18585530；恢复验证通过。旧规则未存权重，未测融合，不据此否定融合贡献。')
node('pure_dl_abd_backbone_module_ablation','ABD·双骨干内部消融完成','冻结B/D；A两分支10项有效模块干预+2项空point数值对照。当前独立MLP早已移除。','完整视图去全局：A -0.50921，ABD +0.025934至80.435073；校准视图去全局ABD -0.131623。两分支均依赖局部注意力/瓶颈/跳连。推理依赖不等于重训收益。',stage='pair')
node('pure_dl_abd_fused_training80_fold5','ABD·原结构融合损失增量训练','冻结B/D，仅A按最终融合NLL更新。','18轮无增益；最佳epoch0 80.40913868。训练后最高80.285611。保存A权重逐张量等于原模型；严格逐元素回放断言失败，未宣称精确复现。',stage='vote')
node('pure_dl_abd_no_global_full80_fold5','ABD·完整视图去全局后适应','仅删除A完整206视图global attention，冻结B/D，同预算低学习率增量训练。','推理起点80.435073（+0.025934），已启动；训练收益待定。采用确定性运算，单模型只诊断。',stage='pair')
edge('pure_dl_abd_backbone_module_ablation','pure_dl_abd_no_global_full80_fold5','evolution','由模块干预定位候选，继续适应；不等于随机初始化训练消融。')
edge('pure_dl_abd_fused_training80_fold5','pure_dl_abd_no_global_full80_fold5','control','同起始checkpoint/预算/融合目标；删去一个全局分支，另开启数值确定性。')
node('pure_dl_abd_no_global_full80_fold5','去全局·确定性训练启动失败','全程确定性运算下首次backward失败，CUDA线性上采样反向不支持。','初始80.43508293；参数更新0次，不是训练结果。保留记录，改用确定性验证。',stage='pair')
node('pure_dl_abd_no_global_full_evaldet80_fold5','ABD·去全局后增量适应','删除完整206视图global attention，B/D冻结；验证确定性，训练沿用普通CUDA。','已启动，训练成绩待定；对照原结构增量训练，候选推理80.435073。',stage='pair')
edge('pure_dl_abd_no_global_full80_fold5','pure_dl_abd_no_global_full_evaldet80_fold5','evolution','修复执行兼容性：仅验证/冻结缓存确定性，训练保留支持的CUDA反向。')
node('pure_dl_abd_scratch_control80_fold5','ABD 模块重训·原结构对照','A从原seed随机初始化，保留原NLL/内部router/辅助监督/动态Q；B/D冻结，只进入验证。','已启动，结果待定。按ABD融合TOTAL选最佳，而非按A单独分数。',stage='pair')
node('pure_dl_abd_scratch_no_global_full80_fold5','ABD 模块重训·完整视图无全局','与原结构对照相同初始化及训练规则，仅移除A完整206视图global attention。','准备完成，等待前一任务结束；尚无训练结果。先验仅推理删除+0.02593。',stage='pair')
edge('pure_dl_abd_backbone_module_ablation','pure_dl_abd_scratch_control80_fold5','evolution','补从随机初始化的训练归因，避免把推理依赖当训练增益。')
edge('pure_dl_abd_scratch_control80_fold5','pure_dl_abd_scratch_no_global_full80_fold5','control','同seed/预算/原始损失与动态Q；单项全局模块删除，按融合验证。')
node('pure_dl_abd_no_global_full_evaldet80_fold5','ABD·去全局增量适应完成','冻结B/D；删除完整视图global，按融合NLL微调A，确定性验证。','80.43508293@0，18轮无训练增益。最佳预测精确回放、B/D不变已验证；+0.025931来自结构删除，不能归功于微调。572.46秒/1798.65MiB。',stage='pair')
node('pure_dl_abd_scratch_no_global_full80_fold5','ABD 模块重训·完整视图无全局','原seed随机初始化、原损失/辅助/动态Q，只关闭完整206视图全局注意力。','已与原结构对照双并发启动；暂无最终成绩。固定B/D，每轮按融合TOTAL选最佳。',stage='pair')
node('pure_dl_abd_fusion_selected_fullcv','ABD 融合选模·完整五折流程','第五折过门槛并恢复核验后，逐折先训练B、冻结同折B/D，再训练A按融合选最佳；复用已有第五折。','流程已准备，补折未启动，尚无完整OOF。原结构第五折初步80.49231；继续等待配对实验结束。',stage='vote')
edge('pure_dl_abd_scratch_control80_fold5','pure_dl_abd_fusion_selected_fullcv','evolution','仅在完整第五折核验通过后补折，不以单折替代完整OOF目标。')
node('pure_dl_abd_scratch_control80_fold5','ABD 原结构·融合选模80.49231','原结构A从头训练，固定B/D；只改变checkpoint按融合TOTAL选择。','完成80.49230854@6/stop24，A单独77.69492；恢复精确，原ABD+0.08315667。主要新收益来自组合选模。',stage='pair')
node('pure_dl_abd_scratch_no_global_full80_fold5','ABD 去全局·匹配训练80.46882','同初始化/训练规则，只删除A完整206视图global attention。','完成80.46881577@6/stop24，恢复精确；比原结构-0.02349276。从头训练不支持删除，保留原结构。',stage='pair')
node('pure_dl_abd_fusion_selected_fullcv','ABD 完整五折·用户暂缓','补折流程已准备，但用户明确暂不补折。','未启动、无完整OOF；后续不得自动开队列，等待恢复。',stage='vote')
node('pure_dl_abd_joint_finetune80_fold5','80.49231·三模型共同训练','同一最佳A6/B/D起点、等权融合NLL，三模型全部允许更新。','已启动21517，待初步结果；max80/pat18，逐模型clip2。增量训练，不是三模型从零训练。',stage='vote')
node('pure_dl_abd_b_only_finetune80_fold5','80.49231·固定A/D训练B','同起点/种子/融合损失/LR，只允许B更新，固定最佳A6与D。','已启动38600，待初步结果；与共同训练匹配对照，checkpoint按融合分数选择。',stage='vote')
edge('pure_dl_abd_scratch_control80_fold5','pure_dl_abd_joint_finetune80_fold5','evolution','用户要求同时融合同时训练，使用已验证的最佳组合起点。')
edge('pure_dl_abd_joint_finetune80_fold5','pure_dl_abd_b_only_finetune80_fold5','control','唯一更新范围差异：全体 versus 仅B；同种子/目标/预算。')
node('pure_dl_abd_scratch_control80_fold5','ABD 融合选模80.49231·收益有集中性','完成80.49230854@6；新CPU分解：PERM+0.21418，POR+0.11342，SW-0.07380。','16井7升9降；最大正贡献井占TOTAL+0.13642，排除它后其余井-0.05685。保留最好结果，但不能声称跨井普遍改善；用户暂缓补折。',stage='pair')
node('pure_dl_abd_b_scratch80_fold5','候选准备·固定A6/D重训B','区别于小学习率微调：B从原seed08随机初始化，恢复原独立损失/辅助/动态Q，按融合选模。','代码准备、未启动，无分数。待当前匹配微调产生结果并释放GPU后决定；不补折。',stage='vote')
edge('pure_dl_abd_b_only_finetune80_fold5','pure_dl_abd_b_scratch80_fold5','evolution','区分B继续微调与从头重新训练，不能混为一个消融。')
node('pure_dl_abd_b_only_finetune80_fold5','固定A/D微调B·无增益','同80.49231起点，只B按最终融合NLL更新。','完成18轮：最佳epoch0，训练最高80.48479059@5（-0.00751794）。精确恢复、A/D不变、覆盖核验通过。923.26秒/1806.81MiB。',stage='vote')
node('pure_dl_abd_b_scratch80_fold5','固定A6/D·B从头训练已启动','B恢复原seed08随机初始化、独立损失、辅助监督和动态Q，每轮按融合TOTAL选模型。','B-only微调无收益后启动，暂无新分数；与联合微调并发，不补折。',stage='vote')
node('pure_dl_abd_joint_finetune80_fold5','三模型共同微调·无增益','从A6/B43/D52共同更新，固定等权融合NLL与小学习率。','完成18轮，最佳仍epoch0 80.49230854；训练最高80.39709517@5。恢复精确，1679.05秒；此结论限于该增量训练方案。',stage='vote')
node('pure_dl_channel_gradient_diagnostic','三通道梯度诊断·SW较弱','64训练井512窗口，A6/A52/B43/D52同输入Q=1；主损失与辅助损失分别测量，无参数更新。','完成87.74秒/660.49MiB。A52梯度范数4.92/3.06/1.62；B的PERM/SW在7/16批冲突，但总梯度未出现全局反向。支持检验温和平衡，尚不能解释早期融合的全部收益。',stage='loss')
edge('pure_dl_abd_scratch_control80_fold5','pure_dl_channel_gradient_diagnostic','evolution','早期A的融合收益引出多任务干扰假设；先测量再做机制消融。')
node('pure_dl_abd_b_fused_scratch80_fold5','从零训练B·最终融合损失','训练中直接融合A/B/D，冻结A6/D，仅B更新；保留原辅助监督，按融合分数选模。','已启动26379，首批真实反向确认只有B骨干/router有梯度、A/D不变；尚无验证分数。记录三通道损失和梯度，暂不叠加通道平衡。',stage='vote')
edge('pure_dl_abd_b_scratch80_fold5','pure_dl_abd_b_fused_scratch80_fold5','control','主训练目标从B独立预测改为ABD最终融合；冻结来源、随机初始化与验证协议相同。')
edge('pure_dl_abd_b_only_finetune80_fold5','pure_dl_abd_b_fused_scratch80_fold5','evolution','已有融合增量微调无收益；本次随机初始化重新训练，不能混为一项对照。')
node('pure_dl_frozen_backbone_channel_heads','冻结骨干分通道头·等待接力','固定B43双骨干和原始路由，三个通道输出头共19350参数、各自早停；A6/D52冻结，按最终融合训练。','CPU队列47285等待旧B训练正常完成；头训练尚未开始、无分数。再次检查GPU容量后启动，不开第三路、不补折。',stage='loss')
edge('pure_dl_channel_gradient_diagnostic','pure_dl_frozen_backbone_channel_heads','evolution','用户根据通道学习速度与梯度差异提出冻结骨干、解耦输出头。')
edge('pure_dl_abd_b_fused_scratch80_fold5','pure_dl_frozen_backbone_channel_heads','evolution','保留用户要求的训练/验证最终融合语义，进一步限制可训练参数为通道头。')
node('pure_dl_abd_b_scratch80_fold5','B从零训练·仅验证融合完成','B独立主损失训练、固定A6/D验证融合；不是warm-start微调。','完成80.484205@47/stop65，较80.492309 -0.008104，最佳预测精确恢复；未取得新增收益。',stage='vote')
node('pure_dl_abd_b_fused_scratch80_fold5','B从零融合损失·用户停止','训练主损失直接融合ABD，仅更新B；原辅助/Q仍保留。','用户在第20轮后停止；日志最高79.947889@10，比基线-0.544420。没有最终恢复核验，不作完整训练结果。',stage='vote')
node('pure_dl_frozen_backbone_channel_heads','B骨干独立头·启动后取消','原B43骨干独立头，融合主训练损失的旧计划。','CPU接力刚启动数据准备即被用户停止，无训练成绩；用户改为A6骨干、头自身损失训练。',stage='loss')
node('pure_dl_a6_channel_heads_own80_fold5','A6冻结骨干·独立通道头','使用80.4923组合的A第6轮，原输出行初始化；仅头自身损失训练，B43/D52只作验证融合。','新实现就绪，待实际启动结果；19350参数、三通道各自最佳checkpoint，固定原路由避免跨头耦合。',stage='loss')
node('pure_dl_b_own_finetune80_fold5','B43微调·仅验证融合','原B43暖启动，训练B自身后验NLL，固定A6/D52只在验证时融合。','准备完成，尚无成绩；同起点/Q/LR对照此前B-only融合NLL微调。',stage='vote')
edge('pure_dl_frozen_backbone_channel_heads','pure_dl_a6_channel_heads_own80_fold5','evolution','用户更正骨干为80.4923来源A6，并改为各头自身损失训练。')
edge('pure_dl_abd_b_only_finetune80_fold5','pure_dl_b_own_finetune80_fold5','control','只将B微调主目标改为自身NLL，融合仍用于验证选模。')
node('pure_dl_a6_channel_heads_own80_fold5','A6独立头·保留运行','A6双骨干和原路由冻结，头自身损失训练，验证仍与B/D投票。','用户要求保留该已启动实验；初始组合80.49230854精确复现。此路线不同于新统一特征backbone。',stage='loss')
node('pure_dl_b_own_finetune80_fold5','B自身损失微调·用户停止','B43暖启动、A6/D52仅验证融合的微调路线。','用户明确停止PID50824；未做最终恢复核验，不自动重启。任务位交给整体特征骨干结构实验。',stage='vote')
node('pure_dl_abd_feature_heads80_fold5','整体ABD特征骨干·新通道头','冻结A6/B43/D52共5骨干，128维特征×5拼接640维；3独立MLP头直接预测，不投票原输出。','运行10715：255819可训练参数，首次反向确认仅新头更新。随机新头epoch0=68.41008，尚无训练收益；保留另一条A6头实验。',stage='loss')
edge('pure_dl_a6_channel_heads_own80_fold5','pure_dl_abd_feature_heads80_fold5','evolution','用户澄清为三个模型整体作为特征backbone，替代输出层投票；旧A头路线保留对照。')
node('pure_dl_a6_channel_heads_own80_fold5','A6独立头完成·80.501552','A6骨干/原路由冻结，仅主头自身损失微调，B/D验证融合；按通道独立最佳轮次。','完成+0.009243；最佳POR/PERM/SW为4/0/11轮，停止22/18/29。组合预测逐元素复现、冻结来源不变、覆盖通过。874.13秒/1730.91MiB；单折微增，不补折。',stage='loss')
node('pure_dl_abd_feature_heads80_fold5','整体ABD特征骨干·新通道头','冻结5条骨干并拼接640维，3个独立新MLP直接输出，不含旧预测投票。','运行10715：第7轮单轮79.218259，独立头候选79.258706（7/5/7），未最终组装核验；持续训练。',stage='loss')
node('pure_dl_abd_feature_stats_heads80_fold5','分通道读出·补回均值与尺度','冻结ABD五骨干；保留640归一化特征，加10个分支均值/logstd，每个头独立零初始化支路。','已准备，尚无成绩；每通道新增1280参数，基础头同初始化/优化/预算。分别检查POR/PERM/SW收益，严格第五折>80.5。',stage='loss')
edge('pure_dl_abd_feature_heads80_fold5','pure_dl_abd_feature_stats_heads80_fold5','control','针对LayerNorm可能丢失的幅值信息；保留原头，零初始化任务专属统计输入。')
# Preserve later editorial extensions when refreshing the original registry.
path=Path(__file__).parent
if (path/'curation.json').exists():
    previous=json.loads((path/'curation.json').read_text(encoding='utf-8'))
    nodes.update(previous.get('nodes',{}))
    known={(e['source'],e['target'],e['type']) for e in edges}
    edges.extend(e for e in previous.get('edges',[]) if (e['source'],e['target'],e['type']) not in known)
    known_dead={d['id'] for d in dead}
    dead.extend(d for d in previous.get('deadends',[]) if d['id'] not in known_dead)
    terms.update(previous.get('terms',{}))
root=path.parents[2]
study=root/'results/deep_learning/quality_readout_v37_20260929'
final=json.loads((study/'assembled_report.json').read_text(encoding='utf-8')) if (study/'assembled_report.json').exists() else None
for target in ('POR','SW'):
    id='pure_dl_quality_readout_v37_'+target+'_fold5'
    out=root/'outputs'/id
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    result='尚无训练成绩' if not state else json.dumps(state.get('modes',state),ensure_ascii=False)
    conclusion='训练快照：'+result
    if final:
        own=[x for x in final['candidates'] if x['target']==target]
        conclusion='训练后最高 '+', '.join(x['mode']+' '+str(x['report']['best_after_update']['target_score']) for x in own)
        conclusion+='；均回退原模型，实际三列TOTAL='+str(final['selected_scores']['TOTAL'])
    node(id,'v37 '+target+' Q50与等权输出头','冻结80.336513基线，全因子有界小头11857参数；同初始化、训练窗，仅质量权重不同。',
         conclusion+'；原第五折，未补其他折。',stage='loss',status='recorded' if final else 'running')
    parent='pure_dl_mechanism_v31_POR_fold5' if target=='POR' else 'pure_dl_split_route_v29_SW_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','冻结原目标预测器，新全因子小头检验训练质量权重')
if final:
    dead=[d for d in dead if d['id']!='dead:v37_full_factor_head']
    end('dead:v37_full_factor_head','v37 全因子小头本次设置关闭','loss',
        ['pure_dl_quality_readout_v37_POR_fold5','pure_dl_quality_readout_v37_SW_fold5'],
        'Q50和等权均未超越80.336513，四项18轮早停；最佳恢复/冻结状态/三列评分通过。',
        '等权相对Q50训练后最佳POR+0.002414、SW+0.006835，但均低于父模型。不继续该设置扫参；不泛化为所有质量加权无效。')
v38study=root/'results/deep_learning/action_value_v38_20260929'
v38final=json.loads((v38study/'assembled_report.json').read_text(encoding='utf-8')) if (v38study/'assembled_report.json').exists() else None
for target,suffix in [('POR','_fixed'),('SW','')]:
    id='pure_dl_action_value_v38_'+target+suffix+'_fold5'
    out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    conclusion='训练快照：'+json.dumps(state.get('modes',state),ensure_ascii=False)
    if (out/'fold_5_report.json').exists():
        report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8'))
        conclusion='最佳恢复完成：'+', '.join(x['mode']+' '+str(x['target_score']) for x in report['arms'])
    if v38final:
        own=[x for x in v38final['candidates'] if x['target']==target]
        conclusion+='；训练后最高 '+', '.join(x['mode']+' '+str(x['report']['best_after_update']['target_score']) for x in own)
        conclusion+='；固定三列TOTAL='+str(v38final['selected_scores']['TOTAL'])
    node(id,'v38 '+target+' 直接动作奖励回归','冻结原预测器，原动作期望奖励加±0.25有界神经修正；point/local5同参数、同初始化，仅隐藏池化不同。',
         conclusion+'。POR首个错误逆变换运行已剔除，修正版从基线重启；不是旧v12密度软动作损失。',stage='loss',status='recorded' if v38final else 'running')
    parent='pure_dl_mechanism_v31_POR_fold5' if target=='POR' else 'pure_dl_split_route_v29_SW_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','固定强密度预测器，直接拟合原官方动作奖励而非似然')
invalid='pure_dl_action_value_v38_POR_fold5'
if (root/'outputs'/invalid/'failure.json').exists():
    node(invalid,'v38 POR 首次执行·标签逆变换错误','POR训练log1p标签被错误用exp还原；第四轮停止，观测分数不纳入选择。',
         '已修复为expm1，在独立新目录从原基线重跑；不使用本次checkpoint。SW原log/exp训练不受影响。',stage='loss',status='failed',highlight=False)
    if not any(e['source']==invalid and e['target']=='pure_dl_action_value_v38_POR_fixed_fold5' for e in edges):
        edge(invalid,'pure_dl_action_value_v38_POR_fixed_fold5','evolution','执行修正，非权重继承；2048训练点奖励标签与官方kernel专项核对')
if v38final:
    dead=[d for d in dead if d['id']!='dead:v38_direct_action_head']
    end('dead:v38_direct_action_head','v38 直接奖励小头本次设置关闭','loss',
        ['pure_dl_action_value_v38_POR_fixed_fold5','pure_dl_action_value_v38_SW_fold5'],
        '修正版POR与SW各point/local5四项均18轮早停，保留父模型TOTAL80.336513。',
        '无绝对收益；局部相对逐点训练后最高POR+0.001128、SW-0.009570。原动作及评分不变，最佳恢复和冻结检查通过；不继续该设置宽度/窗口扫参，不能泛化到所有直接效用学习。')
v39study=root/'results/deep_learning/backbone_norm_v39_20260929'
v39final=json.loads((v39study/'assembled_report.json').read_text(encoding='utf-8')) if (v39study/'assembled_report.json').exists() else None
for mode in ('decoder_norm','all_norm'):
    id='pure_dl_backbone_norm_v39_SW_'+mode+'_fold5';out=root/'outputs'/id
    if not out.exists():continue
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8')) if (out/'config.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:
        conclusion=f"最佳SW={report['target_score']:.12f}@{report['best_epoch']}，训练后最高={report['best_after_update']['target_score']:.12f}，停止{report['stopped_epoch']}轮；最佳恢复与冻结状态通过。"
    if v39final:conclusion+='实际三列TOTAL='+str(v39final['selected_scores']['TOTAL'])
    node(id,'v39 SW '+mode+' 归一化仿射微调',
         '只更新已有骨干归一化scale/bias，参数'+str(cfg.get('trainable_parameters','待定'))+'；原头/路由/卷积固定。不同更新范围，非同容量消融。',
         conclusion,stage='architecture',status='recorded' if report else 'running')
    if not any(e['source']=='pure_dl_split_route_v29_SW_fold5' and e['target']==id for e in edges):
        edge('pure_dl_split_route_v29_SW_fold5',id,'evolution','冻结输出侧，向私有骨干内部移动可训练参数；同基线起点')
if v39final and not v39final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v39_norm_affine_SW']
    end('dead:v39_norm_affine_SW','v39 SW归一化仿射本次微调关闭','architecture',
        ['pure_dl_backbone_norm_v39_SW_decoder_norm_fold5','pure_dl_backbone_norm_v39_SW_all_norm_fold5'],
        '解码端及全骨干两范围均18轮未超过父模型，固定三列80.336513不变。',
        '全骨干相对解码端训练后最高SW差-0.005456；不继续本设置学习率扫描。仅限当前SW起点与Q50/NLL，不泛化为所有骨干适配无效。')
v40study=root/'results/deep_learning/well_invariance_v40_20260929'
v40final=json.loads((v40study/'assembled_report.json').read_text(encoding='utf-8')) if (v40study/'assembled_report.json').exists() else None
for mode in ('blocked','reversed'):
    id='pure_dl_well_invariance_v40_SW_'+mode+'_fold5';out=root/'outputs'/id
    if not out.exists():continue
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:
        conclusion=f"最佳SW={report['target_score']:.12f}@{report['best_epoch']}；训练后最高{report['best_after_update']['target_score']:.12f}；{report['stopped_epoch']}轮。"
        conclusion+='末态分类诊断：'+json.dumps(report['last_domain_diagnostic'],ensure_ascii=False)
    if v40final:conclusion+='固定三列TOTAL='+str(v40final['selected_scores']['TOTAL'])
    node(id,'v40 SW 井分类梯度 '+mode,
         '同5120骨干norm与20608训练分类器参数；64训练井辅助标签，验证井不映射、ID不作输入。前三轮阻断，实验组4至8轮反转系数升至1，域CE权重.1。',
         conclusion,stage='loss',status='recorded' if report else 'running')
    parent='pure_dl_backbone_norm_v39_SW_all_norm_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'control','同当前强SW初始化，增加同容量训练井分类器，只改变骨干是否收到反向井分类梯度；不是旧多目标从头实验')
if v40final and not v40final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v40_SW_domain_reversal']
    end('dead:v40_SW_domain_reversal','v40 SW井梯度反转本次设置关闭','loss',
        ['pure_dl_well_invariance_v40_SW_blocked_fold5','pure_dl_well_invariance_v40_SW_reversed_fold5'],
        '两项完成，未超过原SW，固定三列80.336513保留。辅助分类器的准确率只是训练诊断。',
        '本次归一化适配及固定反转调度未改善，不继续该系数扫描；不得泛化为所有域不变性方法无效。')
v41study=root/'results/deep_learning/perm_family_v41_20260929'
v41final=json.loads((v41study/'assembled_report.json').read_text(encoding='utf-8')) if (v41study/'assembled_report.json').exists() else None
id='pure_dl_perm_family_v41_fold5'
if v41final:
    conclusion='；'.join(x['mode']+' 初始PERM='+str(x['report']['initial_candidate_score'])+'，训练后最高='+str(x['post_update_scores']['PERM']) for x in v41final['candidates'])
    node(id,'v41 PERM Gaussian / Student-t3分布族',
         '冻结原PERM；同初始化11824参数密度头，16分量权重/位置/尺度修正。t3与Gaussian初始均值和标准差相同但函数不同；默认概率、原log10标签/Q50/官方动作固定。',
         conclusion+'；两组18轮结束，候选最佳均第0轮，最终回退原PERM。实际三列TOTAL='+str(v41final['selected_scores']['TOTAL']),stage='loss',status='recorded')
    parent='pure_dl_split_route_v29_PERM_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'control','同容量条件分布族；明确区分初始分布干预与训练收益')
    dead=[d for d in dead if d['id']!='dead:v41_PERM_family']
    end('dead:v41_PERM_family','v41 当前PERM分布族小头关闭','loss',[id],
        'Gaussian及Student-t3训练后均低于原PERM89.396724，保留三列TOTAL80.336513。最佳恢复/冻结状态/源哈希/三列评分通过。',
        '训练后最佳t3相对Gaussian+0.005536，但t3初始已经退步且训练未恢复。不继续自由度/混合系数/网格扫描，结论不推广到所有重尾模型。')
v42study=root/'results/deep_learning/anchored_backbone_v42_20260929'
v42final=json.loads((v42study/'assembled_report.json').read_text(encoding='utf-8')) if (v42study/'assembled_report.json').exists() else None
for mode in ('free','anchored'):
    id='pure_dl_anchored_backbone_v42_SW_'+mode+'_fold5';out=root/'outputs'/id
    if not out.exists():continue
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion=f"最佳SW={report['target_score']:.12f}@{report['best_epoch']}，训练后最高={report['best_after_update']['target_score']:.12f}，停止{report['stopped_epoch']}轮；最后原点惩罚={report['last_normalized_anchor']:.9f}。"
    if v42final:
        conclusion+='固定三列TOTAL='+str(v42final['selected_scores']['TOTAL'])
        conclusion+='；约束组相对自由组训练后最高SW+0.002969807，18/18匹配轮次更高，但两组均未超父模型。'
    node(id,'v42 SW末端块 '+mode,
         '仅原两个dec0块可训练，同容量同初始化；原点惩罚系数0或1，按初始张量RMS归一化，RMS下限.01。不是伪标签，无网络扩容。',
         conclusion,stage='loss',status='recorded' if report else 'running')
    parent='pure_dl_split_route_v29_SW_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','同强SW起点，固定原头/路由，更新已有末端卷积残差块')
if v42final and not v42final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v42_SW_dec0_anchor']
    end('dead:v42_SW_dec0_anchor','v42 SW末端块本次设置关闭','loss',
        ['pure_dl_anchored_backbone_v42_SW_free_fold5','pure_dl_anchored_backbone_v42_SW_anchored_fold5'],
        '两个候选均未超过原SW73.663083，实际三列TOTAL80.336513保持。',
        '约束相对自由微调训练后最高+0.002969807，18/18匹配轮次更高：保留减少退化的局部机制证据，但不采用其预测。关闭本dec0替换，不扫系数/LR；不能泛化为所有骨干适配失败。')
v43study=root/'results/deep_learning/anchored_features_v43_20260929'
v43final=json.loads((v43study/'assembled_report.json').read_text(encoding='utf-8')) if (v43study/'assembled_report.json').exists() else None
for mode in ('free','anchored'):
    id='pure_dl_anchored_features_v43_SW_'+mode+'_fold5';out=root/'outputs'/id
    if not out.exists():continue
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion=f"最佳SW={report['target_score']:.12f}@{report['best_epoch']}，训练后最高={report['best_after_update']['target_score']:.12f}，停止{report['stopped_epoch']}轮；最后原点惩罚={report['last_normalized_anchor']:.9f}。"
    if v43final:
        conclusion+='实际固定三列TOTAL='+str(v43final['selected_scores']['TOTAL'])
        conclusion+='；约束相对自由组训练后最高+0.002517，11/18对应轮次更高，但两组均低于原SW。'
    node(id,'v43 SW完整特征 '+mode,
         '1272694既有特征参数可训练，原输出头/路由固定；同初始化同数据，原点惩罚系数0/1，统一lr1e-5。无新增容量。v42局部减少退化机制迁移，本批内才是匹配对照。',
         conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_split_route_v29_SW_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','同原SW强起点，完整特征适配，输出头与路由固定')
    control='pure_dl_anchored_backbone_v42_SW_'+mode+'_fold5'
    if not any(e['source']==control and e['target']==id for e in edges):edge(control,id,'control','迁移更新约束思路，不继承v42权重；同时改变更新范围与学习率，不称单因素容量对照')
if v43final and not v43final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v43_SW_feature_anchor']
    end('dead:v43_SW_feature_anchor','v43 SW完整特征本次配置关闭','architecture',
        ['pure_dl_anchored_features_v43_SW_free_fold5','pure_dl_anchored_features_v43_SW_anchored_fold5'],
        '两项未超过原SW，保持实际三列TOTAL80.336513。',
        '原点约束训练后最高相对+0.002517，11/18对应轮次更高，中位+0.001985，未证实稳定收益；不以第0轮回退宣称训练收益，不扫描本系数/LR，结论限当前配置。')
v44study=root/'results/deep_learning/state_messages_v44_20260929'
v44final=json.loads((v44study/'assembled_report.json').read_text(encoding='utf-8')) if (v44study/'assembled_report.json').exists() else None
id='pure_dl_state_messages_v44_SW_fold5';out=root/'outputs'/id
if out.exists():
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion='；'.join(x['mode']+' 最佳SW='+str(x['target_score'])+'，训练后最高='+str(x['best_after_update']['target_score']) for x in report['arms'])
    if v44final:conclusion+='；实际三列TOTAL='+str(v44final['selected_scores']['TOTAL'])+'；neighbor训练后最高相对self+0.006936065，11井升/5井降，仍低于原模型。'
    node(id,'v44 SW默认状态 self/neighbor消息',
         '两项同10273参数/初始化，8步有限半径神经状态消息；self或neighbor读取状态不同，原连续分布固定。训练井SW状态切换2.0538%，仅动机不是收益。旧transition_loss未正式训练，本次新推理模块。',
         conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_split_route_v29_SW_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','固定强SW，默认门控增加有限状态消息；不恢复旧转移损失原型')
if v44final and not v44final['selections']:
    dead=[d for d in dead if d['id']!='dead:v44_SW_state_messages']
    end('dead:v44_SW_state_messages','v44 当前消息头未替换基线','architecture',[id],
        '两项未超过原SW，固定组合保持；相对比较见独立记录。',
        'neighbor对self局部+0.006936065，保留小头对照供独立机制研究，但本配置不替换原模型；不扫描消息步数/耦合界，不把第0轮回退当收益，不能泛化为所有序列状态模型无效。')
v45study=root/'results/deep_learning/divergent_models_v45_20260929'
v45final=json.loads((v45study/'assembled_report.json').read_text(encoding='utf-8')) if (v45study/'assembled_report.json').exists() else None
for target,mode in [('SW','long_section'),('POR','curve_axial')]:
    id=f'pure_dl_divergent_v45_{target}_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion=f"原第五折{target}={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}，相对原目标{report['delta_reference_target']:+.12f}。"
    if v45final:
        conclusion+='实际固定三列候选TOTAL='+str(v45final['selected_scores']['TOTAL'])
        if target=='SW':conclusion+='；磁盘重载全144665点一致；8井升/8井降，微小数值增量不构成明确泛化收益。'
    description=('1024点原206特征，局部细节+64个patch全局四层注意力，完整私有SW Gaussian密度网络。' if mode=='long_section' else
                 '512点九基础曲线输入变体，曲线/深度两轴注意力，25%曲线块遮挡与.1联合重建，完整私有POR Student3密度网络。')
    node(id,'v45 发散新模型 '+target+' '+mode,description+'全部随机网络，不加载旧权重或旧预测；原训练隔离/评分/Q50/80-18保持。不是同容量因果消融。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_split_route_v29_SW_fold5' if target=='SW' else 'pure_dl_mechanism_v31_POR_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'control','评分替换基线，非模型权重血缘；用户要求离开冻结小头路线')
if v45final:
    for row in v45final['candidates']:
        if row['report']['delta_reference_target']>=0:continue
        did='dead:v45_'+row['mode'];dead=[d for d in dead if d['id']!=did]
        end(did,'v45 '+row['mode']+' 当前结果未胜原目标','architecture',
            [f"pure_dl_divergent_v45_{row['target']}_{row['mode']}_fold5"],
            f"原第五折{row['target']}={row['report']['target_score']:.9f}，相对原目标{row['report']['delta_reference_target']:+.9f}。",
            '保留真实学习曲线；不能把初始化到训练的增量当作超越旧基线，也不泛化为整类架构无效。')
v46study=root/'results/deep_learning/well_conditioned_v46_20260929'
v46final=json.loads((v46study/'assembled_report.json').read_text(encoding='utf-8')) if (v46study/'assembled_report.json').exists() else None
for mode in ('reference','well'):
    id=f'pure_dl_well_conditioned_v46_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion=f"SW={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v46final:
        row=next(x for x in v46final['candidates'] if x['mode']==mode)
        conclusion+='训练后最高='+str(row['best_after_update']['target_score'])+'；实际三列TOTAL='+str(v46final['selected_scores']['TOTAL'])
        conclusion+='；well较父SW仅+0.016586，7井升9井降，探索区间跨零，未证实稳定新井收益。' if mode=='well' else '；最佳回退第0轮，当前静态条件热启动未产生收益。'
    node(id,'v46 整井条件 '+mode,'v45新SW同起点，128点输入编码生成六层scale/bias；当前井与64训练井固定参考对照，同836921参数，全部骨干训练。无标签/井ID特征。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_divergent_v45_SW_long_section_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','热启动新SW，增加学习整井环境的条件网络，完整骨干继续训练')
if v46final and v46final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v46_reference']
    end('dead:v46_reference','v46 固定参考条件热启动未改善','architecture',
        ['pure_dl_well_conditioned_v46_SW_reference_fold5'],
        '训练后最高SW73.533866，低于起点73.664378；第18轮早停，回退第0轮。',
        '仅停止本静态参考配置延伸；当前井输入仍有局部正差，不能把参考失败泛化为条件网络无效。')
if v46final and not v46final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v46_well_conditioned']
    end('dead:v46_well_conditioned','v46 当前井条件配置未替换基线','architecture',
        ['pure_dl_well_conditioned_v46_SW_reference_fold5','pure_dl_well_conditioned_v46_SW_well_fold5'],
        '两项未超过v45 SW；当前候选TOTAL80.336967保持。',
        '同容量井输入源对照见独立报告，不扫描采样数量或调制界；只限本热启动配置，不能推断所有井条件网络无效。')
v47study=root/'results/deep_learning/well_memory_v47_20260929'
v47final=json.loads((v47study/'assembled_report.json').read_text(encoding='utf-8')) if (v47study/'assembled_report.json').exists() else None
for mode in ('global_query','section_query'):
    id=f'pure_dl_well_memory_v47_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion=f"SW={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v47final:
        row=next(x for x in v47final['candidates'] if x['mode']==mode)
        conclusion+='训练后最高='+str(row['best_after_update']['target_score'])+'；实际三列TOTAL='+str(v47final['selected_scores']['TOTAL'])
    node(id,'v47 任务记忆 '+mode,'v46井条件模型同起点，状态/数值各自4头注意力读128点整井输入记忆；查询取窗口均值或保留局部位置。同900985参数，全共享骨干训练，不是两套完全独立骨干。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','热启动正候选SW，增加任务查询整井输入记忆；无其他模型预测混合')
if v47final and not v47final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v47_task_memory']
    end('dead:v47_task_memory','v47 当前任务记忆读出没有提高','architecture',
        ['pure_dl_well_memory_v47_SW_global_query_fold5','pure_dl_well_memory_v47_SW_section_query_fold5'],
        '两项训练后最高均未超过v46 SW73.680964；实际组合保留80.342772。',
        '停止当前热启动配置的查询维数/点数/头数扫描，不泛化为全部任务记忆无效；两组都含任务分支，不能单独归因任务分离。')
v48study=root/'results/deep_learning/quantile_density_v48_20260929'
v48final=json.loads((v48study/'assembled_report.json').read_text(encoding='utf-8')) if (v48study/'assembled_report.json').exists() else None
for mode in ('linear','rqs'):
    id=f'pure_dl_quantile_density_v48_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    initial=json.loads((out/'initial.json').read_text(encoding='utf-8')) if (out/'initial.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if initial:conclusion+='；新密度初始SW='+str(initial['target_score'])+'（非父函数）'
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v48final:
        row=next(x for x in v48final['candidates'] if x['mode']==mode)
        conclusion+='训练后最高='+str(row['best_after_update']['target_score'])+'；实际三列TOTAL='+str(v48final['selected_scores']['TOTAL'])
    if v48final and mode=='rqs':conclusion+='；较linear含初始最佳SW+1.082068，13井升3井降；保留机制候选但未超原SW，不升级当前配方。'
    node(id,'v48 条件分位密度 '+mode,'v46骨干/FiLM/默认头热启动，原高斯密度换为33有序分位点及指数尾；线性与有理二次CDF同838211参数。全骨干训练，训练侧先验新密度不保持父函数。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','继承训练折骨干/FiLM/默认头，重置连续密度为训练侧分位先验，非函数保持扩展')
if v48final and not v48final['adopted'] and v48final['post_update_rqs_minus_linear']<=0:
    dead=[d for d in dead if d['id']!='dead:v48_quantile_density']
    end('dead:v48_quantile_density','v48 当前条件分位密度未改善','architecture',
        ['pure_dl_quantile_density_v48_SW_linear_fold5','pure_dl_quantile_density_v48_SW_rqs_fold5'],
        '两项均未超过原SW73.680964；实际组合保持80.342772，初始化到训练的改善不等于超过父模型。',
        '停止当前点数/尾部质量扫描；只限本固定CDF高度、调和导数、热启动与训练预算，不能否定所有条件流模型。')
if v48final and v48final['post_update_rqs_minus_linear']>0:
    dead=[d for d in dead if d['id'] not in ('dead:v48_quantile_density','dead:v48_linear_density')]
    end('dead:v48_linear_density','v48 线性分位密度训练退化','architecture',
        ['pure_dl_quantile_density_v48_SW_linear_fold5'],
        'linear初始72.346442、训练后最高72.136428，回退第0轮；RQS训练后73.428510，相对训练后线性+1.292082。',
        '仅关闭本线性CDF配置。RQS对照有益但未超过v46 SW73.680964，保留RQS权重作为机制候选，不替换当前组合、不补折。')
v49study=root/'results/deep_learning/quantile_supervision_v49_20260929'
v49final=json.loads((v49study/'assembled_report.json').read_text(encoding='utf-8')) if (v49study/'assembled_report.json').exists() else None
for mode in ('likelihood','quantile_aux'):
    id=f'pure_dl_quantile_supervision_v49_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v49final:
        row=next(x for x in v49final['candidates'] if x['mode']==mode)
        conclusion+='训练后最高='+str(row['best_after_update']['target_score'])+'；实际三列TOTAL='+str(v49final['selected_scores']['TOTAL'])
    node(id,'v49 分位直接监督 '+mode,'v48 RQS第4轮相同838211参数与原函数；全骨干/FiLM/输出训练，原BCE+NLL加0或0.5有限33分位点pinball，非精确CRPS。共同新lr3e-5，仅SW训练标签。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_quantile_density_v48_SW_rqs_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','携带有消融收益的平滑CDF，测试直接分位监督；未将低于SOTA的父模型误称新基线')
if v49final and v49final['aux_minus_likelihood']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v49_quantile_aux']
    end('dead:v49_quantile_aux','v49 当前直接分位项无同容量收益','architecture',
        ['pure_dl_quantile_supervision_v49_SW_likelihood_fold5','pure_dl_quantile_supervision_v49_SW_quantile_aux_fold5'],
        '同函数起点下，含初始回退的辅助分位监督不优于似然对照；训练后差及绝对分数见报告。',
        '停止当前系数或分位点数扫描；只限v48热启动、当前RQS参数化和预算，不否定所有分位监督或原v48平滑插值收益。')
v50study=root/'results/deep_learning/perm_transfer_v50_20260929'
v50final=json.loads((v50study/'assembled_report.json').read_text(encoding='utf-8')) if (v50study/'assembled_report.json').exists() else None
for mode in ('scratch','sw_transfer'):
    id=f'pure_dl_perm_transfer_v50_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    initial=json.loads((out/'initial.json').read_text(encoding='utf-8')) if (out/'initial.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if initial:conclusion+='；全新PERM输出先验='+str(initial['target_score'])
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v50final:
        conclusion+='；实际三列TOTAL='+str(v50final['selected_scores']['TOTAL'])+'；迁移-随机PERM='+str(v50final['transfer_minus_scratch']['channel_delta'])
    node(id,'v50 PERM骨干 '+mode,'同836921参数长井段/整井FiLM，PERM输出全重置；骨干随机或同64训练井SW权重初始化。输出初始哈希相同、骨干不同是预设变量；全模型私有训练，仅当前PERM损失。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution' if mode=='sw_transfer' else 'control','同训练折跨目标骨干迁移、输出重置' if mode=='sw_transfer' else '同结构随机对照，不继承网络权重')
if v50final and v50final['transfer_minus_scratch']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v50_sw_transfer']
    end('dead:v50_sw_transfer','v50 当前SW到PERM迁移未优于随机','architecture',
        ['pure_dl_perm_transfer_v50_PERM_sw_transfer_fold5'],
        '匹配容量/新PERM头/预算下迁移不优于随机，绝对分数和轮次见报告。',
        '仅关闭当前SW源骨干初始化，不扫描冻结层或LR，也不泛化所有跨目标迁移无效；当前组合是否更新以实际评分为准。')
if v50final and v50final['transfer_minus_scratch']['channel_delta']>0:
    dead=[d for d in dead if d['id']!='dead:v50_random_perm']
    end('dead:v50_random_perm','v50 随机PERM骨干未替代当前模型','architecture',
        ['pure_dl_perm_transfer_v50_PERM_scratch_fold5'],
        '随机PERM88.811296，低当前89.396724；迁移组89.050808较随机+0.239512，但也未超过当前。',
        '清理随机负权重；迁移保留为局部匹配收益候选，9井升7井降、探索区间跨零，不宣称稳定新井收益或开始补折。')
v51study=root/'results/deep_learning/perm_moment_mix_v51_20260929'
v51final=json.loads((v51study/'assembled_report.json').read_text(encoding='utf-8')) if (v51study/'assembled_report.json').exists() else None
for mode in ('plain','interwell'):
    id=f'pure_dl_perm_moment_mix_v51_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v51final:conclusion+='；实际三列TOTAL='+str(v51final['selected_scores']['TOTAL'])+'；增强-对照PERM='+str(v51final['interwell_minus_plain']['channel_delta'])
    node(id,'v51 PERM跨井矩 '+mode,'v50迁移同836921参数和初始函数；stem/FiLM后训练时以0.5概率Beta(.1,.1)混合不同训练井隐藏均值/尺度，统计detach/有效掩码，无新学习参数。验证关闭，井ID只约束配对。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_perm_transfer_v50_PERM_sw_transfer_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留有消融收益的迁移PERM，新增训练域矩增强的匹配对照')
if v51final and v51final['interwell_minus_plain']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v51_moment_mix']
    end('dead:v51_moment_mix','v51 当前跨井矩增强无最终收益','architecture',
        ['pure_dl_perm_moment_mix_v51_PERM_plain_fold5','pure_dl_perm_moment_mix_v51_PERM_interwell_fold5'],
        '同起点下跨井矩增强不优于无增强，训练后与含起点差分别记录；实际应用数量已核验，非失活试验。',
        '不扫描概率/Beta/层位置；只限此层与当前PERM迁移配置，不否定所有井间增强，也不否定v50迁移局部收益。')
v52study=root/'results/deep_learning/perm_token_pool_v52_20260929'
v52final=json.loads((v52study/'assembled_report.json').read_text(encoding='utf-8')) if (v52study/'assembled_report.json').exists() else None
for mode in ('position','content'):
    id=f'pure_dl_perm_token_pool_v52_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}，最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v52final:conclusion+='；实际三列TOTAL='+str(v52final['selected_scores']['TOTAL'])+'；内容-位置PERM='+str(v52final['content_minus_position']['channel_delta'])
    node(id,'v52 PERM汇聚 '+mode,'同837305参数/初始，SW来源迁移与PERM全新头；16点四组24维值学习汇聚，打分读位置或内容。全骨干训练，改变内部压缩不是旧预测投票。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','沿v50迁移配方从SW源重置PERM头，增加内部可学习深度汇聚')
if v52final and v52final['content_minus_position']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v52_content_pool']
    end('dead:v52_content_pool','v52 当前内容汇聚未优于位置','architecture',
        ['pure_dl_perm_token_pool_v52_PERM_position_fold5','pure_dl_perm_token_pool_v52_PERM_content_fold5'],
        '位置88.894217、内容88.868046，内容-位置=-0.026171；两者也低于历史普通平均89.050808。真实梯度与训练后权重已核验。',
        '不扫描组数/步长/温度；只限此打分与通道分组，不泛化所有自适应压缩无效。')
v53study=root/'results/deep_learning/por_transfer_v53_20260929'
v53final=json.loads((v53study/'assembled_report.json').read_text(encoding='utf-8')) if (v53study/'assembled_report.json').exists() else None
for mode in ('scratch','sw_transfer'):
    id=f'pure_dl_por_transfer_v53_POR_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终POR={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v53final:conclusion+='；实际组合TOTAL='+str(v53final['selected_scores']['TOTAL'])+'；迁移-随机POR='+str(v53final['transfer_minus_scratch']['channel_delta'])
    node(id,'v53 POR骨干迁移 '+mode,'同836921参数长井段与整井FiLM，POR全新Student-t3头；随机或同64训练井SW权重初始化，当前只用POR损失全网络训练。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution' if mode=='sw_transfer' else 'control','同训练折SW到POR迁移；随机组不继承网络权重')
if v53final and v53final['transfer_minus_scratch']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v53_por_transfer']
    end('dead:v53_por_transfer','v53 当前SW到POR迁移无匹配收益','architecture',
        ['pure_dl_por_transfer_v53_POR_scratch_fold5','pure_dl_por_transfer_v53_POR_sw_transfer_fold5'],
        '同结构/先验/预算下迁移不优于随机；实际分数与相对当前POR差见报告。',
        '不扫描当前来源的冻结层数或LR；只限该SW骨干到POR，不否定v50的PERM局部收益。')
if v53final and v53final['transfer_minus_scratch']['channel_delta']>0:
    dead=[d for d in dead if d['id']!='dead:v53_random_por']
    end('dead:v53_random_por','v53 随机长井段POR未替代现有模型','architecture',
        ['pure_dl_por_transfer_v53_POR_scratch_fold5'],
        '随机73.640525、迁移73.888601，迁移局部收益+0.248077；均低于现有POR77.551937，实际组合仍80.342772。',
        '随机负权重已清理；迁移完整磁盘重建后保留为机制候选，不宣称新冠军或完整OOF收益。')
v54study=root/'results/deep_learning/sw_stratified_v54_20260929'
v54final=json.loads((v54study/'assembled_report.json').read_text(encoding='utf-8')) if (v54study/'assembled_report.json').exists() else None
for mode in ('uniform','stratified'):
    id=f'pure_dl_sw_stratified_v54_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v54final:conclusion+='；实际组合TOTAL='+str(v54final['selected_scores']['TOTAL'])+'；分层-随机SW='+str(v54final['stratified_minus_uniform']['channel_delta'])
    node(id,'v54 SW状态采样 '+mode,'同836921参数及v46SW完整起点；64随机或32/32状态锚点，按2*N_state/N校正，共同窗口平均风险；全骨干训练。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','同起点与窗口目标，只改训练锚点抽样；非v14井风险权重')
if v54final and v54final['stratified_minus_uniform']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v54_state_sampling']
    end('dead:v54_state_sampling','v54 当前状态分层无最终配对收益','architecture',
        ['pure_dl_sw_stratified_v54_SW_uniform_fold5','pure_dl_sw_stratified_v54_SW_stratified_fold5'],
        '完整选择含初始回退；分层不优于随机，训练后分数与实际组成证据见报告。',
        '不扫描32/32比例/LR；只限当前SW模型的锚点状态分层及窗口风险，不泛化所有采样无效。')
v55study=root/'results/deep_learning/sw_sharpness_v55_20260929'
v55final=json.loads((v55study/'assembled_report.json').read_text(encoding='utf-8')) if (v55study/'assembled_report.json').exists() else None
for mode in ('adamw','sam'):
    id=f'pure_dl_sw_sharpness_v55_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v55final:conclusion+='；实际组合TOTAL='+str(v55final['selected_scores']['TOTAL'])+'；SAM-AdamW='+str(v55final['sam_minus_adamw']['channel_delta'])
    node(id,'v55 SW权重扰动 '+mode,'同836921参数及v46SW完整起点；AdamW一次或SAM同批两遍、rho.05，恢复原权重后更新；原批次点均SW损失，全模型训练，非等计算量。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','当前SW同起点优化对照，复用旧SAM更新实现，不恢复旧多目标训练')
if v55final and v55final['sam_minus_adamw']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v55_sam']
    end('dead:v55_sam','v55 当前SW权重扰动无最终收益','architecture',
        ['pure_dl_sw_sharpness_v55_SW_adamw_fold5','pure_dl_sw_sharpness_v55_SW_sam_fold5'],
        '完整选择含初始回退；SAM不优于AdamW，训练后成绩/扰动实测/耗时见报告。',
        '不扫描rho或优化器组合；只限当前单目标热启动尺度，不泛化所有锐度优化无效。')
v56study=root/'results/deep_learning/sw_interaction_v56_20260929'
v56final=json.loads((v56study/'assembled_report.json').read_text(encoding='utf-8')) if (v56study/'assembled_report.json').exists() else None
for mode in ('additive','multiplicative'):
    id=f'pure_dl_sw_interaction_v56_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v56final:conclusion+='；实际组合TOTAL='+str(v56final['selected_scores']['TOTAL'])+'；乘法-加法SW='+str(v56final['multiplicative_minus_additive']['channel_delta'])
    node(id,'v56 SW内部交互 '+mode,'同839481参数及初始；四个FF随机替换、两支256维加/乘及公共LayerNorm；SW输出全重置，其余源SW迁移，全模型只用SW损失。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_well_conditioned_v46_SW_well_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','迁移TCN/注意力/FiLM，替换四层FF并重置输出；非源函数延续')
if v56final and v56final['multiplicative_minus_additive']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v56_multiplicative']
    end('dead:v56_multiplicative','v56 当前内部乘性交互无匹配收益','architecture',
        ['pure_dl_sw_interaction_v56_SW_multiplicative_fold5'],
        '同容量/初始参数/预算下乘法73.522007低于加法73.713976；加法已采用，组合80.354326，不把正对照列为失败路线。',
        '不扫描分支宽度或层数；只限此新FF/公共归一化，不否定全部门控或乘性交互。')
v57study=root/'results/deep_learning/sw_ffgrowth_v57_20260929'
v57final=json.loads((v57study/'assembled_report.json').read_text(encoding='utf-8')) if (v57study/'assembled_report.json').exists() else None
for mode in ('base','grow128'):
    id=f'pure_dl_sw_ffgrowth_v57_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v57final:conclusion+='；实际组合TOTAL='+str(v57final['selected_scores']['TOTAL'])+'；扩容-基准SW='+str(v57final['grow128_minus_base']['channel_delta'])
    node(id,'v57 SW并行容量 '+mode,'同v56加法完整函数；base839481或989369参数，四个零输出128并行FF保留原256核心归一化，全网络共同lr3e-5/原损失训练。',conclusion,stage='architecture',status='recorded' if report else 'running')
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','沿已采用微增配置做同函数有限容量对照；非单一384维LayerNorm')
if v57final and v57final['grow128_minus_base']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v57_growth']
    end('dead:v57_growth','v57 当前并行扩容无匹配收益','architecture',
        ['pure_dl_sw_ffgrowth_v57_SW_grow128_fold5'],
        '同核心/函数/学习率/预算下扩容不优于基准；完整选择含起点，训练后差单独记录。',
        '不扫描本并行128/256宽度或层数；只限本初始化和共同LR，不泛化所有大网络无效。')
v58study=root/'results/deep_learning/sw_rqs_staging_v58_20260929'
v58final=json.loads((v58study/'assembled_report.json').read_text(encoding='utf-8')) if (v58study/'assembled_report.json').exists() else None
for mode in ('joint','headstart'):
    id=f'pure_dl_sw_rqs_staging_v58_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v58final:conclusion+='；实际组合TOTAL='+str(v58final['selected_scores']['TOTAL'])+'；先适配-共同SW='+str(v58final['headstart_minus_joint']['channel_delta'])
    node(id,'v58 平滑密度适配 '+mode,'同840771参数及初始RQS函数；v56加法骨干/默认头保留，新密度先验；首轮全训练或仅密度5轮后全解冻。原CDF代码未改，骨干更新数不同。',conclusion,stage='architecture',status='recorded' if report else 'running')
    for parent,kind,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','evolution','载入已采用新骨干及默认头，密度更换'),('pure_dl_quantile_density_v48_SW_rqs_fold5','evolution','复用已验证RQS机制与代码，不载入该旧模型权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,kind,detail)
if v58final and v58final['headstart_minus_joint']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v58_headstart']
    end('dead:v58_headstart','v58 当前密度先适配无匹配收益','architecture',
        ['pure_dl_sw_rqs_staging_v58_SW_headstart_fold5'],
        '同完整RQS起点下先密度5轮不优于首轮共同训练；实际冻结/解冻梯度与最终差见报告。',
        '不扫描冻结轮数或LR；只限此密度/骨干，组合绝对分数和历史v48局部平滑收益分开记录。')
if v58final and v58final['headstart_minus_joint']['channel_delta']>0:
    dead=[d for d in dead if d['id']!='dead:v58_joint']
    end('dead:v58_joint','v58 当前新密度立即共同训练落后','architecture',
        ['pure_dl_sw_rqs_staging_v58_SW_joint_fold5'],
        '共同73.587567低于先适配73.708218；先适配有配对+0.120651，但仍低于当前SW73.713976。最佳在仅密度训练第2轮。',
        '清理共同训练负权重，保留已重载验证的先适配候选；不把后续解冻未刷新成绩写成全模型微调收益。')
v59study=root/'results/deep_learning/sw_cached_readout_v59_20260929'
v59final=json.loads((v59study/'assembled_report.json').read_text(encoding='utf-8')) if (v59study/'assembled_report.json').exists() else None
for mode in ('affine','silu'):
    id=f'pure_dl_sw_cached_readout_v59_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v59final:conclusion+='；实际组合TOTAL='+str(v59final['selected_scores']['TOTAL'])+'；SiLU-affine SW='+str(v59final['silu_minus_affine']['channel_delta'])
    node(id,'v59 冻结特征密度读出 '+mode,'同851237总/14852密度参数及初始函数；冻结v58骨干和默认头，密度原线性加128-64-34零输出支路，恒等或SiLU。共享canonical训练缓存改变历史随机上下文；原Q/RQS评分不改。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_rqs_staging_v58_SW_headstart_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留先适配第2轮完整函数，仅训练密度读出；缓存推理需原始输入重建')
if v59final and v59final['silu_minus_affine']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v59_silu']
    end('dead:v59_silu','v59 当前SiLU密度读出无匹配收益','architecture',
        ['pure_dl_sw_cached_readout_v59_SW_silu_fold5'],
        '同参数和完整起点，SiLU不优于仿射；完整选择含回退，训练后差与缓存边界单列。',
        '不扫描本固定骨干的宽度/激活；仅限本RQS读出，非所有非线性头。')
v60study=root/'results/deep_learning/sw_brier_v60_20260929'
v60final=json.loads((v60study/'assembled_report.json').read_text(encoding='utf-8')) if (v60study/'assembled_report.json').exists() else None
for mode in ('likelihood','brier'):
    id=f'pure_dl_sw_brier_v60_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v60final:conclusion+='；实际组合TOTAL='+str(v60final['selected_scores']['TOTAL'])+'；Brier-likelihood SW='+str(v60final['brier_minus_likelihood']['channel_delta'])
    node(id,'v60 SW概率平方损失 '+mode,'同840771总/4515门控密度参数及初始函数；冻结v58骨干，仅原gate/RQS头。原似然或额外0.5*(gate Brier+16*非default条件256格Brier)。共享canonical训练缓存，标签只用训练包，原解码不改。',conclusion,stage='loss',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_rqs_staging_v58_SW_headstart_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','同完整源函数，概率平方监督与似然对照；无新网络参数')
if v60final and v60final['brier_minus_likelihood']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v60_brier']
    end('dead:v60_brier','v60 当前Brier辅助无匹配收益','loss',
        ['pure_dl_sw_brier_v60_SW_brier_fold5'],
        '同参数和完整起点，Brier辅助不优于原似然；完整选择含回退，训练后差单列。',
        '不扫描本256格平方损失系数；仅限当前固定骨干读出，不泛化全部概率评分。')
v61study=root/'results/deep_learning/sw_upsample_v61_20260929'
v61final=json.loads((v61study/'assembled_report.json').read_text(encoding='utf-8')) if (v61study/'assembled_report.json').exists() else None
for mode in ('coarse_query','local_query'):
    id=f'pure_dl_sw_upsample_v61_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v61final:conclusion+='；实际组合TOTAL='+str(v61final['selected_scores']['TOTAL'])+'；局部-粗查询SW='+str(v61final['local_minus_coarse']['channel_delta'])
    node(id,'v61 SW神经深度还原 '+mode,'同842585参数和v56完整初始函数；新增3104参数rank16双邻居插值query/key，query零初始。查询读粗插值或局部TCN，原输出不重置，全网络lr3e-5训练，原Q/Gaussian损失/随机1024上下文。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','完整函数热启动，检验粗到细还原的查询来源，无旧输出投票')
if v61final and v61final['local_minus_coarse']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v61_local_query']
    end('dead:v61_local_query','v61 当前局部查询插值无匹配收益','architecture',
        ['pure_dl_sw_upsample_v61_SW_local_query_fold5'],
        '同参数和完整起点，局部查询不优于粗查询；完整选择含回退，训练后差单列。',
        '不扫描本插值rank/邻居数/温度；仅限当前双邻居结构，不泛化所有神经上采样。')
v62study=root/'results/deep_learning/sw_density_private_v62_20260929'
v62final=json.loads((v62study/'assembled_report.json').read_text(encoding='utf-8')) if (v62study/'assembled_report.json').exists() else None
for mode in ('gate_aux','numeric_only'):
    id=f'pure_dl_sw_density_private_v62_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if (out/'failure.json').exists():conclusion+='；首次并发第8轮CUDA OOM，完整状态存于第7轮；显式串行恢复，源配置不改，失败记录保留。'
    if v62final:conclusion+='；实际组合TOTAL='+str(v62final['selected_scores']['TOTAL'])+'；仅数值-辅助SW='+str(v62final['numeric_minus_aux']['channel_delta'])
    node(id,'v62 SW密度私有骨干 '+mode,'同1678962总/839481优化器参数及v56完整初始函数；固定默认分支，私有密度全骨干lr3e-5训练。密度自己的gate仅训练辅助，系数1或0，0组该129参数无梯度且推理不用；原Q/Gaussian/1024随机上下文。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','完整函数复制并分离默认和密度骨干，检验密度监督干扰；非输出投票')
if v62final and v62final['numeric_minus_aux']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v62_numeric_only']
    end('dead:v62_numeric_only','v62 当前仅数值监督无匹配收益','architecture',
        ['pure_dl_sw_density_private_v62_SW_numeric_only_fold5'],
        '同参数和完整起点，仅数值不优于默认辅助；完整选择含回退，训练后差单列。',
        '不扫描本辅助系数或复制支路数；仅限当前固定默认/私有密度结构，不泛化所有任务解耦。')
v63study=root/'results/deep_learning/por_internal_adapter_v63_20260929'
v63final=json.loads((v63study/'assembled_report.json').read_text(encoding='utf-8')) if (v63study/'assembled_report.json').exists() else None
for mode in ('point','local'):
    id=f'pure_dl_por_internal_adapter_v63_POR_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终POR={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v63final:conclusion+='；实际组合TOTAL='+str(v63final['selected_scores']['TOTAL'])+'；local-point POR='+str(v63final['local_minus_point_selected']['channel_delta'])
    node(id,'v63 POR内部适配器 '+mode,'双专家18个rank8适配器共39888新参数，up零初始，单块新增界0.1；原模型全冻结。point将三核系数用于同点，local读邻点，参数和初始函数相同。锚点路由保留输入梯度但参数不动，原POR/Q50/256上下文，串行。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_mechanism_v31_POR_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留v31 anchored完整POR预测器，只增内部特征适配器；非新输出头')
if v63final and v63final['local_minus_point_selected']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v63_local_adapter']
    end('dead:v63_local_adapter','v63 当前局部内部适配无匹配收益','architecture',
        ['pure_dl_por_internal_adapter_v63_POR_local_fold5'],
        '同参数和初始函数，局部不优于逐点；完整选择含回退，训练后差单列。',
        '不扫描本rank/核宽度/单块界限；仅限当前插入方式，不泛化所有适配器。')
v64study=root/'results/deep_learning/por_interval_v64_20260929'
v64final=json.loads((v64study/'assembled_report.json').read_text(encoding='utf-8')) if (v64study/'assembled_report.json').exists() else None
id='pure_dl_por_interval_v64_POR_interval_fold5';out=root/'outputs'/id
if out.exists():
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终POR={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v64final:conclusion+='；实际组合TOTAL='+str(v64final['selected_scores']['TOTAL'])+'；interval-control POR='+str(v64final['interval_minus_control_selected']['channel_delta'])
    node(id,'v64 POR容差区间质量监督','复用已完成v63 point历史对照；同39888适配器参数和初始函数，仅训练损失加入0.5非default真实POR±8percent区间条件质量负对数。原log1p/t3CDF/Q50/数据/解码不变。代理区间覆盖不等于精确三角效用。',conclusion,stage='loss',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_por_internal_adapter_v63_POR_point_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','历史同配置对照，仅改变训练辅助；均从v31函数和相同新参数初始化，不加载v63训练后权重')
if v64final and v64final['interval_minus_control_selected']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v64_interval']
    end('dead:v64_interval','v64 当前容差质量辅助无匹配收益','loss',[id],
        '同参数和初始函数，区间质量辅助不优于历史似然对照；含起点和训练后差分开报告。',
        '不扫描当前区间宽度或辅助系数；仅限此POR适配器与目标，不泛化全部效用监督。')
v65study=root/'results/deep_learning/sw_transition_v65_20260929'
v65final=json.loads((v65study/'assembled_report.json').read_text(encoding='utf-8')) if (v65study/'assembled_report.json').exists() else None
for mode in ('detached','joint'):
    id=f'pure_dl_sw_transition_v65_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v65final:conclusion+='；实际组合TOTAL='+str(v65final['selected_scores']['TOTAL'])+'；joint-detached SW='+str(v65final['joint_minus_detached']['channel_delta'])
    node(id,'v65 SW相邻状态辅助 '+mode,'同v56完整预测器及8257参数辅助头，相邻中心有效训练点的隐藏均值/绝对差预测default状态XOR；原主损失+0.1平衡Q加权BCE/3。仅差辅助反传是否到骨干，主/辅独立clip2。辅助不参与推理，不输入标签。',conclusion,stage='loss',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','同完整SW函数，仅增训练时状态边界辅助；原解码保持')
if v65final and v65final['joint_minus_detached']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v65_transition']
    end('dead:v65_transition','v65 当前相邻状态辅助无配对收益','loss',
        ['pure_dl_sw_transition_v65_SW_joint_fold5'],'同参数和初始函数，联合辅助未优于detach；含回退和训练后差分开报告。',
        '停止本状态差辅助系数/头宽度扫描；仅限此辅助任务，不泛化全部多任务学习。')
v66study=root/'results/deep_learning/sw_well_order_v66_20260929'
v66final=json.loads((v66study/'assembled_report.json').read_text(encoding='utf-8')) if (v66study/'assembled_report.json').exists() else None
for mode in ('permuted','ordered'):
    id=f'pure_dl_sw_well_order_v66_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v66final:conclusion+='；实际组合TOTAL='+str(v66final['selected_scores']['TOTAL'])+'；ordered-permuted SW='+str(v66final['ordered_minus_permuted']['channel_delta'])
    node(id,'v66 SW整井层序 '+mode,'原v56全冻结，只训85216参数整井FiLM残差；128点原隐表示按深度或固定打乱，Conv64→32/kernel3、均值/标准差、零初始输出1216，增量界0.1。相同点集合/参数/初始函数，原主损失和解码保持，无标签输入。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留原整井统计条件与完整SW函数，仅增层序条件支路')
if v66final and v66final['ordered_minus_permuted']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v66_well_order']
    end('dead:v66_well_order','v66 当前整井层序无配对收益','architecture',
        ['pure_dl_sw_well_order_v66_SW_ordered_fold5'],'同参数和初始函数，深度次序未优于固定打乱；含回退和训练后差分开报告。',
        '停止本整井卷积核宽度/残差幅度扫描；限当前128点稀疏编码，不泛化全部层序信息。')
v67study=root/'results/deep_learning/perm_quantile_v67_20260929'
v67final=json.loads((v67study/'assembled_report.json').read_text(encoding='utf-8')) if (v67study/'assembled_report.json').exists() else None
for mode in ('linear','rqs'):
    id=f'pure_dl_perm_quantile_v67_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v67final:conclusion+='；实际组合TOTAL='+str(v67final['selected_scores']['TOTAL'])+'；rqs-linear PERM='+str(v67final['rqs_minus_linear']['channel_delta'])
    node(id,'v67 PERM分位密度 '+mode,'同838211参数及初始张量，v50 PERM来源骨干/FiLM/默认门控，重置33分位密度；线性或平滑RQS插值/对应正确密度。初始函数可能不同，训练侧PERM先验，原数据/Q/评分/80-18保持。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_perm_transfer_v50_PERM_sw_transfer_fold5','加载已保留同训练折PERM骨干及默认门控；连续密度重置'),('pure_dl_quantile_density_v48_SW_rqs_fold5','复用有局部消融收益的平滑CDF机制与源码，不加载SW RQS权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v67final and v67final['rqs_minus_linear']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v67_perm_rqs']
    end('dead:v67_perm_rqs','v67 当前PERM平滑CDF无配对收益','architecture',
        ['pure_dl_perm_quantile_v67_PERM_rqs_fold5'],'同参数初始张量，平滑密度未优于线性；初始函数差与最终差分开报告。',
        '停止本PERM分位数/尾质量扫描；只限此配置，不泛化其他目标上的密度机制。')
v68study=root/'results/deep_learning/perm_gradient_v68_20260929'
v68final=json.loads((v68study/'assembled_report.json').read_text(encoding='utf-8')) if (v68study/'assembled_report.json').exists() else None
for mode in ('sum','project'):
    id=f'pure_dl_perm_gradient_v68_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v68final:conclusion+='；实际组合TOTAL='+str(v68final['selected_scores']['TOTAL'])+'；project-sum PERM='+str(v68final['project_minus_sum']['channel_delta'])
    node(id,'v68 PERM梯度冲突 '+mode,'同838211参数及v67 RQS第24轮完整函数；同两次autograd分别求门控/密度梯度，直接相加或共享参数全局负点积时用原梯度对称投影。私有头不投影，原数据/Q/损失/评分保持，新3e-5优化器。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_perm_quantile_v67_PERM_rqs_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','携带有消融收益的PERM平滑密度，检验门控/密度共享梯度干扰')
if v68final and v68final['project_minus_sum']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v68_projection']
    end('dead:v68_projection','v68 当前PERM梯度投影无配对收益','optimization',
        ['pure_dl_perm_gradient_v68_PERM_project_fold5'],'同完整初始函数，冲突投影未优于直接相加；训练侧冲突存在不代表验证收益。',
        '停止当前投影阈值/幅度扫描；只限当前PERM两损失，不泛化所有梯度方法。')
v69study=root/'results/deep_learning/perm_curvature_v69_20260929'
v69final=json.loads((v69study/'assembled_report.json').read_text(encoding='utf-8')) if (v69study/'assembled_report.json').exists() else None
id='pure_dl_perm_curvature_v69_PERM_curvature_fold5';out=root/'outputs'/id
if out.exists():
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v69final:conclusion+='；实际组合TOTAL='+str(v69final['selected_scores']['TOTAL'])+'；curvature-sum PERM='+str(v69final['curvature_minus_sum']['channel_delta'])
    node(id,'v69 PERM分位间距曲率','同838211参数及v67完整起点，复用v68 sum历史控制；仅加0.05非default训练log分位间距二阶差分平方均值，Q加权/全部点平均/3。原数据/评分/优化保持，不是物理密度曲率精确积分。',conclusion,stage='loss',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_perm_quantile_v67_PERM_rqs_fold5','加载已验证RQS第24轮完整模型，保留正机制'),('pure_dl_perm_gradient_v68_PERM_sum_fold5','复用未改普通更新控制证据，不载入其训练后权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v69final and v69final['curvature_minus_sum']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v69_curvature']
    end('dead:v69_curvature','v69 当前间距曲率无配对收益','loss',[id],
        '同初始完整函数和优化配置，曲率约束未超过已完成控制；含起点和训练后差单列。',
        '不扫描当前正则系数/差分阶数；仅限此PERM分位间距约束，不泛化所有密度正则。')
v70study=root/'results/deep_learning/perm_cdfwarp_v70_20260929'
v70final=json.loads((v70study/'assembled_report.json').read_text(encoding='utf-8')) if (v70study/'assembled_report.json').exists() else None
for mode in ('affine','nonlinear'):
    id=f'pure_dl_perm_cdfwarp_v70_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v70final:conclusion+='；实际组合TOTAL='+str(v70final['selected_scores']['TOTAL'])+'；nonlinear-affine PERM='+str(v70final['nonlinear_minus_affine_selected']['channel_delta'])
    node(id,'v70 PERM条件CDF校准 '+mode,'冻结当前v29 coupled全预测器，10306参数319→32→2零输出头，中间恒等/GELU；G(F)=F+uF(1-F)+vF(1-F)(1-2F)，u/v界.25、导数≥.5。正确变换似然，原格概率/评分，初始逐位一致。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_split_route_v29_PERM_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','完整加载coupled最佳PERM，再学习条件CDF变换；不修改原分量或门控')
if v70final and v70final['nonlinear_minus_affine_selected']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v70_nonlinear_cdf']
    end('dead:v70_nonlinear_cdf','v70 当前非线性CDF头无配对收益','architecture',
        ['pure_dl_perm_cdfwarp_v70_PERM_nonlinear_fold5'],'同参数和完整起点，GELU未优于仿射；含回退和训练后差分开。',
        '不扫本头宽度/激活；若两者均无绝对收益，不扫当前多项式阶数/幅度；不泛化所有校准。')
v71study=root/'results/deep_learning/sw_depth_attention_v71_20260929'
v71final=json.loads((v71study/'assembled_report.json').read_text(encoding='utf-8')) if (v71study/'assembled_report.json').exists() else None
failedid='pure_dl_sw_depth_attention_v71_SW_symmetric_fold5'
if (root/'outputs'/failedid/'failure.json').exists():
    node(failedid,'v71 深度偏置原后端失败','初始SW73.713975628494逐位一致，零系数128参数；新增SDPA浮点mask要求梯度。','首次反向CUDA LSE strideH对齐错误，无完成轮；失败源码/config保留，math新实现重新开始。',stage='architecture',status='failed')
    if not any(e['source']=='pure_dl_sw_interaction_v56_SW_additive_fold5' and e['target']==failedid for e in edges):edge('pure_dl_sw_interaction_v56_SW_additive_fold5',failedid,'evolution','载入v56完整SW函数，零偏置初始一致，首步反向后端失败')
for mode in ('symmetric','directional'):
    id=f'pure_dl_sw_depth_attention_v71_math_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v71final:conclusion+='；实际组合TOTAL='+str(v71final['selected_scores']['TOTAL'])+'；directional-symmetric SW='+str(v71final['directional_minus_symmetric']['channel_delta'])
    node(id,'v71 SW物理深度注意力 '+mode,'原v56全冻结，仅128系数；tanh(物理token深度差/10m)取绝对值或保留方向，8个归一化RBF基、偏置界.5。原attention+显式FP32偏置与零偏置attention差，完整初始函数保持，三次计算成本实报。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','完整SW父函数，深度只作平移不变两两差'),(failedid,'修复mask梯度后端错误；新源/新运行，无旧状态恢复')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v71final and v71final['directional_minus_symmetric']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v71_directional']
    end('dead:v71_directional','v71 当前方向偏置无配对收益','architecture',
        ['pure_dl_sw_depth_attention_v71_math_SW_directional_fold5'],'同128参数和完整起点，带方向关系未优于绝对距离；后端失败和有效训练分开。',
        '不扫尺度/RBF数；两者都无绝对收益则不扫幅度，仅限当前粗token关系，不泛化所有位置编码。')
v72study=root/'results/deep_learning/perm_observation_v72_20260929'
v72final=json.loads((v72study/'assembled_report.json').read_text(encoding='utf-8')) if (v72study/'assembled_report.json').exists() else None
for mode in ('point','interval'):
    id=f'pure_dl_perm_observation_v72_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终PERM={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v72final:conclusion+='；实际组合TOTAL='+str(v72final['selected_scores']['TOTAL'])+'；interval-point PERM='+str(v72final['interval_minus_point']['channel_delta'])
    node(id,'v72 PERM观测分辨率 '+mode,'同v67 RQS第24轮完整838211参数/初始函数，只变归一化训练标签附近平均密度半宽0/.05，五点Gauss-Legendre和logsumexp近似。原默认BCE/数据/Q50/解码不变，完整网络新3e-5训练；非真实测量精度或精确积分。',conclusion,stage='loss',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_perm_quantile_v67_PERM_rqs_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留正消融RQS完整起点，仅改变训练观测分辨率')
if v72final and v72final['interval_minus_point']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v72_observation']
    end('dead:v72_observation','v72 当前窄区间似然无配对收益','loss',
        ['pure_dl_perm_observation_v72_PERM_interval_fold5'],'同完整起点，窄区间平均密度未优于点似然；含回退和训练后差分开报告。',
        '停止当前区间宽度/积分点数扫描；仅限此PERM RQS和固定尺度，不泛化所有标签噪声建模。')
v73study=root/'results/deep_learning/sw_patch_moments_v73_20260929'
v73final=json.loads((v73study/'assembled_report.json').read_text(encoding='utf-8')) if (v73study/'assembled_report.json').exists() else None
for mode in ('permuted','ordered'):
    id=f'pure_dl_sw_patch_moments_v73_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v73final:conclusion+='；实际组合TOTAL='+str(v73final['selected_scores']['TOTAL'])+'；ordered-permuted SW='+str(v73final['ordered_minus_permuted']['channel_delta'])
    node(id,'v73 SW块内有序矩 '+mode,'同857913参数，v56骨干/FiLM迁移，输出重置训练先验；16点隐藏的一/二阶位置矩拼接192经零Linear→96补token，原均值保留。只差新矩读取固定排列或深度顺序，原局部/位置不改，完整网络新1e-4训练。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留有收益骨干结构，新增token内形态摘要；旧输出重置，不声称保留父函数')
if v73final and v73final['ordered_minus_permuted']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v73_patch_moments']
    end('dead:v73_patch_moments','v73 当前有序矩无配对收益','architecture',
        ['pure_dl_sw_patch_moments_v73_SW_ordered_fold5'],'同完整初始张量和先验，有序矩不优于固定打乱矩；绝对冠军差另列。',
        '停止本矩阶数/patch大小扫描；只限该token摘要，不泛化全部顺序结构。')
v74study=root/'results/deep_learning/sw_key_dropout_v74_20260929'
v74final=json.loads((v74study/'assembled_report.json').read_text(encoding='utf-8')) if (v74study/'assembled_report.json').exists() else None
for mode in ('control','dropkey'):
    id=f'pure_dl_sw_key_dropout_v74_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v74final:
        contrast=v74final['dropkey_minus_control']
        conclusion+='；实际组合TOTAL='+str(v74final['selected_scores']['TOTAL'])+'；dropkey-control SW='+str(contrast['channel_delta'])+'；默认差'+str(contrast['populations']['default']['channel_delta'])+'/数值差'+str(contrast['populations']['numeric']['channel_delta'])+'；保留'+str(v74final['retained'])
    node(id,'v74 SW长程键随机失活 '+mode,'同839481参数，v56骨干/FiLM迁移，输出重置训练先验；训练4层attention每样本丢0/10%key，所有头/查询共mask，全丢恢复原mask，两组相同抽样消耗。验证无丢弃，完整网络新1e-4训练。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_interaction_v56_SW_additive_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','原推理结构完整保留；训练期注意力正则对照，旧输出重置')
if v74final and v74final['dropkey_minus_control']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v74_dropkey']
    end('dead:v74_dropkey','v74 当前长程键失活无配对收益','optimization',
        ['pure_dl_sw_key_dropout_v74_SW_dropkey_fold5'],'同完整初始张量和先验，训练key失活未优于控制；绝对冠军差另列。',
        '停止当前失活概率/施加层数扫描；只限该SW注意力正则，不泛化全部正则化。')
v75study=root/'results/deep_learning/sw_dropout_scope_v75_20260929'
v75final=json.loads((v75study/'assembled_report.json').read_text(encoding='utf-8')) if (v75study/'assembled_report.json').exists() else None
for mode in ('both','gate_only'):
    id=f'pure_dl_sw_dropout_scope_v75_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v75final:
        contrast=v75final['gate_only_minus_both']
        conclusion+='；实际组合TOTAL='+str(v75final['selected_scores']['TOTAL'])+'；gate_only-both SW='+str(contrast['channel_delta'])+'；默认差'+str(contrast['populations']['default']['channel_delta'])+'/数值差'+str(contrast['populations']['numeric']['channel_delta'])+'；保留'+str(v75final['retained'])
    node(id,'v75 SW失活监督范围 '+mode,'同839481参数和v56迁移/重置输出先验，单模型每批两次前向（dropkey/完整），两组门控BCE都用dropkey，密度NLL用dropkey或完整。两图零系数反向保持同遍历，共享骨干仍有梯度耦合，推理单完整前向。原数据/Q/评分，新1e-4。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','加载原骨干，不加载v74训练后权重；输出重置'),('pure_dl_sw_key_dropout_v74_SW_dropkey_fold5','沿用正则机制局部收益，检验默认改善而数值下降的权衡；非权重继承')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v75final and v75final['gate_only_minus_both']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v75_dropout_scope']
    end('dead:v75_dropout_scope','v75 当前门控独用失活无配对收益','optimization',
        ['pure_dl_sw_dropout_scope_v75_SW_gate_only_fold5'],'同初始张量和先验，数值用完整上下文未提高相对同两图控制的最终SW；子群变化另列。',
        '停止本两图密度损失混合比例和监督范围扫描；v74正机制保留，不泛化全部目标解耦。')
v76study=root/'results/deep_learning/sw_dropout_rate_v76_20260930'
v76final=json.loads((v76study/'assembled_report.json').read_text(encoding='utf-8')) if (v76study/'assembled_report.json').exists() else None
for mode in ('p05','p20'):
    id=f'pure_dl_sw_dropout_rate_v76_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v76final:conclusion+='；实际组合TOTAL='+str(v76final['selected_scores']['TOTAL'])+'；最佳新候选-历史10% SW='+str(v76final['best_new_minus_p10']['channel_delta'])+'；保留'+str(v76final['retained'])
    node(id,'v76 SW失活强度 '+mode,'同839481参数和v74单前向全部配置，仅DropKey概率5%/20%；加载v56骨干重置SW先验，非v74最佳权重续训。复用历史10%与0%控制，全点初始和配置核对；原Q/评分/1e-4/80-18。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','原骨干来源，输出重置'),('pure_dl_sw_key_dropout_v74_SW_dropkey_fold5','沿用有局部收益正则机制，复用历史10%控制，不加载其权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v76final and v76final['best_new_minus_p10']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v76_dropout_rate']
    end('dead:v76_dropout_rate','v76 两侧失活强度无新增收益','optimization',
        ['pure_dl_sw_dropout_rate_v76_SW_p05_fold5','pure_dl_sw_dropout_rate_v76_SW_p20_fold5'],'同初始张量和先验，5%/20%均未优于已验证10%；相对零失活及当前冠军另列。',
        '停止当前概率细扫，保留历史10%正候选；不泛化其他正则或来源。')
v77study=root/'results/deep_learning/sw_dropout_extend_v77_20260930'
v77final=json.loads((v77study/'assembled_report.json').read_text(encoding='utf-8')) if (v77study/'assembled_report.json').exists() else None
for mode in ('p30','p40'):
    id=f'pure_dl_sw_dropout_extend_v77_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v77final:
        contrast=v77final['best_new_minus_p20']
        conclusion+='；实际组合TOTAL='+str(v77final['selected_scores']['TOTAL'])+'；最佳新候选-历史20% SW='+str(contrast['channel_delta'])+'；默认差'+str(contrast['populations']['default']['channel_delta'])+'/数值差'+str(contrast['populations']['numeric']['channel_delta'])+'；井胜/负'+str(contrast['well_wins'])+'/'+str(contrast['well_losses'])+'；保留'+str(v77final['retained'])+'；当前独立通道路线的开发折微小提升，非独立泛化确认'
    node(id,'v77 SW失活强度扩展 '+mode,'同839481参数和v76单前向全部配置，仅DropKey概率30%/40%；仍加载v56骨干重置SW先验，非v76权重续训。复用历史20%与0%控制，初始全点及配置核对；原Q/评分/1e-4/80-18。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','原骨干来源，输出重置'),('pure_dl_sw_dropout_rate_v76_SW_p20_fold5','沿用已采用正机制，复用20%历史控制，不加载其训练后权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v77final and v77final['best_new_minus_p20']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v77_dropout_extension']
    end('dead:v77_dropout_extension','v77 向上失活强度无新增收益','optimization',
        ['pure_dl_sw_dropout_extend_v77_SW_p30_fold5','pure_dl_sw_dropout_extend_v77_SW_p40_fold5'],'同初始和先验，30%/40%均未优于20%；原正模型保留。',
        '停止本轮向上扩展和密集概率细扫；不泛化全部正则化或其他初始化。')
v78study=root/'results/deep_learning/sw_dropout_heads_v78_20260930'
v78final=json.loads((v78study/'assembled_report.json').read_text(encoding='utf-8')) if (v78study/'assembled_report.json').exists() else None
for mode in ('shared','per_head'):
    id=f'pure_dl_sw_dropout_heads_v78_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终含起点SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}。"
    if v78final:
        contrast=v78final['per_head_minus_shared']
        conclusion+='；实际组合TOTAL='+str(v78final['selected_scores']['TOTAL'])+'；per_head-shared SW='+str(contrast['channel_delta'])+'；训练后最高差'+str(v78final['post_update_per_head_minus_shared'])+'；保留'+str(v78final['retained'])+'；共同微调与掩码机制收益区分，开发折非独立泛化证明'
    node(id,'v78 SW头间失活掩码 '+mode,'完整v77 p40第25轮同起点，同839481参数，原输出不重置；两组新AdamW3e-5/80-18，每层同BHt随机消耗，只差4头共享或独立40%key掩码。每头全丢回退valid，验证完整；原SW损失/Q50/数据/解码不变。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_dropout_extend_v77_SW_p40_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','加载完整SW函数并重置优化器；公平检验头间随机掩码共享，非旧轨迹续跑')
if v78final and v78final['per_head_minus_shared']['channel_delta']<=0 and v78final['post_update_per_head_minus_shared']<=0:
    dead=[d for d in dead if d['id']!='dead:v78_head_mask']
    end('dead:v78_head_mask','v78 当前独立头失活掩码无收益','optimization',
        ['pure_dl_sw_dropout_heads_v78_SW_per_head_fold5'],'同完整起点、参数和随机消耗，独立掩码含起点及训练后最高均不优于shared；共同微调另记。',
        '停止本头间掩码概率细扫；仅限该起点/学习率/作用方式，保留其他正机制。')
v79study=root/'results/deep_learning/sw_input_products_v79_20260930'
v79final=json.loads((v79study/'assembled_report.json').read_text(encoding='utf-8')) if (v79study/'assembled_report.json').exists() else None
for mode in ('marginal','cross'):
    id=f'pure_dl_sw_input_products_v79_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终含起点SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}；原权重冻结={report['all_original_weights_frozen_exactly']}。"
    if v79final:
        contrast=v79final['cross_minus_marginal']
        conclusion+='；实际组合TOTAL='+str(v79final['selected_scores']['TOTAL'])+'；cross-marginal SW='+str(contrast['channel_delta'])+'；训练后最高差'+str(v79final['post_update_cross_minus_marginal'])+'；保留'+str(v79final['retained'])+'；仅开发折选择，非独立泛化证明'
    node(id,'v79 SW输入幅值交互 '+mode,'完整v77 p40同起点，冻结839481旧参数且eval，仅新增4320参数45→96无偏置零stem适配器，±0.25激活界。九曲线z=tanh(x/3)，单曲线2~6次幂或45个含对角乘积；不新拟合统计。原数据/Q50/评分，新3e-4/80-18。',conclusion,stage='features',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    parent='pure_dl_sw_dropout_extend_v77_SW_p40_fold5'
    if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution','保留完整有效函数，只检验有界输入基适配；非v32局部相关统计输出头或v56隐藏FF乘法')
if v79final and v79final['cross_minus_marginal']['channel_delta']<=0 and v79final['post_update_cross_minus_marginal']<=0:
    dead=[d for d in dead if d['id']!='dead:v79_input_products']
    end('dead:v79_input_products','v79 当前输入乘积基无匹配收益','features',
        ['pure_dl_sw_input_products_v79_SW_cross_fold5'],'同完整起点和4320新参数，显式乘积基不优于单曲线基；含起点及更新后成绩分开。',
        '停止本乘积组合/阶数细扫；若都无绝对收益，也停止此stem幅度/容量细扫，不泛化全部特征工程。')
v80failed='pure_dl_sw_head_utility_v80_SW_likelihood_fold5'
v80out=root/'outputs'/v80failed
if v80out.exists():
    state=json.loads((v80out/'runtime_status.json').read_text(encoding='utf-8')) if (v80out/'runtime_status.json').exists() else {}
    initial=json.loads((v80out/'initial.json').read_text(encoding='utf-8')) if (v80out/'initial.json').exists() else {}
    node(v80failed,'v80 SW效用头首次实现失败','完整v77同起点，冻结骨干仅3225输出参数；原似然/联合效用对照尚未完成。误用仅推理的no_grad区间质量函数，实际辅助密度梯度记录阻止继续。',
        '启动失败，非算法负结果：'+json.dumps(state,ensure_ascii=False)+'；只有源函数初始SW='+str(initial.get('target_score'))+'；未完成任何轮，joint未启动。原源/配置冻结，v80r1另目录修复。',stage='optimization',status='failed')
    if not any(e['source']=='pure_dl_sw_dropout_extend_v77_SW_p40_fold5' and e['target']==v80failed for e in edges):edge('pure_dl_sw_dropout_extend_v77_SW_p40_fold5',v80failed,'evolution','预定冻结有效SW骨干，首次实现失败，不当精度证据')
v80study=root/'results/deep_learning/sw_head_utility_v80r1_20260930'
v80final=json.loads((v80study/'assembled_report.json').read_text(encoding='utf-8')) if (v80study/'assembled_report.json').exists() else None
for mode in ('likelihood','joint'):
    id=f'pure_dl_sw_head_utility_v80r1_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else None
    conclusion='训练快照：'+json.dumps(state,ensure_ascii=False)
    if report:conclusion+=f"；最终含起点SW={report['target_score']:.12f}；最佳{report['best_epoch']}/停止{report['stopped_epoch']}；旧骨干冻结={report['all_backbone_weights_frozen_exactly']}。"
    if v80final:
        contrast=v80final['joint_minus_likelihood']
        conclusion+='；实际组合TOTAL='+str(v80final['selected_scores']['TOTAL'])+'；joint-likelihood SW='+str(contrast['channel_delta'])+'；更新后最高差'+str(v80final['post_update_joint_minus_likelihood'])+'；保留'+str(v80final['retained'])+'；仅开发折选择，非独立泛化证明'
    node(id,'v80r1 SW输出头联合效用 '+mode,'完整v77同起点、839481总参数，仅3225输出线性参数更新。可微8Gaussian区间质量，无限CDF尾常数安全处理，soft动作tau.05、联合收益辅助系数0/.5。固定骨干eval，新3e-5/80-18；原数据/Q50/解码，非重训v12全TCN。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'))
    for parent,detail in [('pure_dl_sw_dropout_extend_v77_SW_p40_fold5','加载完整有效SW函数，固定全部非输出权重'),('pure_dl_independent_sw_v12_joint_utility80_fold5','沿用先前更新后+0.021333的窄局部损失线索，不继承其权重；历史无绝对提升'),(v80failed,'修复no_grad误用，原源码/失败留档，新运行重置，不恢复失败状态')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v80final and v80final['joint_minus_likelihood']['channel_delta']<=0 and v80final['post_update_joint_minus_likelihood']<=0:
    dead=[d for d in dead if d['id']!='dead:v80_head_utility']
    end('dead:v80_head_utility','v80r1 当前联合效用头无匹配收益','optimization',
        ['pure_dl_sw_head_utility_v80r1_SW_joint_fold5'],'同完整起点和3225输出参数，辅助联合效用不优于原似然；含起点与更新后成绩分开。',
        '停止本温度/辅助权重扫描；若都无绝对收益也停止同头拟合LR细扫，不泛化全部效用损失。')
if v80final and not v80final['adopted'] and all(row['report']['best_epoch']==0 for row in v80final['candidates']):
    dead=[d for d in dead if d['id']!='dead:v80_fixed_head_warmstart']
    end('dead:v80_fixed_head_warmstart','v80r1 固定骨干热启动头拟合无绝对改进','optimization',
        ['pure_dl_sw_head_utility_v80r1_SW_likelihood_fold5','pure_dl_sw_head_utility_v80r1_SW_joint_fold5'],
        '两项都恢复原完整函数，训练后最高joint-likelihood='+str(v80final['post_update_joint_minus_likelihood'])+'，微小局部正差与无绝对收益分开，不能当新完整改进。',
        '关闭这组固定头热启动LR/温度/系数密集扫描；不否定其他独立学习场景下的效用线索。')
v81study=root/'results/deep_learning/sw_full_utility_v81_20260930'
v81final=json.loads((v81study/'assembled_report.json').read_text(encoding='utf-8')) if (v81study/'assembled_report.json').exists() else None
for mode in ('likelihood','joint'):
    id=f'pure_dl_sw_full_utility_v81_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    initial=json.loads((out/'initial.json').read_text(encoding='utf-8')) if (out/'initial.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else {}
    conclusion='原第五折SW，启动先验='+str(initial.get('target_score'))+'；训练快照='+json.dumps(report or state,ensure_ascii=False)
    if v81final:
        contrast=v81final['joint_minus_likelihood']
        conclusion+='；实际组合TOTAL='+str(v81final['selected_scores']['TOTAL'])+'；joint-likelihood SW='+str(contrast['channel_delta'])+'；更新后最高差='+str(v81final['post_update_joint_minus_likelihood'])+'；保留='+str(v81final['retained'])+'；仅开发折，不是完整OOF或独立泛化证明'
    shown={}
    if report:
        combined=dict(scope='第五折 · 固定POR/PERM组合',value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],source=f'outputs/{id}/fold_5_report.json',kind='fold',fold=5,epoch=report['best_epoch'])
        channel=dict(scope='第五折 · 单目标SW',value=report['target_score'],scores={'SW':report['target_score']},source=f'outputs/{id}/fold_5_report.json',kind='channel',fold=5,epoch=report['best_epoch'])
        shown=dict(primary=combined,results=[combined,channel],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/fold_5_report.json','results/deep_learning/sw_full_utility_v81_20260930/FINAL.md'])
    else:
        shown['resultText']='第五折SW先验 '+str(initial.get('target_score'))+'；观测最佳 '+str(state.get('best'))+'；尚未恢复验证'
    node(id,'v81 SW完整学习联合效用 '+mode,'同v56训练折内additive骨干/FiLM与重置SW训练先验，不加载v77/v80训练后权重。全839481参数更新，四层40%DropKey保持；两组共同似然及可微联合动作效用图，仅辅助系数0/.5，tau.05。原1e-4/80-18、数据/Q50/评分/解码，POR/PERM固定。',conclusion,highlight=bool(v81final and v81final['selected_mode']==mode),stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running'),**shown)
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','加载训练折内第11轮骨干/FiLM，SW输出重新设为原训练先验'),('pure_dl_sw_dropout_extend_v77_SW_p40_fold5','采用已有效40%DropKey配方和相同初始状态/预算；不继承其训练后权重'),('pure_dl_sw_head_utility_v80r1_SW_joint_fold5','沿用可微有限尾CDF及微小更新后损失线索，不继承权重，不将旧无绝对收益算成新收益')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v81final and v81final['joint_minus_likelihood']['channel_delta']<=0 and v81final['post_update_joint_minus_likelihood']<=0:
    dead=[d for d in dead if d['id']!='dead:v81_full_utility']
    end('dead:v81_full_utility','v81 当前完整DropKey学习联合效用无匹配收益','optimization',
        ['pure_dl_sw_full_utility_v81_SW_joint_fold5'],'同初始函数、839481全可训练参数和预算，系数.5的联合效用未优于系数0；含先验与更新后成绩分别记录。',
        '关闭此v56迁移/重置SW先验、40%DropKey、tau.05和alpha.5的辅助温度/系数扫描；不泛化全部效用学习。共同训练收益不能归给辅助项，历史p40仍为额外绝对对照。')
v82study=root/'results/deep_learning/sw_utility_strength_v82_20260930'
v82final=json.loads((v82study/'assembled_report.json').read_text(encoding='utf-8')) if (v82study/'assembled_report.json').exists() else None
for mode,alpha in (('a025',.25),('a100',1.)):
    id=f'pure_dl_sw_utility_strength_v82_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    initial=json.loads((out/'initial.json').read_text(encoding='utf-8')) if (out/'initial.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else {}
    history=json.loads((out/'validation_history.json').read_text(encoding='utf-8')) if (out/'validation_history.json').exists() else []
    conclusion='原第五折SW先验='+str(initial.get('target_score'))+'；当前快照='+json.dumps(report or state,ensure_ascii=False)
    if v82final:
        row=next(r for r in v82final['candidates'] if r['mode']==mode)
        diagnosis=row['against_alpha05']
        conclusion+='；对已有效.5 SW='+str(diagnosis['channel_delta'])+'；默认/数值差='+str(diagnosis['populations']['default']['channel_delta'])+'/'+str(diagnosis['populations']['numeric']['channel_delta'])+'；井胜/负='+str(diagnosis['well_wins'])+'/'+str(diagnosis['well_losses'])+'；对0 SW='+str(row['against_alpha0']['channel_delta'])+'；实际新组合TOTAL='+str(v82final['selected_scores']['TOTAL'])+'；保留='+str(v82final['retained'])+'；原第五开发折，非完整OOF或独立泛化证明；微小TOTAL改善不代表子群普遍改善'
    shown={}
    if report:
        combined=dict(scope='第五折 · 固定POR/PERM组合',value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],source=f'outputs/{id}/fold_5_report.json',kind='fold',fold=5,epoch=report['best_epoch'])
        channel=dict(scope='第五折 · 单目标SW',value=report['target_score'],scores={'SW':report['target_score']},source=f'outputs/{id}/fold_5_report.json',kind='channel',fold=5,epoch=report['best_epoch'])
        shown=dict(primary=combined,results=[combined,channel],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/fold_5_report.json','results/deep_learning/sw_utility_strength_v82_20260930/README.md'])
    elif history:
        best=max(history,key=lambda r:r['target_score'])
        observed=dict(scope='第五折 · 固定POR/PERM组合 · 观测最佳',value=best['reference_replacement_total'],scores={'SW':best['target_score'],'TOTAL':best['reference_replacement_total']},source=f'outputs/{id}/validation_history.json',kind='observed',fold=5,epoch=best['epoch'],last_epoch=state.get('last_completed_epoch'))
        shown=dict(primary=observed,results=[observed],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/validation_history.json'])
    else:shown['resultText']='第五折SW重置先验 '+str(initial.get('target_score'))+'；尚未完成训练'
    node(id,'v82 SW效用系数 '+str(alpha),'仅系数.25/1对已完成.5；同839481参数/v56训练折内迁移骨干和重置SW先验，不加载v81训练后权重。全SW更新、40%DropKey、共同可微损失图、原.05温度/数据/Q50/评分/1e-4/80-18，POR/PERM固定；显存不足则串行。',conclusion,highlight=bool(v82final and v82final['selected_mode']==mode),stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),**shown)
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','实际加载原训练折内骨干/FiLM，SW先验重置'),('pure_dl_sw_full_utility_v81_SW_joint_fold5','已有效.5作同容量/初始/预算对照；沿用DropKey和损失，不加载其训练后权重'),('pure_dl_sw_full_utility_v81_SW_likelihood_fold5','既有系数0机制对照，比较预测证据，不继承模型权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v82final and not v82final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v82_utility_coefficients']
    end('dead:v82_utility_coefficients','v82 当前效用系数两侧无新增收益','optimization',
        ['pure_dl_sw_utility_strength_v82_SW_a025_fold5','pure_dl_sw_utility_strength_v82_SW_a100_fold5'],
        '同初始函数、容量、40%DropKey和预算，预定.25/1均不超过已有效.5；对0的重复机制增益与新系数收益分开。',
        '结束本原温度/架构的系数网格，不追加临时点；保留v81 .5正模型。不泛化全部效用学习或其他配置。')
v83study=root/'results/deep_learning/sw_population_utility_v83_20260930'
v83final=json.loads((v83study/'assembled_report.json').read_text(encoding='utf-8')) if (v83study/'assembled_report.json').exists() else None
for mode,share in (('balanced',.5),('numeric75',.75)):
    id=f'pure_dl_sw_population_utility_v83_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    initial=json.loads((out/'initial.json').read_text(encoding='utf-8')) if (out/'initial.json').exists() else {}
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8')) if (out/'config.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else {}
    history=json.loads((out/'validation_history.json').read_text(encoding='utf-8')) if (out/'validation_history.json').exists() else []
    conclusion='原第五SW先验='+str(initial.get('target_score'))+'；训练人口权重='+json.dumps(cfg.get('utility_population',{}),ensure_ascii=False)+'；快照='+json.dumps(report or state,ensure_ascii=False)
    if v83final:
        row=next(r for r in v83final['candidates'] if r['mode']==mode);d=row['against_global']
        conclusion+='；对全人口1 SW='+str(d['channel_delta'])+'；默认/数值差='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；井胜/负='+str(d['well_wins'])+'/'+str(d['well_losses'])+'；实际组合TOTAL='+str(v83final['selected_scores']['TOTAL'])+'；仅开发折，非完整OOF或独立泛化证明'
    shown={}
    if report:
        combined=dict(scope='第五折 · 固定POR/PERM组合',value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],source=f'outputs/{id}/fold_5_report.json',kind='fold',fold=5,epoch=report['best_epoch'])
        channel=dict(scope='第五折 · 单目标SW',value=report['target_score'],scores={'SW':report['target_score']},source=f'outputs/{id}/fold_5_report.json',kind='channel',fold=5,epoch=report['best_epoch'])
        shown=dict(primary=combined,results=[combined,channel],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/fold_5_report.json','results/deep_learning/sw_population_utility_v83_20260930/README.md'])
    elif history:
        best=max(history,key=lambda r:r['target_score'])
        observed=dict(scope='第五折 · 固定POR/PERM组合 · 观测最佳',value=best['reference_replacement_total'],scores={'SW':best['target_score'],'TOTAL':best['reference_replacement_total']},source=f'outputs/{id}/validation_history.json',kind='observed',fold=5,epoch=best['epoch'],last_epoch=state.get('last_completed_epoch'))
        shown=dict(primary=observed,results=[observed],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/validation_history.json'])
    else:shown['resultText']='第五折SW重置先验 '+str(initial.get('target_score'))+'；尚未完成训练'
    node(id,'v83 SW数值效用份额 '+str(share),'同839481参数/v56训练折内骨干与SW先验重置，全SW更新、40%DropKey/原输入/Q50/评分/1e-4/80-18。仅效用人口份额.5/.75，比例只用64训练井拟合，人口平均权重1；BCE/NLL不加人口权重，系数1/温度.05不变。固定POR/PERM，推理无人口选择。',conclusion,highlight=bool(v83final and v83final['selected_mode']==mode),stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),**shown)
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','实际加载原训练折内骨干/FiLM，SW输出重新设为训练先验'),('pure_dl_sw_utility_strength_v82_SW_a100_fold5','沿用系数1/DropKey配方并作全人口对照，不加载其训练后权重；数值退步只作机制假设线索'),('pure_dl_sw_full_utility_v81_SW_joint_fold5','旧.5改善默认及数值，作人口权重边界诊断，不继承模型权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v83final and not v83final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v83_utility_population']
    end('dead:v83_utility_population','v83 当前效用人口重权无总体收益','optimization',
        ['pure_dl_sw_population_utility_v83_SW_balanced_fold5','pure_dl_sw_population_utility_v83_SW_numeric75_fold5'],
        '同初始化/容量/预算，全人口系数1对照下，数值份额.5/.75均无总体收益；默认、数值及实际停止轮分别保留。',
        '关闭当前两个训练人口份额，不临时改权重或延长预算凑分；保留原正模型并换结构。此结论只覆盖本原温度/联合代理/架构，不泛化所有不平衡学习。')
v84failed='pure_dl_sw_frozen_gate_utility_v84_SW_likelihood_fold5'
v84out=root/'outputs'/v84failed
if v84out.exists():
    state=json.loads((v84out/'runtime_status.json').read_text(encoding='utf-8')) if (v84out/'runtime_status.json').exists() else {}
    node(v84failed,'v84 私有密度接口初始化失败','当前v82完整函数复制为冻结默认和私有密度；评估接口遗漏length，在初始评估前失败。',
        '启动失败，未完成初始评分或任何训练轮，不是算法负结果；原源码/配置/日志冻结，v84r1新目录只补length1024/c2。'+json.dumps(state,ensure_ascii=False),stage='architecture',status='failed',resultText='初始化失败 · 无训练成绩')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',v84failed,'evolution','计划以保留完整SW函数为同起点，接口失败，不计精度证据')
v84study=root/'results/deep_learning/sw_frozen_gate_utility_v84r1_20260930'
v84final=json.loads((v84study/'assembled_report.json').read_text(encoding='utf-8')) if (v84study/'assembled_report.json').exists() else None
for mode in ('likelihood','balanced'):
    id=f'pure_dl_sw_frozen_gate_utility_v84r1_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8')) if (out/'runtime_status.json').exists() else {}
    initial=json.loads((out/'initial.json').read_text(encoding='utf-8')) if (out/'initial.json').exists() else {}
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else {}
    history=json.loads((out/'validation_history.json').read_text(encoding='utf-8')) if (out/'validation_history.json').exists() else []
    conclusion='原第五SW完整起点='+str(initial.get('target_score'))+'；实际快照='+json.dumps(report or state,ensure_ascii=False)
    if v84final:
        row=next(r for r in v84final['candidates'] if r['mode']==mode)
        conclusion+='；训练后最高SW='+str(row['best_after_update']['target_score'])+'；匹配均衡-似然含起点差='+str(v84final['matched_loss_comparison']['channel_delta'])+'；更新后最高差='+str(v84final['post_update_balanced_minus_likelihood'])+'；实际组合TOTAL='+str(v84final['selected_scores']['TOTAL'])+'；第18轮早停/恢复起点不是跑满80；仅开发折，非完整OOF'
    shown={}
    if report:
        scope='第五折 · 固定POR/PERM组合'+(' · 恢复起点' if report['best_epoch']==0 else '')
        combined=dict(scope=scope,value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],source=f'outputs/{id}/fold_5_report.json',kind='fold',fold=5,epoch=report['best_epoch'])
        channel=dict(scope='第五折 · 单目标SW',value=report['target_score'],scores={'SW':report['target_score']},source=f'outputs/{id}/fold_5_report.json',kind='channel',fold=5,epoch=report['best_epoch'])
        shown=dict(primary=combined,results=[combined,channel],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/fold_5_report.json','results/deep_learning/sw_frozen_gate_utility_v84r1_20260930/FINAL.md'])
    elif initial:
        shown['resultText']='第五折完整起点SW '+str(initial['target_score'])+'；当前观测 '+str(state.get('best'))+'；尚未恢复核验'
    else:shown['resultText']='初始化中 · 无训练成绩'
    node(id,'v84r1 SW固定默认/私有密度 '+mode,'同当前v82第25轮完整SW函数的两份复制，无输出重置。冻结默认eval/no_grad，私有骨干/FiLM/密度更新；1678962总/839481优化器参数，129个私有gate参数不用。两项同损失图只差均衡效用0/1，训练64井人口拟合，原输入/Q50/评分/私有40%DropKey、新3e-5/80-18。',conclusion,highlight=bool(v84final and v84final['selected_mode']==mode),stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),**shown)
    for parent,detail in [('pure_dl_sw_utility_strength_v82_SW_a100_fold5','实际加载当前完整函数，固定默认和私有密度都同来源'),('pure_dl_sw_population_utility_v83_SW_balanced_fold5','数值改善但默认退步，仅作结构假设线索；不加载已删除的负权重'),('pure_dl_sw_density_private_v62_SW_numeric_only_fold5','沿用旧分离结构思路，不继承该旧负权重；来源和损失条件不同'),(v84failed,'评估接口修复，新目录，不修改失败源码或恢复失败训练')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v84final and not v84final['adopted'] and all(r['report']['best_epoch']==0 for r in v84final['candidates']):
    dead=[d for d in dead if d['id']!='dead:v84_private_density']
    end('dead:v84_private_density','v84r1 当前固定默认/私有密度无新增收益','architecture',
        ['pure_dl_sw_frozen_gate_utility_v84r1_SW_likelihood_fold5','pure_dl_sw_frozen_gate_utility_v84r1_SW_balanced_fold5'],
        '两项均18轮早停恢复原函数；真实固定网络当前/最佳hash不变。更新后最高均衡-似然='+str(v84final['post_update_balanced_minus_likelihood'])+'，不将恢复初始分数算新收益。',
        '关闭本v82来源/冻结默认/私有密度/原似然或均衡效用/3e-5组合，不继续扫LR或人口份额；下一步改输出结构。只覆盖实际18轮早停证据，不泛化所有分离模型。')
v85study=root/'results/deep_learning/sw_residual_readout_v85_20260930'
v85final=json.loads((v85study/'assembled_report.json').read_text(encoding='utf-8')) if (v85study/'assembled_report.json').exists() else None
for mode in ('point64','local64'):
    id=f'pure_dl_sw_residual_readout_v85_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read83=lambda name:json.loads((out/name).read_text(encoding='utf-8')) if (out/name).exists() else {}
    state=read83('runtime_status.json');initial=read83('initial.json');cfg=read83('config.json');report=read83('fold_5_report.json')
    history=read83('validation_history.json') or []
    conclusion='原第五SW先验='+str(initial.get('target_score'))+'；新增参数='+str(cfg.get('added_readout_parameters'))+'；快照='+json.dumps(report or state,ensure_ascii=False)
    if v85final:
        row=next(r for r in v85final['candidates'] if r['mode']==mode);d=row['against_linear']
        conclusion+='；对线性SW='+str(d['channel_delta'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；local-point SW='+str(v85final['local_minus_point']['channel_delta'])+'；实际组合TOTAL='+str(v85final['selected_scores']['TOTAL'])+'；容量差256，近容量，不能完全归因邻域；仅开发折，非完整OOF'
    shown={}
    if report:
        r=dict(scope='第五折 · 固定POR/PERM组合',value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],source=f'outputs/{id}/fold_5_report.json',kind='fold',fold=5,epoch=report['best_epoch'])
        shown=dict(primary=r,results=[r],evidence=[f'outputs/{id}/README.md',f'outputs/{id}/config.json',f'outputs/{id}/fold_5_report.json','results/deep_learning/sw_residual_readout_v85_20260930/README.md'])
    elif history:
        b=max(history,key=lambda r:r['target_score'])
        r=dict(scope='第五折 · 固定POR/PERM组合 · 观测最佳',value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},source=f'outputs/{id}/validation_history.json',kind='observed',fold=5,epoch=b['epoch'],last_epoch=state.get('last_completed_epoch'))
        shown=dict(primary=r,results=[r])
    else:shown['resultText']='第五折重置先验SW '+str(initial.get('target_score'))+'；尚无训练成绩'
    node(id,'v85 SW残差输出 '+mode,'原128→25线性输出加MLP64残差；point新增9881/local新增10137，local加3点depthwise卷积初始中心恒等，投影0和原valid遮罩。CPU fork_rng保护原随机态。相同v56骨干/训练先验重置，全SW训练、全人口效用1/40%DropKey/原输入/Q50/评分/1e-4/80-18，不叠人口重权或固定默认。',conclusion,highlight=bool(v85final and v85final['selected_mode']==mode),stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),**shown)
    for parent,detail in [('pure_dl_sw_interaction_v56_SW_additive_fold5','实际加载相同训练折内骨干/FiLM，输出设为原训练先验'),('pure_dl_sw_utility_strength_v82_SW_a100_fold5','保留已有效全人口效用1/40%DropKey与原线性头同预算对照，不加载其训练后权重'),('pure_dl_sw_frozen_gate_utility_v84r1_SW_balanced_fold5','关闭固定默认/私有密度路线后换读出结构；不加载负权重')]:
        if not any(e['source']==parent and e['target']==id for e in edges):edge(parent,id,'evolution',detail)
if v85final and not v85final['adopted'] and v85final['local_minus_point']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v85_residual_readout']
    end('dead:v85_residual_readout','v85 当前两个小残差读出无收益','architecture',
        ['pure_dl_sw_residual_readout_v85_SW_point64_fold5','pure_dl_sw_residual_readout_v85_SW_local64_fold5'],
        '两项都不超历史线性当前函数，近容量local-point='+str(v85final['local_minus_point']['channel_delta'])+'；初始基本状态/预测匹配，实际停止轮与80上限区分。',
        '关闭当前MLP64和3点局部残差读出，不无界扩头或临时扫参数；只覆盖本来源/损失/初始化/预算，不泛化所有非线性输出头。')
v86id='pure_dl_por_joint_utility_v86_POR_fold5';v86out=root/'outputs'/v86id
v86study=root/'results/deep_learning/por_joint_utility_v86_20260930'
if v86out.exists():
    read86=lambda p:json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    state=read86(v86out/'runtime_status.json');report=read86(v86out/'fold_5_report.json')
    final=read86(v86study/'assembled_report.json');results=[]
    for mode in ('likelihood','joint'):
        arm=read86(v86out/mode/'fold_5_report.json')
        if arm:
            results.append(dict(value=arm['reference_replacement_total'],scores={'POR':arm['target_score'],'TOTAL':arm['reference_replacement_total']},scope='第五折 · 固定PERM/SW · '+mode+(' · 恢复起点' if arm['best_epoch']==0 else ''),kind='fold',fold=5,epoch=arm['best_epoch'],source='outputs/'+v86id+'/'+mode+'/fold_5_report.json'))
        elif state.get('modes',{}).get(mode):
            snap=state['modes'][mode]
            results.append(dict(value=80.38209329121432+.3*(snap['best']-77.551936801885),scores={'POR':snap['best']},scope='第五折 · 固定PERM/SW · '+mode+' · 观测快照',kind='observed',fold=5,epoch=snap['best_epoch'],source='outputs/'+v86id+'/runtime_status.json'))
    conclusion='当前快照='+json.dumps(state,ensure_ascii=False)+'；初始POR77.551936801885/组合80.382093291214，未完成不宣称收益。'
    if final:
        conclusion='joint-likelihood POR='+str(final['joint_minus_likelihood']['channel_delta'])+'；训练后最高差='+str(final['post_update_joint_minus_likelihood'])+'；实际组合='+str(final['selected_scores']['TOTAL'])+'；采用='+str(final['selected_mode'])+'；epoch0不是新收益；停止轮按实际，不写跑满80；仅开发折非完整OOF'
    shown=dict(results=results,primary=max(results,key=lambda r:r['value'])) if results else dict(resultText='初始化中 · 尚无新成绩')
    node(v86id,'v86 POR似然/联合动作效用','同v31 anchored完整函数/30819既有路由；原专家/骨干/位置头固定，无新增容量。只有效用系数0/1不同；原t3/256支持800动作、0.08容差、log1p/expm1、Q50/数据/80-18。一个进程共享冻结计算，两头独立更新/选模；新3e-5/cos到1e-5。',conclusion,highlight=bool(final and final['adopted']),stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),evidence=['outputs/'+v86id+'/config.json','results/deep_learning/por_joint_utility_v86_20260930/README.md'],**shown)
    edge('pure_dl_mechanism_v31_POR_fold5',v86id,'evolution','实际加载anchored第7轮完整路由与冻结依赖，原POR函数保持')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',v86id,'evolution','SW效用有收益，迁移训练目标思路；不加载SW权重或输入SW预测')
    if final and not final['adopted'] and final['joint_minus_likelihood']['channel_delta']<=0:
        dead=[d for d in dead if d['id']!='dead:v86_por_route_utility']
        end('dead:v86_por_route_utility','v86 当前POR路由效用未取得收益','optimization',[v86id],
            '真实两项完成，最终和更新后差分开记录；起点恢复不是新收益。',
            '关闭当前冻结专家/anchored路由/系数1/温度.05/3e-5组合，不泛化其他POR分布训练或全骨干效用。')
v87study=root/'results/deep_learning/perm_full_utility_v87_20260930'
v87final=json.loads((v87study/'assembled_report.json').read_text(encoding='utf-8')) if (v87study/'assembled_report.json').exists() else None
for mode in ('likelihood','joint'):
    id=f'pure_dl_perm_full_utility_v87_PERM_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read87=lambda name:json.loads((out/name).read_text(encoding='utf-8')) if (out/name).exists() else {}
    state=read87('runtime_status.json');report=read87('fold_5_report.json');initial=read87('initial.json');cfg=read87('config.json')
    history=read87('validation_history.json') or [];shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],scope='第五折 · 固定POR/SW · '+mode+(' · 恢复起点' if report['best_epoch']==0 else ''),kind='fold',fold=5,epoch=report['best_epoch'],source='outputs/'+id+'/fold_5_report.json')
        shown=dict(primary=row,results=[row])
    elif history:
        b=max(history,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'PERM':b['target_score'],'TOTAL':b['reference_replacement_total']},scope='第五折 · 固定POR/SW · 观测快照',kind='observed',fold=5,epoch=b['epoch'],source='outputs/'+id+'/validation_history.json')
        shown=dict(primary=row,results=[row])
    else:shown['resultText']='完整起点PERM '+str(initial.get('target_score'))+'；初始化或训练中，尚无新成绩'
    conclusion='快照='+json.dumps(report or state,ensure_ascii=False)+'；未完成不宣称收益。'
    if v87final:
        r=next(r for r in v87final['candidates'] if r['mode']==mode)
        conclusion='对当前PERM='+str(r['against_parent']['channel_delta'])+'；joint-likelihood='+str(v87final['joint_minus_likelihood']['channel_delta'])+'；更新后差='+str(v87final['post_update_joint_minus_likelihood'])+'；默认/数值='+str(r['against_parent']['populations']['default']['channel_delta'])+'/'+str(r['against_parent']['populations']['numeric']['channel_delta'])+'；同轮轨迹='+json.dumps(v87final.get('trajectory_diagnostic',{}),ensure_ascii=False)+'；实际采用三列='+str(v87final['selected_scores']['TOTAL'])+'；epoch0不是新收益；原第五开发折非完整OOF'
    node(id,'v87 PERM完整网络效用 '+mode,'同当前v29 coupled完整PERM函数/1324879学习参数，无新增容量。解除旧no_grad/detach，骨干/内部路由/读出及已有效修正更新；packed其他目标行仍是buffer，仅PERM标签。原Gaussian/log10效用/256支持800动作/Q50/数据不变，系数0/1，body1e-5/head3e-5/cos到1e-5/80-18。',conclusion,highlight=bool(v87final and v87final['selected_mode']==mode),stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),evidence=['outputs/'+id+'/config.json','results/deep_learning/perm_full_utility_v87_20260930/README.md'],**shown)
    edge('pure_dl_split_route_v29_PERM_fold5',id,'evolution','实际加载PERM coupled第2轮路由及v8/v26/v28依赖，原完整函数同起点')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',id,'evolution','迁移全网络效用有收益的训练目标思路，不加载SW权重或其他目标预测')
if v87final and not v87final['adopted'] and v87final['joint_minus_likelihood']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v87_perm_full_utility']
    end('dead:v87_perm_full_utility','v87 当前PERM完整微调未改善原模型','optimization',
        ['pure_dl_perm_full_utility_v87_PERM_likelihood_fold5','pure_dl_perm_full_utility_v87_PERM_joint_fold5'],
        '完整两项同起点、原数据/Q50/官方解码，均恢复原函数；训练后最高效用小正诊断与同轮6升/12降分开，不写成采用收益。',
        '关闭本warm-start/全网络/低LR/系数1/温度.05组合，不继续临时扫效用或低LR，不泛化全部PERM优化。')
v88id='pure_dl_perm_grid_readout_v88_PERM_fold5';v88out=root/'outputs'/v88id
v88study=root/'results/deep_learning/perm_grid_readout_v88_20260930'
if v88out.exists():
    read88=lambda p:json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    state=read88(v88out/'runtime_status.json');group=read88(v88out/'fold_5_report.json');final=read88(v88study/'assembled_report.json');results=[]
    for mode in ('likelihood','joint'):
        arm=read88(v88out/mode/'fold_5_report.json')
        if arm:
            results.append(dict(value=arm['reference_replacement_total'],scores={'PERM':arm['target_score'],'TOTAL':arm['reference_replacement_total']},kind='fold',fold=5,epoch=arm['best_epoch'],scope='第五折 · 固定POR/SW · '+mode+(' · 恢复起点' if arm['best_epoch']==0 else ''),source='outputs/'+v88id+'/'+mode+'/fold_5_report.json'))
        elif state.get('modes',{}).get(mode):
            q=state['modes'][mode];results.append(dict(value=80.38209329121432+.35*(q['best']-89.39672404936582),scores={'PERM':q['best']},kind='observed',fold=5,epoch=q['best_epoch'],scope='第五折 · 固定POR/SW · '+mode+'快照',source='outputs/'+v88id+'/runtime_status.json'))
    shown=dict(results=results,primary=max(results,key=lambda r:r['value'])) if results else dict(resultText='初始化中 · 无新成绩')
    conclusion='当前快照='+json.dumps(state,ensure_ascii=False)+'；未完成不宣称收益。'
    if final:
        conclusion='实际采用TOTAL='+str(final['selected_scores']['TOTAL'])+'；最终joint-likelihood='+str(final['joint_minus_likelihood']['channel_delta'])+'；完整训练后最佳匹配差='+str(final['post_update_joint_minus_likelihood'])+'；匹配增益有完整head/pred重载依赖，不与采用增益混同；仅第五开发折非完整OOF'
    node(v88id,'v88 PERM有界概率格读出','冻结当前PERM v29完整函数，仅37185参数MLP64预测256格质量与默认增量；质量log界.5、gate界.25，零输出/保持prior总质量确保初始原函数。原9曲线206+已有幅值63/训练侧拟合/Q50/评分不改。两项同容量同初始化，格软CE/效用0或1，新1e-4/80-18；一个GPU进程共享冻结计算；先验0质量格不能新增质量。',conclusion,stage='architecture',status='recorded' if group else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(final and final['adopted']),evidence=['results/deep_learning/perm_grid_readout_v88_20260930/README.md','outputs/'+v88id+'/config.json'],**shown)
    edge('pure_dl_split_route_v29_PERM_fold5',v88id,'evolution','实际加载当前完整私有PERM作为冻结先验，非其他模型输出投票')
    edge('pure_dl_perm_full_utility_v87_PERM_joint_fold5',v88id,'evolution','完整微调无采用收益，换概率表示与监督；不加载已删除负权重')
    if final and not final['adopted'] and final['post_update_joint_minus_likelihood']<=0:
        dead=[d for d in dead if d['id']!='dead:v88_perm_grid']
        end('dead:v88_perm_grid','v88 当前有界乘性概率格无收益','architecture',[v88id],
            '两项完整对照，绝对与训练后匹配收益分开；实际最佳状态可重载。',
            '关闭本冻结先验/MLP64/乘性log界.5/gate界.25/格软CE/效用0或1组合，先验零格无法新增质量；不泛化所有概率格表达。')
v89id='pure_dl_sw_student4_v89_SW_student4_fold5';v89out=root/'outputs'/v89id;v89study=root/'results/deep_learning/sw_student4_v89_20260930'
if v89out.exists():
    read89=lambda p:json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    state=read89(v89out/'runtime_status.json');report=read89(v89out/'fold_5_report.json');initial=read89(v89out/'initial.json');history=read89(v89out/'validation_history.json') or [];final=read89(v89study/'assembled_report.json');shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,epoch=report['best_epoch'],scope='第五折 · 固定POR/PERM · Student-t4',source='outputs/'+v89id+'/fold_5_report.json');shown=dict(primary=row,results=[row])
    elif history:
        b=max(history,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 固定POR/PERM · 观测快照',source='outputs/'+v89id+'/validation_history.json');shown=dict(primary=row,results=[row])
    else:shown['resultText']='t4自身初始SW '+str(initial.get('target_score'))+'；无新训练成绩'
    conclusion='快照='+json.dumps(state,ensure_ascii=False)+'；t4初始函数不同，不把从自身先验改善当超越Gaussian。'
    if final:
        d=final['against_gaussian'];conclusion='对Gaussian SW='+str(d['channel_delta'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；实际采用三列='+str(final['selected_scores']['TOTAL'])+'；采用='+str(final['adopted'])+'；实际停止轮='+str(report.get('stopped_epoch'))+'；仅开发折非完整OOF'
    node(v89id,'v89 私有SW Student-t4密度','原839481参数/v56训练折内骨干FiLM及SW训练先验重置，同raw权重/mean/std/预算，改条件族为t4、scale=std/sqrt2、NLL和CDF一致。全SW训练/效用1/40%DropKey/原9曲线206/Q50/数据/支持动作不变。旧共享t3 SW-0.086780仅负边界，不重启；新假设限当前私有配方。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(final and final['adopted']),evidence=['results/deep_learning/sw_student4_v89_20260930/README.md','outputs/'+v89id+'/config.json'],**shown)
    edge('pure_dl_sw_interaction_v56_SW_additive_fold5',v89id,'evolution','实际加载原训练折内骨干FiLM，输出重置为原SW训练先验')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',v89id,'evolution','同raw初始化/容量/预算的Gaussian历史对照；不加载其已训练权重，初始CDF不同')
    edge('pure_dl_sw_t3_sequence',v89id,'evolution','旧共享多目标t3负结果作为范围证据，不继承权重或重启旧任务')
    if final and not final['adopted']:
        dead=[d for d in dead if d['id']!='dead:v89_sw_t4']
        end('dead:v89_sw_t4','v89 当前私有SW固定t4未胜Gaussian','architecture',[v89id],
            '同容量/raw初始化/预算的历史Gaussian对照；自身初始CDF不同，训练收益和绝对分开。',
            '关闭当前df4/v56迁移骨干/先验重置/效用1/40%DropKey组合，不扫df或scale，不泛化全部重尾模型；实际停止轮如实记录。')
v90study=root/'results/deep_learning/sw_direct_well_v90_20260930'
v90final=json.loads((v90study/'assembled_report.json').read_text(encoding='utf-8')) if (v90study/'assembled_report.json').exists() else {}
for mode in ('reference','well'):
    id=f'pure_dl_sw_direct_well_v90_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read90=lambda n:json.loads((out/n).read_text(encoding='utf-8')) if (out/n).exists() else {}
    state=read90('runtime_status.json');report=read90('fold_5_report.json');initial=read90('initial.json');history=read90('validation_history.json') or [];shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,epoch=report['best_epoch'],scope='第五折 · 固定POR/PERM · '+mode,source='outputs/'+id+'/fold_5_report.json');shown=dict(primary=row,results=[row])
    elif history:
        b=max(history,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 固定POR/PERM · 观测快照',source='outputs/'+id+'/validation_history.json');shown=dict(primary=row,results=[row])
    else:shown['resultText']='训练先验初始SW '+str(initial.get('target_score'))+'；尚无新成绩'
    conclusion='快照='+json.dumps(state,ensure_ascii=False)+'；未完成不宣称收益。'
    if v90final:
        r=next(r for r in v90final['candidates'] if r['mode']==mode);d=r['against_gaussian']
        conclusion='对Gaussian SW='+str(d['channel_delta'])+'；well-reference='+str(v90final['well_minus_reference']['channel_delta'])+'；训练后最佳匹配差='+str(v90final['post_update_well_minus_reference'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；采用TOTAL='+str(v90final['selected_scores']['TOTAL'])+'；仅开发折非完整OOF'
    node(id,'v90 SW全井直接读出 '+mode,'同841906参数/v56训练折骨干FiLM和SW先验重置，新增零初始化96→25直读头2425参数。旧FiLM两组均本井，新增支路只差本井code或64训练井均值输入code，两code都计算。原9曲线206/输入bank/Q50/评分/全网络/效用1/DropKey.4/1e-4/80-18不改，同初始函数和基本state。',conclusion,stage='architecture',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(v90final and v90final['selected_mode']==mode),evidence=['results/deep_learning/sw_direct_well_v90_20260930/README.md','outputs/'+id+'/config.json'],**shown)
    edge('pure_dl_sw_interaction_v56_SW_additive_fold5',id,'evolution','实际加载原训练折内骨干FiLM，输出重置原SW训练先验')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',id,'evolution','保留Gaussian有效损失/DropKey配方，与无新头历史对照；不加载其训练后权重')
    edge('pure_dl_sw_student4_v89_SW_student4_fold5',id,'evolution','固定t4负结果后转向输入编码直达输出，不继承负权重')
if v90final and not v90final['adopted'] and v90final['post_update_well_minus_reference']<=0:
    dead=[d for d in dead if d['id']!='dead:v90_sw_direct_well']
    end('dead:v90_sw_direct_well','v90 当前全井线性直接支路无收益','architecture',[f'pure_dl_sw_direct_well_v90_SW_{m}_fold5' for m in ('reference','well')],
        '两组完整同容量/同初始函数/同来源预算，绝对与匹配收益分开，实际停止轮记录。',
        '关闭本共享96编码/零线性25输出/v56/训练先验/效用1/DropKey.4组合，不扫宽度LR强度，不泛化所有全井输入机制。')
v91id='pure_dl_sw_frozen_direct_v91_SW_fold5';v91out=root/'outputs'/v91id;v91study=root/'results/deep_learning/sw_frozen_direct_v91_20260930'
if v91out.exists():
    read91=lambda p:json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    state=read91(v91out/'runtime_status.json');group=read91(v91out/'fold_5_report.json');final=read91(v91study/'assembled_report.json');results=[]
    for mode in ('reference','well'):
        r=read91(v91out/mode/'fold_5_report.json');h=read91(v91out/mode/'validation_history.json') or []
        if r:results.append(dict(value=r['reference_replacement_total'],scores={'SW':r['target_score'],'TOTAL':r['reference_replacement_total']},kind='fold',fold=5,epoch=r['best_epoch'],scope='第五折 · 固定POR/PERM · '+mode+(' · 恢复起点' if r['best_epoch']==0 else ''),source='outputs/'+v91id+'/'+mode+'/fold_5_report.json'))
        elif h:
            b=max(h,key=lambda r:r['target_score']);results.append(dict(value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 固定POR/PERM · '+mode+'快照',source='outputs/'+v91id+'/'+mode+'/validation_history.json'))
    shown=dict(primary=max(results,key=lambda r:r['value']),results=results) if results else dict(resultText='初始化/训练中 · 尚无新成绩')
    conclusion='快照='+json.dumps(state,ensure_ascii=False)+'；未完成不宣称收益。'
    if final:conclusion='实际采用TOTAL='+str(final['selected_scores']['TOTAL'])+'；well-reference='+str(final['well_minus_reference']['channel_delta'])+'；训练后完整匹配差='+str(final['post_update_well_minus_reference'])+'；采用='+str(final['selected_mode'])+'；epoch0非新收益；仅开发折非完整OOF'
    node(v91id,'v91 SW冻结全井直接头','当前Gaussian v82 a100第25轮完整函数固定839481原参数，新参考/本井各2425参数零96→25线性输出。原FiLM均本井，两code都计算；参考64训练井输入均值，无标签ID。共享冻结eval前向无DropKey，仅两独立头自身SW优化/早停，原数据/Q50/评分/效用1/1e-4/80-18不变，不加载v90训练后权重。',conclusion,stage='architecture',status='recorded' if group else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(final and final['adopted']),evidence=['results/deep_learning/sw_frozen_direct_v91_20260930/README.md','outputs/'+v91id+'/config.json'],**shown)
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',v91id,'evolution','实际加载完整已采用Gaussian最佳25轮，固定所有原参数')
    for mode in ('reference','well'):edge(f'pure_dl_sw_direct_well_v90_SW_{mode}_fold5',v91id,'evolution','完整匹配well-reference局部SW+0.004915，结构假设迁移到冻结当前函数；不加载其训练后权重')
    if final and not final['adopted'] and final['post_update_well_minus_reference']<=0:
        dead=[d for d in dead if d['id']!='dead:v91_sw_frozen_direct']
        end('dead:v91_sw_frozen_direct','v91 当前冻结Gaussian全井直接头无收益','architecture',[v91id],
            '参考/本井完整同容量同初始函数，绝对/含起点/训练后最佳对照分开，实际停止如实。',
            '关闭本冻结v82/共享96code/零25输出/1e-4/效用1/预算组合，不扫宽度LR强度，不泛化全井机制；v90局部证据单独保留。')
v92study=root/'results/deep_learning/por_full_utility_v92_20260930'
v92final=json.loads((v92study/'assembled_report.json').read_text(encoding='utf-8')) if (v92study/'assembled_report.json').exists() else {}
for mode in ('likelihood','joint'):
    id=f'pure_dl_por_full_utility_v92_POR_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read92=lambda n:json.loads((out/n).read_text(encoding='utf-8')) if (out/n).exists() else {}
    state=read92('runtime_status.json');report=read92('fold_5_report.json');initial=read92('initial.json');h=read92('validation_history.json') or [];shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,epoch=report['best_epoch'],scope='第五折 · 固定PERM/SW · '+mode+(' · 恢复起点' if report['best_epoch']==0 else ''),source='outputs/'+id+'/fold_5_report.json');shown=dict(primary=row,results=[row])
    elif h:
        b=max(h,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'POR':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 固定PERM/SW · '+mode+'快照',source='outputs/'+id+'/validation_history.json');shown=dict(primary=row,results=[row])
    else:shown['resultText']='原始POR '+str(initial.get('target_score'))+'；尚无新成绩'
    conclusion='快照='+json.dumps(state,ensure_ascii=False)+'；完整POR参与训练，不是v86仅路由；未完成不宣称收益。'
    if v92final:
        r=next(r for r in v92final['candidates'] if r['mode']==mode);d=r['against_parent']
        conclusion='对当前POR='+str(d['channel_delta'])+'；joint-likelihood='+str(v92final['joint_minus_likelihood']['channel_delta'])+'；完整训练后最佳匹配差='+str(v92final['post_update_joint_minus_likelihood'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；实际采用TOTAL='+str(v92final['selected_scores']['TOTAL'])+'；同轮轨迹='+json.dumps(v92final.get('trajectory_diagnostic',{}),ensure_ascii=False)+'；epoch0非收益；第五开发折非完整OOF'
    node(id,'v92 POR完整网络效用 '+mode,'当前v31 anchored最佳7轮完整私有POR热启动，v25幅值修正/v30实际读出/v28-v29路由原函数保持。仅旧未用读出副本/anchor参考/其他目标packed行固定；解除source隐藏和输入路由detach/no_grad并绕过v25 frozen train override。两骨干、实际头与有效修正训练，无新容量。原9曲线206+已有幅值63/Q50/t3/log1p/官方.08/default.1/256支持800动作不变，仅损失系数0/1；body1e-5/head3e-5/cos到1e-5/80-18。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(v92final and v92final['selected_mode']==mode),evidence=['results/deep_learning/por_full_utility_v92_20260930/README.md','outputs/'+id+'/config.json'],**shown)
    edge('pure_dl_mechanism_v31_POR_fold5',id,'evolution','实际加载anchored第7轮及v30/v25/v8依赖，完整函数同起点；原权重/交付不改')
    edge('pure_dl_por_joint_utility_v86_POR_fold5',id,'evolution','此前仅路由效用负结果，不足否定全骨干；扩大自身梯度范围，不加载其负权重')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',id,'evolution','SW效用完整匹配收益启发跨目标假设；不加载SW权重或SW预测')
if v92final and not v92final['adopted'] and v92final['post_update_joint_minus_likelihood']<=0:
    dead=[d for d in dead if d['id']!='dead:v92_por_full_utility']
    end('dead:v92_por_full_utility','v92 当前POR完整网络效用无收益','optimization',[f'pure_dl_por_full_utility_v92_POR_{m}_fold5' for m in ('likelihood','joint')],
        '两完整匹配初始化/容量/预算，含起点/训练后/同轮轨迹与绝对差分开；实际梯度覆盖全目标私有网络。',
        '关闭本v31热启动/原t3/低LR/系数1/温度.05/预算配置，不无限扫参，不泛化所有POR效用训练。')
v93study=root/'results/deep_learning/sw_sam_v93_20260930'
v93final=json.loads((v93study/'assembled_report.json').read_text(encoding='utf-8')) if (v93study/'assembled_report.json').exists() else {}
for mode in ('control','sam005'):
    id=f'pure_dl_sw_sam_v93_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read93=lambda n:json.loads((out/n).read_text(encoding='utf-8')) if (out/n).exists() else {}
    state=read93('runtime_status.json');report=read93('fold_5_report.json');initial=read93('initial.json');h=read93('validation_history.json') or [];shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,epoch=report['best_epoch'],scope='第五折 · 固定POR/PERM · '+mode,source='outputs/'+id+'/fold_5_report.json');shown=dict(primary=row,results=[row])
    elif h:
        b=max(h,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 固定POR/PERM · '+mode+'快照',source='outputs/'+id+'/validation_history.json');shown=dict(primary=row,results=[row])
    else:shown['resultText']='原训练先验SW '+str(initial.get('target_score'))+'；尚无新成绩'
    conclusion='快照='+json.dumps(state,ensure_ascii=False)+'；两次计算匹配，未完成不宣称收益。'
    if v93final:
        r=next(r for r in v93final['candidates'] if r['mode']==mode);d=r['against_gaussian']
        conclusion='对当前Gaussian SW='+str(d['channel_delta'])+'；SAM-control='+str(v93final['sam_minus_control']['channel_delta'])+'；训练后完整最佳匹配差='+str(v93final['post_update_sam_minus_control'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；实际采用TOTAL='+str(v93final['selected_scores']['TOTAL'])+'；同轮轨迹='+json.dumps(v93final.get('trajectory_diagnostic',{}),ensure_ascii=False)+'；并发CUDA故障后按完整last串行恢复完成，原失败证据保持；仅开发折非完整OOF'
    node(id,'v93 SW两次计算匹配SAM '+mode,'839481参数/v56训练折骨干FiLM+原训练先验重置，原Gaussian/效用1/DropKey.4/数据/Q50/评分不变。rho0/.05都两次forward/backward同batch与RNG回放mask，未clip全局L2首梯度扰动、第二梯度AdamW，clone精确恢复再更新，随机流消费一个forward。1e-4/80-18/cos不改，额外计算实测记录；旧v55 warm-start SAM best0负边界，不重启。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(v93final and v93final['selected_mode']==mode),evidence=['results/deep_learning/sw_sam_v93_20260930/README.md','results/deep_learning/sw_sam_v93_20260930/HISTORICAL.md','outputs/'+id+'/config.json'],**shown)
    edge('pure_dl_sw_interaction_v56_SW_additive_fold5',id,'evolution','实际加载原训练折内骨干FiLM，输出重置原训练先验')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',id,'evolution','当前有效损失/DropKey配方及历史绝对对照，不加载其训练后权重')
    edge('pure_dl_sw_sharpness_v55_SW_sam_fold5',id,'evolution','旧私有SW warm-start SAM best0/stop18负边界；新原始prior训练/效用/DropKey配方，不继承负权重')
if v93final and not v93final['adopted'] and v93final['post_update_sam_minus_control']<=0:
    dead=[d for d in dead if d['id']!='dead:v93_sw_sam']
    end('dead:v93_sw_sam','v93 当前SW SAM.05无收益','optimization',[f'pure_dl_sw_sam_v93_SW_{m}_fold5' for m in ('control','sam005')],
        '同容量同初始化同两次计算/更新数，绝对/含起点/训练后最佳/同轮差分开；额外计算实测。',
        '关闭本v56-prior/Gaussian/效用1/DropKey.4/AdamW/rho.05/预算变体，不扫rhoLRASAM，不泛化全部SAM；旧v55证据分开。')
v94study=root/'results/deep_learning/sw_sam_radius_v94_20260930'
v94final=json.loads((v94study/'assembled_report.json').read_text(encoding='utf-8')) if (v94study/'assembled_report.json').exists() else {}
for mode in ('r025','r100'):
    id=f'pure_dl_sw_sam_radius_v94_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read94=lambda n:json.loads((out/n).read_text(encoding='utf-8')) if (out/n).exists() else {}
    state=read94('runtime_status.json');report=read94('fold_5_report.json');initial=read94('initial.json');h=read94('validation_history.json') or [];shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,epoch=report['best_epoch'],scope='第五折 · 固定POR/PERM · '+mode,source='outputs/'+id+'/fold_5_report.json');shown=dict(primary=row,results=[row])
    elif h:
        b=max(h,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 固定POR/PERM · '+mode+'快照',source='outputs/'+id+'/validation_history.json');shown=dict(primary=row,results=[row])
    else:shown['resultText']='训练先验SW '+str(initial.get('target_score'))+'；尚无新训练成绩'
    conclusion='快照='+json.dumps(state,ensure_ascii=False)+'；两项显式串行，未完成不宣称收益。'
    if v94final:
        r=next(r for r in v94final['candidates'] if r['mode']==mode);d=r['against_sam05']
        conclusion='对.05 SW='+str(d['channel_delta'])+'；训练后对rho0='+str(r['post_update_against_rho0']['channel_delta'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；实际采用TOTAL='+str(v94final['selected_scores']['TOTAL'])+'；采用='+str(v94final['selected_mode'])+'；仅开发折非完整OOF'
    node(id,'v94 SW SAM半径 '+mode,'同839481参数/v56训练折骨干FiLM+训练先验重置，rho.025/.1两项仅半径不同；已完整.05/0作历史对照，不加载其训练后权重。原Gaussian/效用1/DropKey.4/两次计算/RNG回放/clone恢复/原数据/Q50/评分/1e-4/80-18不变。第二pass RNG终态只读核验，last保存跨恢复peak。上批并发故障，本批强制一个GPU进程、各项显式串行，无自动队列。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(v94final and v94final['selected_mode']==mode),evidence=['results/deep_learning/sw_sam_radius_v94_20260930/README.md','outputs/'+id+'/config.json'],**shown)
    edge('pure_dl_sw_interaction_v56_SW_additive_fold5',id,'evolution','实际加载原训练折内骨干FiLM，输出设原训练先验')
    edge('pure_dl_sw_sam_v93_SW_sam005_fold5',id,'evolution','.05完整正收益后有限半径比较，实际同raw初始化/容量/预算，不加载已训练权重')
    edge('pure_dl_sw_sam_v93_SW_control_fold5',id,'evolution','同两次计算rho0完整对照，固定绝对和机制差，旧控制已核验保留')
if v94final and not v94final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v94_sam_radius_range']
    end('dead:v94_sam_radius_range','v94 两个邻近半径未提高已有效.05','optimization',[f'pure_dl_sw_sam_radius_v94_SW_{m}_fold5' for m in ('r025','r100')],
        '完整两项对.05绝对差与rho0机制差分开；无新采用不抹去对rho0的正机制收益。',
        '结束本.025/.1范围，保留.05及有实际机制亮点的依赖，不继续无限rho/LR/ASAM搜索，不泛化SAM无效。')
v95study=root/'results/deep_learning/sw_factorization_v95_20260930'
v95final=json.loads((v95study/'assembled_report.json').read_text(encoding='utf-8')) if (v95study/'assembled_report.json').exists() else {}
for mode in ('same','separate'):
    id=f'pure_dl_sw_factorization_v95_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    report=json.loads((out/'fold_5_report.json').read_text(encoding='utf-8')) if (out/'fold_5_report.json').exists() else {}
    shown={};conclusion='预定义同1678962参数固定结构，尚未评价，无新收益。'
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,scope='第五折 · 冻结概率分解 · '+mode,source='outputs/'+id+'/fold_5_report.json')
        shown=dict(primary=row,results=[row]);d=report['diagnostics']
        conclusion='未训练，原144665点两个磁盘来源重建；same与r100逐点相同。SW差='+str(d['channel_delta'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；实际采用TOTAL='+str(v95final.get('selected_scores',{}).get('TOTAL'))+'；非独立泛化或完整OOF。'
    else:shown['resultText']='未训练/尚未评价'
    node(id,'v95 固定默认atom/条件密度 '+mode,'完全冻结同容量两分支，r100默认atom+自身或旧v82a100条件密度；同一条件概率公式应用每点，无预测投票/按点选模型/标签前向。原输入/bank/Q50/折/评分/动作不改。旧v84密度训练负范围区分，本批零训练。',conclusion,stage='architecture',status='recorded' if report else 'untrained',highlight=bool(v95final and v95final.get('selected_mode')==mode),evidence=['results/deep_learning/sw_factorization_v95_20260930/PLAN.md','results/deep_learning/sw_factorization_v95_20260930/FINAL.md','outputs/'+id+'/config.json'],**shown)
    edge('pure_dl_sw_sam_radius_v94_SW_r100_fold5',id,'evolution','实际加载固定默认atom；same同时加载其条件密度，是等容量函数对照')
    if mode=='separate':edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',id,'fusion','预定义固定条件密度源，条件概率分解；不是点预测投票或按点选择')
if v95final and not v95final['adopted']:
    dead=[d for d in dead if d['id']!='dead:v95_fixed_factorization']
    end('dead:v95_fixed_factorization','v95 固定两来源分解无收益','architecture',['pure_dl_sw_factorization_v95_SW_separate_fold5'],
        'r100固定默认atom+自身/旧v82密度，同容量/原动作全144665评价，零训练。','仅关闭这两个来源的直接分解，不扫来源或比例，不泛化所有因子模型。')
v96study=root/'results/deep_learning/sw_factor_sam_v96_20260930'
v96final=json.loads((v96study/'assembled_report.json').read_text(encoding='utf-8')) if (v96study/'assembled_report.json').exists() else {}
for mode in ('likelihood','joint'):
    id=f'pure_dl_sw_factor_sam_v96_SW_{mode}_fold5';out=root/'outputs'/id
    if not out.exists():continue
    read96=lambda n:json.loads((out/n).read_text(encoding='utf-8')) if (out/n).exists() else {}
    report=read96('fold_5_report.json');state=read96('runtime_status.json');h=read96('validation_history.json') or [];initial=read96('initial.json');shown={}
    if report:
        row=dict(value=report['reference_replacement_scores']['TOTAL'],scores=report['reference_replacement_scores'],kind='fold',fold=5,epoch=report['best_epoch'],scope='第五折 · 私有密度SAM · '+mode,source='outputs/'+id+'/fold_5_report.json');shown=dict(primary=row,results=[row])
    elif h:
        b=max(h,key=lambda r:r['target_score']);row=dict(value=b['reference_replacement_total'],scores={'SW':b['target_score'],'TOTAL':b['reference_replacement_total']},kind='observed',fold=5,epoch=b['epoch'],scope='第五折 · 私有密度SAM快照 · '+mode,source='outputs/'+id+'/validation_history.json');shown=dict(primary=row,results=[row])
    else:shown['resultText']='初始完整v95 SW '+str(initial.get('target_score'))+'；尚无新训练成绩'
    conclusion='训练快照='+json.dumps(state,ensure_ascii=False)+'；两项未完成前不宣称最终公平收益。'
    if v96final:
        r=next(r for r in v96final['candidates'] if r['mode']==mode);d=r['against_parent'];p=r['report']
        conclusion='对v95 SW='+str(d['channel_delta'])+'；最佳/停止='+str(p['best_epoch'])+'/'+str(p['stopped_epoch'])+'；joint-likelihood含起点='+str(v96final['joint_minus_likelihood']['channel_delta'])+'，更新后='+str(v96final['post_update_joint_minus_likelihood']['channel_delta'])+'；实际采用TOTAL='+str(v96final['selected_scores']['TOTAL'])+'；默认/数值='+str(d['populations']['default']['channel_delta'])+'/'+str(d['populations']['numeric']['channel_delta'])+'；epoch0非新收益/开发折非完整OOF。'
    node(id,'v96 固定SAM默认atom/私有密度 '+mode,'同v95完整函数热启动，1678962总/839481优化器参数（129私有gate不用）。r100默认atom完全冻结，v82来源密度全网SAM.1/1e-4/40%DropKey/80-18，仅效用0/1不同。两pass同batch/RNG回放/精确clone恢复，完整原子last；原输入/Q50/评分/动作保持，单GPU显式串行，无其他折。旧v84旧gate/3e-5/单pass负范围区分。',conclusion,stage='optimization',status='recorded' if report else ('failed' if state.get('status')=='failed' else 'running_snapshot'),highlight=bool(v96final and v96final['selected_mode']==mode),evidence=['results/deep_learning/sw_factor_sam_v96_20260930/PLAN.md','results/deep_learning/sw_factor_sam_v96_20260930/README.md','outputs/'+id+'/config.json'],**shown)
    edge('pure_dl_sw_factorization_v95_SW_separate_fold5',id,'evolution','实际加载已有效完整因子函数，默认atom冻结；私有密度优化')
    edge('pure_dl_sw_sam_radius_v94_SW_r100_fold5',id,'fusion','实际固定默认atom权重，借用已实测有效rho.1更新私有密度，不投票')
    edge('pure_dl_sw_utility_strength_v82_SW_a100_fold5',id,'evolution','实际加载私有密度热启动，不重置先验')
if v96final and not v96final['adopted'] and v96final['post_update_joint_minus_likelihood']['channel_delta']<=0:
    dead=[d for d in dead if d['id']!='dead:v96_private_density_sam']
    end('dead:v96_private_density_sam','v96 当前父函数私有密度SAM无新收益','optimization',[f'pure_dl_sw_factor_sam_v96_SW_{m}_fold5' for m in ('likelihood','joint')],
        '同父完整函数、容量/两pass/SAM.1/1e-4，只差效用0/1，完整最佳/更新后/同轮差分开。','关闭本父函数/密度/SAM.1/学习率/两损失配置，不扫来源/人口/LR，不泛化全部SAM或因子模型。')
if v94final:
    r=next(r for r in v94final['candidates'] if r['mode']=='r025')
    if r['against_sam05']['channel_delta']<=0 and r['post_update_against_rho0']['channel_delta']<=0:
        dead=[d for d in dead if d['id']!='dead:v94_r025']
        end('dead:v94_r025','v94 rho.025 无绝对/匹配收益','optimization',['pure_dl_sw_sam_radius_v94_SW_r025_fold5'],
            '同raw初始化/839481参数/两pass计算，best11/stop29；对.05 SW='+str(r['against_sam05']['channel_delta'])+'，对rho0='+str(r['post_update_against_rho0']['channel_delta'])+'。',
            '关闭这一个.025配置，不扫描该低半径邻域；.1正收益保留，不泛化SAM无效。')
if v96final:
    r=next(r for r in v96final['candidates'] if r['mode']=='joint')
    if r['report']['best_epoch']==0 and v96final['post_update_joint_minus_likelihood']['channel_delta']<=0:
        dead=[d for d in dead if d['id']!='dead:v96_joint_conditional_utility']
        end('dead:v96_joint_conditional_utility','v96 固定atom的密度联合效用未增益','optimization',['pure_dl_sw_factor_sam_v96_SW_joint_fold5'],
            '同v95完整函数/同密度容量/SAM.1/1e-4；joint恢复epoch0，训练后最佳joint-like SW='+str(v96final['post_update_joint_minus_likelihood']['channel_delta'])+'；纯似然实测通过80.5。',
            '关闭这一个固定atom/私有密度alpha1配置，不扫人口份额或系数；保留原全网效用与本批纯似然正收益，不泛化联合效用无效。')

from fullcv_v97 import annotate as annotate_fullcv_v97
annotate_fullcv_v97(root,node,edge)
from por_effective_sam_v98 import annotate as annotate_por_effective_sam_v98
annotate_por_effective_sam_v98(root,node,edge,end)
from por_joint_utility_v99 import annotate as annotate_por_joint_utility_v99
annotate_por_joint_utility_v99(root,node,edge,end)
from compact_v100 import annotate as annotate_compact_v100
annotate_compact_v100(root,node,edge,end)
from compact_mean_scope_v101 import annotate as annotate_compact_mean_scope_v101
annotate_compact_mean_scope_v101(root,node,edge,end)
from compact_source_features_v102 import annotate as annotate_compact_source_features_v102
annotate_compact_source_features_v102(root,node,edge,end)
from compact_cached_features_v103 import annotate as annotate_compact_cached_features_v103
annotate_compact_cached_features_v103(root,node,edge,end)
from compact_frequency_v104 import annotate as annotate_compact_frequency_v104
annotate_compact_frequency_v104(root,node,edge,end)
from compact_por_projection_v105 import annotate as annotate_compact_por_projection_v105
annotate_compact_por_projection_v105(root,node,edge,end)
from compact_local_density_v106 import annotate as annotate_compact_local_density_v106
annotate_compact_local_density_v106(root,node,edge,end)
from compact_attention_v107 import annotate as annotate_compact_attention_v107
annotate_compact_attention_v107(root,node,edge,end)
from compact_por_features_v108 import annotate as annotate_compact_por_features_v108
annotate_compact_por_features_v108(root,node,edge,end)
from compact_query_read_v109 import annotate as annotate_compact_query_read_v109
annotate_compact_query_read_v109(root,node,edge,end)
from compact_hierarchical_utility_v110 import annotate as annotate_compact_hierarchical_utility_v110
annotate_compact_hierarchical_utility_v110(root,node,edge,end)
from compact_perm_mean_v111 import annotate as annotate_compact_perm_mean_v111
annotate_compact_perm_mean_v111(root,node,edge,end)
from compact_por_cauchy_v112 import annotate as annotate_compact_por_cauchy_v112
annotate_compact_por_cauchy_v112(root,node,edge,end)
from compact_por_mixed_family_v113 import annotate as annotate_compact_por_mixed_family_v113
from compact_por_scale_v114 import annotate as annotate_compact_por_scale_v114
from compact_perm_family_v115 import annotate as annotate_compact_perm_family_v115
from compact_sw_dilation_v116 import annotate as annotate_compact_sw_dilation_v116
from compact_perm_density_head_v117 import annotate as annotate_compact_perm_density_head_v117
from compact_por_utility_v118 import annotate as annotate_compact_por_utility_v118
from compact_por_feature_adapt_v119 import annotate as annotate_compact_por_feature_adapt_v119
from compact_sw_well_sampling_v120 import annotate as annotate_compact_sw_well_sampling_v120
from compact_sw_readout_split_v121 import annotate as annotate_compact_sw_readout_split_v121
from training_matmul_timing_v122 import annotate as annotate_training_matmul_timing_v122
from compact_sw_scaled_sam_v123 import annotate as annotate_compact_sw_scaled_sam_v123
from compact_sw_cdf_brier_v124 import annotate as annotate_compact_sw_cdf_brier_v124
from compact_por_family_gate_v125 import annotate as annotate_compact_por_family_gate_v125
from compact_por_output_scope_v126 import annotate as annotate_compact_por_output_scope_v126
from por_convgru_backbone_v127 import annotate as annotate_por_convgru_backbone_v127
from por_joint_context_head_v128 import annotate as annotate_por_joint_context_head_v128
from por_joint_point_ablation_v129 import annotate as annotate_por_joint_point_ablation_v129
from joint_hierarchical_backbone_v130 import annotate as annotate_joint_hierarchical_backbone_v130
from por_joint_density_head_v131 import annotate as annotate_por_joint_density_head_v131
from joint_representation_backbone_v132 import annotate as annotate_joint_representation_backbone_v132
from por_joint_full_density_head_v133 import annotate as annotate_por_joint_full_density_head_v133
from por_joint_density_width_v134 import annotate as annotate_por_joint_density_width_v134
from joint_curve_backbone_v135 import annotate as annotate_joint_curve_backbone_v135
from perm_joint_density_v136 import annotate as annotate_perm_joint_density_v136
from joint_pyramid_backbone_v137 import annotate as annotate_joint_pyramid_backbone_v137
from perm_joint_density_width_v138 import annotate as annotate_perm_joint_density_width_v138
from compact_sw_student_v139 import annotate as annotate_compact_sw_student_v139
from por_joint_density_width_v140 import annotate as annotate_por_joint_density_width_v140
from joint_private_readout_backbone_v141 import annotate as annotate_joint_private_readout_backbone_v141
from por_density_location_bound_v142 import annotate as annotate_por_density_location_bound_v142
from sw_joint_density_transfer_v143 import annotate as annotate_sw_joint_density_transfer_v143
from por_density_extensions_v144_v145 import annotate as annotate_por_density_extensions_v144_v145
from por_joint_density_width_v146 import annotate as annotate_por_joint_density_width_v146
from por_joint_density_width_v147 import annotate as annotate_por_joint_density_width_v147
from por_density_tilt_v148 import annotate as annotate_por_density_tilt_v148
from por_component_readout_v149 import annotate as annotate_por_component_readout_v149
from por_smooth_cdf_readout_v150 import annotate as annotate_por_smooth_cdf_readout_v150
from por_ranked_probability_v151 import annotate as annotate_por_ranked_probability_v151
from por_expert_interaction_v152 import annotate as annotate_por_expert_interaction_v152
from por_ranked_probability_v153 import annotate as annotate_por_ranked_probability_v153
from perm_ranked_probability_v154 import annotate as annotate_perm_ranked_probability_v154
from sw_ranked_probability_v155 import annotate as annotate_sw_ranked_probability_v155
from perm_ranked_probability_v156 import annotate as annotate_perm_ranked_probability_v156
from por_bounded_gate_rps_v157 import annotate as annotate_por_bounded_gate_rps_v157
from perm_laplace_rps_v158 import annotate as annotate_perm_laplace_rps_v158
annotate_compact_por_mixed_family_v113(root,node,edge,end)
annotate_compact_por_scale_v114(root,node,edge,end)
annotate_compact_perm_family_v115(root,node,edge,end)
annotate_compact_sw_dilation_v116(root,node,edge,end)
annotate_compact_perm_density_head_v117(root,node,edge,end)
annotate_compact_por_utility_v118(root,node,edge,end)
annotate_compact_por_feature_adapt_v119(root,node,edge,end)
annotate_compact_sw_well_sampling_v120(root,node,edge,end)
annotate_compact_sw_readout_split_v121(root,node,edge,end)
annotate_training_matmul_timing_v122(root,node,edge,end)
annotate_compact_sw_scaled_sam_v123(root,node,edge,end)
annotate_compact_sw_cdf_brier_v124(root,node,edge,end)
annotate_compact_por_family_gate_v125(root,node,edge,end)
annotate_compact_por_output_scope_v126(root,node,edge,end)
annotate_por_convgru_backbone_v127(root,node,edge,end)
annotate_por_joint_context_head_v128(root,node,edge,end)
annotate_por_joint_point_ablation_v129(root,node,edge,end)
annotate_joint_hierarchical_backbone_v130(root,node,edge,end)
annotate_por_joint_density_head_v131(root,node,edge,end)
annotate_joint_representation_backbone_v132(root,node,edge,end)
annotate_por_joint_full_density_head_v133(root,node,edge,end)
annotate_por_joint_density_width_v134(root,node,edge,end)
annotate_joint_curve_backbone_v135(root,node,edge,end)
annotate_perm_joint_density_v136(root,node,edge,end)
annotate_joint_pyramid_backbone_v137(root,node,edge,end)
annotate_perm_joint_density_width_v138(root,node,edge,end)
annotate_compact_sw_student_v139(root,node,edge,end)
annotate_por_joint_density_width_v140(root,node,edge,end)
annotate_joint_private_readout_backbone_v141(root,node,edge,end)
annotate_por_density_location_bound_v142(root,node,edge,end)
annotate_sw_joint_density_transfer_v143(root,node,edge,end)
annotate_por_density_extensions_v144_v145(root,node,edge,end)
annotate_por_joint_density_width_v146(root,node,edge,end)
annotate_por_joint_density_width_v147(root,node,edge,end)
annotate_por_density_tilt_v148(root,node,edge,end)
annotate_por_component_readout_v149(root,node,edge,end)
annotate_por_smooth_cdf_readout_v150(root,node,edge,end)
annotate_por_ranked_probability_v151(root,node,edge,end)
annotate_por_expert_interaction_v152(root,node,edge,end)
annotate_por_ranked_probability_v153(root,node,edge,end)
annotate_perm_ranked_probability_v154(root,node,edge,end)
annotate_sw_ranked_probability_v155(root,node,edge,end)
annotate_perm_ranked_probability_v156(root,node,edge,end)
annotate_por_bounded_gate_rps_v157(root,node,edge,end)
annotate_perm_laplace_rps_v158(root,node,edge,end)
from perm_scale_readout_v159 import annotate as annotate_perm_scale_readout_v159
annotate_perm_scale_readout_v159(root,node,edge,end)
from por_rps_utility_v160 import annotate as annotate_por_rps_utility_v160
annotate_por_rps_utility_v160(root,node,edge,end)
from perm_gate_coadapt_v161 import annotate as annotate_perm_gate_coadapt_v161
annotate_perm_gate_coadapt_v161(root,node,edge,end)
from perm_local_representation_v162 import annotate as annotate_perm_local_representation_v162
annotate_perm_local_representation_v162(root,node,edge,end)
from perm_low_lr_rps_v163 import annotate as annotate_perm_low_lr_rps_v163
annotate_perm_low_lr_rps_v163(root,node,edge,end)
from por_deep_readout_v164 import annotate as annotate_por_deep_readout_v164
annotate_por_deep_readout_v164(root,node,edge,end)
from perm_sam_rps_v165 import annotate as annotate_perm_sam_rps_v165
annotate_perm_sam_rps_v165(root,node,edge,end)
from perm_private_backbone_rps_v166 import annotate as annotate_perm_private_backbone_rps_v166
annotate_perm_private_backbone_rps_v166(root,node,edge,end)
from perm_aligned_backbone_rps_v167 import annotate as annotate_perm_aligned_backbone_rps_v167
annotate_perm_aligned_backbone_rps_v167(root,node,edge,end)
from perm_sam_gate_rps_v168 import annotate as annotate_perm_sam_gate_rps_v168
annotate_perm_sam_gate_rps_v168(root,node,edge,end)
from perm_utility_raw_tcn_v171 import annotate as annotate_perm_utility_raw_tcn_v171
# v189 early negative cleaned; v190 wavelet finalized/long29 active, pair incomplete. v191 IQF readout active, initial representation change disclosed.
annotate_perm_utility_raw_tcn_v171(root,node,edge,end)
if (root/'results/deep_learning/perm_aligned_backbone_rps_v167_20261001/control_delta.json').exists():
    end('dead:v167_hidden_anchor','v167 hidden anchor no gain','architecture',['pure_dl_perm_aligned_backbone_rps_v167_fold5/perm_conditional'],'PERM89.344760174753 best0/stop14; same-capacity v166 delta0; actual TOTAL80.591193779236 and exact replay verified.','Close weight0.1 hidden anchoring under80/14 only; no broad conclusion about all alignment methods.')
if (root/'results/deep_learning/perm_sam_gate_rps_v168_20261001/control_delta.json').exists():
    end('dead:v168_sam_gate','v168 gate addition did not expand SAM gain','architecture',['pure_dl_perm_sam_gate_rps_v168_fold5/perm_conditional'],'PERM89.350850358669; versus pureSAM -0.001398010212; actual TOTAL80.593325343606, exact replay verified.','Keep v165 standalone SAM local positive; close this gate addition, retain current v154 recipe.')
from perm_wavelet_detail_v190 import annotate as annotate_perm_wavelet_detail_v190
annotate_perm_wavelet_detail_v190(root,node,edge,end)
from perm_quantile_readout_v191 import annotate as annotate_perm_quantile_readout_v191
annotate_perm_quantile_readout_v191(root,node,edge,end)
from perm_multiorder_interaction_v192 import annotate as annotate_perm_multiorder_interaction_v192
annotate_perm_multiorder_interaction_v192(root,node,edge,end)
if (root/'results/deep_learning/perm_wavelet_detail_v190_20261001/control_delta.json').exists():
    end('dead:v190_wavelet_detail','v190 frequency decomposition below matched long kernel','architecture',['pure_dl_perm_wavelet_detail_v190_fold5/wavelet','pure_dl_perm_wavelet_detail_v190_fold5/long29'],'Wavelet88.935724895823 vs long2988.953509395268; delta-0.017784499445, identical1115185 active parameters and initial function; both restored16/30.','Close this first-detail two-level setting only; both below185, no universal wavelet conclusion.')
if (root/'results/deep_learning/perm_quantile_readout_v191_20261001/control_delta.json').exists():
    end('dead:v191_IQF32','v191 quantile representation no learned gain','architecture',['pure_dl_perm_quantile_readout_v191_fold5/iqf'],'Initial/best89.345246688523 best0/stop14; Gaussian17489.364985921273; initial representation delta+0.000486513770 is not training gain.','Close this32node piecewise-linear/exponential-tail SOURCE-conditioned readout with existing loss only; no universal quantile conclusion.')
from perm_adaptive_mass_v193 import annotate as annotate_perm_adaptive_mass_v193
annotate_perm_adaptive_mass_v193(root,node,edge,end)
from perm_curve_core_v194 import annotate as annotate_perm_curve_core_v194
annotate_perm_curve_core_v194(root,node,edge,end)
from perm_additive_crps_v195 import annotate as annotate_perm_additive_crps_v195
annotate_perm_additive_crps_v195(root,node,edge,end)
from perm_frozen_positive_patch_v197 import annotate as annotate_perm_frozen_positive_patch_v197
annotate_perm_frozen_positive_patch_v197(root,node,edge,end)
if (root/'results/deep_learning/perm_adaptive_mass_v193_20261001/control_delta.json').exists():
    end('dead:v193_adaptive_mass','v193 adaptive probability transport no learned gain','architecture',['pure_dl_perm_adaptive_mass_v193_fold5/adaptive'],'PERM89.344788677861 best0/stop14, learnedgain0; Gaussian174 delta-0.020197243412; exact144665 restored/assembled.','Close this16interval bounded SOURCE-CDF transport only; no parameter grid, original154 positive preserved.')
if (root/'results/deep_learning/perm_additive_crps_v195_20261001/control_delta.json').exists():
    end('dead:v195_additive_CRPS','v195 analytic CRPS addition no gain','architecture',['pure_dl_perm_additive_crps_v195_fold5/perm_conditional'],'PERM89.344760174753 best0/stop14; matched165AdamW delta0, initial same. No learned gain; exact144665 restored/assembled.','Close coefficient0.1 conditional analyticCRPS augmentation under80-14 only; retain154 jointRPS learned model.')
from compact_stage_batch_v201 import annotate as annotate_compact_stage_batch_v201
annotate_compact_stage_batch_v201(root,node,edge,end)
# These curated records have no top-level output report discoverable by the builder.
virtual_ids = [f'pure_dl_complete_seeds_v362_fold5/seed_{s}' for s in range(20261010,20261015)]
virtual_ids += ['pure_dl_complete_seeds_v362_fold5/seed_20261010/SW351',
                'original_seed_pair_v363_20261002', 'original_seed_weight_grid_v364_20261002',
                'per_head_seed_weights_v365_20261002', 'stage_gain_analysis_v366_20261002']
for ident in virtual_ids:
    if ident not in nodes:
        continue
    n = nodes[ident]
    n['virtual'] = True
    n['stage'] = 'vote' if ident.startswith(('original_seed_', 'per_head_seed_')) else 'backbone'
    evidence_file = root/'outputs'/ident/'COMPLETE_MODEL.json' if '/seed_' in ident else root/'results/deep_learning'/ident/'FINAL.json'
    result = None
    if evidence_file.is_file():
        result = json.loads(evidence_file.read_text(encoding='utf-8-sig'))
    elif '/seed_' in ident:
        compact_file=evidence_file.parent/'SUMMARY_RESULTS.json'
        if compact_file.is_file():
            compact=json.loads(compact_file.read_text(encoding='utf-8'))
            matches=[r['result'] for r in compact['results'] if r['source']=='COMPLETE_MODEL.json']
            if matches:
                result=matches[0];evidence_file=compact_file
                n['summary'] += ' 用户已清理此种子权重与逐轮记录；仅保留完整训练的历史摘要和结果，不可直接重载。'
                n['artifacts_compacted']=True
                n['evidence']=[evidence_file.relative_to(root).as_posix()]
    if result is not None:
        scores = result.get('scores')
        if scores and isinstance(scores.get('TOTAL'), (int, float)):
            diagnostic = '/seed_' not in ident
            primary = dict(scope='第五折 · 验证权重诊断' if diagnostic else '第五折 · 完整八阶段模型',
                           value=scores['TOTAL'], scores=scores,
                           source=evidence_file.relative_to(root).as_posix(),
                           kind='diagnostic' if diagnostic else 'fold', fold=5)
            n.update(primary=primary, results=[primary])
            if '/seed_' in ident and ident.count('/')==1:
                n.update(status='complete',resultText=f"Fifth-fold complete8stage historical TOTAL {scores['TOTAL']:.12f}; not fullOOF"+('; weights/logs removed, summary retained' if n.get('artifacts_compacted') else ''))
for snapshot_seed in range(20261010,20261015):
    partial_file = root/f'results/deep_learning/stage_gain_analysis_v366_20261002/SEED{snapshot_seed}_STAGE_SNAPSHOT.json'
    if partial_file.is_file() and 'stage_gain_analysis_v366_20261002' in nodes:
        partial = json.loads(partial_file.read_text(encoding='utf-8'))
        n = nodes['stage_gain_analysis_v366_20261002']
        n['evidence'].append(partial_file.relative_to(root).as_posix())
        n['summary'] += ' Saved native prediction stage analysis: '+json.dumps(partial,ensure_ascii=False)
        seed_ident=f'pure_dl_complete_seeds_v362_fold5/seed_{snapshot_seed}'
        if seed_ident in nodes and not nodes[seed_ident].get('artifacts_compacted') and not (root/'outputs'/seed_ident/'COMPLETE_MODEL.json').is_file():
            seed_node=nodes[seed_ident]
            seed_node.update(status='running_snapshot',primary=None,results=[],
                             resultText=f"{partial['completed_stages']}/8阶段已完成；暂无完整模型分数")
            seed_node['evidence'].append(partial_file.relative_to(root).as_posix())
            seed_node['summary'] += ' 原生已完成阶段前缀（不是完整模型成绩）：'+json.dumps(partial['prefix_scores'],ensure_ascii=False)
curves_file = root/'results/deep_learning/stage_gain_analysis_v366_20261002/SOURCE_CURVE_ANALYSIS.json'
if curves_file.is_file() and 'stage_gain_analysis_v366_20261002' in nodes:
    curves = json.loads(curves_file.read_text(encoding='utf-8'))
    n = nodes['stage_gain_analysis_v366_20261002']
    n['evidence'].extend([curves_file.relative_to(root).as_posix(),
                          curves_file.with_name('source_learning_curves.png').relative_to(root).as_posix()])
    n['summary'] += ' Saved SOURCE history analysis (no model replay): '+json.dumps(curves['interpretation'],ensure_ascii=False)
equivalence_file = root/'results/deep_learning/stage_gain_analysis_v366_20261002/INITIALIZER_EQUIVALENCE_SUMMARY.json'
if equivalence_file.is_file() and 'stage_gain_analysis_v366_20261002' in nodes:
    equivalence = json.loads(equivalence_file.read_text(encoding='utf-8'))
    n = nodes['stage_gain_analysis_v366_20261002']
    n['evidence'].append(equivalence_file.relative_to(root).as_posix())
    n['summary'] += ' CPU initializer-state equivalence addendum: '+json.dumps(equivalence,ensure_ascii=False)
decision_file = root/'results/deep_learning/stage_gain_analysis_v366_20261002/STAGE_REMOVAL_DECISION.md'
quality_file = root/'results/deep_learning/stage_gain_analysis_v366_20261002/QUALITY_PROVENANCE.json'
if quality_file.is_file() and 'stage_gain_analysis_v366_20261002' in nodes:
    n = nodes['stage_gain_analysis_v366_20261002']
    n['evidence'].extend([quality_file.relative_to(root).as_posix(),
                          quality_file.with_suffix('.md').relative_to(root).as_posix()])
    n['summary'] += ' 各自训练侧Q来源澄清：新SOURCE继承严格早于最佳轮的Q50/40/36/38，降权1736/1394/1233/1459点；历史Q50为1755点。原训练每2轮观察（亦含第1轮），不是仅第50轮观察；较早停止不等于无Q。仅配置与冻结源码解读，没有重算Q或因果消融，不把Q差异当成分数缺口的已证明原因。'
last_source_provenance=root/'results/deep_learning/stage_gain_analysis_v366_20261002/SEED20261014_SOURCE_PROVENANCE.json'
if last_source_provenance.is_file():
    provenance=json.loads(last_source_provenance.read_text(encoding='utf-8'))
    for ident in ('stage_gain_analysis_v366_20261002','pure_dl_complete_seeds_v362_fold5/seed_20261014'):
        if ident in nodes:
            nodes[ident]['evidence'].append(last_source_provenance.relative_to(root).as_posix())
            nodes[ident]['summary'] += ' 最后seed14 SOURCE已恢复best51/stop61/TOTAL79.968809511716；自身训练侧Q50严格早于51，1761行降权。此为来源阶段，不是完整8阶段或OOF成绩；SOURCE墙钟包含只读GPU融合评估重叠，非单独速度。'
if decision_file.is_file() and 'stage_gain_analysis_v366_20261002' in nodes:
    n = nodes['stage_gain_analysis_v366_20261002']
    n['evidence'].append(decision_file.relative_to(root).as_posix())
    n['conclusion'] += ' 原与三个完整新模型阶段对照：新SW351均无有效增益；原模型去351损失TOTAL0.018151。优先固定七阶段末端消融，再考虑SOURCE直接合并PERM154/336；尚无通用无损删法，合并未实现或训练。'
proposal_file = root/'results/deep_learning/stage_gain_analysis_v366_20261002/FUTURE_CONTINUOUS_READOUT_PROPOSAL.md'
if proposal_file.is_file():
    node('continuous_aux_readout_idea_v367_20261002','连续值辅助读出：未实施方案',
         '保留原有效POR151父预测器与341正收益结构；同一追加阶段内联合新编码器、主分布读出与训练期连续值辅助读出。未来aux0/aux01同结构容量初始化，只差辅助系数0/0.1；非默认真实训练标签监督，辅助输出不参加预测，不宣称Cauchy期望。',
         '仅机制记录，训练入口未实现、未训练、收益尚未证实；当前五种子八阶段批次不变，本文不排队启动。max8阶段/每头最多一次追加，strict第五80.8/fullOOF82仍未完成。',
         stage='backbone',status='prepared',virtual=True,
         evidence=[proposal_file.relative_to(root).as_posix()],
         resultText='未实现 / 未训练 / 尚无成绩或收益')
    edge('stage_gain_analysis_v366_20261002','continuous_aux_readout_idea_v367_20261002',
         'hypothesis','由实际默认/连续收益分解提出；非已验证增益或已执行训练血缘')
    implementation = root/'results/deep_learning/por_continuous_aux_v367_20261002'
    implementation_files = [implementation/'README.md', implementation/'PLAN.json',
                            root/'experiments/pure_dl/por_continuous_aux_v367.py',
                            root/'experiments/pure_dl/train_por_continuous_aux_v367.py']
    if all(f.is_file() for f in implementation_files):
        n = nodes['continuous_aux_readout_idea_v367_20261002']
        n['title'] = '连续值辅助读出：入口已准备'
        n['evidence'].extend(f.relative_to(root).as_posix() for f in implementation_files)
        recorder = root/'experiments/pure_dl/record_por_continuous_aux_v367.py'
        if recorder.is_file():
            n['evidence'].append(recorder.relative_to(root).as_posix())
            n['summary'] += ' 终点CPU组装入口已准备：固定原PERM336/SW351，实际三通道144665点评分与DEFAULT/CONT分解，配对核对同容量配置，禁止自动采用或补折；尚未执行。'
        n['summary'] += ' 实现准备：同容量aux0/aux01，49参数训练期Linear(48,1)，非默认训练POR z的Q加权Huber监督；lambda0/0.1，aux×mean(Q)/3。编码器与主读出共同训练，辅助输出不进推理。'
        n['conclusion'] = '新模块和训练入口已准备并通过语法检查；没有forward/GPU预检/训练或成绩，收益尚未证实。PLAN.launch_enabled=false且训练入口要求五种子完整批次先完成；不排队、不改当前批次。保留原151父函数，替代341追加阶段，整体仍8阶段。'
        n['resultText'] = '入口已准备 / 未训练 / 尚无成绩或收益'
pair_study = root/'results/deep_learning/pairwise_complete_models_v368_20261002'
if (pair_study/'PLAN.json').is_file():
    pair_result = pair_study/'FINAL.json'
    pair_count=10 if (pair_study/'AUGMENT_PLAN.json').is_file() else 6
    pair_summary = f'{pair_count} native probability-mixture pairs of complete eight-stage models; original plus seeds20261010/11/12, additionally seed20261013 when completed during evaluation. Per-head fixed ratios on 0:0.01:1; one ratio for all rows, ties prefer greater A weight. Reuse original+seed10 verified grid; native forwards shared for fresh pairs. No extra training stages; 16 historical stages per pair. Seed14 incomplete and excluded.'
    pair_conclusion = 'Evaluation pending; no measured gain yet. Current seed training continues, maximum two GPU tasks. Same-fifth weight tuning is diagnostic only; no fullOOF, other folds or automatic adoption.'
    kw = dict(stage='vote',status='running',virtual=True,evidence=[(pair_study/'README.md').relative_to(root).as_posix(),(pair_study/'PLAN.json').relative_to(root).as_posix()],resultText='Pending / no score yet')
    if pair_result.is_file():
        r=json.loads(pair_result.read_text(encoding='utf-8'))
        primary=dict(scope='第五折 · 配对与权重调参诊断',value=r['scores']['TOTAL'],scores=r['scores'],source=pair_result.relative_to(root).as_posix(),kind='diagnostic',fold=5)
        kw.update(status='diagnostic',primary=primary,results=[primary],resultText=f"Best pair diagnostic {r['scores']['TOTAL']:.12f}")
        kw['evidence'].extend([pair_result.relative_to(root).as_posix(),(pair_study/'RESULTS.md').relative_to(root).as_posix()])
        pair_conclusion=f"{r['pairs']} pairs completed; best {r['best_pair']} TOTAL{r['scores']['TOTAL']:.12f}, strict>80.74={r['strictly_above_8074']}. Native disk replays/endpoints exact,144665 rows covered once, actual three-head assembly. Same fifth chosen pair/ratios; no independent generalization/fullOOF or automatic adoption. Training continues."
        gain_file=pair_study/'GAIN_INTERPRETATION.json'
        if gain_file.is_file():
            gain=json.loads(gain_file.read_text(encoding='utf-8'))
            kw['evidence'].append(gain_file.relative_to(root).as_posix())
            pair_conclusion += ' Equal50:50 TOTAL'+str(gain['equal_weight_TOTAL'])+' also exceeds80.74. CONT delta vs original '+json.dumps(gain['continuous_delta_vs_original'])+'; POR/SW CONT decrease, PERM CONT improves, not improvement of all continuous heads.'
    node('pairwise_complete_models_v368_20261002',f'{pair_count}对模型原生分布融合',pair_summary,pair_conclusion,**kw)
    edge('per_head_seed_weights_v365_20261002','pairwise_complete_models_v368_20261002','comparison','Extend the verified fixed per-head distribution-weight grid to all six pairs of four complete models.')
    for seed in ((20261010,20261011,20261012,20261013) if pair_count==10 else (20261010,20261011,20261012)):
        edge(f'pure_dl_complete_seeds_v362_fold5/seed_{seed}','pairwise_complete_models_v368_20261002','fusion','Completed native eight-stage model supplies predictions, no row-wise selection.')
depth_study=root/'results/deep_learning/perm_depth_residual_v369_20261002'
if (depth_study/'PREPARATION.json').is_file():
    node('perm_depth_residual_v369_20261002','PERM小型局部深度骨干残差',
         '保留336稀疏原曲线重采样、有理表征/读出与低秩交互设计；在stem交互后添加48→12→48零up初始化的有界残差，读取[-8,-4,0,4,8]有效邻点并归一化。新骨干、适配器和主分布读出共同训练。point/local共享参数数与初始函数，仅改变连通；point核功能冗余，不冒称同等有效容量。',
         '模型/80轮pat10→14训练入口/终点CPU三头组合已准备且语法通过；未forward、未数值预检、未训练、无成绩。原154初始化，禁止载入已训练336；固定POR341/SW351，替换一个PERM追加阶段，总8阶段。当前五种子批次不变，launch_enabled=false且不排队；完整OOF>82未完成。',
         stage='backbone',status='prepared',virtual=True,resultText='已准备 / 未训练 / 尚无收益',
         evidence=[(depth_study/name).relative_to(root).as_posix() for name in ('README.md','PLAN.json','PREPARATION.json')]+[
             'experiments/pure_dl/perm_depth_residual_v369.py','experiments/pure_dl/train_perm_depth_residual_v369.py','experiments/pure_dl/record_perm_depth_residual_v369.py'])
    edge('pure_dl_perm_low_rank_cross_v336_fold5/cross','perm_depth_residual_v369_20261002','code','只继承有效设计代码，不加载已训练336权重；新残差与编码器/读出在同一追加阶段联合训练。')
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional','perm_depth_residual_v369_20261002','evolution','唯一学习参数父模型154；同fold5/数据/Q，完整配方共8阶段且每头最多1追加阶段。')
    # These two directly affected historical parents used stage labels absent
    # from the eight display lanes. Preserve scores, normalize only this scope.
    depth_ids={'pure_dl_perm_low_rank_cross_v336_fold5/cross',
               'pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',
               'perm_depth_residual_v369_20261002'}
    for ident in depth_ids:
        if ident in nodes:nodes[ident]['stage']='backbone'
    depth_seen=set();depth_edges=[]
    for relation in edges:
        if relation['source'] in depth_ids and relation['target'] in depth_ids:
            key=json.dumps(relation,ensure_ascii=False,sort_keys=True)
            if key in depth_seen:continue
            depth_seen.add(key)
        depth_edges.append(relation)
    edges[:]=depth_edges
prepared_launcher=root/'experiments/pure_dl/launch_two_prepared_routes.py'
if prepared_launcher.is_file():
    for ident,study_name in (('continuous_aux_readout_idea_v367_20261002','por_continuous_aux_v367_20261002'),
                             ('perm_depth_residual_v369_20261002','perm_depth_residual_v369_20261002')):
        launcher_record=root/'results/deep_learning'/study_name/'NEXT_LAUNCH_ENTRY.json'
        if ident in nodes and launcher_record.is_file():
            nodes[ident]['evidence'].extend([prepared_launcher.relative_to(root).as_posix(),launcher_record.relative_to(root).as_posix()])
            nodes[ident]['summary'] += ' Latest user plan holds these routes. Previous two-route launch entry is retired unexecuted; finish five seeds then V370 original+seed13 fullCV, not a new optimization launch.'
pair_fullcv=root/'results/deep_learning/two_seed_fullcv_v370_20261002'
if (pair_fullcv/'PLAN.json').is_file():
    state_file=pair_fullcv/'BATCH_STATE.json'
    state=json.loads(state_file.read_text()) if state_file.is_file() else {}
    node('two_seed_fullcv_v370_20261002','Two-seed original grouped fullCV',
         'Latest explicit user authorization: retain original+seed20261013 pair, train original folds1-4 with same eight-stage architecture/loss/decoder; reuse verified fifth. Each fold scans one ratio per target and also saves fifth-fixed POR.56/PERM.44/SW.68 OOF. Two models have16 historical training stages. Fold-local normalization/amplitude/Q; private source weights, no18-stage replacement.',
         'Current status '+str(state.get('status','prepared'))+'; no new fullOOF score yet. Foldwise best weights are tuning diagnostic. Pair and fixed ratios already selected on fifth, disclose that boundary. New SOURCE/head max80/pat10->14 uses latest budget; historical fifth old patience unchanged. V367/V369 held.',
         stage='vote',status='running' if state.get('status')=='running' else 'failed_pending_resume' if state.get('status')=='failed_or_stopped' else 'complete' if state.get('status')=='complete' else 'prepared',virtual=True,
         primary=None,results=[],resultText='FullCV pending; fifth selection diagnostic80.863458 reused',
         evidence=[(pair_fullcv/f).relative_to(root).as_posix() for f in ('README.md','PLAN.json','PREPARATION.json')])
    edge('pairwise_complete_models_v368_20261002','two_seed_fullcv_v370_20261002','evolution','User authorized original+seed13 remaining folds after finite five-seed training completed; two weight-result scopes kept separate.')
if 'two_seed_fullcv_v370_20261002' in nodes:
    fullcv_node=nodes['two_seed_fullcv_v370_20261002']
    completed_folds=[]
    for fold_report in sorted(pair_fullcv.glob('fold_*/FOLD_REPORT.json')):
        verified_fold=json.loads(fold_report.read_text(encoding='utf-8'))
        completed_folds.append(dict(fold=verified_fold['fold'],fixed=verified_fold['fixed_scores']['TOTAL'],diagnostic=verified_fold['diagnostic_scores']['TOTAL']))
        fullcv_node['summary'] += ' Completed native fold '+str(verified_fold['fold'])+': fixed TOTAL '+str(verified_fold['fixed_scores']['TOTAL'])+'; per-head tuned diagnostic '+str(verified_fold['diagnostic_scores']['TOTAL'])+'; best weights '+json.dumps(verified_fold['per_head_best_weights'],ensure_ascii=False)+'. Native replay and output readback exact. This is a fold result, not fullOOF.'
        fullcv_node['evidence'].append(fold_report.relative_to(root).as_posix())
    if completed_folds:
        fullcv_node['resultText']='FullCV pending; completed fold scores '+json.dumps(completed_folds,ensure_ascii=False)+'; tuned values are diagnostics'
    fold_summary=pair_fullcv/'FOLD_RESULTS.md'
    if fold_summary.is_file():
        fullcv_node['evidence'].append(fold_summary.relative_to(root).as_posix())
    stage_gain_summary=pair_fullcv/'STAGE_GAINS.md'
    if stage_gain_summary.is_file():
        fullcv_node['evidence'].append(stage_gain_summary.relative_to(root).as_posix())
        fullcv_node['summary'] += ' Compact actual adjacent-stage gains are preserved separately from optimizer-initial gains; they are not stage-removal/retraining ablations.'
    snapshot_file=pair_fullcv/'LIVE_OBSERVATION.json'
    if snapshot_file.is_file() and state.get('status')=='running':
        live=json.loads(snapshot_file.read_text())
        fullcv_node['summary'] += ' Process-verified running snapshot: '+json.dumps([dict(identity=j['identity'],fold=j['fold'],stage=j['stage'],pid=j['pid'],alive=j['alive'],epoch=j['runtime'].get('epoch'),best=j['runtime'].get('best_score',j['runtime'].get('best'))) for j in live.get('jobs',[])],ensure_ascii=False)+'. Running scores are snapshots, not terminal model/fullOOF scores.'
        fullcv_node['evidence'].append(snapshot_file.relative_to(root).as_posix())
    for prefix_file in sorted(pair_fullcv.glob('fold_*/*_PREFIX_SNAPSHOT.json')):
        prefix=json.loads(prefix_file.read_text())
        prefix_scope='complete single-fold model, not fullOOF' if len(prefix['completed_prefix'])==8 else 'prefix only, not full model or OOF'
        fullcv_node['summary'] += f" Restored terminal stages {prefix['identity']}/fold{prefix['fold']}: {len(prefix['completed_prefix'])}/8 stages, TOTAL {prefix['prefix_scores']['TOTAL']:.12f}; {prefix_scope}."
        fullcv_node['summary'] += ' Actual incremental gains: '+json.dumps([dict(stage=g['stage'],target_gain=g['actual_target_gain'],TOTAL_contribution=g['actual_TOTAL_contribution']) for g in prefix['stage_gains']],ensure_ascii=False)
        fullcv_node['evidence'].append(prefix_file.relative_to(root).as_posix())
    recovery_file=pair_fullcv/'OOM_RECOVERY_PREPARED.json'
    if recovery_file.is_file():
        recovery=json.loads(recovery_file.read_text())
        fullcv_node['summary'] += ' Two first-fold PERM workers failed with CUDA OOM in validation. Complete last states at original PERM336 epoch9 and seed13 PERM154 epoch7 were preserved; read-only single-GPU original-batch best replay exact144621 rows for both.'
        serial_launch=pair_fullcv/'SERIAL_RECOVERY_LAUNCH.json'
        if serial_launch.is_file():
            fullcv_node['summary'] += ' User explicitly authorized checkpoint recovery. Same-recipe serial controller launched with maximum1 GPU worker; completed stages reused, only the two exact failed states authorized once. Any subsequent failure stops further launches. Current state is '+str(state.get('status'))+'.'
            fullcv_node['evidence'].extend([serial_launch.relative_to(root).as_posix(),(pair_fullcv/'OOM_RECOVERY_AUTHORIZATION.json').relative_to(root).as_posix()])
            terminal_recovery=pair_fullcv/'PAIR_RECOVERY_TERMINAL_VERIFIED.json'
            if terminal_recovery.is_file():
                recovery_result=json.loads(terminal_recovery.read_text(encoding='utf-8'))
                if recovery_result.get('both_authorized_recoveries_completed'):
                    fullcv_node['summary'] += ' Both authorized PERM recoveries completed normally: next epochs10/8, stopped14/best0, disk predictions and frozen weights exact; no optimizer gain. Original fold1 complete8-stage TOTAL '+str(recovery_result['original_fold1_scores']['TOTAL'])+'; still not fullOOF.'
                    fullcv_node['evidence'].append(terminal_recovery.relative_to(root).as_posix())
        else:
            fullcv_node['summary'] += ' No training resumed. Serialized same-recipe recovery prepared; explicit user authorization pending under study AGENTS rule.'
        fullcv_node['evidence'].extend([recovery_file.relative_to(root).as_posix(),(pair_fullcv/'OOM_FAILED_BATCH_SNAPSHOT.json').relative_to(root).as_posix()])
    sw_failure=pair_fullcv/'OOM_SW351_FOLD4_PREPARED.json'
    if sw_failure.is_file():
        sw_recovery=json.loads(sw_failure.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Serial seed13/fold4 SW351 CUDA OOM: full state through epoch11, best3/SW74.674643672407 preserved. Read-only144513-row best replay exact. Failed-state scores are not completed8-stage model results.'
        sw_launch=pair_fullcv/'SW351_FOLD4_RECOVERY_LAUNCH.json'
        if sw_launch.is_file():
            fullcv_node['summary'] += ' User directly approved this SW351 checkpoint recovery; unchanged recipe and logical batch64, full optimizer/scheduler/RNG restoration, maximum1 GPU. Recovery launched once; future failures still stop. Current batch status '+str(state.get('status'))+'.'
            fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('OOM_SW351_FOLD4_AUTHORIZATION.json','SW351_FOLD4_RECOVERY_LAUNCH.json','SW351_FOLD4_RECOVERY_ACTIVATION_QA.json') if (pair_fullcv/f).is_file()])
        else:
            fullcv_node['summary'] += ' Guarded single-GPU unchanged-recipe recovery prepared; direct authorization required, prior first-fold PERM authorization does not apply. No automatic retry.'
        fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('OOM_SW351_FOLD4_PREPARED.json','OOM_SW351_FOLD4_FAILED_BATCH_SNAPSHOT.json','SW351_FOLD4_RECOVERY_READBACK.json','resume_sw351_fold4_serial_v370r5.py')])
    maintenance=pair_fullcv/'MAINTENANCE_PREPARATION.json'
    if maintenance.is_file():
        fullcv_node['evidence'].extend([maintenance.relative_to(root).as_posix(),'experiments/pure_dl/compact_source_diagnostics_v370.py'])
        fullcv_node['summary'] += ' Finite CPU maintenance of completed new source diagnostics: exact epoch/Q retained, latest full observation and all best/last weights/input/prediction/exported-Q dependencies untouched. No GPU/new training or old checkpoint deletion.'
    safe_overlap=pair_fullcv/'SAFE_OVERLAP_ACTIVATION_QA.json'
    if safe_overlap.is_file():
        overlap=json.loads(safe_overlap.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Latest explicit user instruction restores maximum2 GPU workers. CPU-only handoff preserved the exact active GPU process without restart; second worker actually launched and both process identities verified. Current dispatcher excludes concurrent SOURCE/SW/scan/PERM families and requires7GiB driver free before launch. Model training code and recipe unchanged; any new failure stops future launches, no automatic retry. Activation snapshot '+json.dumps(overlap['active_jobs'],ensure_ascii=False)+'.'
        fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('SAFE_OVERLAP_AUTHORIZATION.json','SAFE_OVERLAP_HANDOFF.json','SAFE_OVERLAP_ROOT_LAUNCH.json','SAFE_OVERLAP_ACTIVATION_QA.json')])
    overlap_timing=pair_fullcv/'PARALLEL_TIMING_SNAPSHOT.json'
    if overlap_timing.is_file():
        timing_snapshot=json.loads(overlap_timing.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Observational same-fold SOURCE epoch5->15 timing '+json.dumps(timing_snapshot['SOURCE_epoch_intervals'][-1])+'. SW reference uses a proven solo terminal, not earlier overlapped original/fold1. Seeds/folds/Q differ; individual latency rises during overlap, overall acceleration remains unproven until complete same-work batch wall. No extra training benchmark.'
        fullcv_node['evidence'].append(overlap_timing.relative_to(root).as_posix())
    serial_drain=pair_fullcv/'SERIAL_DRAIN_ACTIVATION_QA.json'
    if serial_drain.is_file():
        drain=json.loads(serial_drain.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Latest user override: let current overlapping workers finish, then strictly serial. CPU-only handoff preserved both exact worker processes. Dispatcher gate forbids any new launch while a worker remains active; maximum future GPU workers1. Recipe and source hashes unchanged; no failure retry. Activation snapshot '+json.dumps([dict(identity=j['identity'],fold=j['fold'],stage=j['stage'],pid=j['pid']) for j in drain['active_jobs']])+'.'
        fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('SERIAL_DRAIN_AUTHORIZATION.json','SERIAL_DRAIN_ROOT_LAUNCH.json','SERIAL_DRAIN_ACTIVATION_QA.json')])
    first_solo=pair_fullcv/'SERIAL_DRAIN_FIRST_SINGLE_WORKER.json'
    if first_solo.is_file():
        solo=json.loads(first_solo.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Actual transition verified: original fold2 POR341 is terminal, only the original surviving SOURCE process remains live, no new PERM336 launched while it runs. Current future cap1; controller alive. This verifies draining policy in execution, not only configuration. Snapshot process '+json.dumps(solo['actual_GPU_pids'])+'.'
        fullcv_node['evidence'].append(first_solo.relative_to(root).as_posix())
    for source_pair_file in sorted(pair_fullcv.glob('FOLD*_SOURCE_PAIR_PROVENANCE.json')):
        source_pair=json.loads(source_pair_file.read_text(encoding='utf-8'))
        if 'sources' in source_pair:
            fullcv_node['summary'] += ' Completed SOURCE pair fold'+str(source_pair['fold'])+': '+json.dumps({identity:dict(scores=value['scores'],best_epoch=value['best_epoch'],Q_epoch=value['Q']['epoch']) for identity,value in source_pair['sources'].items()})+'. Same training/validation rows and input normalization, different learned source weights/private Q; Q strictly before own best. SOURCE scores are prerequisites, not full models or fullOOF.'
        fullcv_node['evidence'].append(source_pair_file.relative_to(root).as_posix())
    compression=pair_fullcv/'STAGE_COMPRESSION_ANALYSIS.json'
    if compression.is_file():
        compression_data=json.loads(compression.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Completed-model adjacent-stage compression priorities from '+str(len(compression_data['cases']))+' fold/model cases. These are correlated observations, not causal pruning ablations or full OOF. No lossless8-to5 proof; current recipe unchanged. Candidate comparisons are prepared as hypotheses only and not launched during fullCV.'
        fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('STAGE_COMPRESSION_ANALYSIS.md','STAGE_COMPRESSION_ANALYSIS.json')])
    action_diagnosis=pair_fullcv/'CONTINUOUS_ACTION_DIAGNOSIS.json'
    if action_diagnosis.is_file():
        action_data=json.loads(action_diagnosis.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' Saved-prediction continuous-row action diagnosis covers '+str(len(action_data['cases']))+' late-head contrasts. Original action-class transitions decompose recorded continuous-score changes exactly; uniform predictions unchanged. This is validation diagnosis only, not rowwise choice or proof of internal gate causality.'
        fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('CONTINUOUS_ACTION_DIAGNOSIS.md','CONTINUOUS_ACTION_DIAGNOSIS.json','analyze_continuous_actions.py')])
    training_audit=pair_fullcv/'audit_remaining_training.py'
    if training_audit.is_file():
        fullcv_node['evidence'].append(training_audit.relative_to(root).as_posix())
        canonical_audit=pair_fullcv/'audit_final_oof.py'
        if canonical_audit.is_file():
            fullcv_node['evidence'].append(canonical_audit.relative_to(root).as_posix())
        training_result=pair_fullcv/'FINAL_REMAINING_TRAINING_AUDIT.json'
        if training_result.is_file():
            fullcv_node['evidence'].append(training_result.relative_to(root).as_posix())
            fullcv_node['summary'] += ' Final remaining-fold checkpoint/config/load-trace provenance audit recorded; historical fifth provenance and accuracy selection scope remain separate.'
        else:
            fullcv_node['summary'] += ' Final remaining-fold checkpoint/config/load-trace provenance audit prepared only, not executed; it awaits complete canonical OOF. No verification completion claim.'
    final_report=pair_fullcv/'FINAL.json'
    final_audit=pair_fullcv/'FINAL_CANONICAL_OOF_AUDIT.json'
    if final_report.is_file() and final_audit.is_file() and (pair_fullcv/'FINAL_REMAINING_TRAINING_AUDIT.json').is_file():
        final_data=json.loads(final_report.read_text(encoding='utf-8'))
        primary=dict(scope='Original grouped5fold OOF; fifth-selected fixed pair/ratios',value=final_data['scores']['fixed']['TOTAL'],scores=final_data['scores']['fixed'],kind='full_oof',source=final_report.relative_to(root).as_posix())
        fullcv_node.update(status='complete',primary=primary,results=[primary],resultText=f"Fixed OOF {primary['value']:.6f}; fold-tuned diagnostic {final_data['scores']['diagnostic']['TOTAL']:.6f}")
        fullcv_node['summary'] += ' Final723556-row canonical OOF audit and actual remaining-fold checkpoint/source/load/Q audit passed. Fixed OOF '+str(primary['value'])+'; original8-stage '+str(final_data['scores']['original']['TOTAL'])+'; new8-stage '+str(final_data['scores']['seed13']['TOTAL'])+'. Two-model ensemble has16 historical stages, not8 total. Fold-tuned '+str(final_data['scores']['diagnostic']['TOTAL'])+' is diagnostic only. Fifth selection and historical schedule differences remain disclosed; unknown test accuracy is not verified.'
        fullcv_node['evidence'].extend([(pair_fullcv/f).relative_to(root).as_posix() for f in ('FINAL.json','FINAL_CANONICAL_OOF_AUDIT.json','FINAL_REMAINING_TRAINING_AUDIT.json','FINAL_SOURCE_LINEAGE_REVIEW.json','FINAL_SUMMARY.md','FOLD4_PAIRED_NATIVE_QA.json') if (pair_fullcv/f).is_file()])
    delivery_proof=pair_fullcv/'DELIVERY_VERIFICATION.json'
    if delivery_proof.is_file():
        delivered=json.loads(delivery_proof.read_text(encoding='utf-8'))
        fullcv_node['summary'] += ' User-requested foldwise-tuned82.194651 submission exported to new sibling directory submission_pure_dl_v370_82_194651_20261002. Full portable723556-row OOF predictions exact; independent extracted CLI predicts10 test wells/95948rows with original workspace and training reads blocked, result bytes exact, final ZIP CRC/manifests passed. This delivered score remains tuning diagnostic; fixed primary OOF stays82.167418. Unknown test accuracy not verified.'
        fullcv_node['evidence'].append(delivery_proof.relative_to(root).as_posix())
last_seed_pair=root/'results/deep_learning/original_last_seed_fusion_v371_20261002'
if (last_seed_pair/'FINAL.json').is_file():
    r=json.loads((last_seed_pair/'FINAL.json').read_text())
    primary=dict(scope='Fifth-fold weight tuning diagnostic',value=r['per_head_scores']['TOTAL'],scores=r['per_head_scores'],kind='diagnostic',fold=5,source=(last_seed_pair/'FINAL.json').relative_to(root).as_posix())
    node('original_last_seed_fusion_v371_20261002','Original + last seed20261014 native mixture',
         'Additional user-requested fixed fifth native probability-mixture grid0:0.01:1. Equal/common/per-head best; one ratio per target shared over all rows. No new training,16 historical stages per pair.',
         f"Per-head diagnostic {r['per_head_scores']['TOTAL']:.12f}, original delta {r['delta_vs_original']:+.12f}, original+seed13 delta {r['delta_vs_original_seed13']:+.12f}. Native disk replay/endpoints/coverage and actual three-head disk readback exact. FullCV pair remains original+seed13, no automatic change.",
         stage='vote',status='diagnostic',virtual=True,primary=primary,results=[primary],resultText=f"Diagnostic {r['per_head_scores']['TOTAL']:.6f}",
         evidence=[(last_seed_pair/f).relative_to(root).as_posix() for f in ('README.md','PLAN.json','FINAL.json','RESULTS.md')])
    edge('pairwise_complete_models_v368_20261002','original_last_seed_fusion_v371_20261002','comparison','Extend only original+newly completed lastseed14; remaining-fold pair unchanged.')
    edge('pure_dl_complete_seeds_v362_fold5/seed_20261014','original_last_seed_fusion_v371_20261002','fusion','Completed private eight-stage last seed supplies native predictions; no rowwise/stratum choice.')
ablation=root/'results/deep_learning/base_only_interface_ablation_20261003'
if (ablation/'ABLATION_RESULT.json').is_file():
    a=json.loads((ablation/'ABLATION_RESULT.json').read_text(encoding='utf-8'))
    primary=dict(scope='Original grouped5fold base-only frozen ablation OOF',value=a['base_only_scores']['TOTAL'],scores=a['base_only_scores'],kind='full_oof',source=(ablation/'ABLATION_RESULT.json').relative_to(root).as_posix())
    node('base_only_interface_ablation_20261003','Frozen base-only output interface ablation','Remove added categorical encoder/readout and state corrections; retain original seed POR151/PERM154/SW104 priors, matched distribution family, no training.', 'Full723556-row OOF81.937732 vs full81.989553, delta-0.051821; all3 heads decline. Not a pure interface-format test; A-board effect unverified. Test10wells95948rows per-well inference, old deliveries unchanged.', stage='head',status='complete',virtual=True,primary=primary,results=[primary],resultText='Base-only OOF81.937732; delta-0.051821',evidence=[(ablation/n).relative_to(root).as_posix() for n in ('README.md','ABLATION_RESULT.json','OOF_VERIFICATION.json','BASE_EXTRACTION_AUDIT.json','delivery_validation.json')])
    edge('two_seed_fullcv_v370_20261002','base_only_interface_ablation_20261003','comparison','Original-seed full8-stage model versus retained base priors only; no new seed fusion, no training.')

# Deleted raw records now point to their retained compact historical summary.
for annotation in nodes.values():
    updated=[]
    for reference in annotation.get('evidence',[]):
        item=root/reference
        if not item.is_file():
            parent=item.parent
            while parent.is_relative_to(root) and parent!=root:
                if (parent/'SUMMARY_RESULTS.json').is_file():
                    reference=(parent/'SUMMARY_RESULTS.json').relative_to(root).as_posix();break
                parent=parent.parent
        updated.append(reference)
    annotation['evidence']=list(dict.fromkeys(updated))
(path/'curation.json').write_text(json.dumps({'nodes':nodes,'edges':edges,'deadends':dead,'terms':terms},ensure_ascii=False,indent=2),encoding='utf8')
