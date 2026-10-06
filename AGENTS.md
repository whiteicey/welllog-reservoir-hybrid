# Pure-DL workspace rules

## Latest goal edit: strict fifth-fold >80.5

- Latest added objective: optimize POR/PERM/SW separately, including output-head architectures and optimization methods. Use independent channel losses/optimizers/stopping/checkpoint selection; keep useful channel gains even when other channels do not improve. Validate any final fixed head assembly explicitly. Maintain the frozen whole-ABD feature backbone/no legacy-output voting on the structural route, and do not change delivered artifacts.

- User raised the screening threshold to STRICT >80.5, superseding historical80.4 for future decisions. New trainers use fold5_screening_805.py; preserve old active/completed source/config hashes.
- Full original grouped5fold OOF>82 remains unachieved. Remaining folds require clear gain as well as the threshold; the user-requested no-additional-fold delivery and current deferred continuation remain unchanged. Do not auto-promote the tiny80.501552 improvement merely because it barely crosses80.5.
- Keep max80/patience18/min10, actual per-fold/per-head best checkpoint, normalCUDA, max2GPU, no test-label use. Current feature-backbone run continues unchanged; delivered result/code archives immutable.

## Latest baseline and ensemble-aware training

- LATEST structural clarification: keep the active A6-head experiment running, stop the B-own-finetune route, and start a separate model whose BACKBONE consists of frozen A6+B43+D52 feature extractors. Concatenate the five branch features before original prediction heads and train independent new target heads. Original model predictions/routers must NOT be added/voted into this new model's answer. Select each target head by its own score and validate their fixed assembly. This new structural route differs from the retained A6-head + frozen B/D voting control. No remaining folds, max2GPU.

- LATEST OVERRIDE: user stopped old routes and now explicitly prefers OWN-model/channel training loss with fusion ONLY in validation. Warm-start B43 fine-tuning uses B own predictive NLL, fixed A6/D52 validation. Independent-head experiment must use A6 (the A checkpoint from80.4923), freeze its two backbones/calibration/original routes, train each head's own target loss, and select each head by that channel's ABD fused validation score with B43/D52 fixed. Supersedes the immediately older mandatory fused TRAINING loss. All old tasks/queue stopped; new outputs/directories, no source mutation or fullCV launch.

- Latest user clarification: training itself must use the final three-model fused predictive loss with the other two models frozen, not merely ensemble-aware validation selection. For current B-from-scratch route, freeze A6/D, differentiate final equal A/B/D likelihood only into B; select by fused validation TOTAL. Existing solo-loss B run is a historical/control experiment, not the requested production recipe. Retain documented original auxiliary supervision; do not silently add channel balancing while correcting this objective.

- Latest user steering: temporarily DO NOT continue remaining folds. Keep full-CV tools prepared but do not launch them under the persistent goal until the user resumes that step. Current authorized paired study starts from verified ABD80.49230853582785: jointly update A/B/D versus freeze best A(epoch6) and D and update B. Same initial ensemble/loss/LR/data for a matched incremental comparison; from-scratch joint training is a separate untested question. Overall >82 goal remains active, not paused or achieved.

- User adopts A+B+D as the baseline: original posterior dual seed20261006 (A), posterior dual seed20261008 (B), and single global-cosine (D). Fixed fifth TOTAL=80.40915186344968; equal posterior distributions, four cosine windows, unchanged fine decoder. This supersedes the four-source80.403220 and five-source80.386892 development baselines, not historical evidence.
- When improving A, freeze B/D and validate A+B+D every epoch. Prefer training on fused prediction loss with gradients only into A; select the checkpoint by fused official TOTAL. Solo scores are diagnostic and must not veto an ensemble improvement. Keep candidate best weights until fusion assessment, even if solo score is below80.4.
- Improvement means positive delta against this identical ABD baseline; remaining folds require fused fifth>80.4 AND positive gain. A selected fifth-fold baseline is not independent full OOF evidence. Preserve normal CUDA, max2 GPU jobs, max80/patience18/min10, original groups and data isolation.

- Goal: maintain reproducible pure neural well-log models predicting POR/PERM/SW. Best complete verified OOF is81.71323318479078;82 target unmet.
- User stop directive: last posterior-router run is FINISHED; start NO new training/continuation unless explicitly requested. Repository cleanup/publication does not authorize more training.
- Keep PROJECTS.md current after each batch; describe completed work, next/pending items and achieved/unachieved objectives.
- Never use test labels, tree predictions, pretrained weights, pseudo-labels, rowwise model selection or OOF-fitted blending. Preserve original groupedfolds/populations/seed/scoring.
- Future trials, only if requested: fixedfifth strict>80.4 before remainingfolds, max80/patience18/min10, actualbestcheckpoint, max2GPU. Historical frozen experiments retain original settings.
- Do not modify frozen experiments while running or silently invalidate recorded hashes. Portable wrappers may redirect paths/admission only and must record it.
- Record each experiment focus, evidence, gain and abandoned directions. See docs/deep_learning and results/deep_learning; no redundant per-trial preflight/review.
- Source and compact summaries only in Git; exclude data, weights, raw predictions, ZIPs, logs and caches. Keep key/milestone models locally; delete redundant failed artifacts rather than move to C.
- Every delivery uses a NEW directory; existing delivered artifacts are immutable.
- Use normal CUDA PyTorch locally. Never stop training merely for disk cleanup.


## Explicit user resumption
The user explicitly requests continuing the80.2508937748 posterior-router direction. This supersedes the previous stop-after-last-run instruction for newly authorized exploration. Preserve fixedfifth strict>80.4, max80/patience18/min10/bs64/64steps,normalGPU,max2jobs,originalsplits/seeds/scoring,no testlabels. Currenttrial local+posterior router only; deliveries immutable.

## Updated gain criterion for exploration
The user now also accepts a mechanism as useful when a fixed, fair ablation shows positive gain even if its absolute fifth-fold score is below the current SOTA. Predefine and report both the same-capacity ablation delta and the absolute score; do not call an early or partial comparison a final gain. Such a trick may be carried into a new isolated experiment. This does not relax the strict original fifth-fold >80.4 gate before training remaining folds or the >82 full-OOF goal.

## Current feature-first direction
User now directs exploration back to the original two-expert posterior router fifth score80.25089377481923. Prior three-member point41 versus uncalibrated68 ablation showed a local+0.1113103704 view benefit but both remained below original twoexpert and strict80.4; do not extend the triple route. New trials should test input feature mechanisms on the original twoexpert architecture, preserving original nine curves, fixed fifth/seed/scoring,80/18 and max2GPUjobs. Fair mechanism ablation gain can justify another isolated trial even if it is below SOTA; remainingfolds still require fifth>80.4.

## Explicit validation-grid authorization
User explicitly permits grid search of fixed expert voting weights against fold5 validation labels. This overrides the older no OOF-fitted blending rule for this isolated diagnostic. Report tuned-validation scores honestly; do not claim independent OOF or auto-promote on this evidence alone.

## Latest user direction: finish active ideas, return to80.39
Finish the already-running full-expert RMS normalization experiment (session76251). Do not launch additional voting/window-weight/decoder-resolution or optimizer sweeps. Then use frozen five-model learned-vote recipe80.38689192147532 (rounded80.39) as the comparison baseline and prioritize feature engineering and neural architecture. This is an ensemble score, not one backbone checkpoint. Preserve its five source checkpoints, learned router.pt, preprocessing and cosine4-window/fine-decoder recipe. First screen isolated feature/architecture changes on original fixed fifth; assess fusion benefit only with evidence. Original9 curves/206-feature contract remains unless a separately documented derived-feature variant is justified; no DEVI/AZIM/BIT/CASE restoration. New training max80/patience18, strict>80.4 before remainingfolds, original fullOOF>82 goal, max2GPU and train-only fitting unchanged. Do not start previously mentioned further learned-weight supervision routes under this direction.


## DL 演进图谱持续维护（2026-09-27 用户要求）

- 每一次新的纯 DL 尝试，都必须同步补充 `docs/deep_learning/atlas/` 演进图谱：机制重点、父节点/对照/融合关系、配置和初步结果；训练中标记快照，结束或转向时补最终结论。
- 每个节点必须显示初步结果；未训练、尚无成绩、启动失败、提前停止要明确区分。首折、第五折、完整五折 OOF、验证调参诊断不可混为同一种分数。
- 对已关闭路线建立 dead end 节点，写明实测证据、为何不继续及结论适用范围；不得把单次负结果泛化为整类方法无效。
- 大意相似的几次尝试允许归组，但必须保留子尝试的配置、结果和来源；证据不足的血缘标为推定或待确认。
- 新增/变更人工摘要与关系时编辑 `curate.py` 并运行它；每批实验后运行 `build_atlas.py` 更新离线 `index.html`、数据快照，并更新 PROJECTS.md。只记录 DL，排除树模型及树/DL 混合。


## LATEST: explicit overnight training stop
User stopped all training to sleep; tomorrow continuation requires explicit user resumption. No auto-restart, new training or queued launch. Preserve saved heads/source models and delivered ZIPs.

## 2026-09-28 Latest explicit resumption: dual80.22 independent readouts
The user resumes only head-structure optimization on the approximately80.22 dual-expert method. Retained source: pure_dl_paired_local_context_router80_sequence/fold_5.pt, epoch52, fifth80.22186108303967. The compact posterior80.227015 run has no saved checkpoint. Freeze the two source backbones/calibration/internal input router; independently optimize POR/PERM/SW readouts, with no external ABD blending. Predefined first comparison: equal-parameter pointwise versus local-depth residual heads with original-function initialization. Same original source evaluation (single coverage/256bins/800actions), fold5/seed/ninecurves/206features, own channel loss, fixed source training-Q, max80/pat18/min10 and max2GPU. Explicit authorization does not resume old ABD/statistics or fullCV tasks. Strict>80.5 and clear gain remain; no extra folds or changes to delivered artifacts. Save resumable current head/optimizer/scheduler/RNG/early-stop state atomically and record all actual failures, updates and score scope in PROJECTS/README/atlas.


## 2026-09-28 第二轮独立头优化
用户继续优化的授权限定在当前双专家。先补linear-only对照，另做每目标联合两专家隐藏特征的cross80头；保持原来源、数据、Q、评分和训练预算。已有逐点/局部结果冻结不改。两任务上限，无旧ABD重启和补折。更新PROJECTS、实验README与演进图谱。


## 2026-09-28 双专家继续优化：门控与密度分离
linear/cross两项已完成，依当前继续优化授权，在同一双专家和原线性头上预先定义gate_only与density_only的参数更新范围对照；评分/数据/种子/Q/预算不变。只启动此两项，无旧路线或补折。各输出因子冻结需最终校验；分别记录通道收益，维护PROJECTS。


## 2026-09-28 本批有界POR子空间实验
继续优化授权用于三个预先固定对照：mean_only、mean_scale、density_all，全部从已验证80.257638选定头热启动，仅POR更新；PERM/SW/双骨干/路由/Q/原评分冻结。独立新优化器和80/18预算，不冒称旧轨迹续跑。最多2GPU，显式分批启动，不开自动队列/旧ABD/补折。按真实最终恢复记录结果并维护PROJECTS、README与图谱。


## 2026-09-28 当前继续：有界位置残差读出
本批实现方案限linear/point32/local32三项，从80.268734508852同起点新优化，只POR位置更新；三组共享线性位置更新，新增非线性残差限制为0.25固定分量标准差。该界仅约束新增支路。原9曲线/折/评分/Q/80-18/2GPU保持；其余头和混合尺度冻结，无外部模型或旧ABD恢复、无补折。原子完整状态保存，逐批更新PROJECTS/README/图谱，保留实际失败记录。


## 2026-09-28 用户授权复杂预测头和attention
在当前双专家80.274135基础上，本批预定义linear/deep_mlp/attention三个头对照；先优化POR位置，PERM/SW/骨干/旧残差/门控/混合尺度固定。新复杂头可使用同井两专家联合隐藏特征和两层四头局部attention，64维、33点窗口。共同线性lr3e-5，新深头lr3e-4；其余原数据/评分/Q/80-18/最多2GPU不变。三组同起点，明确分批启动，无旧路线或补折；维护PROJECTS、独立记录与图谱，不宣称复杂度本身有收益。


## 2026-09-28 当前有界优化：分量query读出v7
继续优化本批预定义center/local两项，完全冻结80.274135旧预测器（含线性位置与旧支路），仅训练分量查询交叉注意力模块。两方案结构、参数、初始化、掩码一致，仅记忆读取当前点或邻近点不同。原输入/分折/评分/Q/80-18/max2GPU保持；新模块lr3e-4，不扩大本批候选。无旧ABD重启/补折，维护PROJECTS与独立记录。


## 2026-09-28 最新授权：通道独立模型，无共享骨干
用户允许不同模型分别预测与训练POR/PERM/SW，无须共享backbone。覆盖此前仅POR/骨干冻结限制。本批六项：各目标single完整视图和dual双视图私有模型；全骨干、当前目标门控/密度和私有路由可训练，各自优化器/早停/完整checkpoint，最后固定三模型共同验证。初始化来自原训练折预训练，不声称从零。原9曲线/分折/评分/80-18/max2保持，新低学习率全骨干微调配置须记录。无额外数据/自动补折/旧任务或交付覆盖，维护PROJECTS与图谱。


## 2026-09-28 继续推进：独立SW分布模型v9
v8六项与三模型恢复完成后，本批预定义ce/ce_utility两项：SW私有冻结双专家先验+直接读取原206特征的可训练TCN概率格修正；只差训练容差效用项0/0.1，其余原数据/分折/评分/Q/80-18/max2保持。仅新TCN训练，POR/PERM独立且不更新，不把当前设计扩成自动搜索或补折。维护PROJECTS/README/EXPLORATIONS/图谱与真实失败记录。


## 2026-09-28 本轮继续优化：独立SW从头训练v10
本轮预定义tcn/bigru两项，均从原207维输入直接学习SW门控与256格连续分布；不加载旧网络或预测，复用原训练侧预处理/网格/Q以保持评分边界。POR/PERM保留独立模型。原第五折、seed20261010、80/18/min10、batch64/64步、最多2GPU；无旧任务恢复、无自动搜索/补折。每批维护PROJECTS/README/EXPLORATIONS/图谱，严格区分冷启动改善与超越旧SW。


## 2026-09-28 当前继续：SW状态/数值表征解耦v11
本批预定义shared/split两个匹配候选，各有两套独立48维TCN、同初始化和92273参数；仅改变两个头读共享均值还是各自编码器。训练与解码相同，按整个SW官方分选checkpoint。原输入/分折/Q/80-18/max2保持，POR/PERM独立且不变；训练侧梯度诊断不读验证标签，无自动搜索/旧任务/补折。维护PROJECTS、README与图谱。


## 2026-09-28 当前继续：SW联合动作效用v12
本批likelihood/joint_utility两个匹配实验，v11shared第13轮同起点，92273参数同结构；全SW参数训练，损失分别L_like与L_like+0.5*L_joint，softaction温度0.05。原动作/分折/Q/80-18/max2不改，新lr1e-4优化器重置；POR/PERM独立不变，无旧任务、补折或交付覆盖。明确默认与连续动作都参与联合项；维护PROJECTS/README/图谱，代理损失不等同精确官方argmax。


## v15用户要求拓宽网络结构：本轮执行范围
用户要求从网络结构拓宽探索。本轮方案为三种完整可训练私有SW骨干：自适应采样、潜变量读写记忆、双向门控对角状态空间。固定v14内层48/16和损失/评分/Q=1/80-18/max2，预定义三项显式分批。当前不修改POR/PERM模型，不恢复旧任务或补折；构想与完成事实分开记录。


## v16本轮推进到原第五折
继续推进最近架构候选：原样v15latent与TCN对照在原64井重新训练，Q=1，原第五折144665点选模。只读既有POR/PERM作为固定模型，结束后实际三通道合并验证。本批不运行内层或原其他四折，不覆盖旧交付。详细边界见latent_fold5_v16_20260928/AGENTS.md。


## 最新方向：回到80.274135，停止无收益替换路线
用户明确要求回到80.27继续改进。冷启动替换、继续扩容及专家训练方式的后续实验停止延伸，不恢复既有队列。本批v25只做amplitude/shape两项小型POR位置修正；所有原冠军参数固定，只训练12816新参数，原数据/Q50/分折/解码不变。无其他折和交付覆盖，维护champion_shape_v25_20260928/PROJECTS.md。

## v25保留配方
回到原80.274135后，幅值统计位置修正实测TOTAL80.295581058278（+0.021445720487），POR77.520401442467，其他通道不变。当前开发候选为outputs/pure_dl_champion_shape_v25_amplitude_fold5/model_best.pt第36轮，配套champion_shape_v25.py及v20幅值侧视图元数据；完整依赖见results/deep_learning/champion_shape_v25_20260928/CURRENT_RECIPE.json。原v5/v8检查点和交付不覆盖，负shape路线不追加，停止冷启动/扩容延伸。不自动补折，继续维护PROJECTS与独立实验记录。


## v26 最新授权：三个目标的不同输出头均可优化
用户明确不必局限某个头。当前直接从v25组合80.295581出发，预定义POR gate/weight/scale，PERM和SW各gate/weight/location/scale共11项；全部旧权重冻结，幅值条件小头独立训练/早停。每目标只采用一个完整候选并实际合并，不直接相加不同因子。max80/pat18/max2正常GPU，无冷启动/扩容/旧队列或其他折，维护all_heads_v26_20260928/PROJECTS.md。


## v26最新完成配方
原第五折TOTAL=80.300730612039，相对v25+0.005149553761。当前每目标选择见results/deep_learning/all_heads_v26_20260928/CURRENT_RECIPE.json。11项有限小头已完成，旧来源和交付保留；不自动叠加同目标因子、启动其他折或恢复冷启动/扩容。维护PROJECTS和图谱。


## v27基于80.30的后验条件增量
当前用户要求继续基于80.30优化。本批预定义三目标各self/pair六项，冻结v26全预测器，仅有效位置/门控/权重因子的小头更新；同参数同初始状态，只差第二份后验输入来源。原数据/Q50/分折/评分/80-18/max2保持，无冷启动/扩容/旧队列或其他折。维护champion_posterior_v27_20260929/PROJECTS.md。


## v27最终研究配方
原第五折TOTAL=80.300730612039，相对v26+0.000000000000。六项有限比较已完成，当前配方见results/deep_learning/champion_posterior_v27_20260929/CURRENT_RECIPE.json。旧模型/交付不改，不恢复负路线或其他折。


## v28当前80.30的路由更新范围对照
继续优化授权下，本批预定义三目标joint/conditional六项；只新增内部路由log-odds小头，原专家和v25/v26头冻结。原数据/Q50/解码/80-18/max2保持，显式分批，无其他折、冷启动、扩容和交付覆盖。详见champion_route_v28_20260929/PROJECTS.md。


## v28最新完成研究配方
TOTAL=80.324397278674，当前引用results/deep_learning/champion_route_v28_20260929/CURRENT_RECIPE.json。六项已结束，旧来源不变，无其他折和旧队列。


## v29继续当前有效模型：有限路由解耦
用户继续优化，本批固定v28 TOTAL80.324397，三目标coupled/split六项。全部父权重冻结，新路由增量±0.125，同20546参数和初始函数；原数据/Q50/评分/80-18/max2保持。不恢复冷启动/扩容，不补折或覆盖交付。维护champion_split_route_v29_20260929/PROJECTS.md。


## v29最新研究配方
当前TOTAL=80.335082954320，相对v28+0.010685675646。引用results/deep_learning/champion_split_route_v29_20260929/CURRENT_RECIPE.json；六项完成，无其他折和旧队列，原交付保留。


## v30已有有效模块协同微调
本批从v29 TOTAL80.335083出发，三目标route_only/coadapt六项；无新增学习模块，保留主骨干，仅路由或路由与已有效输出修正共同更新。统一3e-5、分组clip2，原数据/Q50/评分/80-18/max2不变。无其他折、旧队列或交付覆盖。维护champion_coadapt_v30_20260929/PROJECTS.md。


## v30当前研究配方
TOTAL=80.336211945752，相对v29+0.001128991432。当前配方results/deep_learning/champion_coadapt_v30_20260929/CURRENT_RECIPE.json。六项已结束，旧模型/交付保留，无其他折及旧队列。


## v31用户采纳诊断机制验证
基于80.336211945752，仅POR同边界tanh/anchored和SW affine/pwl4共四项。PERM冻结，原9曲线/数据/Q50/80-18/max2/解码不变。新目标变换计入完整密度Jacobian；不得将参数化变化误算成初始收益。不补折、旧交付保留。维护mechanism_checks_v31_20260929/PROJECTS.md。


## v31当前研究配方
实际TOTAL=80.336513436816；当前配方results/deep_learning/mechanism_checks_v31_20260929/CURRENT_RECIPE.json。四项已结束，无其他折或队列，旧交付保留。


## v32跨曲线局部相关特征对照
继续授权下，本批从v31 TOTAL80.336513436816开始，三目标marginal/crosscorr六项，同结构同初始化；原模型冻结，仅有限小头读取18维相关或已有统计冗余对照。原数据/Q50/评分/80-18/max2不变；暂不加显式交互，不补折、不改交付。维护crosscurve_features_v32_20260929/PROJECTS.md。


## v32当前研究配方
TOTAL=80.336513436816；配方results/deep_learning/crosscurve_features_v32_20260929/CURRENT_RECIPE.json。六项结束，无自动后续实验或补折，旧模型及交付保留。


## v33当前研究配方
四项边界统计对照未获新增收益，保持TOTAL80.336513436816及原v31权重；配方见results/deep_learning/boundary_context_v33_20260929/CURRENT_RECIPE.json。近边界困难不等于总损失集中，暂停本次边界侧统计扫描，不自动补折或修改交付。


## v34当前配方
当前TOTAL=80.336513436816；cross_target_v34_20260929/CURRENT_RECIPE.json。辅助OOF与整体泛化区别见报告，有限候选完成，不自动开其他方向/原其他折。


## v36当前配方
当前开发TOTAL=80.336513436816；配方retrieval_smooth_gate_v36_20260929/CURRENT_RECIPE.json。v35门控失活不作为充分负证据，v36已真实训练。无自动后续扫描或其他折，旧交付不变。


## v86当前有限范围
# v86目标与约束
仅POR既有anchored路由30819参数的likelihood/joint两项；原专家/骨干/位置头冻结，无扩容或旧路线。原9曲线206+已有幅值侧63/训练侧拟合/Q50/原第五折/seed/评分/Student-t3原解码保持。normal GPU/max2/80-18/min10/b64/64步；新AdamW3e-5、cos80*64到1e-5，优化器重置。原子完整状态显式恢复，维护PROJECTS/独立记录/图谱；负权重清单删除，正来源/交付保留，无其他折或自动队列。第五严格>80.5且明确收益后再考虑补折，原完整OOF>82未完成。



## v86当前研究配方
当前TOTAL=80.382093291214，变化+0.000000000000；joint-likelihood POR +0.000000000000，更新后最高差-0.000746818167。当前配方results/deep_learning/por_joint_utility_v86_20260930/CURRENT_RECIPE.json；无其他折。


## v87 PERM完整模型效用范围
# v87目标与约束
仅PERM current v29完整私有函数的likelihood/joint两项，所有PERM骨干/内部输入路由/读出及有效小头可训练，原其他目标输出行仍是冻结buffer。原206/已有幅值63/分折/训练侧统计/Q50/评分不改，默认0.01和log10官方效用、原256支持800动作解码保持。无新增容量，低LR body1e-5/head3e-5、新80/18/min10/bs64/64步/cos到1e-5，原子last与显式恢复。normal最多2GPU，以实测显存选择串/并行；无旧队列、其他折或覆盖交付。维护PROJECTS/独立探索/EXPLORATIONS/图谱，负权重清单删除，正源保留。第五严格>80.5和明确收益才考虑补折，完整原OOF>82未完成。



## v87当前研究配方
当前TOTAL=80.382093291214，变化+0.000000000000；最终joint-likelihood PERM+0.000000000000，更新后差+0.015989004499。配方results/deep_learning/perm_full_utility_v87_20260930/CURRENT_RECIPE.json；无其他折或旧队列。


## v88当前有限范围
# v88目标与约束
仅PERM冻结当前v29完整私有函数的概率格读出，两项同37185参数/同初始化，只差效用0/1。原9曲线206+已有幅值63/训练侧统计/原第五/seed/Q50/官方评分与支持动作保持。新head1e-4/80-18/min10/bs64/64步/cos到1e-5，其他目标不训练。normal/max2GPU，本批一进程共享冻结计算。原子完整可训练状态与冻结依赖，显式恢复，另存训练后最佳state/pred。维护PROJECTS/独立探索/图谱；负权重清单删除，正源/交付不改，无其他折/旧队列。原第五严格>80.5且明确收益后考虑其他折，原完整OOF>82未完成。



## v88当前研究配方
当前TOTAL=80.382093291214，变化+0.000000000000；joint-likelihood PERM +0.000000000000，更新后最高差-0.001626163619。当前配方results/deep_learning/perm_grid_readout_v88_20260930/CURRENT_RECIPE.json；无其他折。


## v89当前有限范围
# v89目标与约束
仅新SW Student-t4候选，同839481参数、v56训练折内骨干/FiLM和训练SW先验原始权重初始化；当前Gaussian v82 a100作历史对照，旧共享t3负路线不重启。std/scale一致，t4 NLL和CDF同步，原9曲线206/输入bank/分折/seed/Q50/官方支持动作不改。效用1/DropKey0.4/1e-4/80-18/min10/bs64/64步/cos到1e-5，normal/max2GPU。初始函数不强行等于Gaussian。原子完整last、训练后最佳state/pred、显式恢复；维护PROJECTS/独立记录/图谱，负权重清单删除、正源/交付保留，无其他折/旧队列。原第五严格>80.5且明确收益后考虑其余四折，完整原OOF>82未完成。



## v89当前研究配方
SW73.779643062735，对Gaussian-0.013667889753，最佳/停止14/32；采用=False，当前TOTAL80.382093291214。配方results/deep_learning/sw_student4_v89_20260930/CURRENT_RECIPE.json；无其他折。


## v90当前有限范围

# v90目标与约束
仅SW Gaussian参考/本井直接读出两项，同841906参数；原v56骨干FiLM和训练先验，全SW训练。原FiLM两组均本井，新零输出96→25头只差reference/well来源，两种code都计算。原9曲线206/输入bank/64-16/seed/Q50/评分/效用1/DropKey.4/80-18/min10/bs64/64步，normal/max2GPU，依据实测显存分批。原子完整last和训练后最佳，恢复显式。维护PROJECTS/独立记录/图谱；负项先落盘清理计划再删。无其他折/旧队列/交付覆盖，第五严格>80.5且明确收益、原OOF>82仍未完成。



## v90当前研究配方
well-reference SW+0.004915085801；采用=None，当前TOTAL80.382093291214，保留['reference', 'well']。配方results/deep_learning/sw_direct_well_v90_20260930/CURRENT_RECIPE.json；无其他折，完整OOF>82未完成。


## v91当前有限范围
# v91目标与约束
仅当前v82 Gaussian SW冻结全模型，参考/本井两独立2425参数零线性头。原FiLM用本井，两个code均计算，共享冻结推理；无DropKey训练随机化。原9曲线206/输入bank/分折/Q50/官方评分/Gaussian效用1/80-18/min10/bs64/64步/1e-4cos到1e-5不变。normal/max2GPU，本批一个进程，两头各自优化/早停/选模。原子last含current/best/优化器/scheduler/RNG/停止，恢复显式。维护PROJECTS/独立记录/图谱，清理计划先落盘。无其他折/旧队列/交付覆盖，第五严格>80.5且明确收益，完整OOF>82未完成。



## v91当前研究配方
当前TOTAL=80.382093291214，变化+0.000000000000；well-reference SW +0.000000000000，更新后最高差-0.091901136443。当前配方results/deep_learning/sw_frozen_direct_v91_20260930/CURRENT_RECIPE.json；无其他折。


## v92 POR完整模型效用范围
# v92目标与约束
仅当前v31 anchored完整POR likelihood/joint两项；两私有骨干/输入路由/实际读出和有效修正更新，未用读出副本/route anchor/其他输出行固定，无新增容量或先验重置。原9曲线206+幅值63/训练侧统计/原分折/Q50/评分/t3/log1p/官方.08/default.1保持。body1e-5/head3e-5/cos到1e-5/80-18/min10/b64/64steps，normal/max2，以实测显存显式分批。原子完整last、训练后最佳/显式恢复；维护PROJECTS/独立README/EXPLORATIONS/图谱；清理计划先落盘，正源和交付保留。第五严格>80.5且明确收益后才考虑其他折，原完整OOF>82未完成，无旧队列/交付覆盖。



## v92当前研究配方
当前TOTAL80.382093291214，采用=None；完整训练后joint-likelihood POR-0.029103461105，同轮均值-0.040396054022。配方results/deep_learning/por_full_utility_v92_20260930/CURRENT_RECIPE.json；无其他折，原OOF>82未完成。


## v93 SW完整模型效用范围
# v93目标与约束
仅SW同839481参数/原v56训练折内骨干FiLM+训练先验，两次forward/backward rho0/SAM.05同计算控制，原Gaussian/效用1/DropKey.4/训练窗口和回放mask完全相同。首未clip全局L2扰动、次梯度更新，clone精确恢复，RNG每更新消费一个forward；原9曲线206/输入bank/64-16/Q50/评分/1e-4/80-18/min10/b64/64steps/cos不改。只SW标签，无released权重/其他预测；normal/max2GPU，实测分批，原子完整last/显式恢复。维护PROJECTS/独立记录/EXPLORATIONS/图谱；删除计划先落盘，正源/旧交付保留。第五严格>80.5且明确收益后才考虑其他折，原OOF>82未完成。



## v93当前研究配方
sam005-control SW+0.109559154892；采用=sam005，当前TOTAL80.417837208128，保留['control', 'sam005']。配方results/deep_learning/sw_sam_v93_20260930/CURRENT_RECIPE.json；无其他折，完整OOF>82未完成。


## v94 SW完整模型效用范围
# v94目标与约束
仅SW SAM半径.025/.1两项，相同839481参数/v56训练折骨干FiLM+训练先验重置，历史已完成rho.05/0作对照；不加载v93已训练权重。原9曲线206/输入bank/64-16/Q50/评分/Gaussian/效用1/DropKey.4/两次计算/RNG回放/clone恢复/1e-4/80-18/min10/b64/64steps/cos不改，仅自身SW标签。强制一个GPU进程、每项显式启动，normal/最多2GPU，不自动队列/关闭别的应用。原子完整last含跨恢复peak，恢复显式；维护PROJECTS/独立记录/EXPLORATIONS/图谱，清理计划先落盘，正源和交付保留。第五严格>80.5且有收益后才考虑其他折，原OOF>82未完成，无旧任务或覆盖交付。



## v94当前研究配方
当前TOTAL80.442433883814，采用=r100，保留=['r100']。配方results/deep_learning/sw_sam_radius_v94_20260930/CURRENT_RECIPE.json；两项真实串行，无其他折，原OOF>82未完成。


## v95当前研究配方
原第五TOTAL80.472576222471，采用=separate；两同容量固定来源均已原144665输入评价，同来源control与r100精确一致。未训练/其他折，完整OOF>82未完成。配方results/deep_learning/sw_factorization_v95_20260930/CURRENT_RECIPE.json。



## v96固定atom私有密度范围
# v96目标与约束
只做likelihood/joint两项，从当前v95完整函数相同热启动，冻结r100默认atom，更新v82来源的私有SW密度网络；同SAM rho.1，两组只差效用系数0/1。原9曲线206/相对深度/128输入bank/64-16/Q50/原第五/原评分动作不变；无其他标签或按点选择。总1678962/优化器839481参数（129密度gate输出不用），normal单GPU串行、原最多2上限，80/18/min10/b64/64步/lr1e-4/cos80*64。原子完整last含current/best/optimizer/scheduler/RNG/停止与跨段显存，恢复显式。维护PROJECTS、独立记录、EXPLORATIONS、图谱；负/重复清理计划先落盘，正源/交付不改。无旧路线重启/自动其他折，完整原OOF>82未完成。



## v96当前研究配方
当前原第五TOTAL80.504197424477，采用=likelihood；两项真实恢复/评分/清理完成，未补折/完整OOF。配方results/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json。


## v97 用户明确恢复其余四折
用户答复“恢复其余四折训练”。授权限定当前v96第五80.504197完整配方的原1~4折；第五预测核验复用，其余来源/预处理/Q只能各折64训练井重训。固定18有效阶段加1来源骨干；逐目标选实际最佳checkpoint，80/18/b64/64steps，normal最多2GPU，当前执行串行。旧暂停在此范围被覆盖；其他旧队列不启动，交付不覆盖，维护PROJECTS/独立记录/图谱。完整OOF>82仍须实际723556点证明。


## v97 当前并发执行配置
原四折补全范围不变。当前以新CPU调度入口接管已有训练进程，原PID/训练代码/配置/模型hash保留；主阶段依序执行，与下一折来源预训练重叠，GPU最多2且SW阶段最多1。空闲门槛来源6.8GiB、阶段4GiB；不重跑已完成项、不启动旧路线，失败不自动重试。此执行配置覆盖此前“当前串行”的描述；依据用户已明确授权最多2GPU且恢复其余四折。维护PROJECTS/独立记录/图谱；完整原OOF>82仍须实际证明。

## v100 最新目标：最多五个总训练阶段与第五折80.8
用户要求大量精简18阶段且保留增益。新v100定义来源预训练、联合POR头、联合PERM头、SW直接联合训练、固定本批atom的密度优化，总计5阶段。来源训练计数/成本不隐去，只允许复用原双专家来源，禁止加载旧18阶段后续权重冒充紧凑训练。原输入/第五分折/Q50/自身通道损失/80-18/min10/b64/64步保持，normal最多2GPU且SW最多1，显式启动。新第五筛选门槛严格>80.8且有明确收益，覆盖旧80.5的新推广门槛；旧源码/报告/交付冻结。实际三通道组合后才判断收益或补其他折。维护PROJECTS、独立记录、EXPLORATIONS与DL图谱。

## Temporary training keep-awake runtime
New GPU trainer entrypoints use `training_awake_guard.TrainingAwake` around main to request Windows SYSTEM_REQUIRED only for the process lifetime; release on exit. Do not modify frozen trained source retroactively or permanent power settings. User stop/suspend instructions still take precedence. Record observed long idle/standby gaps separately from typical epoch time.

## Latest user focus: one target until gain exceeds0.25
Concentrate subsequent bounded exploration on POR; keep PERM(v100) and SW(v104) fixed. Compact control POR77.54806949489308; also preserve historical original-fifth POR77.551936801885. Require POR>77.801936801885 (strict gain>0.25 against stronger historical reference), not just improved TOTAL or partial snapshot. Mechanism-positive ablations can guide continued POR work below this milestone. No switching to PERM/SW before this milestone absent new user direction. Original fullOOF>82 and strict original fifth TOTAL>80.8+gain promotion gate unchanged; five recipe training phases incl source. No automatic remaining folds. Maintain PROJECTS/exploration/atlas, max2GPU and frozen history/deliveries.

Latest two-route authorization: route1 concentrates POR output optimization until historical original-fifth gain>0.25; route2 independently replaces neural backbone, starting random privatePOR ConvGRU v127. Old backbone need not be retained. Max2 jobs across routes, no automatic queue, original fifthTOTAL>80.8+gain before remainingfolds; recipe<=5 phases includingsource, original OOF>82 unmet. Other-target fixed predictions retained for fair POR assembly. This overrides any prior ban on new random/backbone attempts within this scoped route.

PORhistorical+0.25milestone verified:v13477.869090 (+0.317154). Firstroute now independentPERMconditionaldensityv136 on frozenoriginalsource; fixedpositivePOR/SW, finalrecipe5phases. Secondroute curve-tokennewbackbonev135 retainingpositivealignment. Max2jobs, strictTOTAL80.8/OOF82 goalsstillunmet; no otherfolds/deliveryedits.

V136 final positive retained: PERM89.387936095784, compact gain+0.009541884640; actual TOTAL80.5891182349535; restored-best replay/frozen integrity verified. Five phases including source; no other folds. Next v138 width256 replaces PERM phase from same original source, no v136 warm-start.

Latest user direction: prioritize analysis and further expansion of proven POR head gains; transfer to all output heads only after validated POR evidence. v139 SW early-stop requested at epoch24 boundary (below control0.546008), no forcedkill. v140 width256 from same v113 is ready, launch only after SW handle terminal. Current v137 backbone may finish unchanged; v141 remains prepared, no automatic launch under this priority. Original strictfifthTOTAL80.8+gain/fullOOF82/max2/five total phases remain.

Latest user authorizes POR width512 and residualdistribution mechanism in parallel max2, both from frozenv113 andreplace head phase. Preserveverified256/5phaserecipe, no512 or residual warmstart fromtrainedv140. Holdunrelatedpreparedbackbone routes; transferallhead after measuredgain.

Latest explicituser: stop512, test192.512 was already naturallycomplete35/best17 POR77.854566, no resume. v146192 prepared same frozenv113/loss/LR/budget; no256/512trainedhead loaded, replacephase/five total. Residualv145 continuesunchanged, no more512 launches.

Latest explicituser requests384PORhead. v147 prepared samefrozenv113/loss/budget,444704activeparams, no256/512headwarmstart. Residualv145 alreadycomplete35/POR77.676996 below256; new384usesfreeGPUslot beside192. Five total phases, no more512/no otherfolds.

V149 distribution readout: [study](results/deep_learning/por_component_readout_v149_20261001/README.md). Two predefined matched shared/communicate component heads, same frozenv113 source; five total phases. Prepared/untrained, no score; focusPOR readout, otherbackbone remains held, no otherfolds.

Latest user: if new ideas lack gain, augment mechanisms already proven useful. V151 [jointRPS study](results/deep_learning/por_ranked_probability_v151_20261001/README.md) keeps effective256location/logweight structure and capacity, adds true default/continuous CDF supervision only, fromsameparent/fivephases. No negative module stacking or extra sixthphase. Prepared/untrained; v150 continues unchanged, max2.

Latest explicit user two lines: continuePOR on positivefoundation; other heads may use different mechanisms, MAIN PERM. SW155prioritystop atsavedepoch21/best20, exactsavedbestreplay73.670352; early/incomplete, not convergence conclusion, no resume. V157 [PORpositiveRPS plus smallgate](results/deep_learning/por_bounded_gate_rps_v157_20261001/README.md), prepared/untrained,231201parameters/+257, fivephases, max2 besidePERM156.


## 用户最新早停：微调patience14

微调连续14轮无改善早停，max80/min10保持，覆盖旧patience18。已完成v157/v159不追改历史；v159在收到新指令处理期间自然完成best0/stop18。已启动v160保留冻结80/18源码/配置，专项只读监控PID62172在bad14写已有轮间停止请求，保存最优并报告实际轮次，不更改训练源码/不重启/不自动队列；后续新源码直接14。


## 用户撤销阶段压缩

用户回复“算了，就这样吧”，保留当前阶段数，不再推进压缩。当前有效POR链路3阶段、PERM2阶段；共享原来源只计一次，全方法共5阶段。保留已验证配方和权重，继续原有优化/80-14/第五折严格80.8筛选。未启动阶段压缩训练。


## 用户撤销阶段压缩

用户回复“算了，就这样吧”，保留当前阶段数，不再推进压缩。当前有效POR链路3阶段、PERM2阶段；共享原来源只计一次，全方法共5阶段。保留已验证配方和权重，继续原有优化/80-14/第五折严格80.8筛选。未启动阶段压缩训练。

## Latest user: retain proven mechanisms, small-module or additive changes
Keep verified best architecture/recipe/weights. Subsequent trials retain positive mechanisms and change only one small module or add a residual/auxiliary term; broad readout/backbone replacement is held. Current192multiply/193adaptive may complete already-authorized fixed comparisons unchanged, max2. Prepared194 random-private curve-core pair is HELD, no launch under this direction. New additive experiments must count all training phases honestly; do not load trained154 and call another stage free. Source-only initialization with same effective architecture plus one change can replace the existing head phase and remain5 stages. Original splits/Q/scoring/80-14/promotion strict80.8/OOF>82 preserved.

Latest user explicitly raises total training stage cap to8, superseding5. Prioritize freezing learned positive functions with small additive repairs. New197 warm154+one patch honestly6stages, ownPERM loss/fixedscore/max2/80-14. Existing sources/deliveries unchanged. Prepared196 source-retrained variant HELD in favor of197 preserving learned154 weights; no automatic queue/remainingfolds.

## Latest user: each output head may add at most one stage
Use verified five-stage154 combined recipe as baseline. POR current SOURCE+113+151 may add one; PERM current SOURCE+154 may add one; SW104 may add one. SOURCE shared once; whole method at most8. Warm154+197 is the only added PERM stage, whole6. Mean/weighted candidates are parallel alternatives from154, never sequential197mean->197weighted or197->another trainedrepair. Future PERM attempts may be siblings from original154; adopted chain only one addition. POR/SW each retain one possible addition. Do not conceal retraining/refitting as no stage, no autoqueue/otherfolds; strict80.8+gain/fullOOF82/max2/80-14 unchanged.

Latest user prioritizes simplePORstageablation and no unnecessaryextra stages. HoldPORscale200 (entry failed wrongparentfolder beforeGPU admission/firstbatch, no score, preservefailure source/log/status, no automaticrepair/retry). Default practicalrecipe remains five-stage15480.612475417994; optional198sixstage80.612568631546 retained as micropositive, not necessarydefaultphase. New201source/prefix/directskipmiddle ablation, noaddedPORphase; othernewPORaugmentationheld.

## 2026-10-01 latest user direction: backbone and readout together
User explicitly says prediction-head-only fine-tuning is bottlenecked; all future optimization candidates must train a feature backbone/encoder together with its readout. Do not launch new head-only repair/fine-tune jobs. Preserve proved gains and original source/checkpoint evidence. Existing v271 jointly trains fresh raw-input encoders and direct conditional distribution heads, thus continues unchanged. New v272 jointly updates full expert feature encoder copies and direct density readout, retaining original154 prior as frozen safeguard. Prior frozen-only restrictions superseded for newly scoped trials; active/completed sources immutable. One channel route plus backbone exploration, max2GPU/noOOM, max80/pat14, original scoring/data isolation/fifthstrict80.8+meaningfulgain before otherfolds/fullOOF82, total<=8 phases and each head at most one additional phase remain. Never chain a trained repair then another repair stage while hiding the extra phase.


## Latest user: conditional early-stopping patience
All future NEW training: max80/min10, patience10 until strictcurrentSOTA is broken, then latchpatience14 for rest ofsame run. OfficialwholeTOTAL isSOTAcriterion; for an independenttarget with otherretainedheads fixed, use mathematically equivalent targetthreshold (POR34178.04643071793232/PERM33689.51838939879066/SW35174.17361205223327; assembledTOTAL80.70612972323806). Freeze reference atlaunch, do not useCONTbest/oracle. Active357 frozen trainer/config remainshistoricalpat14; external existingepoch-boundary stoprequest enforces10untilPERMSOTA then14, no restart. Sourceimmutability/explicitresume/strictfifth80.8beforeotherfolds/max2/<=8phases unchanged.

Latest user explicitly authorizes five COMPLETE models of80.706130 eight-stage recipe: seeds20261010..20261014, newSOURCE and all7downstream stages perseed, fixedoriginal fifth, no otherfolds. Finite serial8x5 dependency batch authorized/no unrelatedqueue/no retry/autoresume, max2. Latest80/10->14 budget common to allnewseeds; historical budget differs. Eachseed own train-onlyQ/weights, no cross-seed head splice. Study results/deep_learning/complete_seed_compare_v362_20261002; old best/deliveries preserved.


## Latest user: obsolete artifacts retain only summaries and results

The user explicitly authorizes deleting obsolete experiment details and weights, including unused seeds20261010/11/12/14. Preserve current original + seed20261013 full-CV and required dependency artifacts, data and delivered packages. Cleaned runs are historical score evidence and cannot be replayed without retraining. This overrides older preservation of obsolete detailed logs/negative weights; do not re-create removed artifacts or restart their queues. Manifest: D:/match_predict/maintenance_cleanup_20261002/DELETE_PLAN.json.
