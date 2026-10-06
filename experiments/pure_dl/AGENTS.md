# Experiment rules

## Latest goal edit

User adds channel-wise optimization objective: POR/PERM/SW heads, training methods and best checkpoints may differ. Report matched per-channel gains and actual assembled result; never per-row/per-well oracle selection. Current normalized-feature head run continues; a separate zero-initialized per-channel branch-statistics input tests lost feature scale with the other GPU slot.

User changes fixed fifth-fold gate to STRICT >80.5; future gating uses fold5_screening_805.py. Clear gain is still required; full original OOF>82 is unachieved. Preserve active/completed source hashes, existing no-additional-fold delivery and deferred fullCV. Do not treat the tiny80.501552 increase as automatically sufficient for continuation. Max80/pat18/min10,normalCUDA,max2GPU unchanged.

## Latest: A+B+D baseline and frozen-companion training

LATEST: user wants all A6/B43/D52 feature extractors as one frozen backbone with new independent target heads; no legacy prediction voting in this new route. Preserve currently active A6-head voting experiment, stop B-own fine-tune, use freed GPU slot for feature concatenation640 ->3new heads. Training own-target loss, per-target checkpoint selection, final joint reconstruction. Historical scripts unchanged; max2GPU, no fullCV.

LATEST OVERRIDE: user now specifies own-model loss for training and fusion only at validation; old experiments/queue stopped. A6 from80.4923 is the required frozen backbone for independent channel heads; B43/D52 fixed validation companions. Separate warm-start B43 own-loss route uses A6/D52 fixed validation companions. Preserve source histories and independent output directories; no remaining folds.

Latest clarification: the TRAINING main loss must also be computed from the three-model fused distribution with the other two sources frozen. Validation-only fusion is insufficient. New B route uses random B with fixed A6/D, final fused main NLL gradients to B alone, original auxiliary supervision, and fused validation checkpoint selection. Leave active/completed source hashes untouched; no channel-balancing changes until this objective correction has its own evidence.

Latest user explicitly defers remaining-fold continuation. Do not run the prepared full-CV queue until later user resumption. Current comparison: start from verified80.49230853582785(A epoch6+B original+D original), jointly train all3 versus freezeA/D and trainB. Both use identical initialization/seed/fused loss/LR and fused validation checkpoint selection. This is incremental training; do not call it all-model random-initialization training. Overall fullOOF>82 remains unproved.

User adopts A+B+D fixed equal posterior ensemble as the baseline (fifth80.40915186344968): A posterior dual seed20261006, B posterior dual seed20261008, D single global-cosine. This supersedes four-source80.403220/five-source80.386892 for future comparisons. Preserve four cosine windows and fine decoder.
New candidate A training freezes B and D; final fused prediction loss may update only A. Validation, early stopping and best checkpoint selection use fused official TOTAL; solo scores are diagnostic. Always retain the candidate best weights until fusion assessment, regardless of solo80.4. Promotion needs positive fused gain versus unchanged ABD plus strict>80.4; no fullOOF claim from this exploratory fifth. Existing single-model ablations remain isolated controls and cannot establish fusion failure.

## Latest user baseline and attribution priority

The user additionally requests all four leave-one-out subsets of this four-source baseline. Evaluate4 triples +6 pairs +the baseline (11 arms), reuse the same source predictions, fixed equal posterior weights and identical decoder. This supersedes the earlier pairs-only scope.
User adopts the four-source fifth80.403220094 combination as the subsequent development baseline: original posterior dual seed20261006, posterior dual seed20261008, calibrated75 dual, and single global-cosine; exclude seed20261007. Preserve the source checkpoints and fixed equal posterior weights, cosine4-window/fine decoder. First evaluate all six fixed equal two-source subsets without retraining; two sources may contain four backbones. Continue controlled Q/loss attribution, max2 GPU tasks including inference. This is an adopted exploratory fifth-fold baseline, not independent validation; report selection history honestly and do not silently overwrite deliveries or claim fullOOF>82. Do not launch further unstructured mechanism sweeps while these requested ablations are pending.

Follow root AGENTS.md. Keep PROJECTS.md current. Historical files preserve original methods/configuration; do not launch old queues or batch scripts automatically. Current user directive: finish last posterior-router trial then stop training. Use documented entrypoints and read status before any execution.


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

## 2026-09-28 Current authorized study: dual80.22 head architectures
Resume only the new dual-expert head study requested by the user. Source retained local-context-router80.221861@52; compact posterior80.227015 has no weights. Two frozen128dim branches, original frozen input router, copied independent channel output rows plus zero-output residual readouts. Compare pointwise width80 versus depthwise-kernel5 width77 at identical93420 trainable parameters total, same data/LR/own-target dual likelihood/80-18 budget and source single-window decoder; report the width/context trade-off, not a pure context-only intervention. Keep training-Q from source epoch50 fixed; no fresh auxiliary supervision, no external B/D votes. Source and all old experiments immutable, old paused trainers stay paused, no fullCV. Explicit per-run resume only with saved current parameters/optimizer/scheduler/RNG/stopping state. Maintain independent README records and PROJECTS/EXPLORATIONS/atlas with real startup/failure/early/final results.


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


## v97 用户明确恢复其余四折
用户答复“恢复其余四折训练”。授权限定当前v96第五80.504197完整配方的原1~4折；第五预测核验复用，其余来源/预处理/Q只能各折64训练井重训。固定18有效阶段加1来源骨干；逐目标选实际最佳checkpoint，80/18/b64/64steps，normal最多2GPU，当前执行串行。旧暂停在此范围被覆盖；其他旧队列不启动，交付不覆盖，维护PROJECTS/独立记录/图谱。完整OOF>82仍须实际723556点证明。


## v97 当前并发执行配置
原四折补全范围不变。当前以新CPU调度入口接管已有训练进程，原PID/训练代码/配置/模型hash保留；主阶段依序执行，与下一折来源预训练重叠，GPU最多2且SW阶段最多1。空闲门槛来源6.8GiB、阶段4GiB；不重跑已完成项、不启动旧路线，失败不自动重试。此执行配置覆盖此前“当前串行”的描述；依据用户已明确授权最多2GPU且恢复其余四折。维护PROJECTS/独立记录/图谱；完整原OOF>82仍须实际证明。
