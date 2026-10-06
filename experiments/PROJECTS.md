# Experiment status

All training finished by user request. Final posteriorrouter fifth80.25089377481923; best completeOOF81.71323318479078. No automatic new runs. See repository root PROJECTS.md, docs/deep_learning/EXPERIMENTS.md and results/deep_learning. Source/compactreports preserved, non-key bulky artifacts removed. Future work needs explicit user authorization.


## Resumed research by explicit user request
Current experiment: pure_dl_paired_local_posterior_router80_sequence; originalposterior80.25089377481923 control plus zero-init depthwise9 residual in routerhidden32 (+288params). Fromscratch, originalfifth/max80/18, no change to expertloss/Q/seed/LR. PID71028/session75871 launched with normalPython; no other training job at admission. Results pending; no gain/fullOOF claim. No second trial queued. Next inspect training, compare matchedepochs then bestscore; only strict>80.4 qualifies for remainingfolds. Preserve previous GitHub release and immutablelocaldeliveries.

- Live continuation: PID71028 confirmed running; localposterior80 epoch11, best78.182559651147 at11, matchedposteriorcontrol delta-0.069357472024. Saved comparison; too early for mechanism conclusion. No errors, singleGPUjob; fullOOF>82 unproven.


- Resumed-route CDF posterior candidate launched PID68156/session87910: original80.2508937748 posteriorrouter directcontrol, append42 detached conditionalCDF values at standardizedtargets[-3,-2,-1,0,1,2,3] (2members*3targets*7) to92inputs. Firstlayer134 preserves original92columns/bias, newcolumnszeros, originalfinalrouter preserved afterfork_rng. +1344params expected1330462; no addedbackbone. PORt3CDF usesatan-basedformula withactualstd, Gaussianndtr, float32/clamp. No defaultatom insideconditionalCDF; existinggate descriptor retained. Original ownexpertNLL/separaterouterNLL/Qfeedback/80-18/LR.0006/originalfixedfifth804 unchanged. No standalonepreflight/review, no claimedgain. Other localposteriorPID71028 live: epoch48 best79.9778593226 observed. Two distinctmechanisms, max2GPU; timing must identify overlap versus previoussingle phase. No delivery edits.


- localposterior80 COMPLETE: {"scores": {"POR": 77.07023223011392, "PERM": 89.17061248871573, "SW": 73.54956056042865, "TOTAL": 80.0731302362347}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1305.1545322000165, "peak_gpu_allocated_mib": 776.1279296875, "best_checkpoint_predictions_reproduced": true}. Delta vs originalposterior80.25089377481923 = -0.17776353858452865. No gain from combining localcontext withposteriorstats; no kernelgrid/remainingfolds. Deleted977050766bytes redundantweights/logs/largecaches, preservedsource/config/reports/predictions. Time is trainerfold time with initialsolo thenCDF overlap, not a controlled concurrency benchmark.

- NEW splitposterior80 launched PID52016/session84208 only after localrunnerterminalexit0. Original80.250894control, separatedefault/conditionalrouting (92->32->12,rolelabelsdefault/continuous), validhurdleposterior; defaultmass usespi_g,conditionalcomponents use normalizedpi_c*survival. +198params expected1329316,zeroequalinit; original24detachstats/expertownNLL/separaterouteroptimizer/Qfeedback unchanged. Newmax80/18/fixedfifthstrict804. SingleotherCDFPID68156 remains; noGPUpreflight/review,no resultclaim. FullOOF>82 stillunproven; deliveriesimmutable.


- CDFposterior80 COMPLETE: {"scores": {"POR": 77.31185546665223, "PERM": 89.35265012588025, "SW": 73.56708502035238, "TOTAL": 80.21546394117709}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1489.089324200002, "peak_gpu_allocated_mib": 822.67138671875, "best_checkpoint_predictions_reproduced": true}. Delta vs posterior80.25089377481923=-0.03542983364214081. No netbenefit, below804; noCDFpointgrid or remainingfolds. Deleted978286630bytes redundantweights/rawlogs/dynamics/distribution, retain source/config/reports/predictions. Timing includes mixedtwo-job overlap (localposterior then split), not controlledserial/parallelbenchmark.

- Shared symmetricposterior80 launched PID88644/session24551 afterCDFrunnerexit0, soleother splitPID52016. Original68+own12+other12 views, same92->32->3 scorer forbothmembers, swap-equivariantroutes, removesfixedpositionbias; -99expectedparams1329019. Originalfirstlayer/expertinit preserved, newlastzero; ownexpertNLL/separaterouterNLL/fullQfeedback and80/18/fixedfifth804 unchanged. NoGPUpreflight/review; equalityconstraint mayhurt real206/68 specialization, gainunproven. DeliveredfullOOF81.713233 stillbest; target82notmet.


- Splitposterior80 COMPLETE: {"scores": {"POR": 76.72913416494663, "PERM": 89.21734257695437, "SW": 73.47349524819485, "TOTAL": 79.96053348828622}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1549.940439099999, "peak_gpu_allocated_mib": 776.125, "best_checkpoint_predictions_reproduced": true}. Delta vs80.25089377481923=-0.2903602865330157. Separate default/conditional routing notuseful inthiscontrol; no rolecapacitygrid/remainingfolds. Deleted982271764bytes redundantweights/logs/largecaches, compactevidence/source/predictions kept. Trainerfoldtime includes successiveCDF/symmetric overlap; not a controlledconcurrencycomparison. SharedsymmetricPID88644 continues (epoch20 observed). Router-only utility objective is being prepared, NOTlaunched: originalexpertNLLunchanged, routerNLL plusfixed0.25 expectedscore surrogate ondetachedexpertposteriors, eightnodeCDF layercake withclippingatoms. Related oldutility_mass full-expertfailure acknowledged; no coefficientgrid.


- Router-only utility80 launched PID60060/session81907 after splitrunnerterminal; originalposterior80.250894 architecture/init/params unchanged. RouterNLL+fixed0.25*finalQ*mean(1-sum routingweight*detachedmemberutility), ownexpertNLL/aux40 unchanged. 8fixedGL layercakeCDF, defaultatom included, physicalBOUNDSclippedprediction atoms accounted; train-only rawtargettensor, no validationlosslabels. Same80/18/fixedfifthstrict804/LR.0006/bs64; separateutilitylog, noextraexpertforward. Existingutility_mass full-expert failure79.299712 remains negative; thisisolatesnewgradient to router, approximate randomizedactionutility not exactargmaxdecodegradient. No coefficientgrid/preflight/review. Soleother sharedsymmetricPID88644 continues; max2GPU, fullOOF>82 unmet.

## 2026-09-27 SW消融结束与局部特征重跑
- SW仅均值组件消融第五折已完整结束：best52/stop70，官方TOTAL80.0468714929，较原双专家80.2508937748为-0.2040222819，较门控加均值80.0967730451为-0.0499015522。训练器已恢复最佳checkpoint并复现预测；未过严格>80.4，不补折。记录 outputs/pure_dl_paired_sw_mean_conditioned80_sequence/README.md。
- 新井内101/401点局部稳健特征首次实验epoch1验证后遭遇D盘空间不足：全训练点探测的临时CDF需约1.78GiB，而当时仅余约1.41GiB；epoch1不是完成分数。审批系统拒绝删除三个已完成负实验的大诊断缓存，拒绝理由为blocked by policy，未绕过删除。
- 改为独立内存探测重跑：精确float32全点CDF保存在RAM，Q和训练/评分不变，最终仅保留紧凑状态；等待均值消融结束释放RAM后单独启动PID73824/session31259。记录 outputs/pure_dl_paired_local_robust_memory80_sequence/README.md，结果待定。
- 完整五折最好仍81.7132331848，>82目标未完成。只在固定第五折严格>80.4并完成必要核验后续训其他四折。

## 2026-09-27 局部特征结束与曲线关系续试
- 九曲线井内101/401点局部稳健位置特征已完成第五折：best52/stop70，官方TOTAL80.1069906584，较原双专家80.2508937748为-0.1439031164；PERM+0.058649而POR/SW下降，未过严格>80.4，不补折。内存版全点float32 Q无磁盘临时CDF，GPU峰值1129.83MiB，单任务958.42s。记录 outputs/pure_dl_paired_local_robust_memory80_sequence/README.md。
- 下一项特征机制已启动PID17316/session32628：四组原曲线在101/401点窗口的井内滚动相关，仅对后验路由 logits 提供零初始化8维侧信息；专家206/68输入及损失不变。记录 outputs/pure_dl_paired_local_curve_relation80_sequence/README.md。最多2GPU，此刻只有1项。
- 完整五折最高81.7132331848，>82目标未完成；仅固定第五折严格>80.4并核验后补折。

## 2026-09-27 局部相关负结果与无监督原型试验
- 四组曲线局部相关路由第五折完整结束：best52/stop70，官方TOTAL80.0558691617，较原后验双专家80.2508937748为-0.1950246131；最佳checkpoint预测已复现，新增路由权重L2=0.231098但未改善，未过严格>80.4、不补折。单任务901.28s/显存1099.50MiB。记录 outputs/pure_dl_paired_local_curve_relation80_sequence/README.md。
- 新的井内九曲线组合软原型实验已启动PID8992/session35829：MiniBatchKMeans仅拟合64训练井输入，12个中心的软归属零初始化注入原双专家逐目标路由；验证井只做变换，目标标签不参与聚类。单折结果待定，记录 outputs/pure_dl_paired_facies_router80_sequence/README.md。
- 完整五折最高仍81.7132331848，>82目标未完成；严格第五折>80.4门槛继续适用。

## 2026-09-27 原型特征的注入位置对照
- 仅训练井拟合的12个曲线组合软原型用于路由，第五折完整结果80.2430688700（best52/stop70），较原双专家80.2508937748为-0.0078249048；最佳checkpoint预测复现，未过严格>80.4，不补折。路由新增权重L2=0.505334，单任务909.34s/显存1111.35MiB。记录 outputs/pure_dl_paired_facies_router80_sequence/README.md。
- 同一原型特征改为仅注入第二专家输入校准层的单次机制对照已启动PID37756/session3073；原型数、拟合方法、分折/seed/损失/Q/LR/预算不变。记录 outputs/pure_dl_paired_facies_calibrated80_sequence/README.md，结果待定。
- 完整五折最好仍81.7132331848；严格第五折>80.4门槛与完整OOF>82目标不变。

## 2026-09-27 特征优先批次结论
- 固定第五折四项新特征机制均完整完成、最佳checkpoint复现：局部稳健位置80.106991、局部曲线关系路由80.055869、12训练井无监督原型路由80.243069、同一原型注入第二专家校准80.117536。原后验双专家80.250894仍最好；全部低于严格>80.4，不补其余四折，也不宣称完整OOF收益。两项SW条件化收尾同样为负：80.096773/80.046871。
- 逐项README及汇总 docs/deep_learning/FEATURE_FIRST_20260927.md 记录假设、对照、分数、耗时、显存、无收益方向和后续计划。内存探测全训练点float32 Q解决本批D盘临时CDF不足问题；原始失败epoch1不可当完整成绩。
- 当前无GPU训练任务。D盘约余1.28GiB；精准删除三个已完成负实验的大诊断缓存被自动审批以blocked by policy拒绝，未执行且未绕过。下一项长训练前需先有可靠存储余量；在该约束下可继续代码/机制设计。手工侧特征继续微调的证据不足，下一步考虑训练井内学习的跨曲线表征。完整五折最高81.7132331848，>82目标未完成。目标达成后可用独立种子/留井验证稳定性并整理新版独立交付。

## 2026-09-27/28 学习型跨曲线残差与轻量产物
- 在原后验双专家第五折80.250894上新增九曲线逐点16维/2头attention残差，仅修正full206专家输入，另一专家及92维后验路由原样；区别于已失败的独立curve-token网络、离线masked预训练和单曲线卷积。固定第五折/seed+4/bs64/LR.0006/80轮cosine/patience18/Q/官方评分。PID87612/session13408单GPU运行，实验独立记录 outputs/pure_dl_paired_curve_attention80_sequence/README.md。
- 新训练器使用全训练点float32内存CDF与紧凑末态；如果第五折未过严格>80.4，则保存配置、每轮分数、预测及紧凑诊断，不落盘负实验checkpoint和大分布。若过门槛，保存完整最佳权重及分布供必要核验与其他四折；不改变训练与最佳状态选择。D盘启动约1.28GiB，故本批避免无用大产物。
- 完整OOF>82仍未完成；后续是否并发第二GPU任务视本实验显存/内存峰值和候选独立性决定，最多2。

## 2026-09-27/28 attention注入位置双并发
- full206专家九曲线attention残差第五折运行PID87612/session13408，前期全点探测峰值约959MiB；校准68专家的同模块同参数注入对照已启动PID38088/session92651。GPU守卫确认最多2任务且无OOM迹象；两路均80轮上限/最佳验证状态/严格第五>80.4。
- 第二路模型仅覆写注入位置，不修改活动第一路源码；源与结果记录 outputs/pure_dl_paired_curve_attention_calibrated80_sequence/README.md。未过门槛不落盘大分布/权重；没有新完整五折分数，目标>82仍未完成。

## 2026-09-27/28 九曲线学习型残差结果
- full206专家注入路在核对PID87612/新鲜history后于epoch64提前终止，best observed80.059460/epoch52、12轮未刷新、距严格门槛0.340540；runner预期退出1，进程消失，无完整最佳状态导出，标注PARTIAL，不作为完成第五折分数。记录 outputs/pure_dl_paired_curve_attention80_sequence/manual_early_stop.json 与README。
- 同模块注入校准68专家的完成第五折80.1837020319，best52/stop70，最佳状态预测复现；较原后验双专家80.250894为-0.067192，POR-0.012977/PERM-0.055329/SW-0.125524。单项1147.44s、显存峰值970.96MiB；未过严格>80.4，不补折，负实验未落盘checkpoint/大分布。记录 outputs/pure_dl_paired_curve_attention_calibrated80_sequence/README.md。
- 当前两GPU任务均已退出；D盘约1.24GiB剩余。新轻量训练器成功避免无收益权重/分布占盘；完整五折最好仍81.7132331848，>82目标未完成。下一步改用针对SW输出/损失的机制，而非继续扫曲线attention容量。

## 2026-09-27/28 SW目标特异九曲线attention
- 校准68输入attention第五折完整80.1837020319，best52/stop70，较原双专家80.2508938为-0.067192，其中SW-0.125524最大；负实验按新规则未写checkpoint/大分布，显存峰值970.96MiB，时间1147.44s（含部分并发）。full206输入同模块路epoch64人工早停PARTIAL best observed80.059460，不充作完整结果。
- 新试验仅修正每位专家SW主门控与8个均值：输入为九曲线逐点attention而非先前失败的POR/PERM后验条件化。固定第五折、seed+4、80轮上限、patience18、原Q/NLL/评分及轻量负产物。PID72412/session27270单GPU运行；独立记录 outputs/pure_dl_paired_sw_curve_attention80_sequence/README.md。严格>80.4才核验/补四折，完整OOF>82未完成。

## 2026-09-27/28 SW目标头与先验门控损失双路线
- SW九曲线主头attention PID72412/session27270持续；第二路原双专家不改网络，仅以64训练井SW默认率0.7211513设均值归一的门控BCE权重(default0.8550/nondefault1.3750)，POR/PERM及连续NLL/Q/decoder不变。第二路PID85936/session35960，GPU守卫确认最多2任务、内存/显存启动安全。独立记录 outputs/pure_dl_paired_sw_gate_prior80_sequence/README.md。
- 两路都只跑固定第五折80轮上限/最佳状态/严格>80.4门槛，负实验不落盘大权重/分布。完整五折OOF>82仍未完成，不把用户反馈A榜分数当本地验证。

## 2026-09-27/28 井级SW默认先验机制
- 64训练井九曲线逐井分位数/缺失率的五折井级Ridge检查alpha100：预测SW默认比例R²约0.156、MAE0.0457，常数基线MAE0.0523；弱但非零的输入可识别性。第五折低分井的SW默认比例多为0.60~0.64，训练总体约0.72。验证标签只用于事后提出假设，不输入模型/拟合归一。
- 新试验原后验双专家加零初始化36→16→2井级SW主门控偏置，原206特征、POR/PERM、连续分布/损失/Q/路由直接输入不变；训练井拟合侧特征归一，验证井只用自身无标签输入。PID6588/session15382单GPU，第五折80轮上限、严格>80.4才核验补折。记录 outputs/pure_dl_paired_well_sw_prior80_sequence/README.md。完整OOF>82未完成。

## 2026-09-27/28 井级环境的门控/路由注入对照
- 井级36维无标签输入上下文SW门控PID6588/session15382继续；使用完全相同特征归一/网络容量但只加到SW专家路由的对照PID83984/session61461已作为第二GPU任务启动。原双专家初始输出经零末层保持，固定第五折/seed/原Q/NLL/评分/80轮上限/strict>80.4一致。记录 outputs/pure_dl_paired_well_sw_router80_sequence/README.md；两路结果待定，最多2任务。
- D盘低余量下继续沿用失败候选不写权重/大分布的规则；不以验证井真实默认率影响拟合或推理。完整OOF>82未完成。

## 2026-09-27/28 目标分工损失新路线
- 同一36维整井输入的SW门控/路由两路第五折均已完成：79.8943138225与80.0716546023，后者在匹配特征/小网络对照+0.1773407799但仍低于原后验双专家80.2508938；两者未过>80.4，不补折。逐项README/预测证据保留，失败权重/大分布未落盘，原始里程碑模型不动。
- 换思路为不加特征/不增参数的两专家按目标互补监督：full206权重POR/PERM/SW=[1.25,1.25,0.5]，校准68=[0.75,0.75,1.5]，每目标两专家平均1；router损失/原Q/decoder不变。PID7012/session83665单GPU，第五折80轮上限与严格>80.4不变，记录 outputs/pure_dl_paired_target_specialist80_sequence/README.md。完整OOF>82仍未完成。

## 2026-09-27/28 双专家目标分工角色对照并发
- 主路线目标权重full206=[1.25,1.25,.5]、calibrated68=[.75,.75,1.5]运行PID7012/session83665；完全反向的匹配角色对照运行PID65480/session38700。模型参数/seed/fold/Q/routerNLL/训练预算完全一致；均只用第五折筛选且门槛严格>80.4。第二路独立记录 outputs/pure_dl_paired_target_specialist_reverse80_sequence/README.md，最多2GPU、负结果不留权重/大分布。目标完整OOF>82未完成。

- ?????????????50/50????????80+??75???TOTAL80.2706366072???+0.0197428324?????seed??80.3103506050???80.4?????100%???144665???????GPU40.07?/601.56MiB???checkpoint???outputs/pure_dl_fixed_distribution_blend_fold5/README.md????????????????????????OOF>82?????

- ?????????????25%????stride2???stride1????Q??stride1?????/80???/18patience????>80.4????????????outputs/pure_dl_paired_sampling_stride80_sequence/README.md????????????

- Completed stride-sampling augmentation: fold5 TOTAL79.9262807508, best52/stop70, delta original=-0.3246130240; compact control=-0.3007343847. Best predictions reproduced;144665 rows covered. Single-GPU1093.58s/776.25MiB. No checkpoint retained; no remaining folds or stride sweep. Full OOF>82 unmet. See outputs/pure_dl_paired_sampling_stride80_sequence/README.md.

- PCGrad implementation prepared and syntax parsed; NOT trained yet. Earlier combined code-write/background-launch rejected, but separate code-only edits succeeded. Separate expert target losses, project full expert parameter gradients, router unchanged. Record outputs/pure_dl_paired_posterior_pcgrad80_sequence/README.md; gains unmeasured, OOF>82 unmet.

- Completed PCGrad fold5: TOTAL80.1582855947, best52/stop70, delta original=-0.0926081801, compact=-0.0687295408. Best predictions reproduced;144665 rows covered once;1551.86s single GPU/823.92MiB; no checkpoint retained. No final gain, no projection sweeps or remaining folds; next route must change mechanism. Full OOF>82 unmet. See outputs/pure_dl_paired_posterior_pcgrad80_sequence/README.md.

- Completed fixed1024 posterior quadrature: equal3 fold5 TOTAL80.3194134427 versus256 baseline80.3103506050, delta+0.0090628377. Original single also+0.0079939077. Fixed weights/no label-tuned resolution;144665 rows covered.54.03s/1438.96MiB single GPU. Small positive precision trick retained, no remainingfolds (<80.4), full OOF>82 unmet. See outputs/pure_dl_posterior_fine_quadrature_fold5/README.md.

- Completed action-grid4x refinement with fixed1024-bin equal3: fold5 TOTAL80.3220631614, +0.0026497187 versus fine integration and +0.0117125564 versus original equal3.144665 rows,54.31s/1466.40MiB; no weights trained or label tuning. Below80.4; no remaining folds. Diminishing precision gains, retain decoder but change next modeling route. See outputs/pure_dl_posterior_fine_actions_fold5/README.md.

- Completed fixed2-window overlap ensemble: fold5 TOTAL80.3538481813, +0.0317850199 vs same fine-decoder nonoverlap80.3220631614. All3 channels improve,144665 points each exactly2 predictions.62.13s/3163.29MiB single GPU; no weights fitted/no checkpoint. Positive mechanism, still0.0461518187 below80.4; no remaining folds. Record outputs/pure_dl_posterior_overlap_fold5/README.md.

- Completed fixed4-window overlap: fold5 TOTAL80.3627226809, +0.0088744997 versus2-window; POR/PERM up, SW slightly down. Exactly4 coverage of144665 points,79.54s/3163.25MiB, fixed votes/no retraining. Below80.4 by0.0372773191; no remainingfolds. Gains diminishing, retain positive inference recipe; next mechanism rather than endless overlap density. outputs/pure_dl_posterior_overlap4_fold5/README.md.

- Completed geometric4-window pool: fold5 TOTAL80.3595001347, -0.0032225462 vs arithmetic4. No overall gain; do not sweep pooling temperature/floor/channel switches. Current fixed arithmetic4 recipe80.3627226809 remains.144665 points exactly4 coverage;79.29s/3163.25MiB; no new checkpoint/no remainingfolds. Record outputs/pure_dl_posterior_logpool4_fold5/README.md; full OOF>82 unmet.

- Prepared conditioned rank8 numeric cross mixing in calibrated expert only; zero-init residual759params, original posterior network/loss/Q otherwise unchanged. Fixedfold5/80-18, no PCGrad/stride augmentation. Pending, outputs/pure_dl_paired_conditioned_mix80_sequence/README.md.

- Completed conditioned rank8 mixing: fold5 TOTAL80.1954269324, best52/stop70, delta original=-0.0554668424, compact=-0.0315882031. No gain; no rank/width sweep or remaining folds. Best predictions reproduced/144665rows covered,1119.55s single GPU/794.94MiB, no checkpoint retained. outputs/pure_dl_paired_conditioned_mix80_sequence/README.md. Full OOF>82 unmet.

- Prepared Lookahead on original posterior network: k5/alpha0.5, current fast iterate validation, original AdamW moments/schedule retained, no extra forward/backward. Fixedfold5/80-18; pending. outputs/pure_dl_paired_lookahead80_sequence/README.md.

- Lookahead manually terminated after45 completed epochs: observed best79.3084432291@45, substantial matched-control deficit; no complete-fold convergence/restore claim. Original budget80, user-authorized early termination. Session72874 exit1 confirmed after PID59080 stop; no checkpoint exported, no remainingfolds, no k/alpha sweep. Compact partial evidence outputs/pure_dl_paired_lookahead80_sequence/manual_early_stop.json. Full OOF>82 unmet.

- Completed fixed4-model overlap4 equal blend: fold5 TOTAL80.3794015534, +0.0166788725 versus3-model80.3627226809. POR improves, PERM/SW slight drops.144665 rows exactly4 contexts,91.68s/3189.37MiB; no weight fitting/new checkpoints. Positive complementarity; still0.0205984466 below strict80.4, no remainingfolds. outputs/pure_dl_four_model_overlap4_fold5/README.md. Full OOF>82 unmet.

- Completed fixed cosine-window weights on4 equal models: fold5 TOTAL80.3818486966, +0.0024471432 vs uniform80.3794015534. POR/PERM up,SW down; tiny singlefold benefit.144665 points coverage4/weight sum2 verified;90.37s/3189.94MiB. No label-fit/new weights. Still0.0181513034 below80.4; no rounding or remainingfolds. Record outputs/pure_dl_four_model_cosine_window_fold5/README.md; full OOF>82 unmet.

- Completed fixed5-model equal mixture: fold5 TOTAL80.3868607384, +0.0050120418 vs4-model. POR up,PERM/SW down; tiny net benefit.144665 rows coverage4,99.03s/3193.61MiB, no fit/new weights. Still0.0131392616 below strict80.4, no remainingfolds or validation-selected channel switches. outputs/pure_dl_five_model_cosine_window_fold5/README.md; full OOF>82 unmet.

- Completed model-geometric pooling: TOTAL80.3551652795, -0.0316954589 vs arithmetic5; all3 targets decrease. Exactly4 context coverage/144665rows,98.19s/3193.97MiB. No new checkpoint/no remainingfolds/no geometric-temperature grid. Keep arithmetic5 best80.3868607384. outputs/pure_dl_five_model_geometric_fold5/README.md; full OOF>82 unmet.

- Prepared full-expert-only affine RMS normalization (9 depth-local blocks, same parameter count), calibrated expert unchanged. Fixedfold5/80-18, original optimizer/loss/Q, no failed mechanisms. Pending, outputs/pure_dl_paired_rms_full80_sequence/README.md.

- Completed user-requested small-step learned5-model weights:15 logits,all578891 train points,lr1e-4/bs16384; best80.3868919215@update8, initial80.3868541939 (+0.00003773),stop19/28validations; trainNLL improves but validation worsens. Best tiny router checkpoint restored/saved, near20%weights in learned_weights.json.460.85s overlapping RMS/3170.28MiB; CPU cache RAM only released on exit. Below80.4, no remainingfolds; full OOF>82 unmet. outputs/pure_dl_five_learned_slow_vote_fold5/README.md.

- User reprioritized: finish current RMS experiment only, then return to five-model80.3868919215 baseline and prioritize features/neural architecture. No further vote/window/decoder/optimizer sweeps queued. Baseline is ensemble, preserve tiny learned vote checkpoint and source models; strictfold5>80.4/fullOOF>82 remain. RMS live {"status": "running", "session_id": 76251, "epoch": 35, "best_score": 79.57381986395312, "best_epoch": 32, "next_direction": "finish RMS; feature engineering and architecture around frozen80.38689 ensemble; no new voting or decoder sweeps"}

- Completed final in-hand RMS route: fold5 TOTAL79.9742648537, best52/stop70, delta original=-0.2766289212, compact=-0.2527502819. Best predictions reproduced,144665rows covered;1358.13s (partly concurrent learned-vote),797.44MiB. No checkpoint retained/no remainingfolds/no RMS sweeps. Prior active routes all closed; next feature engineering/neural architecture around frozen80.3868919215 ensemble per latest user direction. outputs/pure_dl_paired_rms_full80_sequence/README.md. Full OOF>82 unmet.

- Prepared input missing-gap geometry feature trial:18 previous/next valid-depth distances for9 curves, zero-init calibrated-expert injection; original206 intact/total224. Feature-first isolated ablation for80.39 ensemble, fixedfold5/80-18. No vote/decoder/optimizer sweeps. Pending, outputs/pure_dl_paired_gap_distance80_sequence/README.md.

- Completed gap-distance feature trial: fold5 TOTAL79.9680979184, best52/stop70, delta original=-0.2827958564, compact=-0.2589172171. No total gain; do not scan gap caps or do validation-selected channel mixing. Best restored/144665rows covered,1125.20s singleGPU/825.27MiB; no checkpoint retained/no remainingfolds. Keep80.3868919 ensemble; next distinct feature/architecture mechanism. outputs/pure_dl_paired_gap_distance80_sequence/README.md. Full OOF>82 unmet.

- Prepared patch extrema structural ablation: full-expert mean token plus bounded high/low contrast through zero-init2048-param projection. Calibrated expert/input/loss/Q unchanged; fixedfold5/80-18. Pending, outputs/pure_dl_paired_patch_extrema80_sequence/README.md; preserves80.39 ensemble as later comparison.

- Prepared point-specific patch retrieval: +4284 parameters in full206 expert, original broadcast retained; zero-init residual output. Fifth80/18, unchanged baseline protocol. outputs/pure_dl_paired_point_patch_retrieval80_sequence/README.md. No proven gain yet; preserve80.39 ensemble.

- Early terminated point-patch retrieval after validation62: observed best80.0112295005 at52, delta original=-0.2396642744, compact=-0.2157856351. Ten stale epochs and late scores79.5-79.7; no gain, no head/width sweep. Session42235 exit1; partial trial, final restored-best verification not completed, no fullCV. Peak789.27MiB; last printed training wall974.5s at60 includes preprocessing/probes, singleGPU, no standalone warmup; not final elapsed. Preserve80.3868919 ensemble; return to a different feature/architecture mechanism. Full OOF>82 unmet.

- Prepared conditional-residual input features: nine signed within-well ridge conditional residuals, no labels, +279 parameters to calibrated expert numeric inputs; fixedfifth80/18. Pending, outputs/pure_dl_paired_conditional_residual80_sequence/README.md. Preserve80.39 ensemble; no anomaly weight change.

- Conditional residuals early terminated after validation62: observed best80.1572448302 at52, delta original=-0.0936489446, compact=-0.0697703053. Ten stale epochs; no demonstrated feature benefit; no ridge/clip/injection-width scan. Session62389 exit1, no restored-best final verification; partial trial, no remainingfolds. Peak805.93MiB. Last printed wall1487.5s at60 includes preprocessing/probes, singleGPU with other desktop GPU usage, not final elapsed or matched speed benchmark. No checkpoint retained. Preserve80.3868919 ensemble; fullOOF>82 unmet.

- Prepared adaptive tied-kernel dilation: full-expert bottleneck only, +4635 parameters, original initialization preserved; fixed fifth80/18. outputs/pure_dl_paired_adaptive_dilation80_sequence/README.md. Pending/no gain; preserve80.39 ensemble.

- Feature expansion study completed (not model training): inventory206=9raw+9missing+1relativeDepth+22transforms/proxies+27wellRelative+138local; 64 fold5 train wells584411 rawrows, all depth increments0.1m. Prepared24 directional-depth features (18 signed1m/4m side contrasts,2jointstrength,2participation,2coverage), +33.43MiB float16 inputs; audit4.47s CPU, no labels, no accuracy-gain claim. Source experiments/pure_dl/directional_depth_features.py and analyze_feature_expansion.py; report outputs/pure_dl_feature_expansion_study/README.md. Next isolated feature candidate; do not combine with active adaptive dilation before evidence.

- Prepared isolated24 directional-depth features training on original pair, +1536 zero-init full-expert stem parameters, input230, calibrated expert/router original inputs. Fixedfifth80/18; no adaptive-dilation combination. outputs/pure_dl_paired_directional_depth80_sequence/README.md; gains untested.

- Adaptive dilation early terminated after validation62: observed best79.9977511654 at52, delta original=-0.2531426094/compact=-0.2292639701, ten stale epochs. Session70695 exit1; no final restored-best verification, no checkpoint retained, no dilation/group/strength grid. Peak787.54MiB, printed wall1579.1s at60 includes preprocessing/probes and mixed solo/concurrent phases, not final elapsed. Feature trial63047 remains active; preserve80.39. FullOOF>82 unmet.

- Directional24 feature trial early terminated after validation61: observed best80.0843932001 at52, delta original=-0.1665005747/compact=-0.1426219354. Late stagnation; no demonstrated feature benefit, no radius grid or remainingfolds. Session63047 exit1; partial trial, final restored-best verification incomplete. No checkpoint retained. Peak812.78MiB; last printed wall1525.3s at60 includes preprocessing/probes and mixed parallel/solo phases, not final elapsed. No active training remains from these two routes. Preserve80.3868919 ensemble; fullOOF>82 unmet.

- Prepared increment-texture36 input trial: percurve direction persistence,9/51 energy fraction, spike concentration and coverage; +2304 full-expert stem parameters,242inputs. Original pair/loss/splits80/18, no prior failed mechanisms. outputs/pure_dl_paired_increment_texture80_sequence/README.md; gains untested.


## 2026-09-27 Completed ABD internal-module attribution
- Completed baseline+12 arms: 10 genuine A-module interventions and2 numerical controls (point is already EmptyPoint; independent MLP absent). B/D unchanged. Source weights preserved,144665 rows each covered4times. 419.719s total including data/cache,1705.00MiB peak, concurrent with incremental trainer.
- Full206 A branch removal deltas in ABD: local -0.140824; global +0.025934 (80.43507290); whole hierarchy -0.008891; bottleneck -0.066270; skips -0.047187. Calibrated68: local -0.065327; global -0.131623; hierarchy -0.147241; bottleneck -0.056000; skips -0.133221. All genuine removals hurt A alone. Global-full candidate gains mainly SW+0.08924,9/16 wells improve. Inference dependency only, not training causal gain.
- Original fused-loss incremental trainer finished18 epochs without gain. Bestepoch0=80.40913868, all trained epochs lower (max80.285611). Strict prediction replay assertion failed: CPU readback confirms every saved A tensor equals original A and saved predictions recompute exact saved score. Report explicitly marks exact prediction reproduction false; no false success claim. No need to retrain this negative control.
- Started single-module adapted candidate session18573: train_abd_no_global_full80.py, full206 global attention structurally removed; B/D frozen, same incremental budget/loss. Deterministic operations added to avoid repeat-index accumulation jitter, documented difference. Initial match and training results pending. User-adopted baseline remains80.40915186, not silently changed to selected candidate.
- Next: observe candidate best fused checkpoint and restoration. Random-initialization matched module retraining remains unperformed; fullOOF>82 unmet. Further modules worth later attribution: anti-aliasing downsampling and depth-only normalization; no new sweeps launched.

V370/V371批次：原模型+末种子14分头融合80.849582637629（第五折调参诊断），低于原+13的80.863457557382。V362完成5/5；已启动原+13两模型原1–4折补折，控制器30132/首折SOURCE27184，已实测第6轮。固定比重OOF和逐折最优诊断分别保存。V367/V369启动入口退役未执行。curate/build已更新，2节点局部评分口径检查通过。


## 2026-10-02 obsolete experiment cleanup

User explicitly authorized removal of useless attempts and unused seeds. Deleted 30.588 GiB (13147 files); compact results/summary retained in 1105 artifact directories. Retain original + seed20261013, entire active V370 tree and historical-model dependencies; remove seeds20261010/11/12/14 weights/caches/logs. Data, delivered packages and reusable source unchanged. Verification passed for 282 retained files. Training continues; no stop/restart. Manifest and summary: D:/match_predict/maintenance_cleanup_20261002/README.md.

V370 latest terminal milestone: original/fold1 prefix5/8 TOTAL82.828785889629 (POR80.254410985303/PERM90.989052707754/SW76.875126132353). SW104 best25/stop39, actual delta vs own SOURCE +0.715437013802, TOTAL contribution +0.250402954831; DEFAULT +0.535845326923, CONT +1.210251469116. This differs from +2.209069808162 optimizer gain against cold SW104 initialization. Old delivered full fold1 TOTAL82.823076877363 is comparison only, not a matched budget ablation. New seed13 own SOURCE Q40<best42, original Q54<best55; both normalizations exactly match shared input cache, no learned weights shared. Source2 compaction completed without deleting required checkpoints/Q. Actual POR341 PID77132 and seed13 POR113 PID46888 now run, maximum2; full8/fivefold fixed and diagnostic OOF unfinished.

## V370 verified CUDA OOM failure and prepared recovery

Two first-fold PERM workers failed during validation with CUDA OOM: original PERM336 PID18788 (last complete epoch9), seed20261013 PERM154 PID49708 (last complete epoch7). Controller86104 exited; no GPU training currently running. Best states, optimizer/scheduler/RNG and window states retained. Both saved best predictions were reloaded sequentially using original validation batch32 and reproduced exactly on144621 rows; current frozen weights exact. Read-only peak reserved812MiB each, not a training-memory benchmark. No failed trial was resumed. New CPU controller resume_pair_serial_v370r2.py is prepared/syntaxchecked, maximum1 GPU worker with driver free-memory7GiB gate; unchanged source/config/BS64/validationBS32/80-epoch budget/pat10->14. It reuses completed stages, resumes only the two exact authorized last states once, and stops on another failure. Previous failures/checkpoints copied and SHA verified. Explicit user recovery confirmation required by study AGENTS.md: ??/???????????. FullOOF remains unproven; goal remains active. See OOM_RECOVERY_PREPARED.json and both RECOVERY_READBACK files.
