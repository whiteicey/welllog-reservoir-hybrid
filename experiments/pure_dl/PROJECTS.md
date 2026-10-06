# Experiment status

All training finished by user request. Final posteriorrouter fifth80.25089377481923; best completeOOF81.71323318479078. No automatic new runs. See repository root PROJECTS.md, docs/deep_learning/EXPERIMENTS.md and results/deep_learning. Source/compactreports preserved, non-key bulky artifacts removed. Future work needs explicit user authorization.


## Resumed research by explicit user request
Current experiment: pure_dl_paired_local_posterior_router80_sequence; originalposterior80.25089377481923 control plus zero-init depthwise9 residual in routerhidden32 (+288params). Fromscratch, originalfifth/max80/18, no change to expertloss/Q/seed/LR. PID71028/session75871 launched with normalPython; no other training job at admission. Results pending; no gain/fullOOF claim. No second trial queued. Next inspect training, compare matchedepochs then bestscore; only strict>80.4 qualifies for remainingfolds. Preserve previous GitHub release and immutablelocaldeliveries.

- Live continuation: PID71028 confirmed running; localposterior80 epoch11, best78.182559651147 at11, matchedposteriorcontrol delta-0.069357472024. Saved comparison; too early for mechanism conclusion. No errors, singleGPUjob; fullOOF>82 unproven.

- Verified live training observation: epoch20,best79.158968643472 at20,sameepochposteriorcontrol delta+0.012812153462. Early gaps change sign; continue unchanged, no stablegainclaim. Updated matched_epoch_comparison.json.

- Live midtraining comparison: epoch30,best79.540970595519 at28; sameepochdelta-0.204102094515, best-so-fardelta-0.082612095770 versus posterior80control. Training continues unchanged; not a completed result or proof of gain.

- Live later-midstage observation: epoch40,best79.694192433292 at36; sameepochdelta-0.084942818370, best-so-fardelta-0.048835366281 versus originalposterior80. Recent refresh supports continuing; no finalgain/qualification claim. Comparison JSON updated, no new model or changed hyperparameter.


- Resumed-route CDF posterior candidate launched PID68156/session87910: original80.2508937748 posteriorrouter directcontrol, append42 detached conditionalCDF values at standardizedtargets[-3,-2,-1,0,1,2,3] (2members*3targets*7) to92inputs. Firstlayer134 preserves original92columns/bias, newcolumnszeros, originalfinalrouter preserved afterfork_rng. +1344params expected1330462; no addedbackbone. PORt3CDF usesatan-basedformula withactualstd, Gaussianndtr, float32/clamp. No defaultatom insideconditionalCDF; existinggate descriptor retained. Original ownexpertNLL/separaterouterNLL/Qfeedback/80-18/LR.0006/originalfixedfifth804 unchanged. No standalonepreflight/review, no claimedgain. Other localposteriorPID71028 live: epoch48 best79.9778593226 observed. Two distinctmechanisms, max2GPU; timing must identify overlap versus previoussingle phase. No delivery edits.

- Two live jobs continue without errors; matched-epoch comparisons refreshed: [{"experiment": "paired_local_posterior_router80", "epoch": 60, "best_epoch": 52, "best_score": 80.0731302362347, "bad_epochs": 8, "best_so_far_delta": -0.17776353858452865}, {"experiment": "paired_cdf_posterior_router80", "epoch": 6, "best_epoch": 6, "best_score": 77.64909788001657, "bad_epochs": 0, "best_so_far_delta": 0.08157505069870297}]. Local/posterior combination not outperforming80.250894control; CDF earlystage. Max2GPU, no new candidatequeued; fullOOF>82 unproven.


- localposterior80 COMPLETE: {"scores": {"POR": 77.07023223011392, "PERM": 89.17061248871573, "SW": 73.54956056042865, "TOTAL": 80.0731302362347}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1305.1545322000165, "peak_gpu_allocated_mib": 776.1279296875, "best_checkpoint_predictions_reproduced": true}. Delta vs originalposterior80.25089377481923 = -0.17776353858452865. No gain from combining localcontext withposteriorstats; no kernelgrid/remainingfolds. Deleted977050766bytes redundantweights/logs/largecaches, preservedsource/config/reports/predictions. Time is trainerfold time with initialsolo thenCDF overlap, not a controlled concurrency benchmark.

- NEW splitposterior80 launched PID52016/session84208 only after localrunnerterminalexit0. Original80.250894control, separatedefault/conditionalrouting (92->32->12,rolelabelsdefault/continuous), validhurdleposterior; defaultmass usespi_g,conditionalcomponents use normalizedpi_c*survival. +198params expected1329316,zeroequalinit; original24detachstats/expertownNLL/separaterouteroptimizer/Qfeedback unchanged. Newmax80/18/fixedfifthstrict804. SingleotherCDFPID68156 remains; noGPUpreflight/review,no resultclaim. FullOOF>82 stillunproven; deliveriesimmutable.

- Two verifiedlive routes continue: [{"experiment": "paired_cdf_posterior_router80", "epoch": 30, "best_epoch": 30, "best_score": 79.55318909809255, "bad_epochs": 0, "best_so_far_delta": -0.07039359319709604}, {"experiment": "paired_split_posterior_router80", "epoch": 6, "best_epoch": 6, "best_score": 77.63008314742237, "bad_epochs": 0, "best_so_far_delta": 0.06256031810450224}]. New splitrouter train/probe running without error; matchedcomparisons refreshed; no additionalGPUjob. FullOOFgoal unmet.

- Live jobs68156/52016 observed progressing: [{"experiment": "paired_cdf_posterior_router80", "epoch": 38, "best_epoch": 37, "best_score": 79.62930601210336, "bad_epochs": 1, "best_so_far_delta": -0.04520228561402462}, {"experiment": "paired_split_posterior_router80", "epoch": 13, "best_epoch": 13, "best_score": 78.63743060863854, "bad_epochs": 0, "best_so_far_delta": -0.003710151462044564}]. Keep currentruns, no thirdGPU, no provenfullgoal.

- Verifiedlive comparison: [{"experiment": "paired_cdf_posterior_router80", "epoch": 46, "best_epoch": 44, "best_score": 80.02054965847276, "bad_epochs": 2, "best_so_far_delta": 0.15414912008279202}, {"experiment": "paired_split_posterior_router80", "epoch": 20, "best_epoch": 20, "best_score": 79.02302413482458, "bad_epochs": 0, "best_so_far_delta": -0.12313235518534782}]. CDF recent improvement justifies continuing; split remains earlierstage. No gatepass/fivefold claim, maximum2jobs.

- Both knownlive processes progressing; [{"experiment": "paired_cdf_posterior_router80", "epoch": 54, "best_epoch": 52, "best_score": 80.21546394117709, "bad_epochs": 2, "best_so_far_delta": -0.03542983364214081}, {"experiment": "paired_split_posterior_router80", "epoch": 28, "best_epoch": 28, "best_score": 79.52451672165343, "bad_epochs": 0, "best_so_far_delta": 0.002666388853683088}]. Comparison updated, no configchange/thirdjob, originalfullOOFgoal unmet.

- Verifiedlive wait and comparison update: [{"experiment": "paired_cdf_posterior_router80", "epoch": 62, "best_epoch": 52, "best_score": 80.21546394117709, "bad_epochs": 10, "best_so_far_delta": -0.03542983364214081}, {"experiment": "paired_split_posterior_router80", "epoch": 36, "best_epoch": 36, "best_score": 79.54110961828306, "bad_epochs": 0, "best_so_far_delta": -0.13339867943432182}]. Both remainwithinpatience, no restart/thirdjob; completedbestfullOOF81.713233 unchanged.


- CDFposterior80 COMPLETE: {"scores": {"POR": 77.31185546665223, "PERM": 89.35265012588025, "SW": 73.56708502035238, "TOTAL": 80.21546394117709}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1489.089324200002, "peak_gpu_allocated_mib": 822.67138671875, "best_checkpoint_predictions_reproduced": true}. Delta vs posterior80.25089377481923=-0.03542983364214081. No netbenefit, below804; noCDFpointgrid or remainingfolds. Deleted978286630bytes redundantweights/rawlogs/dynamics/distribution, retain source/config/reports/predictions. Timing includes mixedtwo-job overlap (localposterior then split), not controlledserial/parallelbenchmark.

- Shared symmetricposterior80 launched PID88644/session24551 afterCDFrunnerexit0, soleother splitPID52016. Original68+own12+other12 views, same92->32->3 scorer forbothmembers, swap-equivariantroutes, removesfixedpositionbias; -99expectedparams1329019. Originalfirstlayer/expertinit preserved, newlastzero; ownexpertNLL/separaterouterNLL/fullQfeedback and80/18/fixedfifth804 unchanged. NoGPUpreflight/review; equalityconstraint mayhurt real206/68 specialization, gainunproven. DeliveredfullOOF81.713233 stillbest; target82notmet.

- Verifiedlive processes52016/88644 advance: [{"experiment": "paired_split_posterior_router80", "epoch": 58, "best_epoch": 52, "best_score": 79.96053348828622, "bad_epochs": 6, "best_so_far_delta": -0.2903602865330157}, {"experiment": "paired_symmetric_posterior_router80", "epoch": 5, "best_epoch": 5, "best_score": 77.48454920514943, "bad_epochs": 0, "best_so_far_delta": -0.06593771422402028}]. Noerrors observed, max2jobs; newsharedscorer train/probe passedactualexecution. No fifth804 qualification or fullOOFgoal achievement.


- Splitposterior80 COMPLETE: {"scores": {"POR": 76.72913416494663, "PERM": 89.21734257695437, "SW": 73.47349524819485, "TOTAL": 79.96053348828622}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1549.940439099999, "peak_gpu_allocated_mib": 776.125, "best_checkpoint_predictions_reproduced": true}. Delta vs80.25089377481923=-0.2903602865330157. Separate default/conditional routing notuseful inthiscontrol; no rolecapacitygrid/remainingfolds. Deleted982271764bytes redundantweights/logs/largecaches, compactevidence/source/predictions kept. Trainerfoldtime includes successiveCDF/symmetric overlap; not a controlledconcurrencycomparison. SharedsymmetricPID88644 continues (epoch20 observed). Router-only utility objective is being prepared, NOTlaunched: originalexpertNLLunchanged, routerNLL plusfixed0.25 expectedscore surrogate ondetachedexpertposteriors, eightnodeCDF layercake withclippingatoms. Related oldutility_mass full-expertfailure acknowledged; no coefficientgrid.


- Router-only utility80 launched PID60060/session81907 after splitrunnerterminal; originalposterior80.250894 architecture/init/params unchanged. RouterNLL+fixed0.25*finalQ*mean(1-sum routingweight*detachedmemberutility), ownexpertNLL/aux40 unchanged. 8fixedGL layercakeCDF, defaultatom included, physicalBOUNDSclippedprediction atoms accounted; train-only rawtargettensor, no validationlosslabels. Same80/18/fixedfifthstrict804/LR.0006/bs64; separateutilitylog, noextraexpertforward. Existingutility_mass full-expert failure79.299712 remains negative; thisisolatesnewgradient to router, approximate randomizedactionutility not exactargmaxdecodegradient. No coefficientgrid/preflight/review. Soleother sharedsymmetricPID88644 continues; max2GPU, fullOOF>82 unmet.

- Verifiedlive training update: [{"experiment": "paired_symmetric_posterior_router80", "epoch": 33, "best_epoch": 28, "best_score": 79.53133046522386, "bad_epochs": 5, "best_so_far_delta": -0.09225222606578143}, {"experiment": "paired_router_utility80", "epoch": 5, "best_epoch": 5, "best_score": 77.45648040846771, "bad_epochs": 0, "best_so_far_delta": -0.09400651090574286}]. Utility loss actually executed andfinite logged; no errors, maximum2jobs. No fifth804qualification/fullOOFclaim.

- Both live routes continued with unchanged settings: [{"experiment": "paired_symmetric_posterior_router80", "epoch": 40, "best_epoch": 39, "best_score": 79.70650362039726, "bad_epochs": 1, "best_so_far_delta": -0.03652417917595585}, {"experiment": "paired_router_utility80", "epoch": 12, "best_epoch": 12, "best_score": 78.513657466279, "bad_epochs": 0, "best_so_far_delta": 0.02826808907632028}]. Sameepoch histories refreshed; no qualification or fullOOFgoal claim.

- Verifiedlive continuing both routes: [{"experiment": "paired_symmetric_posterior_router80", "epoch": 49, "best_epoch": 44, "best_score": 79.91082159049672, "bad_epochs": 5, "best_so_far_delta": -0.09121272924100765}, {"experiment": "paired_router_utility80", "epoch": 20, "best_epoch": 20, "best_score": 79.09536423669482, "bad_epochs": 0, "best_so_far_delta": -0.05079225331510884}]. Recentrefreshes justify ongoingtraining, no changes or thirdGPUjob. FullOOFgoal still unmet.


- Shared symmetric posterior router COMPLETE:80.06368214872943,best52/stop70,delta-0.18721162608980535; best checkpoint restored and reproduced. No promotion or shared-scorer grid. Cleanup blocked by automatic approval policy; artifacts still present. Router-only utility PID60060 continues: epoch45,best79.823617357764 at44. No third job; original fullOOF81.713233 remains best, goal>82 unmet.


- Latent posterior router80 launched PID628/session81443: original80.25089377481923 control with detached main-head input representations from both experts, each LayerNorm128->Linear16/GELU, concatenated32->6 zero-initialized router residual; +4838 parameters expected1333956. Explicit equivalent backbone/calibrated68 preprocessing, no hooks or frozen-source changes. Original expert-own NLL, router NLL, separate optimizers, Q feedback,80/18/fifth strict804 unchanged; no utility-loss combination. GPU guard admitted with sole other training PID60060 and10.80GiB free. Utility candidate best80.2115804825 at52, observed56, still running. No promotion or fullOOFgain; fullbest81.713233 unchanged. Next verify actual new-model training and compare original posterior control; no extra preflight/review.


## Final result
Router-only utility auxiliary completed: TOTAL80.21158048250079, POR77.30317230550095/PERM89.3009553284588/SW73.61512693111412; best52/stop70, best checkpoint predictions reproduced. Delta versus original posterior80.25089377481923: -0.03931329231843961. No gain, below strict80.4, no remainingfolds or utility coefficient/node grid. Peak804.331MiB; fold1528.949s with mixed solo/two-job overlap (symmetric then latent), not a controlled concurrency benchmark. Source/config/compact evidence retained. Redundant checkpoint/cache cleanup pending because automatic approval policy blocked the preceding equivalent deletion; no bypass attempted.

- Latent posterior router PID628/session81443 continues: epoch16,best78.754266259328 at14. No second new GPU job scheduled; focus on this distinct learned-representation mechanism. Full verified OOF81.71323318479078 remains best; target>82 unmet.


- Latent router live PID628: {"epoch": 24, "best_epoch": 23, "best_score": 79.21195491610008, "bad_epochs": 1, "status": "running", "matched_best_delta": 0.006133205216940496}. Other GPU slot free. Preparing ONE complementary-block expert-training weighting trial (not launched): independent training-only per-well256-raw-row block balanced signs; own expert losses Q*(1+/-0.5sign), all points retainpositiveweight, router onlyQ. Different from bad-point Q; no extra parameters, no amplitude/blocklength grid. No change to active latent or original posterior source. FullOOFgoal still unmet.


- Complementary-blocks80 launched PID54204/session41117; original posterior80.250894 architecture/init, no new parameters. Each training well raw256-row blocks assigned balanced signs using independent RNG seed20261006+foldindex+73000, recorded in config. Own expert final main/aux losses Q*A with complementary0.5/1.5, all trainingpoints nonzero; router NLL onlyQ, Q/probe unchanged. Equal member average1 perpoint, individual point-average may differ due tailblock sizes. Max80/18/strict804, no blocksize/amplitude grid. SoleotherGPU latent PID628. Gain pending; fullOOFbest81.713233 unchanged.


- Both verifiedlive jobs628/54204 progressed without errors: [{"epoch": 53, "best_epoch": 52, "best_score": 80.20707445856918, "bad_epochs": 1, "experiment": "paired_latent_posterior_router80", "status": "running", "matched_best_delta": -0.04381931625005109}, {"epoch": 18, "best_epoch": 18, "best_score": 78.93475106082802, "bad_epochs": 0, "experiment": "paired_complementary_blocks80", "status": "running", "matched_best_delta": -0.044269302155342416}]. No model/configchanges or extraGPUjob; fullOOFgoal unmet, strictfifth804 before remainingfolds.


## Completed latent-router result
{"scores": {"POR": 77.19460787792981, "PERM": 89.37384153998788, "SW": 73.62242158912709, "TOTAL": 80.20707445856918}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1380.5393081000075, "peak_gpu_allocated_mib": 800.13134765625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.04381931625005109, "qualified": false}. Detached expert hidden representations did not improve originalposterior80.25089377481923. No latent-size/grid/remainingfolds. Timing includes solo and two-job phases, not a controlled concurrency benchmark. Source/config/compact evidence retained; redundant artifacts pending prior automatic deletion-policy block, no bypass attempted.

- Complementary-blocks PID54204/session41117 continues: epoch40,best79.719355133220 at39. No second new job; full verifiedOOF81.71323318479078 remains best, goal>82 unmet.

- Complementary-blocks PID54204/session41117 verified live: {"epoch": 54, "best_epoch": 48, "best_score": 79.95256297808557, "bad_epochs": 6, "status": "running", "matched_best_delta": -0.2983307967336657}. No otherGPUjob or configchanges; no finalgain/804qualification claim. FullOOFgoal unmet.


## Completed complementary-block result
{"scores": {"POR": 76.52249314578376, "PERM": 89.39835427846991, "SW": 73.44683153395995, "TOTAL": 79.95256297808557}, "best_epoch": 48, "stopped_epoch": 66, "seconds": 1236.7837184999953, "peak_gpu_allocated_mib": 781.72509765625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_posterior": -0.2983307967336657, "qualified": false}. Fixed random-block complementary learning priority did not improve original posterior control. No blocklength/strength grid or remainingfolds. Timing has solo/two-job phases and is not a controlled concurrency benchmark. Compact evidence retained; redundant weight/cache deletion pending prior automatic-policy rejection.

- Preparing independent Q0/Q1/mergedQ diagnostic trial, NOT launched. Original expert-own thresholds/recovery and final loss denominator unchanged, router uses originalmergedQ. One memberforward per batch, sequential roleCDF; approximately5GiB scratch, no frozen source edits. No active training after block runner completion; next finish candidate then admit safely. FullOOFbest81.713233 remains, goal>82 unmet.


- Independent-Q80 launched PID25512/session21529, sole GPU job; disk16.88885498GiB available at launch. Original posterior80.250894 architecture/init, three original policies/tables member0/member1/merged. Expert main+aux final point losses use ownQ, router usesmergedQ. Same thresholds/recovery/defaultmask/population denominator, alltrainingpoint diagnostics from one pairforward perbatch, sequential roleCDF/RNG restoration. Float32 previousCDF scratch~5GiB; compressed evidence perrole/perrecord, latest onlyRAM; normalfinalize cleans only ownscratch. New80/18/fixedfifthstrict804; no preflight/review or gainclaim, futurepromotion must adapt three-role audit. No frozen sources/delivery changes.

- Actual execution: independentQ completed epochs1/2 and both full three-role probes; eachrole578891trainingpoints, warmupallQ1. Probes21.55/22.59seconds, observedGPUpeak779.141MiB, scratch5335059456bytes, remainingdisk12682543104bytes. Noerror; no gainclaim before Q divergence. Continue original80/18/fixedfifth804.

- Verifiedlive independentQ PID25512 progress: {"epoch": 14, "best_epoch": 14, "best_score": 78.6600808936658, "bad_epochs": 0, "latest_quality_update": {"observed_epoch": 12, "applies_from_epoch": 13, "roles": {"member0": {"downweighted_by_channel": [0, 0, 0], "downweighted_points": 0, "minimum": 1.0}, "member1": {"downweighted_by_channel": [0, 0, 0], "downweighted_points": 0, "minimum": 1.0}, "merged": {"downweighted_by_channel": [0, 0, 0], "downweighted_points": 0, "minimum": 1.0}}}, "status": "running"}. Allthree train-only populations fully covered, noerrors; no proven gain/804qualification.

- Mechanism diagnostic from immutableepoch16 weightshards (identical578891rawids): {"observed_epoch": 16, "applies_from_epoch": 17, "points": 578891, "downweighted_points": {"member0": 356, "member1": 262, "merged": 235}, "pairwise": {"member0__member1": {"intersection": 77, "union": 541, "only_first": 279, "only_second": 185}, "member0__merged": {"intersection": 132, "union": 459, "only_first": 224, "only_second": 103}, "member1__merged": {"intersection": 116, "union": 381, "only_first": 146, "only_second": 119}}}. Different downgrade sets demonstrate expert-specific decisions, not correctness of anomaly labels or validation gain. Currentepoch18 best78.9017215; continue.

- IndependentQ PID25512 verifiedlive midtraining: {"epoch": 27, "best_epoch": 27, "best_score": 79.25551320093996, "bad_epochs": 0, "status": "running", "matched_best_delta": -0.12677612159951934}. No errors/configchanges, no extraGPUjob, no currentgainclaim. Preserve original fifth804 gate and fullOOFgoal.

- User explicitly continues; independentQ PID25512 unchanged and progressing: {"epoch": 36, "best_epoch": 34, "best_score": 79.63463010605592, "bad_epochs": 2, "status": "running", "matched_best_delta": -0.03987819166145812}. No addedGPUjobs or variant; fullOOFgoal unmet.


- Independent-Q80 current verified observation: {"status": "running", "epoch": 50, "best_epoch": 48, "best_score": 79.89650740310552, "bad_epochs": 2, "matched_best_delta": -0.1055269166322148, "fifth_threshold": 80.4}. Quality tables did diverge after warmup but no terminal benefit yet. Keep training unchanged, original strict fifth >80.4 gate.


## Completed independent-Q result
{"scores": {"POR": 76.7797753364015, "PERM": 89.25273312751027, "SW": 73.59760014451211, "TOTAL": 80.03154924612828}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1468.3607338000147, "peak_gpu_allocated_mib": 783.37353515625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_posterior": -0.21934452869095367, "qualified": false}. Expert-specific downgrades were real (epoch16 member0/1 overlap77 of356/262) but did not improve fifth official score. No remaining folds or Q threshold grid. Three-role CDF scratch removed on normal finalize; source/config/compact evidence remain. Redundant failed checkpoint/cache cleanup remains pending prior automatic-policy rejection.

- Three-view posterior router80 launched PID76124/session73782 after independentQ runner exit0; full206+calibrated68+point41 specialists,104->32->9 router initialized uniform, 24 conditional components. Own-NLL averaged across3, original mergedQ, fixed original fifth80/18/.0006/strict>80.4, normalGPU. GPU guard admitted sole job, free10.8GiB; actual fifth score and memory pending. No historical delivery edits or fivefold promotion.

- Three-view fifth live midpoint: {"status": "running", "epoch": 16, "best_epoch": 15, "best_score": 78.82091691194584, "bad_epochs": 1, "matched_best_delta": 0.1797761518452603, "fifth_threshold": 80.4}. Positive matched-epoch evidence at epochs12 and15, not completed gain or strict80.4 qualification. Actual peak at first full probe981.98MiB; no training error. Continue original80/18; no concurrent speculative GPU job.

- New user criterion: an ablated trick can justify further isolated experimentation if it beats its fair same-capacity control, even when both are below SOTA. The original fifth-fold strict>80.4 promotion gate is unchanged. Primary point41 three-view PID76124 currently running; matched control68 PID39512/session47513 launched as secondGPUjob, same third globalcosine member/router/loss/init/seed with only third input mask41 vs68. Compare completed bestcheckpoint scores, not cherry-picked epochs. No fullCV promotion from below-gate scores.

- Fixed same-capacity third-view ablation running: {"primary_epoch": 50, "primary_best": 79.86918331281122, "primary_best_epoch": 47, "delta_to_two_member_matched_best": -0.13285100692651497, "control_epoch": 11, "control_best": 78.29323237378337, "control_best_epoch": 11, "same_epoch_control_delta_provisional": -0.008841605933270102, "status": "both_running", "fifth_promotion_requires_strictly_more_than": 80.4}. The same-epoch control delta is provisional, not final gain; report completed best scores for point41 and control68. FullOOFgoal still unmet.

- Three-view fixed fifth live ablation: {"primary_epoch": 64, "primary_best": 80.14126652398703, "primary_best_epoch": 52, "delta_to_two_member_best": -0.10962725083220448, "control_epoch": 24, "control_best": 79.29567291736666, "control_best_epoch": 23, "same_epoch_control_delta_provisional": -0.032519996150540464, "status": "both_running", "promotion_fifth_gt": 80.4}. Control is still early; provisional matched-epoch delta shrank from+0.146689 at18 to current. Only final best-checkpoint comparison establishes trick gain; neither run currently qualifies for remainingfolds.


## Completed 41-view three-member result
{"scores": {"POR": 77.09638180433555, "PERM": 89.3505422865114, "SW": 73.54189194973536, "TOTAL": 80.14126652398703}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1749.3763701000134, "peak_gpu_allocated_mib": 984.41845703125, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_two_member": -0.10962725083220448, "qualified": false, "view_ablation": "pending_control68_final"}. Fifth best checkpoint restored and reproduced; does not pass strict80.4, so no remainingfolds. Final restricted-view trick conclusion waits for same-capacity control68. Peak is allocated GPU memory, not total device occupancy; reported foldtime includes mixed solo/concurrent phases and cannot prove concurrency speedup. Source/config/results preserved; failed weight/cache cleanup remains pending prior automatic-policy rejection.

- Same-capacity third-member view ablation progress: {"point41_complete_score": 80.14126652398703, "control68_current_best": 80.02995615359912, "control68_epoch": 62, "control68_best_epoch": 52, "control68_bad_epochs": 10, "provisional_point41_minus_control68": 0.11131037038791192, "status": "control_running"}. Point41 completed at80.1412665; control68 still running, so positive delta is provisional. Both currently below strict fifth80.4; no remainingfolds.


## Completed same-capacity view ablation
{"primary_point41_score": {"POR": 77.09638180433555, "PERM": 89.3505422865114, "SW": 73.54189194973536, "TOTAL": 80.14126652398703}, "control_uncalibrated68_score": {"POR": 76.82851317373027, "PERM": 89.26959969957728, "SW": 73.53440659036569, "TOTAL": 80.02995615359912}, "two_member_reference_score": {"POR": 77.2904355563676, "PERM": 89.36766804409764, "SW": 73.67165512135652, "TOTAL": 80.25089377481923}, "point41_minus_control68_total": 0.11131037038791192, "point41_minus_two_member_total": -0.10962725083220448, "control68_minus_two_member_total": -0.2209376212201164, "primary_best_epoch": 52, "control_best_epoch": 52, "primary_stopped_epoch": 70, "control_stopped_epoch": 70, "both_best_checkpoints_reproduced": true, "fifth_promotion_gate": 80.4, "qualified_for_remaining_folds": false, "control68_router_usage": {"rows": 144665, "channel_mean_weight": [[0.5765758156776428, 0.32503142952919006, 0.0983920618891716], [0.4956466853618622, 0.38557133078575134, 0.11878447979688644], [0.4361644983291626, 0.36888808012008667, 0.1949477344751358]], "third_member_selected_fraction": [0.0, 0.0, 0.19839629488818997]}}. Restricting the third uncalibrated globalcosine expert to point41 improved the fixed fifth by0.1113103704 against uncalibrated68 with same size/seed/loss/route. This is a local positive feature trick, not an absolute improvement over original twoexpert80.2508938 or proof of fivefold gain. Neither passes strict80.4; no continuation. Do not add further triple expert sweeps. Redundant failed checkpoints/caches remain pending previous automatic deletion-policy rejection.

- User steers back to original posterior twoexpert80.2508937748 and feature-first mechanisms. Spike-side candidate launched PID77408/session19506 as soleGPUjob: nine raw curve per-well local robust impulse scores (median11 residual/robust MAD51 with well-IQR floor), appended input-only to original standardized206; original model initialized first then fork_rng zero linear9->206 correction, same original twoexperts/router/loss/Q/split/seed/80-18/strict804. Preprocessing CPU cost and final score pending. Exact41-vs68 tripleview positive local ablation+0.111310 but both below twoexpert; no more triple exploration.

- Feature-first spike-side twoexpert live: {"status": "running", "epoch": 9, "best_epoch": 9, "best_score": 77.92968087362118, "bad_epochs": 0, "matched_best_delta": 0.10363317050951082, "feature_preprocess_seconds": 25.831178699998418, "initial_probe_peak_mib": 787.189453125, "fifth_promotion_gt": 80.4}. Actual preprocessing/first probes complete; no terminal gain or qualification. The nine scores come only from original curves and are distinct from existing missing flags. Keep fixed control and avoid a threshold/window grid if negative.

- Feature-first same-capacity ablation launched PID62208/session68419 as secondGPUjob; original spike-side primary PID77408 remains active. Control recomputes same nine raw-curve spike scores, then repeats per-well mean across depth, preserving well-level quality but removing pointwise local variation. Exact same twoexpert model, side adapter, parameter count, original206, init/seed/loss/Q/fifth/80-18. GPU guard admitted primary as soleother job; no gain claim until both final best checkpoints. Positive primary-control delta can support local feature trick even below SOTA; remainingfolds require strict>80.4.

- Feature-first pointwise spike versus per-well mean side ablation live: {"pointwise_epoch": 40, "pointwise_best": 79.7509418184367, "pointwise_best_epoch": 39, "delta_vs_base_matched_best": 0.007914018863488081, "wellmean_epoch": 15, "wellmean_best": 78.63037956441434, "wellmean_best_epoch": 15, "pointwise_minus_wellmean_sameepoch_provisional": 0.15536868672269577, "status": "both_running", "gate_gt": 80.4}. No terminal gain, control remains earlier stage; both jobs normal, at most2GPU. Final pointwise gain only after both best checkpoints are restored.

- Feature-side fifth live comparison: {"pointwise_epoch": 53, "pointwise_best": 80.10351805827565, "pointwise_best_epoch": 52, "delta_vs_base_matched_best": -0.1473757165435785, "wellmean_epoch": 26, "wellmean_best": 79.07925246742936, "wellmean_best_epoch": 26, "pointwise_minus_wellmean_sameepoch_provisional": 0.09844017633677993, "status": "both_running", "gate_gt": 80.4}. Pointwise reached its epoch52 best but falls below original posterior baseline; pointwise-vs-wellmean feature ablation remains provisional until both terminal best checkpoints. No promotion/fullCVclaim.


## Completed pointwise spike-side result
{"scores": {"POR": 77.08775334117843, "PERM": 89.29748555298232, "SW": 73.49449174965234, "TOTAL": 80.10351805827565}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1384.2067233000125, "peak_gpu_allocated_mib": 796.62939453125, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.1473757165435785, "qualified": false, "local_feature_ablation": "pending_wellmean_control_final"}. Final pointwise fifth under originalposterior80.2508938 and strict80.4. Best checkpoint predictions reproduced. Same-capacity per-well mean control at epoch52 best80.0379267 gives provisional local gain+0.0655913, but final feature ablation waits control terminal. No remainingfolds. Actual time includes mixed solo/concurrent phases, not controlled speed benchmark. Source/config/compact scores preserved; redundant negative checkpoints/cache deletion pending earlier automatic-policy rejection.

- Spike gate feature followup launched PID32384/session16300 as secondGPUjob, concurrent with wellmean control PID62208. It reuses nine local impulse scores on original twoexpert posterior, adds zero-initialized Linear9->9 reliability gate only on nine raw-curve positions (90params), leaves other197 input features unchanged. Fixed fifth80/18/strict804/seed/LR/labels/Q. Tests whether carrying the provisional local feature gain via a constrained curve-wise transform improves the dense adapter80.103518/original80.250894; no final gain or fullCVclaim. GPU guard admitted soleother wellmean control.


## Completed local spike feature ablation
{"pointwise_scores": {"POR": 77.08775334117843, "PERM": 89.29748555298232, "SW": 73.49449174965234, "TOTAL": 80.10351805827565}, "wellmean_scores": {"POR": 76.98819551065021, "PERM": 89.10954413639719, "SW": 73.58036469526414, "TOTAL": 80.03792674427653}, "original_posterior_scores": {"POR": 77.2904355563676, "PERM": 89.36766804409764, "SW": 73.67165512135652, "TOTAL": 80.25089377481923}, "pointwise_minus_wellmean_by_channel": {"POR": 0.09955783052821232, "PERM": 0.18794141658513297, "SW": -0.08587294561179704, "TOTAL": 0.06559131399912133}, "pointwise_minus_original_total": -0.1473757165435785, "pointwise_best_epoch": 52, "wellmean_best_epoch": 52, "both_stopped_epoch": [70, 70], "best_predictions_reproduced": true, "fifth_gate_gt": 80.4, "qualifies_remaining_folds": false, "interpretation": "local feature benefit in fixed fifth and matched capacity only; no fivefold gain claim"}. Pointwise spike side gained+0.0655913140 vs per-well mean side with same model/parameter count/fold/seed/training budget; POR and PERM improved, SW decreased. Both below original posterior80.2508938 and strict80.4. Carry only into separate curve-specific gate experiment, not remainingfolds. Redundant negative weight/cache cleanup remains pending earlier automatic-policy rejection.


## Completed curve-specific feature gate
{"scores": {"POR": 77.01290410215469, "PERM": 89.27153740550203, "SW": 73.54011793935105, "TOTAL": 80.08795060134499}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1130.9077053999936, "peak_gpu_allocated_mib": 799.0341796875, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.1629431734742468, "delta_vs_dense_spike": -0.0155674569306683, "qualified": false}. This constrained9raw-feature gate did not beat original posterior80.2508938 or dense pointwise spike adapter80.1035181. No gate-width/threshold/window grid or remainingfolds. Best checkpoint prediction reproduced; timing includes mixed solo/concurrent phases, not controlled throughput comparison. No additional claim from interim matched epochs.

- Feature-first physical-proxy relative-input trial launched PID87896/session53887 as soleGPUjob, originalposterior twoexpert80.2508938 directcontrol. Four composite proxies CNL_DENSITY_POROSITY_DIFF/MEAN, ARCHIE_PROXY, INVASION_LOG_DIFF obtain per-well ZIQR/QNORM/rank input-only transforms (12sidechannels), robust side ZIQR/QNORM clipped[-8,8], raw206 unchanged, original network init plus fork_rng zero Linear12->206. No extra curves/labels, original fixed fifth/seed/80-18/strict804/normal GPU. Same capacity control only if positive. Prior curve_token core mechanism had negative first-fold result, so no duplicate run. GPU guard admitted with no other training. Result pending.

- Same-capacity physical-proxy side ablation control launched PID85604/session44585 as secondGPUjob; primary pointwise12 side PID87896 continues. Control derives same four composite proxies and12 ZIQR/QNORM/rank input-only features then repeats each well mean across all depths. Exact original206/model/params/initialization/fold/seed/loss/Q/80-18/strict804 retained. GPU guard admitted with primary as sole other job. Differences in side variance are explicit; final local-feature trick gain requires both best-checkpoint reports, no promotion without strictfifth>80.4.

- Physical proxy relative side ablation live: {"pointwise_epoch": 31, "pointwise_best": 79.46530734956603, "pointwise_best_epoch": 30, "delta_vs_base_matched_best": -0.15827534172360913, "wellmean_epoch": 15, "wellmean_best": 78.61815460235586, "wellmean_best_epoch": 14, "pointwise_minus_wellmean_sameepoch_provisional": -0.023254950576756528, "status": "both_running", "gate_gt": 80.4}. Current pointwise physical-proxy rank/ZIQR/QNORM does not beat original posterior matched epoch; early wellmean control is slightly ahead. Continue unchanged; no positive trick claim or promotion.

- Prepared but NOT launched pruned-local-view80 source: originalposterior twoexpert member0 masks exactly90 raw9 residual/rolling-range/slope engineered columns, retaining116; member1 calibrated68, original206 tensor and router68 unchanged. Same initialized modules/params, original fixedfifth/seed/80-18/strict804. Hypothesis: explicit local engineered features duplicate U-Net sequence extraction and cause overfit. No grid or twoexpert fullCV unless gatepassed. Two current GPU jobs proxy_relative PID87896 and proxy_wellmean PID85604 still occupy max2; parent will admit only after one is terminal.

- Physical proxy relative-side live comparison: {"pointwise_epoch": 54, "pointwise_best": 80.0770028682628, "pointwise_best_epoch": 52, "delta_vs_base_matched_best": -0.17389090655643713, "wellmean_epoch": 37, "wellmean_best": 79.57473309727872, "wellmean_best_epoch": 30, "pointwise_minus_wellmean_sameepoch_provisional": 0.0001174437843189935, "status": "both_running", "gate_gt": 80.4}. Pointwise currently below originalposterior best and matched wellmean is essentially tied, so no proven trick; keep both until terminal. Prepared member0 local90 feature prune is waiting for GPU slot.


## Completed pointwise physical-proxy result
{"scores": {"POR": 76.9043057173098, "PERM": 89.18637291520439, "SW": 73.68708752213807, "TOTAL": 80.0770028682628}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1290.6329288000125, "peak_gpu_allocated_mib": 1113.3173828125, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.17389090655643713, "qualified": false, "proxy_ablation": "pending_wellmean_control_final"}. Best checkpoint predictions reproduced; originalposterior80.2508938 and strictfifth80.4 not beaten. Same-capacity wellmean comparison awaits control terminal; no remaining folds.

- Pruned-local-view80 launched PID23076/session42907 after primary proxy runner exit0; exact90 local engineered residual/range/slope cols masked only for full206 member0, leaving116; member1 calibrated68 and router68, model params/init/lossQ/fold/seed/80-18/804 unchanged. GPU guard admitted soleother wellmean controlPID85604. No result/promotion claim.


## Completed proxy-relative same-capacity ablation
{"pointwise_scores": {"POR": 76.9043057173098, "PERM": 89.18637291520439, "SW": 73.68708752213807, "TOTAL": 80.0770028682628}, "wellmean_scores": {"POR": 76.61085831497607, "PERM": 89.05409318987304, "SW": 73.54710271085264, "TOTAL": 79.89367605974681}, "original_posterior_scores": {"POR": 77.2904355563676, "PERM": 89.36766804409764, "SW": 73.67165512135652, "TOTAL": 80.25089377481923}, "pointwise_minus_wellmean_by_channel": {"POR": 0.2934474023337259, "PERM": 0.1322797253313439, "SW": 0.13998481128543006, "TOTAL": 0.18332680851598582}, "pointwise_minus_original_total": -0.17389090655643713, "pointwise_best_epoch": 52, "wellmean_best_epoch": 52, "both_stopped_epoch": [70, 70], "best_predictions_reproduced": true, "fifth_gate_gt": 80.4, "qualifies_remaining_folds": false, "interpretation": "fixed fifth local feature benefit; no absolute/fivefold gain claim"}. Pointwise within-well physical proxy side improves fixed fifth by+0.1833268085 against same-capacity wellmean side, including gains in all three channels. Both below originalposterior80.2508938 and strict80.4; carry once into isolated combination with separately positive local spike side, not remainingfolds.

- Combined positive local feature tricks candidate launched PID23476/session77043 as secondGPUjob with pruned-local PID23076. Originalposterior twoexpert80.2508938 architecture/init, original206 unchanged; original raw9 spike side9 and four composite proxy within-well relative side12 enter via separate fork_rng zero Linear9->206 and12->206 corrections. Fixedfifth/seed/bs64/LR.0006/80-18/strict804, same Q/loss/router. Prior matched single-trick gains+0.065591 and+0.183327 are real local ablations but not additive proof; combined score pending. GPU guard admitted with soleother prune job.

- Current feature-first candidates verifiedlive: [{"experiment": "pure_dl_paired_pruned_local_view80_sequence", "epoch": 38, "best_epoch": 37, "best_score": 79.47031766229671, "bad_epochs": 1, "status": "running", "matched_best_delta": -0.20419063542067306}, {"experiment": "pure_dl_paired_combined_side80_sequence", "epoch": 13, "best_epoch": 13, "best_score": 78.54850510158283, "bad_epochs": 0, "status": "running", "matched_best_delta": -0.09263565851775013}]. Pruning90local columns currently negative; combined two positive side features early positive but not proven. No thirdGPUjob, original strict804 gate/fullOOF>82 unchanged.


## Partial negative result and early stop
{"status": "partial_terminated", "pid": 23076, "last_epoch": 51, "best_epoch": 47, "best_score": 79.6781314736185, "bad_epochs": 4, "comparison_control_score_at_50": 80.00203431973773, "reason": "At least three stagnant epochs and >0.7 below strict fifth80.4; explicit user permits early stop when unlikely to recover. Partial trial only; no restored-checkpoint score claim.", "terminal_runner_exit_code": 1, "trainer_process_missing_after_stop": true, "best_checkpoint_not_restored_or_exported": true, "below_gate": true}. Masking90 full-member local features was substantially behind the original twoexpert in fixed fifth at epoch51; no verified best checkpoint export, no score promotion, no remainingfolds, no mask grid. Early termination freed one GPU slot for the stronger ongoing combined feature trial. Source/config/history/decision retained; redundant incomplete checkpoint cleanup pending prior automatic-policy rejection.

- Specialist-aligned side features candidate launched PID53424/session71872 as secondGPUjob concurrent with shared combined-side PID23476. Same9 local spike +12 physical-proxy relative side/original206, same zero adapter params and original twoexpert init. Proxy adapter only corrects member0 full206; spike adapter only member1 calibrated68; router receives uncorrected original68 plus detached posterior descriptors. OwnNLL trains respective adapters, direct router-input gradient differs from shared combo; mechanism-level comparison not falsely presented as one-factor assignment. Original Q/loss/fold/seed/bs64/LR.0006/80-18/strict804, normalGPU. Admission guard saw soleother combined job, result pending.

- Specialist-aligned vs shared combined side feature live: {"aligned_epoch": 11, "aligned_best": 78.25248593239, "aligned_best_epoch": 11, "aligned_vs_base_matched": 0.0005688092188762539, "aligned_vs_shared_matched": -0.20506423927267292, "shared_epoch": 56, "shared_best": 80.04636824958686, "shared_best_epoch": 52, "both_running": true, "fifth_gate_gt": 80.4}. Early alignment advantage does not prove terminal mechanism gain; shared combination at epoch52 remains under original80.2508938. No promotion or fullOOFgain claim.


## Completed shared combined-side result
{"scores": {"POR": 76.91530325242242, "PERM": 89.2133734033448, "SW": 73.56313309339848, "TOTAL": 80.04636824958686}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1327.3541269999987, "peak_gpu_allocated_mib": 1138.2529296875, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.2045255252323699, "delta_vs_best_single_spike": -0.05714980868879138, "delta_vs_best_single_proxy": -0.030634618675932757, "qualified": false}. Both locally positive feature tricks did not combine additively in the shared correction path; no gain over either single trick or originalposterior. Best checkpoint predictions reproduced. No remainingfolds or feature-window/grid sweep.

- SW cross-target posterior conditioner launched PID26408/session34835 as secondGPUjob with aligned-side PID53424. Original posterior twoexpert init, two independent zero8->16GELU->9 modules read detached own POR/PERM posterior descriptors to correct only SW main default logit/8means; no target truth input, no SWscale/weight or aux changes. Shared backbone can alter POR/PERM trajectories indirectly. Fixed fifth/seed/bs64/LR.0006/80-18/strict804; GPU guard admitted soleother aligned job. Outcome pending.

- Two active fifth-fold mechanisms verifiedlive: [{"experiment": "pure_dl_paired_aligned_side80_sequence", "epoch": 54, "best_epoch": 52, "best_score": 79.9487424422805, "bad_epochs": 2, "status": "running", "matched_best_delta": -0.3021513325387275}, {"experiment": "pure_dl_paired_sw_conditioned80_sequence", "epoch": 18, "best_epoch": 17, "best_score": 78.84210946840801, "bad_epochs": 1, "status": "running", "matched_best_delta": -0.136910894575351}]. Aligned feature route remains below original posterior; SW target-conditioner early matched-epoch gain has not been established at final checkpoint. Max2GPU and strict804/fullOOF>82 invariants unchanged.


## Partial negative aligned-side result
{"status": "partial_terminated", "pid": 53424, "last_epoch": 63, "best_epoch": 52, "best_score": 79.9487424422805, "bad_epochs": 11, "comparison_original_posterior_best": 80.25089377481923, "strict_fifth_gate": 80.4, "reason": "Stagnant >=8 epochs, >0.45 below strict fifth gate and behind original; user permits early stop when unlikely to recover. Partial trial only, no restored-checkpoint score claim.", "terminal_runner_exit_code": 1, "trainer_process_missing_after_stop": true, "best_checkpoint_not_restored_or_exported": true, "below_gate": true}. Different per-expert feature assignment remained behind original posterior and far below strict fifth gate, so no completed score/remainingfolds or aligned-feature grid claim. Source/config/history/decision retained.

- SW means-only conditioner launched PID33536/session54298 as secondGPUjob concurrently with original SW gate+mean conditioner PID26408. Exact same detached own POR/PERM descriptors/hidden weights init, but zero-last output removes SW default gate correction and retains8 SW conditional component mean corrections. OwnNLL/Q/fold/seed/bs64/LR.0006/80-18/strict804 unchanged. Only34 fewer params; component ablation, not exact same-size control. GPU guard admitted soleother SW-conditioned job. Result pending.

- SW posterior conditioning component ablation live: {"gate_mean_epoch": 41, "gate_mean_best": 79.74221713470723, "gate_mean_best_epoch": 37, "gate_mean_delta_vs_base_matched": -0.0008106648659804705, "mean_only_epoch": 9, "mean_only_best": 77.9721864220753, "mean_only_best_epoch": 9, "mean_only_delta_vs_gate_mean_sameepoch_provisional": 0.15941979359693903, "both_running": true, "fifth_gate_gt": 80.4}. Early mean-only edge is provisional; final best-checkpoint scores and channels required, no fifthgate/fullOOFclaim.

## 2026-09-27 双专家特征优先续进
- 原后验双专家第五折对照80.2508937748。SW gate+means条件化已完整结束：best48/stop66，官方TOTAL80.0967730451，较原双专家-0.1541207297；最佳checkpoint预测已复现，未过严格>80.4，不补折。独立记录 outputs/pure_dl_paired_sw_conditioned80_sequence/README.md。
- SW仅均值消融继续运行PID33536；新局部稳健特征实验运行PID7748，GPU守卫最多2任务。新试验对9曲线计算井内101/401点稳健位置，只注入第二专家校准层；原206、第一专家及路由输入保持。独立记录 outputs/pure_dl_paired_local_robust80_sequence/README.md，结果待定。
- 当前完整OOF最好仍81.7132331848，>82目标未完成。下一步盯完两个固定第五折；仅严格>80.4的候选核验后进入剩余折。

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

- 2026-09-27 用户指定原80.2508938后验双专家新增两种子、等权分布投票：新seed20261007/08固定第五折79.891599/79.924450；三checkpoint重算动作100%一致。三等权144665行TOTAL80.3103506（较原+0.059457），未过严格>80.4。随后用户明确停止增种子、冻结三专家只学当前折权重：64训练井32768点静态NLL权重80.2955842、静态官方动作软奖励80.3078684、48/16训练井内部分割的小849参数逐点后验路由80.2236404；验证标签拟合0行，三方案均未胜等权。动态负router.pt已删除，关键三专家权重保留；不补其余四折，完整OOF最好81.7132332仍低于>82目标。源码及完整独立记录见outputs/pure_dl_posterior_three_seed_equal80、pure_dl_posterior_learned_vote_fold5、pure_dl_posterior_metric_vote_fold5、pure_dl_posterior_dynamic_vote_fold5。

- 2026-09-27 冻结三专家元路由慢速训练校正：全578891训练点bs8192/LR.001首轮80.247013、best1/stop19；用户要求更慢并捕捉中间状态后，固定bs16384/LR.0001、更新0和每8步第五折官方TOTAL验证，96次中best更新16=80.310378（初始80.310355仅+0.000023）、stop19、最佳预测复现、GPU峰3980MiB，无权重落盘。验证标签仅选checkpoint不进梯度。三等权80.310351有局部收益但不足>80.4；静态/动态/慢速票权路线均不应继续扫参，不补剩余折。

- ???????????????????????????3???0.1??+0.02???????/????????????????????OOF>82???????????????????outputs/pure_dl_posterior_validation_grid_fold5/README.md?

- ??????????????3????0.1/?0.02???80.3552484461????80.3957094753???80.3103506050?+0.0853588703??POR[.7,.2,.1]/PERM[.3,.36,.34]/SW[.8,0,.2]??????/07/08??144665???????????52.06?/1893.52MiB????80.4???????????????OOF???????????????????????/????????????OOF>82??????outputs/pure_dl_posterior_validation_grid_fold5/README.md?

- ?????????????50/50????????80+??75???TOTAL80.2706366072???+0.0197428324?????seed??80.3103506050???80.4?????100%???144665???????GPU40.07?/601.56MiB???checkpoint???outputs/pure_dl_fixed_distribution_blend_fold5/README.md????????????????????????OOF>82?????

- ?????????????25%????stride2???stride1????Q??stride1?????/80???/18patience????>80.4????????????outputs/pure_dl_paired_sampling_stride80_sequence/README.md????????????

- ????PID81932????epoch 6???best 77.77040699643632??????773.32MiB??????????????????

- Live stride-sampling comparison (not final): {"status": "running", "pid": 81932, "epoch": 14, "best_epoch": 14, "best_score": 78.60641851039014, "control_same_epoch_best": 78.65071599486306, "matched_delta": -0.0442974844729207, "bad_epochs": 0, "decision": "continue; early evidence insufficient for termination or promotion"}

- Verified live observation: {"status": "running", "pid": 81932, "epoch": 28, "best_epoch": 28, "best_score": 79.40327594211027, "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.06345054447118059, "bad_epochs": 0, "decision": "continue live training; no completed gain or promotion claimed"}

- Verified live observation: {"status": "running", "pid": 81932, "epoch": 44, "best_epoch": 38, "best_score": 79.78546102252159, "control_same_epoch_best": 79.89736194351701, "matched_delta": -0.11190092099542426, "bad_epochs": 6, "decision": "continue; recent gains, below promotion threshold"}

- Completed stride-sampling augmentation: fold5 TOTAL79.9262807508, best52/stop70, delta original=-0.3246130240; compact control=-0.3007343847. Best predictions reproduced;144665 rows covered. Single-GPU1093.58s/776.25MiB. No checkpoint retained; no remaining folds or stride sweep. Full OOF>82 unmet. See outputs/pure_dl_paired_sampling_stride80_sequence/README.md.

- PCGrad implementation prepared and syntax parsed; NOT trained yet. Earlier combined code-write/background-launch rejected, but separate code-only edits succeeded. Separate expert target losses, project full expert parameter gradients, router unchanged. Record outputs/pure_dl_paired_posterior_pcgrad80_sequence/README.md; gains unmeasured, OOF>82 unmet.

- PCGrad observable session86736 launched successfully after code preparation; {"status": "running", "session_id": 86736, "epoch": 2, "best_score": 76.03787183543336, "observed_peak_gpu_mib": 823.705078125, "launch": "foreground observable tool session; accepted", "claim": "early result only, gain unproven"}. Earlier combined-command rejection did not recur. First-epoch projection conflict fraction0.0208333; not accuracy-gain evidence.

- Verified live PCGrad observation: {"status": "running", "session_id": 86736, "epoch": 9, "best_score": 78.00179343013366, "best_epoch": 9, "control_same_epoch_best": 78.02272835633327, "matched_delta": -0.020934926199615234, "bad_epochs": 0, "peak_gpu_mib_observed": 823.92431640625, "decision": "continue, near baseline; gain unproven"}. First5 epochs wall160.6s including preprocessing; serial single GPU.

- Verified live PCGrad: {"status": "running", "session_id": 86736, "epoch": 16, "best_epoch": 15, "best_score": 78.69891102391998, "control_same_epoch_best": 78.73471136844955, "matched_delta": -0.03580034452957648, "bad_epochs": 1, "observed_peak_gpu_mib": 823.92431640625, "decision": "continue; no final gain claim"}; first15 epochs380.6s including preprocessing/probes, serial.

- Verified live PCGrad: {"status": "running", "session_id": 86736, "epoch": 27, "best_epoch": 27, "best_score": 79.31530419693895, "control_same_epoch_best": 79.33956050045694, "matched_delta": -0.02425630351798702, "bad_epochs": 0, "decision": "continue live trial; final gain unproven"}

- Verified live PCGrad: {"status": "running", "session_id": 86736, "epoch": 36, "best_epoch": 36, "best_score": 79.6633109215126, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.0608547436038549, "bad_epochs": 0, "decision": "continue, final gain unproven"}

- Verified live PCGrad: {"status": "running", "session_id": 86736, "epoch": 46, "best_epoch": 44, "best_score": 79.95880288118829, "control_same_epoch_best": 79.89736194351701, "matched_delta": 0.06144093767127856, "bad_epochs": 2, "decision": "continue; no complete gain claim"}

- Verified live PCGrad: {"status": "running", "session_id": 86736, "epoch": 57, "best_epoch": 52, "best_score": 80.15828559470478, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.06872954082429317, "bad_epochs": 5, "decision": "continue; below original and promotion threshold"}

- Completed PCGrad fold5: TOTAL80.1582855947, best52/stop70, delta original=-0.0926081801, compact=-0.0687295408. Best predictions reproduced;144665 rows covered once;1551.86s single GPU/823.92MiB; no checkpoint retained. No final gain, no projection sweeps or remaining folds; next route must change mechanism. Full OOF>82 unmet. See outputs/pure_dl_paired_posterior_pcgrad80_sequence/README.md.

- Prepared fixed1024-bin posterior integration versus original256, same800-action grid and equal3 experts, no weight search/training. Record outputs/pure_dl_posterior_fine_quadrature_fold5/README.md, result pending.

- Completed fixed1024 posterior quadrature: equal3 fold5 TOTAL80.3194134427 versus256 baseline80.3103506050, delta+0.0090628377. Original single also+0.0079939077. Fixed weights/no label-tuned resolution;144665 rows covered.54.03s/1438.96MiB single GPU. Small positive precision trick retained, no remainingfolds (<80.4), full OOF>82 unmet. See outputs/pure_dl_posterior_fine_quadrature_fold5/README.md.

- Prepared fixed4x action-grid refinement on1024-bin equal3 ensemble; no weight fitting. Compare80.3194134427. Pending, record outputs/pure_dl_posterior_fine_actions_fold5/README.md.

- Completed action-grid4x refinement with fixed1024-bin equal3: fold5 TOTAL80.3220631614, +0.0026497187 versus fine integration and +0.0117125564 versus original equal3.144665 rows,54.31s/1466.40MiB; no weights trained or label tuning. Below80.4; no remaining folds. Diminishing precision gains, retain decoder but change next modeling route. See outputs/pure_dl_posterior_fine_actions_fold5/README.md.

- Prepared overlapping-window fixed equal posterior averaging: stride64/core128, exactly2 predictions per raw point, frozen3 experts and fine decoder. Compare80.3220631614; pending. Record outputs/pure_dl_posterior_overlap_fold5/README.md.

- Completed fixed2-window overlap ensemble: fold5 TOTAL80.3538481813, +0.0317850199 vs same fine-decoder nonoverlap80.3220631614. All3 channels improve,144665 points each exactly2 predictions.62.13s/3163.29MiB single GPU; no weights fitted/no checkpoint. Positive mechanism, still0.0461518187 below80.4; no remaining folds. Record outputs/pure_dl_posterior_overlap_fold5/README.md.

- Prepared fixed4-window overlap (stride32/core128), compare positive2-window80.3538481813. Same3 equal experts/fine decoder; no trained or label-fitted weights. Pending, outputs/pure_dl_posterior_overlap4_fold5/README.md.

- Completed fixed4-window overlap: fold5 TOTAL80.3627226809, +0.0088744997 versus2-window; POR/PERM up, SW slightly down. Exactly4 coverage of144665 points,79.54s/3163.25MiB, fixed votes/no retraining. Below80.4 by0.0372773191; no remainingfolds. Gains diminishing, retain positive inference recipe; next mechanism rather than endless overlap density. outputs/pure_dl_posterior_overlap4_fold5/README.md.

- Prepared geometric4-window consensus versus arithmetic4 baseline80.3627226809; models remain arithmetic equal3, fixed probability floor1e-12, no fitting. Pending. outputs/pure_dl_posterior_logpool4_fold5/README.md.

- Completed geometric4-window pool: fold5 TOTAL80.3595001347, -0.0032225462 vs arithmetic4. No overall gain; do not sweep pooling temperature/floor/channel switches. Current fixed arithmetic4 recipe80.3627226809 remains.144665 points exactly4 coverage;79.29s/3163.25MiB; no new checkpoint/no remainingfolds. Record outputs/pure_dl_posterior_logpool4_fold5/README.md; full OOF>82 unmet.

- Prepared conditioned rank8 numeric cross mixing in calibrated expert only; zero-init residual759params, original posterior network/loss/Q otherwise unchanged. Fixedfold5/80-18, no PCGrad/stride augmentation. Pending, outputs/pure_dl_paired_conditioned_mix80_sequence/README.md.

- Launched and verified live: {"status": "running", "session_id": 71759, "epoch": 5, "best_score": 77.46493622360362, "best_epoch": 5, "matched_delta": -0.1473460869176222, "peak_gpu_mib_observed": 782.7939453125, "decision": "continue; early gain unproven"}

- Verified live conditioned-mix: {"status": "running", "session_id": 71759, "epoch": 20, "best_epoch": 20, "best_score": 79.07873304423393, "control_same_epoch_best": 79.0916641270535, "matched_delta": -0.012931082819577, "bad_epochs": 0, "decision": "continue; no final gain claim"}

- Verified live conditioned-mix: {"status": "running", "session_id": 71759, "epoch": 35, "best_epoch": 28, "best_score": 79.64502255064163, "control_same_epoch_best": 79.56187209225561, "matched_delta": 0.0831504583860152, "bad_epochs": 7, "decision": "continue; no complete gain claim"}

- Verified live conditioned-mix: {"status": "running", "session_id": 71759, "epoch": 50, "best_epoch": 48, "best_score": 79.94440142742305, "control_same_epoch_best": 79.93028980255141, "matched_delta": 0.014111624871631534, "bad_epochs": 2, "decision": "continue; no completed gain claim"}

- Completed conditioned rank8 mixing: fold5 TOTAL80.1954269324, best52/stop70, delta original=-0.0554668424, compact=-0.0315882031. No gain; no rank/width sweep or remaining folds. Best predictions reproduced/144665rows covered,1119.55s single GPU/794.94MiB, no checkpoint retained. outputs/pure_dl_paired_conditioned_mix80_sequence/README.md. Full OOF>82 unmet.

- Prepared Lookahead on original posterior network: k5/alpha0.5, current fast iterate validation, original AdamW moments/schedule retained, no extra forward/backward. Fixedfold5/80-18; pending. outputs/pure_dl_paired_lookahead80_sequence/README.md.

- Verified live Lookahead: {"status": "running", "session_id": 72874, "epoch": 4, "best_score": 76.72847189448831, "best_epoch": 4, "observed_peak_gpu_mib": 778.30908203125, "decision": "continue; early evidence only"}

- Verified live Lookahead: {"status": "running", "session_id": 72874, "epoch": 15, "best_epoch": 14, "best_score": 77.98954458290324, "control_same_epoch_best": 78.73471136844955, "matched_delta": -0.7451667855463171, "bad_epochs": 1, "decision": "continue, early convergence slower; no gain claimed"}

- Verified live Lookahead: {"status": "running", "session_id": 72874, "epoch": 29, "best_epoch": 29, "best_score": 78.91231174273345, "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.5544147438479996, "bad_epochs": 0, "decision": "continue monitoring; slower learning with recent gains, no benefit proven"}

- Lookahead manually terminated after45 completed epochs: observed best79.3084432291@45, substantial matched-control deficit; no complete-fold convergence/restore claim. Original budget80, user-authorized early termination. Session72874 exit1 confirmed after PID59080 stop; no checkpoint exported, no remainingfolds, no k/alpha sweep. Compact partial evidence outputs/pure_dl_paired_lookahead80_sequence/manual_early_stop.json. Full OOF>82 unmet.

- Prepared fixed4-model equal overlap4: add historical calibrated75 to original/07/08, same fine decoder. Compare80.3627226809. No coefficient search or new training; pending, outputs/pure_dl_four_model_overlap4_fold5/README.md.

- Completed fixed4-model overlap4 equal blend: fold5 TOTAL80.3794015534, +0.0166788725 versus3-model80.3627226809. POR improves, PERM/SW slight drops.144665 rows exactly4 contexts,91.68s/3189.37MiB; no weight fitting/new checkpoints. Positive complementarity; still0.0205984466 below strict80.4, no remainingfolds. outputs/pure_dl_four_model_overlap4_fold5/README.md. Full OOF>82 unmet.

- Prepared fixed cosine-position window weighting on equal4 models: sin^2 core weights, per-row normalization, no label fit. Compare80.3794015534. Pending, outputs/pure_dl_four_model_cosine_window_fold5/README.md.

- Completed fixed cosine-window weights on4 equal models: fold5 TOTAL80.3818486966, +0.0024471432 vs uniform80.3794015534. POR/PERM up,SW down; tiny singlefold benefit.144665 points coverage4/weight sum2 verified;90.37s/3189.94MiB. No label-fit/new weights. Still0.0181513034 below80.4; no rounding or remainingfolds. Record outputs/pure_dl_four_model_cosine_window_fold5/README.md; full OOF>82 unmet.

- Prepared fixed5-model test adding single global-cosine expert79.989867 to4-model80.381849, equal1/5 and same cosine-window/fine decoder. No coefficient search or training. Pending, outputs/pure_dl_five_model_cosine_window_fold5/README.md.

- Completed fixed5-model equal mixture: fold5 TOTAL80.3868607384, +0.0050120418 vs4-model. POR up,PERM/SW down; tiny net benefit.144665 rows coverage4,99.03s/3193.61MiB, no fit/new weights. Still0.0131392616 below strict80.4, no remainingfolds or validation-selected channel switches. outputs/pure_dl_five_model_cosine_window_fold5/README.md; full OOF>82 unmet.

- Prepared geometric pool across5 models, keep cosine arithmetic window weights; compare80.3868607384. Distinct from earlier window-geometric negative, fixed/no temperature grid. Pending, outputs/pure_dl_five_model_geometric_fold5/README.md.

- Completed model-geometric pooling: TOTAL80.3551652795, -0.0316954589 vs arithmetic5; all3 targets decrease. Exactly4 context coverage/144665rows,98.19s/3193.97MiB. No new checkpoint/no remainingfolds/no geometric-temperature grid. Keep arithmetic5 best80.3868607384. outputs/pure_dl_five_model_geometric_fold5/README.md; full OOF>82 unmet.

- Prepared full-expert-only affine RMS normalization (9 depth-local blocks, same parameter count), calibrated expert unchanged. Fixedfold5/80-18, original optimizer/loss/Q, no failed mechanisms. Pending, outputs/pure_dl_paired_rms_full80_sequence/README.md.

- User supersedes fixed expert voting: prepared5-frozen-model learned targetwise logits, train578891 points NLL,lr1e-4/bs16384,80max/18patience, choose validation best including initialization. Fine4-window validation cache CPU, one target toGPU. Pending; outputs/pure_dl_five_learned_slow_vote_fold5/README.md. RMS experiment continues separately.

- Learned-vote session6338 live during frozen-expert cache preparation; no weights-update result yet. Concurrent RMS session76251 live, max2 jobs.

- Live learned-vote: {"status": "training_weights", "session_id": 6338, "epoch": 19.0, "update": 684, "best_score": 80.38689192147532, "best_epoch": 0.2222222222222222, "initial_equal_score": 80.38685419388258, "validation_evaluations": 28, "trainable_logits": 15, "learning_rate": 0.0001, "batch_size": 16384, "validation_gradients": false}. CPU full cache initialized successfully; initial score agrees prior recipe within0.000007. System RAM was tight (~1.5GB free) during cache construction; no additional jobs.

- Completed user-requested small-step learned5-model weights:15 logits,all578891 train points,lr1e-4/bs16384; best80.3868919215@update8, initial80.3868541939 (+0.00003773),stop19/28validations; trainNLL improves but validation worsens. Best tiny router checkpoint restored/saved, near20%weights in learned_weights.json.460.85s overlapping RMS/3170.28MiB; CPU cache RAM only released on exit. Below80.4, no remainingfolds; full OOF>82 unmet. outputs/pure_dl_five_learned_slow_vote_fold5/README.md.

- User reprioritized: finish current RMS experiment only, then return to five-model80.3868919215 baseline and prioritize features/neural architecture. No further vote/window/decoder/optimizer sweeps queued. Baseline is ensemble, preserve tiny learned vote checkpoint and source models; strictfold5>80.4/fullOOF>82 remain. RMS live {"status": "running", "session_id": 76251, "epoch": 35, "best_score": 79.57381986395312, "best_epoch": 32, "next_direction": "finish RMS; feature engineering and architecture around frozen80.38689 ensemble; no new voting or decoder sweeps"}

- Verified live RMS: {"status": "running", "session_id": 76251, "epoch": 52, "best_epoch": 52, "best_score": 79.97426485365254, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.2527502818765299, "bad_epochs": 0, "decision": "finish active idea; then features/architecture around80.38689, no new vote sweeps"}

- Completed final in-hand RMS route: fold5 TOTAL79.9742648537, best52/stop70, delta original=-0.2766289212, compact=-0.2527502819. Best predictions reproduced,144665rows covered;1358.13s (partly concurrent learned-vote),797.44MiB. No checkpoint retained/no remainingfolds/no RMS sweeps. Prior active routes all closed; next feature engineering/neural architecture around frozen80.3868919215 ensemble per latest user direction. outputs/pure_dl_paired_rms_full80_sequence/README.md. Full OOF>82 unmet.

- Prepared input missing-gap geometry feature trial:18 previous/next valid-depth distances for9 curves, zero-init calibrated-expert injection; original206 intact/total224. Feature-first isolated ablation for80.39 ensemble, fixedfold5/80-18. No vote/decoder/optimizer sweeps. Pending, outputs/pure_dl_paired_gap_distance80_sequence/README.md.

- Launched gap-feature trial, verified live: {"status": "running", "session_id": 10938, "features_total": 224, "added_features": 18, "added_parameters": 2654, "baseline_ensemble": 80.38689192147532, "gain": "unproven", "epoch": 2, "best_score": 76.63893252314224, "best_epoch": 2}

- Verified live gap-feature trial: {"status": "running", "session_id": 10938, "epoch": 14, "best_epoch": 12, "best_score": 78.59573864520242, "control_same_epoch_best": 78.65071599486306, "matched_delta": -0.05497734966064627, "bad_epochs": 2, "decision": "continue; feature gain not yet established"}

- Verified live gap-feature trial: {"status": "running", "session_id": 10938, "epoch": 25, "best_epoch": 23, "best_score": 79.10922139080138, "control_same_epoch_best": 79.14794099275363, "matched_delta": -0.03871960195225199, "bad_epochs": 2, "decision": "continue; no final feature gain claim"}

- Verified live gap-feature trial: {"status": "running", "session_id": 10938, "epoch": 35, "best_epoch": 30, "best_score": 79.48205522815707, "control_same_epoch_best": 79.56187209225561, "matched_delta": -0.0798168640985466, "bad_epochs": 5, "decision": "continue; incomplete feature ablation"}

- Verified live gap-feature trial: {"status": "running", "session_id": 10938, "epoch": 46, "best_epoch": 44, "best_score": 79.87666990306893, "control_same_epoch_best": 79.89736194351701, "matched_delta": -0.020692040448082594, "bad_epochs": 2, "decision": "continue; recent improvement, final benefit unproven"}

- Verified live gap-feature trial: {"status": "running", "session_id": 10938, "epoch": 57, "best_epoch": 52, "best_score": 79.96809791843255, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.25891721709652415, "bad_epochs": 5, "decision": "observe late convergence; no feature gain proven, no variants queued"}

- Completed gap-distance feature trial: fold5 TOTAL79.9680979184, best52/stop70, delta original=-0.2827958564, compact=-0.2589172171. No total gain; do not scan gap caps or do validation-selected channel mixing. Best restored/144665rows covered,1125.20s singleGPU/825.27MiB; no checkpoint retained/no remainingfolds. Keep80.3868919 ensemble; next distinct feature/architecture mechanism. outputs/pure_dl_paired_gap_distance80_sequence/README.md. Full OOF>82 unmet.

- Prepared patch extrema structural ablation: full-expert mean token plus bounded high/low contrast through zero-init2048-param projection. Calibrated expert/input/loss/Q unchanged; fixedfold5/80-18. Pending, outputs/pure_dl_paired_patch_extrema80_sequence/README.md; preserves80.39 ensemble as later comparison.

- Verified live patch-extrema: {"status": "running", "session_id": 34480, "epoch": 1, "best_epoch": 1, "best_score": 74.26429380187552, "added_parameters": 2048, "decision": "continue; first-epoch result only"}

- Verified live patch-extrema trial: {"status": "running", "session_id": 34480, "epoch": 10, "best_epoch": 10, "best_score": 78.32303595159317, "control_same_epoch_best": 78.2303687909218, "matched_delta": 0.09266716067136826, "bad_epochs": 0, "decision": "continue; early structure ablation only"}

- Verified live patch-extrema trial: {"status": "running", "session_id": 34480, "epoch": 22, "best_epoch": 22, "best_score": 79.07545734344352, "control_same_epoch_best": 79.0916641270535, "matched_delta": -0.01620678360998795, "bad_epochs": 0, "decision": "continue; no completed structural gain claim"}

- Live patch-extrema update: {"status":"running","session_id":34480,"epoch":40,"best_epoch":37,"best_score":79.6243262494133,"control_same_epoch_best":79.74282298156136,"matched_delta":-0.11849673214806,"bad_epochs":3,"decision":"finish current structural trial; retain 80.38689192147532 ensemble; prioritize features and architecture, no further voting sweeps"}

- Verified live patch-extrema trial: {"status":"running","session_id":34480,"epoch":47,"best_epoch":44,"best_score":79.80945012092758,"control_same_epoch_best":79.89736194351701,"matched_delta":-0.08791182258943,"bad_epochs":3,"decision":"continue; recent best improvement, structural benefit still unproven"}

- Verified live patch-extrema trial: {"status":"running","session_id":34480,"epoch":57,"best_epoch":52,"best_score":79.97058959474765,"control_same_epoch_best":80.227015135529072,"matched_delta":-0.25642554078145,"bad_epochs":5,"decision":"observe final convergence; no parameter extensions queued, no structural gain yet"}

- USER STOP: patch-extrema process13912 terminated, session34480 exit1. Observed best fold5=79.97058959474765 at52, last completed validation70; final restored-best verification incomplete. No gain vs original80.2508938 or compact80.2270151; no further variants, no remaining folds, do not auto-resume. Preserve80.3868919 baseline. Full OOF>82 unmet.

- Prepared point-specific patch retrieval: +4284 parameters in full206 expert, original broadcast retained; zero-init residual output. Fifth80/18, unchanged baseline protocol. outputs/pure_dl_paired_point_patch_retrieval80_sequence/README.md. No proven gain yet; preserve80.39 ensemble.

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":3,"best_epoch":3,"best_score":76.60328069074778,"added_parameters":4284,"observed_gpu_peak_mib":786.4755859375,"decision":"continue; early results only, no gain claim"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":9,"best_epoch":9,"best_score":78.04198641257656,"control_same_epoch_best":78.02272835633327,"matched_delta":0.01925805624329,"bad_epochs":0,"decision":"continue; early learning comparison only, no final gain claim"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":15,"best_epoch":14,"best_score":78.6681929656765,"control_same_epoch_best":78.73471136844955,"matched_delta":-0.06651840277305,"bad_epochs":1,"decision":"continue; monitor mid-training comparison, no completed gain"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":20,"best_epoch":20,"best_score":79.09333487567802,"control_same_epoch_best":79.0916641270535,"matched_delta":0.00167074862452,"bad_epochs":0,"decision":"continue; recent progress but no established benefit"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":26,"best_epoch":23,"best_score":79.2641143766807,"control_same_epoch_best":79.14794099275363,"matched_delta":0.11617338392707,"bad_epochs":3,"decision":"continue; benefit not established, no extra variants queued"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":30,"best_epoch":28,"best_score":79.41053237847584,"control_same_epoch_best":79.5169041745716,"matched_delta":-0.10637179609576,"bad_epochs":2,"decision":"continue; monitor late convergence before judging structural benefit"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":36,"best_epoch":36,"best_score":79.61876801291866,"control_same_epoch_best":79.72416566511646,"matched_delta":-0.10539765219780,"bad_epochs":0,"decision":"continue toward late convergence; no structural gain established"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":40,"best_epoch":39,"best_score":79.68457292070298,"control_same_epoch_best":79.74282298156136,"matched_delta":-0.05825006085838,"bad_epochs":1,"decision":"continue; assess later convergence, no extensions queued"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":46,"best_epoch":44,"best_score":79.87659320989175,"control_same_epoch_best":79.89736194351701,"matched_delta":-0.02076873362526,"bad_epochs":2,"decision":"continue; approaching baseline best-epoch region, no extension planned"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":52,"best_epoch":52,"best_score":80.0112295004615,"control_same_epoch_best":80.22701513552907,"matched_delta":-0.21578563506757,"bad_epochs":0,"decision":"observe remaining convergence; no gain established, no variants queued"}

- Verified live point-patch retrieval: {"status":"running","session_id":42235,"epoch":57,"best_epoch":52,"best_score":80.0112295004615,"control_same_epoch_best":80.227015135529072,"matched_delta":-0.2157856350676,"bad_epochs":5,"decision":"observe late convergence; no gain, no variant or remaining-fold launches"}

- Early terminated point-patch retrieval after validation62: observed best80.0112295005 at52, delta original=-0.2396642744, compact=-0.2157856351. Ten stale epochs and late scores79.5-79.7; no gain, no head/width sweep. Session42235 exit1; partial trial, final restored-best verification not completed, no fullCV. Peak789.27MiB; last printed training wall974.5s at60 includes preprocessing/probes, singleGPU, no standalone warmup; not final elapsed. Preserve80.3868919 ensemble; return to a different feature/architecture mechanism. Full OOF>82 unmet.

- Prepared conditional-residual input features: nine signed within-well ridge conditional residuals, no labels, +279 parameters to calibrated expert numeric inputs; fixedfifth80/18. Pending, outputs/pure_dl_paired_conditional_residual80_sequence/README.md. Preserve80.39 ensemble; no anomaly weight change.

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":2,"best_epoch":2,"best_score":76.65892925452033,"observed_gpu_peak_mib":792.17333984375,"decision":"continue; first-epoch feature trial only, no gain claim"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":5,"best_epoch":5,"best_score":77.39262144833572,"control_same_epoch_best":77.61228231052124,"matched_delta":-0.21966086218552,"bad_epochs":0,"decision":"continue; early feature ablation only, no final gain established"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":8,"best_epoch":8,"best_score":77.75052753542641,"control_same_epoch_best":77.74331901761342,"matched_delta":0.00720851781299,"bad_epochs":0,"decision":"continue; early feature learning, final gain unproven"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":12,"best_epoch":12,"best_score":78.50945652779306,"control_same_epoch_best":78.48230718472027,"matched_delta":0.02714934307279,"bad_epochs":0,"decision":"continue; observe whether early feature gains persist"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":15,"best_epoch":15,"best_score":78.71347745927508,"control_same_epoch_best":78.73471136844955,"matched_delta":-0.02123390917447,"bad_epochs":0,"decision":"continue; mid-training feature effect remains unproven"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":18,"best_epoch":17,"best_score":78.78768558988729,"control_same_epoch_best":78.7476960609833,"matched_delta":0.03998952890399,"bad_epochs":1,"decision":"continue; no stable feature gain established"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":22,"best_epoch":20,"best_score":79.12914840899751,"control_same_epoch_best":79.0916641270535,"matched_delta":0.03748428194401,"bad_epochs":2,"decision":"continue; middle-stage comparison only, no final gain claim"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":25,"best_epoch":23,"best_score":79.26524755173939,"control_same_epoch_best":79.14794099275363,"matched_delta":0.11730655898576,"bad_epochs":2,"decision":"continue; middle-stage result only, no final gain established"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":29,"best_epoch":28,"best_score":79.6401154339048,"control_same_epoch_best":79.46672648658145,"matched_delta":0.17338894732335,"bad_epochs":1,"decision":"continue; no completed feature benefit established"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":32,"best_epoch":28,"best_score":79.6401154339048,"control_same_epoch_best":79.56187209225561,"matched_delta":0.07824334164919,"bad_epochs":4,"decision":"continue; assess persistence of feature advantage, no final gain claim"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":36,"best_epoch":35,"best_score":79.69227976749073,"control_same_epoch_best":79.72416566511646,"matched_delta":-0.03188589762573,"bad_epochs":1,"decision":"continue; no final feature gain established"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":39,"best_epoch":39,"best_score":79.74703028690371,"control_same_epoch_best":79.74282298156136,"matched_delta":0.00420730534235,"bad_epochs":0,"decision":"continue single training; system GPU6144MiB at spot check, model allocator peak794MiB, no OOM"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":44,"best_epoch":42,"best_score":79.88829497267294,"control_same_epoch_best":79.89736194351701,"matched_delta":-0.00906697084407,"bad_epochs":2,"decision":"continue; observe late convergence, no proven feature gain"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":48,"best_epoch":48,"best_score":79.91216012746975,"control_same_epoch_best":79.93028980255141,"matched_delta":-0.01812967508166,"bad_epochs":0,"decision":"continue; approaching late selection region, no final gain established"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":52,"best_epoch":52,"best_score":80.15724483023966,"control_same_epoch_best":80.22701513552907,"matched_delta":-0.06977030528941,"bad_epochs":0,"decision":"continue late convergence; no additional variants or remaining folds"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":56,"best_epoch":52,"best_score":80.15724483023966,"control_same_epoch_best":80.22701513552907,"matched_delta":-0.06977030528941,"bad_epochs":4,"decision":"continue late convergence; no feature gain or promotion yet"}

- Verified live conditional-residual trial: {"status":"running","session_id":62389,"epoch":60,"best_epoch":52,"best_score":80.15724483023966,"control_same_epoch_best":80.22701513552907,"matched_delta":-0.06977030528941,"bad_epochs":8,"decision":"observe closing phase; no ridge or feature-width sweep, no promotion"}

- Conditional residuals early terminated after validation62: observed best80.1572448302 at52, delta original=-0.0936489446, compact=-0.0697703053. Ten stale epochs; no demonstrated feature benefit; no ridge/clip/injection-width scan. Session62389 exit1, no restored-best final verification; partial trial, no remainingfolds. Peak805.93MiB. Last printed wall1487.5s at60 includes preprocessing/probes, singleGPU with other desktop GPU usage, not final elapsed or matched speed benchmark. No checkpoint retained. Preserve80.3868919 ensemble; fullOOF>82 unmet.

- Prepared adaptive tied-kernel dilation: full-expert bottleneck only, +4635 parameters, original initialization preserved; fixed fifth80/18. outputs/pure_dl_paired_adaptive_dilation80_sequence/README.md. Pending/no gain; preserve80.39 ensemble.

- Verified live adaptive-dilation trial: {"status":"running","session_id":70695,"epoch":2,"best_epoch":2,"best_score":75.79081405027141,"observed_gpu_peak_mib":784.2099609375,"decision":"continue; early architecture experiment only, no gain claim"}

- Verified live adaptive-dilation trial: {"status":"running","session_id":70695,"epoch":6,"best_epoch":6,"best_score":77.6918836781125,"control_same_epoch_best":77.61228231052124,"matched_delta":0.07960136759126,"bad_epochs":0,"decision":"continue; early architecture comparison only, no final benefit claim"}

- Verified live adaptive-dilation trial: {"status":"running","session_id":70695,"epoch":10,"best_epoch":10,"best_score":78.19794615805795,"control_same_epoch_best":78.2303687909218,"matched_delta":-0.03242263286385,"bad_epochs":0,"decision":"continue; early architecture result only, no established benefit"}

- Verified live adaptive-dilation trial: {"status":"running","session_id":70695,"epoch":14,"best_epoch":14,"best_score":78.65569246727291,"control_same_epoch_best":78.65071599486306,"matched_delta":0.00497647240985,"bad_epochs":0,"decision":"continue; no stable architecture gain established"}

- Feature expansion study completed (not model training): inventory206=9raw+9missing+1relativeDepth+22transforms/proxies+27wellRelative+138local; 64 fold5 train wells584411 rawrows, all depth increments0.1m. Prepared24 directional-depth features (18 signed1m/4m side contrasts,2jointstrength,2participation,2coverage), +33.43MiB float16 inputs; audit4.47s CPU, no labels, no accuracy-gain claim. Source experiments/pure_dl/directional_depth_features.py and analyze_feature_expansion.py; report outputs/pure_dl_feature_expansion_study/README.md. Next isolated feature candidate; do not combine with active adaptive dilation before evidence.

- Prepared isolated24 directional-depth features training on original pair, +1536 zero-init full-expert stem parameters, input230, calibrated expert/router original inputs. Fixedfifth80/18; no adaptive-dilation combination. outputs/pure_dl_paired_directional_depth80_sequence/README.md; gains untested.

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 1, "best_epoch": 1, "best_score": 74.73066014985054, "control_same_epoch_best": 74.81389087114947, "matched_delta": -0.08323072129893205, "bad_epochs": 0, "concurrency": "twoGPU tasks, independent feature/architecture routes", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 36, "best_epoch": 28, "best_score": 79.56697767761044, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.15718798750602048, "bad_epochs": 8, "concurrency": "twoGPU tasks, independent feature/architecture routes", "decision": "continue; no completed gain or promotion"}

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 4, "best_epoch": 3, "best_score": 76.87745627080585, "control_same_epoch_best": 76.62364985406369, "matched_delta": 0.2538064167421652, "bad_epochs": 1, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 39, "best_epoch": 39, "best_score": 79.83645304905673, "control_same_epoch_best": 79.74282298156136, "matched_delta": 0.09363006749536851, "bad_epochs": 0, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 7, "best_epoch": 5, "best_score": 77.65140177849756, "control_same_epoch_best": 77.74331901761342, "matched_delta": -0.09191723911585825, "bad_epochs": 2, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 42, "best_epoch": 39, "best_score": 79.83645304905673, "control_same_epoch_best": 79.76759879965745, "matched_delta": 0.06885424939927987, "bad_epochs": 3, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 11, "best_epoch": 10, "best_score": 78.30373768319913, "control_same_epoch_best": 78.25898459131336, "matched_delta": 0.044753091885766594, "bad_epochs": 1, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 46, "best_epoch": 44, "best_score": 79.97399552262817, "control_same_epoch_best": 79.89736194351701, "matched_delta": 0.07663357911116009, "bad_epochs": 2, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 14, "best_epoch": 12, "best_score": 78.52718089006667, "control_same_epoch_best": 78.65071599486306, "matched_delta": -0.12353510479638885, "bad_epochs": 2, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 50, "best_epoch": 44, "best_score": 79.97399552262817, "control_same_epoch_best": 79.93028980255141, "matched_delta": 0.043705720076758325, "bad_epochs": 6, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 18, "best_epoch": 17, "best_score": 78.85193388128934, "control_same_epoch_best": 78.7476960609833, "matched_delta": 0.10423782030603945, "bad_epochs": 1, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 53, "best_epoch": 52, "best_score": 79.99775116538291, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.22926397014616384, "bad_epochs": 1, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_directional_depth: {"status": "running", "session_id": 63047, "epoch": 21, "best_epoch": 20, "best_score": 79.09520413206015, "control_same_epoch_best": 79.0916641270535, "matched_delta": 0.0035400050066414224, "bad_epochs": 1, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Verified live paired_adaptive_dilation: {"status": "running", "session_id": 70695, "epoch": 57, "best_epoch": 52, "best_score": 79.99775116538291, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.22926397014616384, "bad_epochs": 5, "concurrency": "twoGPU tasks", "decision": "continue; no completed gain or promotion"}

- Adaptive dilation early terminated after validation62: observed best79.9977511654 at52, delta original=-0.2531426094/compact=-0.2292639701, ten stale epochs. Session70695 exit1; no final restored-best verification, no checkpoint retained, no dilation/group/strength grid. Peak787.54MiB, printed wall1579.1s at60 includes preprocessing/probes and mixed solo/concurrent phases, not final elapsed. Feature trial63047 remains active; preserve80.39. FullOOF>82 unmet.

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 32, "best_epoch": 28, "best_score": 79.51991831070023, "control_same_epoch_best": 79.56187209225561, "matched_delta": -0.0419537815553781, "bad_epochs": 4, "concurrency": "soleGPU training after adaptive-dilation stop", "decision": "continue; no established final feature gain"}

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 36, "best_epoch": 36, "best_score": 79.67248226327531, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.05168340184114584, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; no established final feature gain"}

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 40, "best_epoch": 39, "best_score": 79.72461393340976, "control_same_epoch_best": 79.74282298156136, "matched_delta": -0.01820904815160418, "bad_epochs": 1, "concurrency": "soleGPU training", "decision": "continue; later convergence comparison, no final feature gain"}

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 45, "best_epoch": 44, "best_score": 80.05548544794402, "control_same_epoch_best": 79.89736194351701, "matched_delta": 0.15812350442701018, "bad_epochs": 1, "concurrency": "soleGPU training", "decision": "continue; no established final feature gain"}

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 50, "best_epoch": 44, "best_score": 80.05548544794402, "control_same_epoch_best": 79.93028980255141, "matched_delta": 0.1251956453926084, "bad_epochs": 6, "concurrency": "soleGPU training", "decision": "continue; final feature benefit still unproven"}

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 55, "best_epoch": 52, "best_score": 80.08439320011678, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.1426219354122935, "bad_epochs": 3, "concurrency": "soleGPU training", "decision": "observe late convergence; no feature gain or promotion yet"}

- Verified live directional-depth trial: {"status": "running", "session_id": 63047, "epoch": 60, "best_epoch": 52, "best_score": 80.08439320011678, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.1426219354122935, "bad_epochs": 8, "concurrency": "soleGPU training", "decision": "observe closing phase; no radius grid or promotion"}

- Directional24 feature trial early terminated after validation61: observed best80.0843932001 at52, delta original=-0.1665005747/compact=-0.1426219354. Late stagnation; no demonstrated feature benefit, no radius grid or remainingfolds. Session63047 exit1; partial trial, final restored-best verification incomplete. No checkpoint retained. Peak812.78MiB; last printed wall1525.3s at60 includes preprocessing/probes and mixed parallel/solo phases, not final elapsed. No active training remains from these two routes. Preserve80.3868919 ensemble; fullOOF>82 unmet.

- Prepared increment-texture36 input trial: percurve direction persistence,9/51 energy fraction, spike concentration and coverage; +2304 full-expert stem parameters,242inputs. Original pair/loss/splits80/18, no prior failed mechanisms. outputs/pure_dl_paired_increment_texture80_sequence/README.md; gains untested.

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 1, "best_epoch": 1, "best_score": 74.47660846303351, "concurrency": "soleGPU training", "decision": "continue; initial texture-feature result only, no gain claim"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 5, "best_epoch": 5, "best_score": 77.4363322552908, "control_same_epoch_best": 77.61228231052124, "matched_delta": -0.1759500552304445, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; early texture-feature result only, no final gain"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 10, "best_epoch": 10, "best_score": 78.11198259381652, "control_same_epoch_best": 78.2303687909218, "matched_delta": -0.11838619710528064, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; early texture-feature result only, no final gain"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 13, "best_epoch": 13, "best_score": 78.63165473514114, "control_same_epoch_best": 78.65071599486306, "matched_delta": -0.019061259721922852, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 17, "best_epoch": 17, "best_score": 78.7745563340515, "control_same_epoch_best": 78.7476960609833, "matched_delta": 0.02686027306819483, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 22, "best_epoch": 22, "best_score": 79.12987258701625, "control_same_epoch_best": 79.0916641270535, "matched_delta": 0.03820845996274613, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 26, "best_epoch": 23, "best_score": 79.21250970456951, "control_same_epoch_best": 79.14794099275363, "matched_delta": 0.06456871181588042, "bad_epochs": 3, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 30, "best_epoch": 28, "best_score": 79.55069935778111, "control_same_epoch_best": 79.5169041745716, "matched_delta": 0.03379518320952002, "bad_epochs": 2, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 33, "best_epoch": 28, "best_score": 79.55069935778111, "control_same_epoch_best": 79.56187209225561, "matched_delta": -0.01117273447449918, "bad_epochs": 5, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 37, "best_epoch": 37, "best_score": 79.67004616205892, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.05411950305753521, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 42, "best_epoch": 42, "best_score": 79.80505103055951, "control_same_epoch_best": 79.76759879965745, "matched_delta": 0.037452230902061956, "bad_epochs": 0, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 46, "best_epoch": 44, "best_score": 79.83855655658894, "control_same_epoch_best": 79.89736194351701, "matched_delta": -0.05880538692807136, "bad_epochs": 2, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}

- Verified live increment-texture trial: {"status": "running", "session_id": 62361, "epoch": 50, "best_epoch": 48, "best_score": 79.94123704503018, "control_same_epoch_best": 79.93028980255141, "matched_delta": 0.010947242478763997, "bad_epochs": 2, "concurrency": "soleGPU training", "decision": "continue; no stable texture-feature gain established"}


- Increment texture36 stopped after validation64: observed best 79.9929897287 at52; original delta -0.2579040462, compact delta -0.2340254069. Session62361 exit1, final restored-best verification incomplete. No checkpoint retained; no texture-window grid or remaining folds. Peak831.08MiB; last printed wall1433.0s at60 includes preprocessing/probes, single training job, no standalone warmup, not final elapsed. Preserve80.3868919 ensemble. FullOOF>82 unmet.


- Prepared and launched GR-conditioned16 feature trial: own-well conditional mid-CDF and median/IQR residual for8 curves given GR rank,9 anchors/Gaussian bandwidth.125, no labels. Original206 +16; zero-init full-expert16to64 projection(+1024params), other expert/router unchanged. Fixedfifth original seed,80/18,LR.0006cosine; oneGPU job. No gain claimed; strict>80.4 before otherfolds. Texture36 closed negative. Preserve80.39. Source train_paired_gr_conditional80_sequence.py; record outputs/pure_dl_paired_gr_conditional80_sequence/README.md. FullOOF>82 outstanding.


- GR-conditional16 verified live: {"status": "running", "session_id": 63984, "epoch": 11, "best_epoch": 10, "best_score": 78.26407129024733, "control_same_epoch_best": 78.25898459131336, "matched_delta": 0.005086698933965295, "bad_epochs": 1, "concurrency": "soleGPU training", "decision": "continue; early conditional-feature result, no gain established"}.

- User-requested intermediate cleanup audited45 files/29.618GiB; automatic approval rejected both bulk cleanup and narrowed2 stopped temporaryCDF files (generic blocked by policy). No files deleted. Exact plan: outputs/cleanup_training_intermediates_plan_20260927.json; optional user-run cleanup_training_intermediates.ps1 prepared, NOT executed. Training63984 remains live; active/model/data/submission/scoring/source artifacts preserved. Cleanup pending external policy or local user execution.


- GR-conditioned16 complete: {"status": "complete", "session_id": 63984, "exit_code": 0, "best_epoch": 52, "stopped_epoch": 70, "scores": {"POR": 76.68826801782241, "PERM": 89.21107289034347, "SW": 73.60560908947653, "TOTAL": 79.99231909828372}, "delta_original": -0.2585746765355168, "delta_compact": -0.2346960372453566, "best_checkpoint_predictions_reproduced": true, "all_validation_rows_covered_once": true, "seconds": 1447.4187803000095, "peak_gpu_allocated_mib": 800.02783203125, "checkpoint_persisted": false}. SoleGPU training; timing includes preprocessing/probes and concurrent browser/document work, no separate warmup. No feature gain; no bandwidth/anchor sweep or remainingfolds. Preserve80.3868919 ensemble; fullOOF>82 still outstanding.


- Delivered offline DL lineage atlas: docs/deep_learning/atlas/index.html. Current inventory316 experiment/subconfiguration records +8 explicit dead-end nodes,648 directed relationships;307 records have numeric results and9 show explicit diagnostic/preparation/failure states.47 parent relationships remain unknown instead of invented; all source links exist and lineage has no cycles. View supports mainline/all/stage/dead-end filters, search, SVG pan/zoom/minimap, list, node details, source links and ancestry tracing. Root/repo/pure_dl AGENTS now require every new attempt to update graph and initial results; similar attempts may be grouped without losing sub-results. GR conditional trial finalized79.9923191 and included. No remote push. Final browser results in atlas/browser-check.json; project notes in atlas/PROJECTS.md. FullOOF>82 training goal remains unmet.

- Final atlas acceptance:324 nodes/648 edges,8 explicit dead-end nodes; numeric scores307 experiment records,9 explicit non-score states. Final Edge checks passed including SVG click/keyboard/zoom, search/detail/lineage, all/empty, and390/1366/1440/1920 widths. Screenshots visually reviewed; data has no cycles or missing evidence. See docs/deep_learning/atlas/VISUAL_ACCEPTANCE.md.


- New local-feature-gate80 trial launched session46097 as soleGPU task: original206 inputs, learned perpoint/percurve gain1+tanh(shared MLP19->16->1),9x4 curve embedding, full expert local90 columns only; +373params, zero-last identity initialization. Original9 curves, other expert/router/Q/seed/split/decoder unchanged;80max/18patience/bs64/LR.0006cosine, strictfifth>80.4 before otherfolds. No gain yet. Record outputs/pure_dl_paired_local_feature_gate80_sequence/README.md; atlas updated. Previous GR-conditioned16 completed negative79.9923191. FullOOF>82 remains unmet.


- Verified live local-feature-gate: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:10:00+08:00", "epoch": 4, "best_epoch": 4, "best_score": 77.05803312821072, "control_same_epoch_best": 76.62364985406369, "matched_delta": 0.4343832741470379, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; initial results only, no established benefit"}. Actual1329491 parameters, +373 vs compact control. First printed elapsed60.8s at1 includes preprocessing/probes, no separate warmup; not final elapsed. No OOM observed.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:12:29+08:00", "epoch": 11, "best_epoch": 11, "best_score": 78.4294059904545, "control_same_epoch_best": 78.25898459131336, "matched_delta": 0.17042139914113363, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; early lead not stable final gain"}. Keep original protocol; no second launch or restart. Fifth>80.4/fullOOF>82 not achieved.


- User-directed atlas layout revision complete: one horizontal swimlane per theme; topological constraints put every child to the right of all parents; cross-theme branches/merges cross lanes. Long same-lane edges pass above intermediate cards. Browser verified649 edges:0 backward edges,0 reversing SVG paths,8 themes each one row; navigation/search/trace/mobile checks passed. Updated atlas AGENTS/README/PROJECTS and visual acceptance. Training46097 left unchanged.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:22:28+08:00", "epoch": 36, "best_epoch": 36, "best_score": 79.68708215484588, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.03708351027057688, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; small middle-stage difference is not final benefit"}. No restart/extraGPU trial; no fifth804 or fullOOF82 qualification. Last printed fold elapsed868.8s at35 includes preprocessing/probes and concurrent documentation/browser work, no separate warmup; not final elapsed.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:25:17+08:00", "epoch": 44, "best_epoch": 44, "best_score": 80.03050256733324, "control_same_epoch_best": 79.89736194351701, "matched_delta": 0.1331406238162316, "bad_epochs": 0, "observed_peak_gpu_mib": 793.6552734375, "concurrency": "soleGPU training", "decision": "continue; no stable late-stage gain established"}. Original training unchanged. Last printed wall979.2s at40 includes preprocessing/probes, no separate warmup; not final elapsed. FullOOF goal unmet.


- Verified live local-feature-gate update: {"status": "running", "atlas_current_trial": true, "session_id": 46097, "verified_at": "2026-09-27T16:29:00+08:00", "epoch": 54, "best_epoch": 48, "best_score": 80.03119287568968, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.19582225983938883, "bad_epochs": 6, "observed_peak_gpu_mib": 795.0791015625, "concurrency": "soleGPU training", "decision": "continue briefly; early lead has not become late gain, terminate if stagnation persists; no gate parameter sweep"}. Original baseline80.2508938 also remains higher; no promotion. Last printed wall1199.2s at50 includes preprocessing/probes, no separate warmup; not final elapsed. FullOOF>82 unmet.


- Local-feature-gate80 early terminated: {"status": "early_terminated_negative", "atlas_current_trial": false, "session_id": 46097, "pid": 88964, "exit_code": 1, "epoch": 58, "best_epoch": 48, "best_score": 80.03119287568968, "bad_epochs": 10, "delta_original": -0.21970089912954904, "delta_compact": -0.19582225983938883, "best_checkpoint_restore_verified": false, "observed_gpu_peak_mib": 795.826171875, "checkpoint_files_retained": 0, "reason": "Late stagnation for10 epochs; observed best below both original and compact controls; user-authorized early termination. No gate-width/embedding/gain sweep."}. Partial trial only; final restored-best verification not completed. Source/config/validation history retained, no checkpoints. Last printed fold wall1326.3s at55 includes preprocessing/probes and concurrent browser/doc work, soleGPU training, no separate warmup; not final elapsed. No remainingfolds. Preserve80.3868919215 five-model recipe. Full original OOF>82 unmet; next exploration must change mechanism rather than gate parameters.


- Context-calibration80 launched session95689: second expert calibration hidden h gains zero-init32x32*(masked_mean17(h)-h), +1024parameters. Original206 inputs/68view, original full expert/router/loss/Q/seed/split/decoder unchanged. Sole requested GPU task,80max/18patience/bs64/LR.0006cosine, strictfifth>80.4 before remainingfolds. No gain yet, no extra preflight. Distinct from raw-curve smoothing/local robust hand features/router context. Record outputs/pure_dl_paired_context_calibration80_sequence/README.md. No17-window/width grid if negative. Previous local-feature gate closed negative; fullOOF>82 unmet.


- Verified live context-calibration: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:37:16+08:00", "epoch": 2, "best_epoch": 2, "best_score": 76.63408234332434, "control_same_epoch_best": 76.62364985406369, "matched_delta": 0.010432489260651323, "bad_epochs": 0, "observed_peak_gpu_mib": 774.03564453125, "concurrency": "soleGPU training", "decision": "continue; initial contextual-calibration result only, no accuracy gain established"}. Actual1330142parameters (+1024); first fold elapsed67.5s at1 includes preprocessing/probes, no standalone warmup, not final elapsed. No OOM. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:39:10+08:00", "epoch": 7, "best_epoch": 6, "best_score": 77.72937727954321, "control_same_epoch_best": 77.74331901761342, "matched_delta": -0.013941738070201382, "bad_epochs": 1, "observed_peak_gpu_mib": 774.150390625, "concurrency": "soleGPU training", "decision": "continue; early result only, no established benefit"}. Last printed elapsed176.9s at5 includes preprocessing/probes, no standalone warmup; not final elapsed. Original protocol unchanged, no additional GPU job. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:40:58+08:00", "epoch": 12, "best_epoch": 12, "best_score": 78.46814231998061, "control_same_epoch_best": 78.48230718472027, "matched_delta": -0.014164864739655059, "bad_epochs": 0, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; early comparison is not a final efficacy result"}. Last printed fold elapsed287.2s at10 includes preprocessing/probes, no separate warmup; not final elapsed. No restart, no otherGPU job, no promotion. FullOOF>82 remains outstanding.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:42:53+08:00", "epoch": 16, "best_epoch": 15, "best_score": 78.66626503594227, "control_same_epoch_best": 78.73471136844955, "matched_delta": -0.06844633250727838, "bad_epochs": 1, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; no stable contextual-calibration gain established"}. Last printed fold elapsed422.9s at15 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:44:42+08:00", "epoch": 20, "best_epoch": 20, "best_score": 79.06045166721934, "control_same_epoch_best": 79.0916641270535, "matched_delta": -0.03121245983416543, "bad_epochs": 0, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; not enough evidence of lasting contextual-calibration benefit"}. Last printed fold elapsed535.6s at20 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; no promotion, fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:46:30+08:00", "epoch": 25, "best_epoch": 23, "best_score": 79.25280825505253, "control_same_epoch_best": 79.14794099275363, "matched_delta": 0.10486726229889598, "bad_epochs": 2, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; intermediate lead needs late-training confirmation"}. Protocol unchanged; no remainingfolds or secondGPU trial. Intermediate scores do not establish fullOOF improvement; original OOF>82 goal remains unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:48:09+08:00", "epoch": 29, "best_epoch": 28, "best_score": 79.35778300279328, "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.10894348378816687, "bad_epochs": 1, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; middle-stage comparison does not establish final gain"}. Last printed elapsed673.1s at25 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; no promotion. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:49:57+08:00", "epoch": 33, "best_epoch": 31, "best_score": 79.45711440123819, "control_same_epoch_best": 79.56187209225561, "matched_delta": -0.10475769101742571, "bad_epochs": 2, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; recent new best but no established gain over matched control"}. Last printed fold elapsed788.5s at30 includes preprocessing/probes, no separate warmup; not final elapsed. Protocol unchanged; fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:51:53+08:00", "epoch": 38, "best_epoch": 37, "best_score": 79.72132588702519, "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.002839778091271228, "bad_epochs": 1, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; near matched control, no demonstrated final benefit"}. Last printed elapsed930.1s at35 includes preprocessing/probes, no separate warmup; not final elapsed. No protocol changes or promotion; fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:53:39+08:00", "epoch": 42, "best_epoch": 42, "best_score": 79.78053044891338, "control_same_epoch_best": 79.76759879965745, "matched_delta": 0.012931649255932598, "bad_epochs": 0, "observed_peak_gpu_mib": 774.17626953125, "concurrency": "soleGPU training", "decision": "continue; recent improvement but no stable matched-control gain"}. Last printed elapsed1041.8s at40 includes preprocessing/probes, no separate warmup; not final elapsed. No promotion or protocol change. FullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:55:30+08:00", "epoch": 48, "best_epoch": 47, "best_score": 79.91417887365749, "control_same_epoch_best": 79.93028980255141, "matched_delta": -0.01611092889392296, "bad_epochs": 1, "observed_peak_gpu_mib": 783.85595703125, "concurrency": "soleGPU training", "decision": "continue; near matched control, await later-stage comparison"}. Last printed fold elapsed1152.4s at45 includes preprocessing/probes, no separate warmup; not final elapsed. Original protocol unchanged; no promotion and fullOOF>82 unmet.


- Verified live context-calibration update: {"status": "running", "atlas_current_trial": true, "session_id": 95689, "verified_at": "2026-09-27T16:57:16+08:00", "epoch": 54, "best_epoch": 52, "best_score": 80.18459751154603, "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.04241762398304161, "bad_epochs": 2, "observed_peak_gpu_mib": 783.85595703125, "concurrency": "soleGPU training", "decision": "continue after new best; still below original and compact baselines, no promotion"}. Last printed elapsed1235.9s at50 includes preprocessing/probes, no separate warmup; not final elapsed. No proven gain or qualification; fullOOF>82 unmet.


- USER OVERRIDE for context-calibration80: continue this exact active trial to configured completion; withdraw discretionary early-stop plan. Keep originalmax80/patience18/best-validation checkpoint selection, no configuration or source mutation. Session95689 verifiedlive; {"epoch": 62, "best_epoch": 52, "best_score": 80.18459751154603, "bad_epochs": 10}. This is not qualification for otherfolds; fullOOF>82 remains outstanding.


- Context-calibration80 COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 95689, "exit_code": 0, "best_epoch": 52, "stopped_epoch": 70, "scores": {"POR": 77.21970642705807, "PERM": 89.36964247878716, "SW": 73.54088775958033, "TOTAL": 80.18459751154603}, "delta_original": -0.06629626327320182, "delta_compact": -0.04241762398304161, "best_checkpoint_predictions_reproduced": true, "all_validation_rows_covered_once": true, "seconds": 1651.5405914000003, "peak_gpu_allocated_mib": 783.85595703125, "checkpoint_persisted": false, "decision": "Completed without manual interruption per user request; original patience18 triggered at70. No measured net benefit; no remainingfolds or window/width sweep."}. User-requested completion honored: no manual termination. SingleGPU fold time1651.54s includes preprocessing/training/validation/probes/final export, no separate warmup; monitoring/docs also ran, not a concurrency benchmark. Negative checkpoint not persisted; source/config/compact results retained. Preserve80.3868919215 ensemble; full original OOF>82 unmet.


- Started first-downsample Haar detail trial (session70576), singleGPU; original paired80 baseline, no context-calibration stacking. Fixedfold5/max80/pat18/strict>80.4; initial score pending. FullOOF>82 remains unmet; next inspect actual training then update final evidence. [Experiment](outputs/pure_dl_paired_haar_detail80_sequence/README.md).


- Haar detail trial live update: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; early comparison only; continue unchanged80/18 protocol.", "epoch": 10, "best_epoch": 9, "best_score": 78.1287160339638, "observed_peak_gpu_mib": 775.78466796875, "verified_at": "2026-09-27T17:14:16.699687", "control_same_epoch_best": 78.2303687909218, "matched_delta": -0.10165275695800347, "bad_epochs": 1}. No final gain established; fullOOF>82 remains unmet.


- Haar detail verified live: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; recent successive best scores; continue, no final gain established.", "epoch": 19, "best_epoch": 18, "best_score": 78.93776797376363, "observed_peak_gpu_mib": 775.89501953125, "verified_at": "2026-09-27T17:17:54.000455", "control_same_epoch_best": 78.7476960609833, "matched_delta": 0.19007191278032565, "bad_epochs": 1}. SingleGPU, no restart/config change. Latest printed elapsed403.4s at15 includes preprocessing/probes, not final elapsed; fullOOF>82 unmet.


- Haar detail live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Session70576 verified live; continue original80/18; no final mechanism gain or promotion established.", "epoch": 28, "best_epoch": 28, "best_score": 79.4562704980624, "observed_peak_gpu_mib": 776.12158203125, "verified_at": "2026-09-27T17:21:23.827229", "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.010455988519055381, "bad_epochs": 0}. Training remains singleGPU; fullOOF>82 outstanding.


- Haar detail verified live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; continue unchanged80/18; no final gain or gate qualification.", "epoch": 36, "best_epoch": 36, "best_score": 79.6643116418347, "observed_peak_gpu_mib": 776.12158203125, "verified_at": "2026-09-27T17:24:48.844609", "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.05985402328175837, "bad_epochs": 0}. SoleGPU; last printed754.0s at30 includes preprocessing/probes, not final wall time. FullOOF>82 unmet.


- Haar detail live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; recent improvements warrant continuation; no final gain or promotion.", "epoch": 45, "best_epoch": 44, "best_score": 79.88202580803328, "observed_peak_gpu_mib": 776.12158203125, "verified_at": "2026-09-27T17:28:11.370947", "control_same_epoch_best": 79.89736194351701, "matched_delta": -0.015336135483735802, "bad_epochs": 1}. SingleGPU, unchanged80/18; fullOOF>82 unmet.


- Haar detail live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 70576, "decision": "Verified live session70576; continue late-stage comparison; no demonstrated benefit, no promotion.", "epoch": 54, "best_epoch": 52, "best_score": 80.10618716475575, "observed_peak_gpu_mib": 776.27880859375, "verified_at": "2026-09-27T17:31:38.829700", "control_same_epoch_best": 80.22701513552907, "matched_delta": -0.12082797077331975, "bad_epochs": 2}. SoleGPU; last printed elapsed1213.6s at50 includes preprocessing/probes, not final elapsed. FullOOF>82 unmet.


- Haar detail CLOSED negative partial: {"status": "stopped", "atlas_current_trial": false, "session_id": 70576, "exit_code": 1, "verified_at": "2026-09-27T17:34:51.286891", "epoch": 62, "best_epoch": 52, "best_score": 80.10618716475575, "scores": {"POR": 77.01702176403668, "PERM": 89.28892837567209, "SW": 73.57130201159866, "TOTAL": 80.10618716475575}, "bad_epochs": 10, "delta_original": -0.14470661006347996, "delta_compact": -0.12082797077331975, "observed_peak_gpu_mib": 778.01318359375, "best_checkpoint_restore_verified": false, "partial_result": true, "checkpoint_files": [], "decision": "Manually stopped after62 validations, ten stale epochs and below both controls. No remainingfolds, no detail scale/layer sweep. Preserve partial evidence, do not claim final restore."}. Last printed1417.0s at60 is not final elapsed; singleGPU, preprocessing/probes included, no separate warmup. No negative checkpoint retained. FullOOF>82 unmet; next change mechanism rather than extend this branch.


- Started global head-exchange trial session46372; singleGPU admission verified no other trainers, free10.80GiB. Baseline paired posterior80, fixedfifth/80epochs/pat18, no Haar stacking. Score pending, strict>80.4 promotion, fullOOF>82 unmet. Next monitor actual validation and update atlas.


- Head exchange verified live: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Live session46372 verified; early comparison only; continue unchanged80/18.", "epoch": 12, "best_epoch": 12, "best_score": 78.43859407671914, "verified_at": "2026-09-27T17:42:26.447882", "control_same_epoch_best": 78.48230718472027, "matched_delta": -0.043713108001128376, "bad_epochs": 0, "observed_peak_gpu_mib": 773.955078125}. SingleGPU/no OOM; no final mechanism gain claimed. FullOOF>82 unmet.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; continue80/18, intermediate evidence only.", "epoch": 21, "best_epoch": 20, "best_score": 79.07513986699209, "verified_at": "2026-09-27T17:45:58.589588", "control_same_epoch_best": 79.0916641270535, "matched_delta": -0.016524260061416385, "bad_epochs": 1, "observed_peak_gpu_mib": 773.955078125}. SingleGPU; fullOOF>82 unmet; no final gain claimed.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; continue original80/18; no final mechanism gain.", "epoch": 29, "best_epoch": 28, "best_score": 79.45907444818177, "verified_at": "2026-09-27T17:49:22.862268", "control_same_epoch_best": 79.46672648658145, "matched_delta": -0.007652038399683647, "bad_epochs": 1, "observed_peak_gpu_mib": 773.955078125}. SingleGPU; last printed634.4s at25 includes preprocessing/probes, not final elapsed. FullOOF>82 unmet.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; continue original80/18, no final gain established.", "epoch": 38, "best_epoch": 35, "best_score": 79.5323083696878, "verified_at": "2026-09-27T17:52:46.500151", "control_same_epoch_best": 79.72416566511646, "matched_delta": -0.1918572954286617, "bad_epochs": 3, "observed_peak_gpu_mib": 773.955078125}. SingleGPU; last printed876.4s at35 includes preprocessing/probes, not final elapsed. FullOOF>82 unmet.


- Head exchange live snapshot: {"status": "running", "atlas_current_trial": true, "session_id": 46372, "decision": "Verified live session46372; recent improvements, continue unchanged80/18; no final gain or promotion.", "epoch": 48, "best_epoch": 44, "best_score": 79.89786675506092, "verified_at": "2026-09-27T17:56:19.499196", "control_same_epoch_best": 79.93028980255141, "matched_delta": -0.032423047490496515, "bad_epochs": 4, "observed_peak_gpu_mib": 774.5}. SoleGPU, no OOM; fullOOF>82 outstanding.


- Head exchange CLOSED negative partial: {"status": "stopped", "atlas_current_trial": false, "session_id": 46372, "exit_code": 1, "verified_at": "2026-09-27T17:59:34.324441", "epoch": 54, "best_epoch": 44, "best_score": 79.89786675506092, "scores": {"POR": 76.5404944107509, "PERM": 89.30011532035928, "SW": 73.37336591345687, "TOTAL": 79.89786675506092}, "bad_epochs": 10, "delta_original": -0.35302701975831496, "delta_compact": -0.32914838046815476, "observed_peak_gpu_mib": 776.375, "best_checkpoint_restore_verified": false, "partial_result": true, "checkpoint_files": [], "decision": "Manually stopped after54 validations, ten stale epochs and below both controls. No promotion or head-exchange parameter sweep. Final best restoration not performed."}. SingleGPU; last printed1204.0s at50 is not final elapsed, includes preprocessing/probes, no separate warmup. No checkpoint retained. FullOOF>82 unmet; next change mechanism.


- Started overlap distribution consistency trial session57644, singleGPU admission no other trainers. Original paired model; extra16 windows/step, shift32, symmetric gradients on training-point predictions, no pseudo labels. Fixedfifth/80epochs/pat18/strict>80.4; first score pending. FullOOF>82 unmet.


- User-directed frozen80.39 numeric patch launched session64486: preserve all five experts and learned weights, train zero-init212to32to3 bounded residual MLP on training labels only. Cache construction first; initial prediction must exactly reproduce baseline80.3868919215. Up to2GPU tasks with window-consistency57644; no claim of final gain. Remainingfolds strict>80.4, fullOOF>82 unmet.


- Dual live update: window-consistency {"status": "running", "atlas_current_trial": true, "session_id": 57644, "decision": "Verified live57644; now concurrent with frozen prediction caching64486; continue80/18, no final gain.", "epoch": 26, "best_epoch": 25, "best_score": 79.19331817293609, "verified_at": "2026-09-27T18:14:50.766092", "control_same_epoch_best": 79.14794099275363, "matched_delta": 0.045377180182455845, "bad_epochs": 1, "observed_peak_gpu_mib": 907.31494140625}. Frozen80.39 numeric patch session64486 is still computing train-only four-window prediction cache; no patch score yet. At most2GPU tasks, fullOOF>82 unmet.


- Frozen80.39 numeric patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 64486, "exit_code": 0, "verified_at": "2026-09-27T18:19:26.213685", "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "best_epoch": 0, "stopped_epoch": 18, "gain": 0.0, "best_trained_epoch": 1, "best_trained_score": 80.3769038022867, "last_trained_score": 80.21606532826233, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "cache_bytes": 17365472, "decision": "No benefit: zero patch best. Close smoothL1 bounded numeric correction; preserve compact frozen prediction cache for distinct incremental mechanisms."}. Total524.6965s, includes roughly507.4s through final well cache generation; remainder also includes setup/evaluation, not pure training time. Peak1549.06MiB, concurrent with consistency training. No learned patch retained; fullOOF>82 unmet.


- Started frozen official-metric patch session30303: same numeric MLP, only loss changed from smoothL1 to weighted official clipped error; shared verified17MB cache, no frozen model re-inference. Fixedfifth,80/18, at most2GPU with consistency57644. Await score; fullOOF>82 unmet.


- Frozen metric patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 30303, "exit_code": 0, "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0, "best_epoch": 0, "stopped_epoch": 18, "best_trained_score": 80.38603591067616, "best_trained_epoch": 1, "last_trained_score": 80.36217060265068, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "wall_seconds": 89.4641632999992, "peak_gpu_allocated_mib": 1530.28955078125, "decision": "No net gain: original zero patch best. Official loss more stable than smoothL1 but stop this bounded pointwise numeric correction family; no parameter sweep."}. Total89.46s includes preprocessing/cache checks/training/validation, concurrent with consistency57644; not pure optimization time. FullOOF>82 remains unmet.


- Started frozen80.39 profile patch session75337: change correction information to within-well9/33 raw-row frozen prediction contrasts and signed coefficients. No neighbor labels; same official loss/Q/optimizer/bounds; cache reused. Fixedfifth80/18, max2GPU with consistency57644, no result yet; fullOOF>82 unmet.


- Frozen profile patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 75337, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59455116702611, "PERM": 89.57780436763036, "SW": 73.5910143119091, "TOTAL": 80.38745188794665}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0005599664713287211, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 80.74168310000096, "peak_gpu_allocated_mib": 1563.1240234375, "checkpoint_persisted": true, "decision": "Tiny positive+0.00055997, not substantive and below80.4. Retain small positive patch as control; no window/width sweep."}. Runtime80.74s includes preprocessing/cache checks/training/validation, concurrent with consistency57644; no fullCV qualification. FullOOF>82 unmet.


- Window consistency CLOSED negative partial: {"status": "stopped", "atlas_current_trial": false, "session_id": 57644, "exit_code": 1, "epoch": 62, "best_epoch": 52, "best_score": 80.1021567909661, "scores": {"POR": 77.0691196166394, "PERM": 89.2051634411578, "SW": 73.59889629019732, "TOTAL": 80.1021567909661}, "bad_epochs": 10, "delta_original": -0.1487369838531265, "delta_compact": -0.1248583445629663, "best_checkpoint_restore_verified": false, "partial_result": true, "observed_peak_gpu_mib": 907.31494140625, "checkpoint_files": [], "decision": "Stopped after62 validations and10 stale epochs; no net gain, no remainingfolds or coefficient/shift sweep."}. Last printed1729.7s at60 is not final elapsed; includes preprocessing/probes and concurrent frozen cache/patch work during part of run. Additional16windows/step did not improve score. Frozen baseline error analysis now saved: train85.3684 vs validation80.3869, SW both-numeric validation zero-score fraction60.82%; diagnostic only, no validation gradients. Next target a distinct frozen80.39 repair mechanism; fullOOF>82 unmet.


- Started frozen SW action-value patch session16373, soleGPU. Freeze80.39 models/weights, preservePOR/PERM, learn66 expected candidate utilities on training labels only. Original action favored at initialization; zero-change baseline included in selection. Record changedSW fraction. Fixedfifth80/18, strict>80.4 promotion/fullOOF>82 unmet.


- Frozen SW action patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 16373, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0, "best_epoch": 0, "stopped_epoch": 18, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 61.80240950000007, "peak_gpu_allocated_mib": 1530.28955078125, "checkpoint_persisted": false, "best_trained_score": 80.36705851680067, "best_trained_epoch": 2, "best_trained_SW": 73.53323004140574, "decision": "No net improvement; zero patch best. Close this SW soft-BCE action-value setup, no candidate-grid/initial-bias sweep."}. SoleGPU total61.80s includes preprocessing/cache checks/training/evaluation; no negative checkpoint retained. changed_SW_fraction logs use exact floating comparison, so include float32 representation changes and are not a semantic correction rate. Restored final prediction exactly equals baseline. FullOOF>82 unmet; no remainingfolds.


- Started private-target frozen profile patch24620, soleGPU. Same zero-initial function and sampling RNG as shared profile, independent hidden gradients for3targets; parameter count increases, not capacity-matched. Same frozen80.39/cache/Q/official loss/80epochs/pat18. Compare shared80.3874519 and baseline80.3868919; strict>80.4/fullOOF>82 unmet.


- Private profile patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 24620, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59433412427276, "PERM": 89.57789619385682, "SW": 73.59094125291199, "TOTAL": 80.38739334365091}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0005014221755885728, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 53.60223920003045, "peak_gpu_allocated_mib": 1563.1240234375, "checkpoint_persisted": true, "delta_shared_profile": -5.85442957401483e-05, "decision": "No incremental mechanism gain over shared profile; tiny baseline gain only. Close target-decoupling direction, no width/LR sweep. Keep small positive control checkpoint."}. SoleGPU total53.60s includes preprocessing/cache checks/training/validation; not a controlled speed comparison to prior concurrent runs. FullOOF>82 unmet; no remainingfolds.


- Started frozen128-block correction72985, soleGPU: same212to32to3 model as pointwise metric patch, shared input summary/correction per raw128row block. No label aggregation; loss remains train-point/Q weighted, zero initial80.3868919. Max80/pat18, strict>80.4/fullOOF>82 unmet.


- Frozen block calibration COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 72985, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.6044252537888, "PERM": 89.57632943119414, "SW": 73.58696168941397, "TOTAL": 80.38847946834947}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0015875468741484156, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 47.49249970004894, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_profile": 0.0010275804028196944, "decision": "Small+0.00158755 baseline gain, best epoch1, later degradation; retain small best checkpoint, no80.4 promotion. No block-size sweep yet."}. SoleGPU total47.49s includes preprocessing/cache checks/training/validation; peak1540.78MiB. FullOOF>82 remains unmet.


- User authorized six penalty strengths0/.001/.003/.01/.03/.1, max2GPU. Reuse0 score80.3884794683 and.01 score80.3885472393. Batch1 launched .001 session38362/.003 session78079; batch2 .03/.1 pending. All same split/seed/model/80epochs/pat18. FullOOF>82 unmet.


- Six-strength anchor sweep COMPLETE: {"screening_fold": 5, "coefficients": [{"coefficient": 0, "TOTAL": 80.38847946834947, "gain_original": 0.0015875468741484156, "delta_no_penalty": 0.0, "best_epoch": 1, "stopped_epoch": 19, "seconds": 47.49249970004894, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": true}, {"coefficient": 0.001, "TOTAL": 80.38848480137887, "gain_original": 0.0015928799035407337, "delta_no_penalty": 5.333029392318167e-06, "best_epoch": 1, "stopped_epoch": 19, "seconds": 64.3387407999835, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}, {"coefficient": 0.003, "TOTAL": 80.38850238097564, "gain_original": 0.0016104595003127997, "delta_no_penalty": 2.2912626164384164e-05, "best_epoch": 1, "stopped_epoch": 19, "seconds": 64.3800079000066, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}, {"coefficient": 0.01, "TOTAL": 80.38854723925408, "gain_original": 0.001655317778755716, "delta_no_penalty": 6.777090460730051e-05, "best_epoch": 1, "stopped_epoch": 19, "seconds": 51.5190433000098, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": true}, {"coefficient": 0.03, "TOTAL": 80.38866398629858, "gain_original": 0.0017720648232568692, "delta_no_penalty": 0.00018451794910845365, "best_epoch": 1, "stopped_epoch": 19, "seconds": 50.87739680003142, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}, {"coefficient": 0.1, "TOTAL": 80.38898528235002, "gain_original": 0.0020933608746958043, "delta_no_penalty": 0.0005058140005473888, "best_epoch": 1, "stopped_epoch": 19, "seconds": 50.15694000001531, "peak_gpu_mib": 1540.77880859375, "reused_prior_result": false}], "best_coefficient": 0.1, "best_score": 80.38898528235002, "all_passed_restore": true, "all_passed_gate": false, "maximum_gpu_tasks": 2, "new_runs": 4, "reused_runs": 2, "new_process_seconds_sum": 229.75308550003683, "timing_note": "Two pairs concurrent; individual times include preprocessing/training/validation. No matched serial rerun, no speedup claim."}. Preserve small positive checkpoints and compact results; no promotion. Full originalOOF>82 unmet.


- Started well-balanced frozen block patch74759, soleGPU. Train-only inverse-well-count weights applied to data and anchor.1, fixed original validation population. Compare point-weighted80.3889853;80epochs/pat18, no coefficient expansion. FullOOF>82 unmet.


- Well-balanced block patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 74759, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.60473022199899, "PERM": 89.57744955650682, "SW": 73.58785428364669, "TOTAL": 80.38927541065343}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0023834891781007173, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 48.93912729999283, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_point_weighted_control": 0.0002901283034049129, "decision": "Tiny positive+0.00029013 over point-weighted anchor.1; no substantive gain, no80.4 gate. Preserve small best, no well-weight exponent sweep."}. SoleGPU total48.94s includes preprocessing/cache checks/training/evaluation. All predictions restored exactly; fullOOF>82 unmet.


- Started frozen block-variation features52705, soleGPU. Add3spatial std descriptors,96zero-init input weights; original inputs/init/RNG/loss/anchor.1/well-balancing unchanged. Compare80.3892754; no label-derived context.80epochs/pat18, strict>80.4/fullOOF>82 unmet.


- Frozen block variation COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 52705, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.60444285891165, "PERM": 89.57744694086408, "SW": 73.58776066579821, "TOTAL": 80.38915552000529}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0022635985299643835, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 49.80855990003329, "peak_gpu_allocated_mib": 2143.19140625, "checkpoint_persisted": true, "delta_no_variation": -0.00011989064813633377, "decision": "No feature gain versus matched no-variation control; do not add higher moments or scan block lengths. Small positive-vs-original artifact retained as control."}. SoleGPU total49.81s includes preprocessing/cache checks/training/validation; peak2143.19MiB reflects descriptor construction too. Best epoch1/stop19, restore verified. FullOOF>82 unmet, no promotion.


- Started frozen linear block patch52216, soleGPU. Reduce MLP to212to3+tanh, zero initialization; same well-balanced block inputs/Q/official loss/anchor.1/80epochs/pat18. Structure changes RNG consumption, documented not update-identical ablation. Compare80.3892754, fullOOF>82 unmet.


- Frozen linear block patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 52216, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59361541567479, "PERM": 89.57758548553893, "SW": 73.59055551542714, "TOTAL": 80.38693397504056}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 4.205356523812043e-05, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 49.21806949999882, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_mlp_control": -0.002341435612862597, "decision": "No benefit from reduced capacity; essentially unchanged from frozen baseline, worse than MLP. Close capacity-adjustment direction, no width/LR sweep."}. SoleGPU total49.22s includes preprocessing/cache checks/training/evaluation;639parameters. Best1/stop19, restored prediction verified. FullOOF>82 unmet, no remainingfolds.


- Started frozen state-repair patch50014, soleGPU. One alternate per target from input-only33row neighbor predictions/default; trainMSE expected utility difference, inference threshold0 (no validation oracle/tuning). Start exact80.3868919;max80/pat18, strict>80.4/fullOOF>82 unmet.


- Frozen state-repair COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 50014, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.0, "best_epoch": 0, "stopped_epoch": 18, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 48.7050696000224, "peak_gpu_allocated_mib": 1563.69921875, "checkpoint_persisted": false, "best_trained_score": 80.37943689195787, "best_trained_epoch": 2, "decision": "No net gain from state-repair expected-advantage patch; initial baseline best. No threshold/window scan; no checkpoint retained."}. SoleGPU total48.71s includes preprocessing/cache checks/training/validation. No trained state beats original80.3868919. FullOOF>82 unmet; no promotion.


- Started frozen smooth-score patch57430, soleGPU. Replace only training clipped kernel with Gaussian convolution sigma.25; original official validation remains. Same well-balanced block/Q/anchor.1/80epochs/pat18. Compare80.3892754, no validation gradient. FullOOF>82 unmet.


- Frozen smooth-score patch COMPLETE: {"status": "complete", "atlas_current_trial": false, "session_id": 57430, "exit_code": 0, "screening_fold": 5, "scores": {"POR": 77.6042669423747, "PERM": 89.57816415742207, "SW": 73.58768332216735, "TOTAL": 80.3893267005687}, "initial_scores": {"POR": 77.59356202942969, "PERM": 89.57816969565643, "SW": 73.58989691190476, "TOTAL": 80.38689192147532}, "gain": 0.002434779093377415, "best_epoch": 1, "stopped_epoch": 19, "best_checkpoint_predictions_reproduced": true, "screening_passed": false, "training_points": 578891, "validation_points": 144665, "validation_points_used_for_gradient": 0, "wall_seconds": 66.73988799995277, "peak_gpu_allocated_mib": 1540.77880859375, "checkpoint_persisted": true, "delta_hard_kernel_control": 5.1289915276697684e-05, "decision": "Only+0.00005129 over hard-kernel control, no substantive gain; best1/stop19. No sigma sweep, retain small positive checkpoint."}. SoleGPU total66.74s includes preprocessing/cache checks/training/validation; original official scores, not smoothed proxy scores. FullOOF>82 unmet, no promotion.


Frozen latent patch COMPLETE: fifth TOTAL80.3868919215, best epoch0, natural stop18; best trained80.3759376707@1. Best prediction restoration verified. No gain and no checkpoint retained. Single GPU total66.724s including preprocessing/extraction/training/validation; extraction5.846s, peak3306.041MiB. All723556 valid rows covered once; normalization uses578891 training rows only. No remaining-fold promotion. FullOOF>82 unmet.

- Started rank8 cross-feature calibration, session77606, sole GPU. New720-parameter context-controlled multiplicative residual in calibrated member; zero output preserves initial baseline. Fixed fifth/max80/pat18/bs64, train-only preprocessing. No result yet; strict>80.4 before promotion, fullOOF>82 remains unmet. See outputs/pure_dl_paired_lowrank_calibration80_sequence/README.md.

- Live snapshot: rank8 calibration session77606 confirmed training, epoch4 best fifth76.93051414. Early result only, no promotion; no OOM, latest reported PyTorch peak776.264MiB.

- Running rank8 calibration: last epoch20, best78.91579401@20; last same-epoch delta-0.17587012. Continue because best is still improving; no final ablation gain or promotion claimed. Session77606 polled live.

- Prepared next mechanism, not launched: weak train-only cross-well hidden feature-statistics mixing (MixStyle-inspired). See outputs/pure_dl_well_style_plan/README.md for evidence, fixed starting design, padding/RNG/domain safeguards and physical-information risk. Current rank8 calibration session77606 must finish naturally under80/pat18; no manual early stop per latest user request.

- Started weak well-style training session69486 concurrently with lowrank77606 (max2). Fixed fifth,80/pat18; no extra parameters. Cross-well donor IDs from training anchor well starts, masked statistics, dedicated RNG, disabled in eval/probe. Initial score pending. Preserve lowrank natural completion. FullOOF>82 unmet.

- Live paired_lowrank_calibration80: epoch49, best79.80864783@44; session77606 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Live paired_weak_well_style80: epoch5, best77.53947410@5; session69486 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Rank8 calibration COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.72788441696548, "PERM": 89.24684557961116, "SW": 73.38376598362012, "TOTAL": 79.93907937222059}, "seconds": 1198.6207101999898, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 782.7607421875, "status": "complete", "atlas_current_trial": false, "session_id": 77606, "exit_code": 0, "screening_fold": 5, "delta_compact_control": -0.28793576330848225, "delta_original": -0.31181440259864246, "decision": "No gain. Natural patience18 stop at70; no rank/capacity sweep, no remaining folds."}. Total1198.62s includes preprocessing/training/probe/validation; partially concurrent with weak-style, no matched serial speedup claim. No checkpoint retained.

## Historical family clarification
A previous conditioned rank8 numeric-mixing trial already completed at80.1954269324 (outputs/pure_dl_paired_conditioned_mix80_sequence). That version applies mixing before the original affine calibration and has biases; this later720-parameter version adds a bounded correction after affine calibration. They are related variants, not separate unexplored mechanism families. Both below compact80.2270151; close this family, no further rank/placement sweeps. Historical PCGrad80.1582856 also failed compact control; do not relaunch gradient projection as a new idea.

- Started80.39 attribution session51546: fixed5/checkpoint hashes, single/equal/learned/leave-one-out and single-vs-four-window/coarse-vs-fine decoding; diagnostic only, no promotion. Runs alongside weak-style69486, max2GPU tasks. Results pending; models/loss attribution requires matched training controls, not historical score subtraction.

- Weak style COMPLETE: {"fold": 5, "best_epoch": 48, "stopped_epoch": 66, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 75.7366886208511, "PERM": 89.13228344192615, "SW": 73.48030727511235, "TOTAL": 79.6354133372188}, "seconds": 1278.4947431000182, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 777.9208984375, "status": "complete", "atlas_current_trial": false, "session_id": 69486, "exit_code": 0, "delta_compact": -0.591601798310279, "decision": "No gain; natural stop66, best48 restored. No augmentation strength/layer sweep."}. Partial concurrency with lowrank and source attribution; no speedup claim. User direction now prioritizes80.39 attribution, not new mechanism search.

-80.39 attribution inference completed26 arms (214.40s/1461.84MiB): strongest single80.331454/equal5 80.386864/learned5 80.386898; see outputs/pure_dl_8039_source_ablation/ANALYSIS.md. Reference score difference6.03e-6, not elementwise identical. Diagnostic dropseed07 80.403220 is NOT auto-promoted. Training ablations Q=1(session82504) and POR Gaussian(session69566) now running, max2; no results yet.

- User adopts four-source80.403220094 baseline (indices0,2,3,4). Six equal two-source subset ablations prepared and queued in observable session28718; waits for GPU capacity while Q=1 session82504 and GaussianPOR69566 run. No backbone retraining for pair study. Preserve original sources/deliveries; diagnostic selection history explicit. See outputs/pure_dl_8040_pair_ablation/README.md.

- Attribution monitoring: [{"status": "running", "atlas_current_trial": true, "session_id": 82504, "screening_fold": 5, "last_epoch": 26, "best_score": 79.15393682729156, "best_epoch": 23, "ablation": "no_quality", "note": "Partial result; no final contribution claim."}, {"status": "running", "atlas_current_trial": true, "session_id": 69566, "screening_fold": 5, "last_epoch": 22, "best_score": 79.07819692255848, "best_epoch": 20, "ablation": "gaussian_por", "note": "Partial result; no final contribution claim."}]. Pair inference28718 verified CPU-waiting while these2GPU tasks run. Historical paired module config/shared-hash audit completed without shared hash changes; source analysis updated, one-seed evidence only.

- Expanded80.4032 subset attribution per user:6pairs+4leave-one-out triples+baseline=11arms. Replaced only CPU-waiting inference28718 with94542, confirmed waiting; training82504/69566 uninterrupted. Same source inference will serve all11arms.

- Attribution runs observed live: [{"ablation": "no_quality", "epoch": 42, "best": 79.7362617128407}, {"ablation": "gaussian_por", "epoch": 40, "best": 79.57860036215014}]. Subset11arms session94542 remains queued; no thirdGPU task. Added config-based source/subset parameter counts to outputs/pure_dl_8040_pair_ablation/model_costs.json (baseline4.646M/7backbones, pairs1.987-2.658M/3-4backbones), not a measured speedup.

- Training attribution live snapshot [{"mode": "no_quality", "epoch": 56, "best": 80.09251227357417}, {"mode": "gaussian_por", "epoch": 55, "best": 80.04144251747569}]. Both still running; do not finalize Q/distribution contributions yet. Added historical attention audit (+0.2158936568 at matched single-backbone75, not current ensemble causal delta) to source ANALYSIS.md. Removed2completed-negative point-diagnostic caches44,965,265bytes with manifest outputs/cleanup_negative_calibration_style_attribution.json; code/config/validation curves/scores and all active files retained. Subset11 queue94542 unchanged.

- Training ablations COMPLETE: [{"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.94295729632633, "PERM": 89.27312796013906, "SW": 73.61151513893603, "TOTAL": 80.09251227357417}, "seconds": 1529.0026964999852, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 82504, "exit_code": 0, "delta_control": -0.1345028619548998, "ablation": "no_quality"}, {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.71596748229165, "PERM": 89.4341072589125, "SW": 73.49918494905378, "TOTAL": 80.04144251747569}, "seconds": 1362.7202495000092, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 69566, "exit_code": 0, "delta_control": -0.18557261805338499, "ablation": "gaussian_por"}]. Q and t3 retained on this single-seed evidence, no additive contribution claim. Subset11 inference94542 now admitted, soleGPU. FullOOF>82 unmet.

- All requested attribution completed:11 subset arms, baseline80.403220094 exactly reproduced. Best pairAB80.3801885; best tripleABD80.4091519 (+0.0059318, parameters3.320M versus4.646M, exploratory selection not robust improvement claim). Q-off80.0925123 and GaussianPOR80.0414425 versus matched dual80.2270151, bothbest52/stop70 restore verified. Reports outputs/pure_dl_8040_pair_ablation/SUMMARY.md and outputs/pure_dl_8039_source_ablation/ANALYSIS.md. All3sessions terminal, no new training launched; fullOOF>82 still unmet.

- Started targeted attribution follow-ups: main-only auxiliary-loss removal62826 (same original dual80/Q/t3, max80/pat18) and frozen internal expert intervention60959 (Aseed06/Bseed08 each full206/calibrated68/equal/routed). Max2GPU; not a new unstructured mechanism sweep. Four-source80.4032 baseline unchanged, results pending, fullOOF>82 unmet.

- Internal experts COMPLETE8arms, exact A reference reproduced. [{"source": "A", "equal_gain_over_best_branch": 0.3514727762118639, "router_gain_over_equal": 0.09321302034933865, "routed_gain_over_best_branch": 0.44468579656120255}, {"source": "B", "equal_gain_over_best_branch": 0.31227311776699196, "router_gain_over_equal": 0.009940617467393054, "routed_gain_over_best_branch": 0.322213735234385}]. Full206 alone stronger total; calibrated68 complements SW. This is frozen intervention, not retrained causal removal. Source report retained. Main-only62826 continues, no new GPU task.

- Main-only attribution62826 confirmed live, epoch13 best78.56925593@13, now soleGPU. Internal-expert inference60959 completed153.71s/1444.27MiB, exact reference; report ANALYSIS.md updated with branch complement and routing gains. FullOOF>82 unmet.

- Added and visually inspected attribution_summary.png/.svg (3separate comparison panels, completed results only), linked from SUMMARY.md and ANALYSIS.md. Main-only62826 continues as soleGPU, epoch25 best79.28455813@25; no final auxiliary-loss contribution claim.

- Started matched no-calibration attribution66172 alongside main-only62826, max2GPU. Original posterior network, calibration zero-output initialization frozen; 68view/Q/t3/auxiliary/router retained. Fills interaction gap beyond older no-router75/80 comparisons; no rank/width sweep. Result pending, original80.4032 four-source baseline unchanged and fullOOF>82 unmet.

- Live final attribution pair: [{"mode": "main_only", "epoch": 40, "best": 79.69091378066616}, {"mode": "no_calibration", "epoch": 2, "best": 76.68573766110542}]. Both actual train/validation/probe cycles confirmed; max2GPU and no OOM observed. Summary renderer prepared to include completed optional auxiliary/calibration ablations only, no provisional results in final comparison charts.

- Verified live attribution wait: [{"mode": "main_only", "epoch": 54, "best_epoch": 52, "best": 79.87490529017835}, {"mode": "no_calibration", "epoch": 14, "best_epoch": 12, "best": 78.49377082386238}]. Both valid tool handles active, no restart or premature final contribution claim. Main-only refreshed best52; calibration still early. No new experiments added.

- Main-only attribution COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.68933694757615, "PERM": 89.15215965438021, "SW": 73.32813807677836, "TOTAL": 79.87490529017835}, "seconds": 1235.02869440004, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.43896484375, "status": "complete", "atlas_current_trial": false, "session_id": 62826, "exit_code": 0, "delta_control": -0.352109845350725, "decision": "All3channels worse. Retain auxiliary supervision in current setup; no auxiliary-weight sweep."}. Partial concurrency; time includes preprocessing/training/probes/validation. Input-calibration66172 remains soleGPU task; fullOOF>82 unmet.

- Final main-only result integrated into SUMMARY/ANALYSIS/summary.json and regenerated chart; chart visually checked with extended delta axis (includes-0.3521). No-calibration66172 still live, epoch34 best79.63841458@28, soleGPU.


## 2026-09-27 ABD baseline and ensemble-aware module attribution
- Goal: original grouped full OOF>82 remains unmet; user adopts fixed-fifth ABD80.40915186344968 as development baseline (not independent test).
- Implemented train_abd_fused_training80.py: freeze B/D, warm-start A, train final equal-mixture likelihood with frozen train-only Q, bs64/64steps/max80/pat18, lr3e-5 cosine1e-5. Checkpoint/early stop by fused TOTAL, solo diagnostic. Session90559 started; initial readback80.4091386805 (delta -0.00001318 from historical, numerical rebatching; not elementwise exact), gradient check A-only passed. Improvement pending.
- User requests deeper attribution: evaluate_abd_backbone_modules.py started session30376, baseline+12 one-factor inference interventions in A's two members with B/D fixed. Local/global/hierarchy/pointMLP/bottleneck/skips; pending results and selected controlled retraining. Inference dependency is not causal retraining gain.
- No-calibration control completed80.0411598317@52/stop70, -0.1858553038 versus matched80.2270151; restored/coverage verified. Old policy omitted checkpoint so no fusion conclusion. New candidates retain best weights for fusion assessment.
- Updated root/repo/experiment AGENTS, README, individual exploration READMEs and atlas. Next: complete module table and compare solo/fused channel/well changes; select a focused training ablation. No remaining-fold or delivery changes yet.


## 2026-09-27 Completed ABD internal-module attribution
- Completed baseline+12 arms: 10 genuine A-module interventions and2 numerical controls (point is already EmptyPoint; independent MLP absent). B/D unchanged. Source weights preserved,144665 rows each covered4times. 419.719s total including data/cache,1705.00MiB peak, concurrent with incremental trainer.
- Full206 A branch removal deltas in ABD: local -0.140824; global +0.025934 (80.43507290); whole hierarchy -0.008891; bottleneck -0.066270; skips -0.047187. Calibrated68: local -0.065327; global -0.131623; hierarchy -0.147241; bottleneck -0.056000; skips -0.133221. All genuine removals hurt A alone. Global-full candidate gains mainly SW+0.08924,9/16 wells improve. Inference dependency only, not training causal gain.
- Original fused-loss incremental trainer finished18 epochs without gain. Bestepoch0=80.40913868, all trained epochs lower (max80.285611). Strict prediction replay assertion failed: CPU readback confirms every saved A tensor equals original A and saved predictions recompute exact saved score. Report explicitly marks exact prediction reproduction false; no false success claim. No need to retrain this negative control.
- Started single-module adapted candidate session18573: train_abd_no_global_full80.py, full206 global attention structurally removed; B/D frozen, same incremental budget/loss. Deterministic operations added to avoid repeat-index accumulation jitter, documented difference. Initial match and training results pending. User-adopted baseline remains80.40915186, not silently changed to selected candidate.
- Next: observe candidate best fused checkpoint and restoration. Random-initialization matched module retraining remains unperformed; fullOOF>82 unmet. Further modules worth later attribution: anti-aliasing downsampling and depth-only normalization; no new sweeps launched.


- Module adaptation launch18573 failed BEFORE any optimizer update: CUDA linear upsampling backward lacks deterministic implementation. Preserved source/config/epoch0 and failure record. Replaced with train_abd_no_global_full_evaldet80.py in NEW outputs/pure_dl_abd_no_global_full_evaldet80_fold5, session51252. Only validation/frozen cache deterministic; training ordinary CUDA. No second live GPU job. Source architecture/budget/loss unchanged. User-adopted ABD baseline unchanged. Compact module table published locally in docs/deep_learning/ABD_MODULE_ATTRIBUTION.md and results/deep_learning/abd_module_ablation_summary.json; no Git push requested.

- Replacement51252 verified actual training: deterministic initial fused80.4350829326; A-only gradient check passed; epoch1 fused80.343809445, best remains epoch0. Training continues max80/pat18; no trained gain claimed.


## 2026-09-27 Random-initialization module attribution
- Previous goal turn classified progress: completed13-arm module evidence, candidate deletion showed solo/fusion disagreement. Current goal remains full original OOF>82, not yet achieved.
- Verified51252 live; no-global incremental training still bestepoch0. Do not replace baseline or claim training gain.
- Implemented FrozenABDValidation and train_abd_module_retrain80.py. Two predefined matched arms: original A control vs only full206 global attention removed AFTER identical initialization. Restore original per-member/router NLL, annealed auxiliaries and full-training-point dynamicQ, not the failed incremental fused-loss objective. B/D frozen and cached; each epoch selects best A by fixed equal ABD full-distribution official score. Same seed/max80/pat18/min10/bs64/64steps/lr.0006cosine. Validation deterministic, ordinary CUDA training.
- Control started64442 while51252 active, total2GPU. Removal arm prepared, not yet started; waits for51252 natural completion. No redundant preflight or independent review, only syntax and actual runtime invariants. Actual teacher preprocessing must match training split. Source hashes frozen at launch.
- Next: launch removal once slot free; compare fused and solo curves and final restored results. Preserve both best candidates until fused assessment. No fold continuation or release replacement until clear gain.


- Incremental candidate51252 COMPLETE exit0: best80.4350829326@0, stop18. Exact restored predictions, frozenB/D unchanged and4-window coverage verified. Delta originalABD+0.0259311 comes from removing module, zero extra training gain.572.4575s including setup/cache/training/restore,1798.65MiB peak. Preserve candidate; no incremental LR sweep.
- Paired from-scratch no-global-full launched60531 after51252 terminal; control64442 already training. Two GPU tasks only. Original A losses/aux/dynamicQ retained, B/D frozen, checkpoint selected by ABD. Both80/pat18/min10 and originalseed. Final comparison pending; fullOOF>82 still unmet.

- Deleted only two redundant epoch0 checkpoint copies after tensor-for-tensor comparison with preserved sources, freed23,909,896bytes. Manifest outputs/cleanup_abd_duplicate_epoch0_20260927.json. Positive80.43508 candidate, original A/B/D milestones and both active training directories preserved; no training interruption or C migration.

- Live paired-training snapshot (not final/restored): [{"arm": "control", "last_epoch": 8, "best_epoch": 6, "best_fused": 80.49230853582785, "solo_at_fused_best": 77.69492368909336, "delta_baseline": 0.08315667237816626}, {"arm": "no_global_full", "last_epoch": 3, "best_epoch": 3, "best_fused": 80.39633467723046, "solo_at_fused_best": 76.81323320745568, "delta_baseline": -0.012817186219223231}]. Both processes confirmed live via tool sessions64442/60531. Keep fullOOF>82 active; no early promotion before best-checkpoint restoration.


## ABD full-CV continuation prepared, not launched
- Live paired screen still running64442/60531. Original control observed80.49230854@6; no-global80.46881577@6. Not final or promoted yet. Last goal turn=progress, fullOOF>82 not proved.
- Implemented fold-aware frozen validation, single-fold B and A continuation entrypoints, promotion verifier, two-slot dependency queue and original-label fullOOF verifier. Existing D fivefold models reused only on their own fold; B must train each remainingfold before its A. Fifth reused unchanged after exact restored checkpoint/population proof. Per-fold delta is not calculated against fifth baseline.
- New outputs/pure_dl_abd_fusion_selected_fullcv README defines source roles,80/18 budget, score scope, admission, no-overwrite and failure behavior. No GPU continuation launch before completed fifth gate. New files syntax-checked only, actual other-fold runtime pending. Historical active source hashes untouched.
- Next: finish paired fifth, choose one predeclared architecture by fused score, verify it, then fill4folds with max2GPU and score full723556-row OOF. No need to ask permission: continuation is already in user goal, conditional on actual gate.


## Latest user steering: defer fullCV; compare joint training versus B-only
- User explicitly says no remaining folds for now. Promotion and queue NOT launched; no promotion.json. Root/repo/experiment AGENTS updated. FullOOF>82 remains active/unproved; this is not a pause of the overall goal.
- Both scratch attribution arms COMPLETE exit0, bestepoch6/stop24, exact restored predictions and coverage verified. Original80.4923085358 vs no-global80.4688157742; keep original. Matching delta -0.0234927616 for deletion. Rendered and visually checked results/deep_learning/abd_retrain_attribution/checkpoint_selection.png; A solo improves later while fused declines.
- Latest user asks jointly updating3models versus freeze bestA6/D and trainB. Implemented one shared trainer train_abd_joint_or_b80.py with mode joint/b_only. Both warm-start the identical80.49231 composition, same batch RNG, fused mixture NLL, 3e-5 cosine1e-5,80/pat18/min10/bs64, per-updated-model clip2. Uses frozen Q active during A best6 (probe epoch4, all1), no auxiliary restart. Explicitly incremental, not random-initialization joint training.
- Launched joint21517 and B-only38600, max2GPU. Initial cache/gradient scope/results pending. Same ordered A+(B+D) evaluation, deterministic validation. Best epoch0 remains candidate; only one ensemble_best.pt per route. B-only must verify A/D unchanged at end.
- Next: compare actual fused score gains and gradients/restore; no fullCV launch even if positive until user resumes that step. Docs/EXPLORATIONS/atlas updated, no remote push.

- New matched runs verified epoch0 EXACTLY reproduces80.49230853582785, zero differing prediction elements in both arms. Gradient masks confirmed: joint[0,1,2], B-only[1]. First epoch joint80.28915356, B-only80.36295906; both currently retain epoch0. Early observations only, no training gain yet; sessions21517/38600 remain live. First-epoch wall90.12s versus44.06s under partial concurrent setup, not an equal-workload speed benchmark.


## CPU gain decomposition while two live GPU routes continue
- Previous goal turn=progress (new matched runs launched, exact epoch0/gradient masks verified). Current sessions21517/38600 and process IDs85532/45368 confirmed live; no restart on observation timeout. No remaining-fold queue launched.
- Added analyze_abd_checkpoint_gain.py and results/deep_learning/abd_retrain_attribution/GAIN.md/json. Original ABD80.40915186 ->80.49230854: POR+0.113417, PERM+0.214180, SW-0.073804. Default contributions to TOTAL +0.150041 and nondefault -0.066885; PERM nondefault improves but POR/SW nondefault worsen. All contribution decompositions exactly sum to official delta.
- Only7/16 wells improve; well index69 contributes+0.136418 to total, greater than net+0.083157. Excluding this largest positive well diagnostically leaves -0.056845 on15wells. No per-well prediction selection or metric changes. This weakens a broad-generalization claim and supports withholding fullCV as user requested.
- Prepared distinct next route train_abd_b_scratch80.py / abd_frozen_ad_validation.py: B random seed08 initialization, original losses/aux/dynamicQ, frozen bestA6/D, fused selection. NOT launched; current matched incremental pair has priority. Not a one-factor comparison to incremental because initialization/LR/aux/Q differ; explicitly documented.
- Joint observed best trained80.377424@3 (last4); B-only best trained80.484791@5 (last8); both best saved remains epoch0 80.49230854. Continue within budget, no new score claimed.


- B-only incremental38600 COMPLETE exit0: best80.4923085358@0, stop18; trained best80.48479059@5. Exact restored prediction, A/D unchanged and coverage checks passed.923.261s including setup/cache,1806.81MiB peak. No incremental gain; do not call initialization score a training improvement.
- Deleted its duplicate epoch0 ensemble_best.pt only after all3 saved state dicts matched preserved A6/B/D exactly, freed20,066,351bytes. Manifest outputs/cleanup_abd_b_only_epoch0.json. Sources/milestones/positive controls/active outputs intact.
- Started B-from-scratch62769 after38600 terminal; total2GPU with joint21517. FrozenAD_READY confirmed; firstepoch fused80.06379145, Bsolo75.34658365. New seed08 initialization, original losses/aux/dynamicQ; not comparable to incremental as a one-factor intervention. No fullCV.
- Joint21517 remains live; last12, trained best80.39709517@5, saved best still epoch0. Respect natural pat18; source untouched. Completed CPU gain analysis confirms concentration, so no broad-generalization claim.


## Channel learning-speed hypothesis
- Joint incremental training completed18 epochs, best remains epoch0=80.49230854; no gain, exact restored predictions verified. B from scratch continues; all-three-from-scratch launch deferred in favor of user-requested channel conflict diagnostic. No remaining folds.
- New benchmark_channel_gradients.py uses512 fixed train-well windows, Q=1, saved A6/A52/B43/D52, per-channel main/auxiliary shared gradients; no optimizer updates. Session16001 started, result pending. Distinguish norm imbalance from negative cosine and from ensemble complementarity.

- Channel-gradient diagnostic16001 COMPLETE exit0:87.739s,660.494MiB,512 train-only windows, four checkpoints unchanged. A52 mean norms POR/PERM/SW4.920/3.063/1.619. B PERM-SW negative cosine7/16 batches, but no shared equal-gradient opposing any task in sampled batches. Q=1/raw gradients only; no causal or Adam-update claim.
- Same A24-epoch history: epoch6 achieves51.1%/79.0%/57.0% of observed POR/PERM/SW epoch1-to-best validation gain; first90% at17/12/20. Evidence favors faster early PERM validation learning; per-channel training-loss convergence not recorded. Report and README updated. Next: instrument channel losses/gradients, then isolated mild shared-gradient balancing with fixed B/D. No new training launched this batch.


## Correct training fusion semantics
- User clarifies main TRAINING objective must fuse A/B/D with other two frozen. Existing B scratch solo-loss run retained unchanged as control; new train_abd_b_fused_scratch80.py launched separately. A6/D52 frozen eval/no_grad, B random seed08, final mixture likelihood differentiates into B backbones+router; original B auxiliary/Q/80-epoch budget retained. Validation uses final fused official TOTAL.
- Added channel BCE/NLL per epoch and first-batch shared-gradient diagnostics; no adaptive channel weights yet. Prepared all-three-joint-scratch remains unlaunched. No remaining folds. First real backward/scores pending.

- Corrected fused-loss B session26379 reached real first backward: B experts/router nonzero gradients, A/D no gradients and tensor-equal to source. Validation pending; two-job GPU snapshot9444/12282MiB withoutOOM.


## Planned frozen backbone with independent target heads
- User proposes future POR/PERM/SW heads with independent architectures, optimizers, stopping and per-channel checkpoint selection on one frozen backbone. Plan recorded at outputs/pure_dl_frozen_backbone_channel_heads/README.md and atlas; not implemented or launched. Current fused B training unchanged.
- First isolate freezing/per-channel checkpoint selection with same head capacity; heterogeneous heads later. In ABD use final fused per-channel likelihood and scores with companions frozen. Reassemble fixed three heads and verify full population prediction; no per-row/well oracle. Existing router descriptors span all channels: frozen router parameters alone do not remove cross-head dependency; use frozen original routes or independent routes.

- Frozen-head route now implemented: frozen_channel_heads.py + train_abd_channel_heads80.py, syntax parsed; NOT launched (two liveGPU jobs). Freeze original B43 feature extractor and original routing outputs, train19350 head params, per-channel fused-score best checkpoint/earlystop, original A6/D52 frozen. Actual initialization/reassembly checks pending runtime.
- Live status refresh: solo-loss B at54, best80.484205@47 (-0.008104 vs80.492309); final-fused B at8, best79.902002@7 (-0.590306). Both processes live, no final results.


## CPU handoff and comparison snapshot
- Prior turn made progress: implemented frozen heads and real live checks. Added CPU-only run_channel_heads_after_control.py; session47285/queuePID85556 waits for controlPID47108 termination + successful report, then verifies<2GPU tasks and>=3GiB before launching heads once. Does not retry failures or run fullCV. If user stops training, also stop this waiting process.
- Added summarize_abd_b_training_routes.py and results/deep_learning/abd_b_training_routes/{snapshot.json,README.md,fused_channel_curves.png}; image visually inspected. Same epoch12 final-fused79.657911 versus solo-loss-control80.080111, delta-0.422199. Early evidence only, not final.
- Removed duplicate completed joint epoch0 ensemble_best.pt after all3states matched preserved A6/B43/D52;20,066,351bytes freed, manifest outputs/cleanup_abd_joint_epoch0.json. Active/milestone models untouched.


## User stopped and replaced routes; A6 backbone correction
- Stopped queue85556, fused-scratchB66088 and newly admitted old head trainer89900 at user request; old solo-lossB47108 had already completed65epochs with verified80.484205@47. Manifest outputs/stop_old_b_routes_20260927.json.
- User requires own-model training loss and validation-only fusion, and specifically A6 from80.4923 as frozen head backbone. Implemented separate train_a6_channel_heads_own80.py + train_b_own_finetune80.py; historical sources unchanged. A6heads use B43/D52 validation, B43warmstart uses A6/D52 validation. Both80/18,LR3e-5,normalGPU; no remaining folds.
- Both launched; first runtime/score checks pending. Head route trains19350params, independent channel stopping/selection and original frozen routing. Source/checkpoint provenance documented in separate README files.


## Unified frozen feature backbone route
- User clarified whole ABD ensemble as feature backbone, no old output averaging. Stopped B-own fine-tune PID50824 only; A6 head PID20076 preserved. Implemented abd_feature_backbone_heads.py + train_abd_feature_heads80.py:5frozen128dim branches->640features->3independent640/128/25 heads (255819 trainable). Original posterior routers not executed; original prediction tensors discarded.
- New run session10715 launched as secondGPU job, normalCUDA,80/18,bs64,LR.0006. New head initialization uses training-only priors; do not expect baseline80.4923 at epoch0. A6 control initial baseline exactly reproduced, awaiting/latest live scores tracked separately. Final head assembly and frozen source checks required; no fullCV.

- Structural feature-head run10715 passed real first backward: only255819 new-head params trainable; all5feature branches frozen, legacy voting absent. Initial random-head TOTAL68.41007972 is epoch0, not a trained result or expected80.4923 reproduction. GPU snapshot8227/12282MiB. A6-head control continues (initial exact80.49230854; per-channel best assembled candidate80.501552 byepoch11, final restore pending).


## Completed A6 target-head selection and ongoing composite backbone
- A6-head session92282 COMPLETE exit0: fixedfifth80.50155201341903 (+0.009243477591184 vs80.49230853582785). Best epochs POR/PERM/SW4/0/11, stopped22/18/29. Final assembled predictions exactly reproduced, all frozen sources unchanged,4window coverage verified.874.133s/1730.905MiB, partially concurrent. Keep positive heads_best.pt and A6/B43/D52 sources. This small development gain is not fullOOF or independent superiority; remaining folds still user-deferred.
- Whole-ABD feature-backbone session10715 confirmed live; epoch7 own-output TOTAL79.2182588, per-channel best candidate79.2587058. No legacy output voting; candidate not final reconstructed score. Continue monitoring only, do not launch further routes while user asks to watch experiments.


## New submission delivered without remaining folds
- Created D:/match_predict/submission_a6_heads_fold5_80_501552 with result.zip + submission_code.zip + README. A6 independent heads4/0/11 + B43/D52, fixedfold5 only. Test10wells/95948rows, exact schema/depth order and resultZIP root checked; actual CLI ran from new ZIP extraction in normalCUDA.
- Exported fullfifth reconstruction matches all144665saved predictions exactly after matching checkpoint row selection; score80.50155201341903. Test labels unavailable. No old delivery overwritten, no additional folds. Structural feature-head training PID78432 preserved.


## User goal threshold updated
- Fixed fifth-fold screening is now STRICT>80.5, clear gain required before remaining folds; full original groupedOOF>82 still unmet. Added fold5_screening_805.py and latest rules without editing any active/completed trainer or delivery ZIP.
- Current structural feature-backbone session10715/PID78432 confirmed live; last observed epoch37, candidate79.708274, final assembled verification pending. Continue existing run. No auto-promotion of tiny80.501552 submitted-model gain; no remaining folds launched.


## Channel-wise optimization objective and first mechanism
- User explicitly adds separate optimization of POR/PERM/SW heads/methods. New statistics-readout route train_abd_feature_stats_heads80.py launched session74235 alongside existing10715; max2GPU. Frozen5branches, same base initialization/head/loss/LR/budget; per-channel zero10->128 adapter restores branch means/logstd discarded by normalization (+3840 params, total259659). Own losses and independent checkpoint selection preserved; first runtime evidence pending.
- Added fold5_screening_805.py to new trainer source hashes/config; strict>80.5, clear gain required. Existing source/config/delivery immutable, no extra folds. Compare per-channel outcomes after completion, do not require all3channels to improve simultaneously.


## Explicit overnight stop
User requests stop ALL training and continue tomorrow. All project trainers and waiting launchers terminated, process list verified empty; outputs/overnight_stop.json and per-run paused_snapshot.json record state. Best saved heads and source models retained, no final reconstruction claimed. Do not auto-resume or launch on persistent goal continuation; wait for explicit user resumption. Delivery ZIPs unchanged.

V370/V371批次：原模型+末种子14分头融合80.849582637629（第五折调参诊断），低于原+13的80.863457557382。V362完成5/5；已启动原+13两模型原1–4折补折，控制器30132/首折SOURCE27184，已实测第6轮。固定比重OOF和逐折最优诊断分别保存。V367/V369启动入口退役未执行。curate/build已更新，2节点局部评分口径检查通过。

V370续盯快照：实际SOURCE进程27184已核验，原模型原第1折第28轮，最佳看到第27轮TOTAL81.653359695217，pat14。SOURCE仍活跃，未恢复终点，不是完整八阶段模型或OOF成绩。180秒观察到期后再次核实同一PID/创建时间/命令仍活跃，未重启。CPU维护6196等待8个完成SOURCE，未修改活跃训练输出；当前空闲约4.208GiB，暂时无磁盘失败。下一步继续现有批次，在SOURCE结束后核对本折缓存/Q/幅值与首个POR113衔接。

V370接续调度已准备并实测接管：仅替换CPU控制器30132→86104，实际GPU SOURCE27184的PID/创建时间/命令不变，未停止/重启/恢复训练，原七个训练上下文入口哈希不变。SOURCE结束后允许下一SOURCE与本模型头链重叠，最多2GPU，最多1个SOURCE/1个SW/1个扫描；仍有限8个model-fold、同8阶段/损失/预算/分头选模。当前首SOURCE仍活跃，因此尚未发生训练重叠、尚不能声称提速。每个阶段记录overlapped/solo及全批实际墙钟，最终按相同工作量计时评价。代码与接管证据：experiments/pure_dl/run_overlap_pair_fullcv_v370r1.py、OVERLAP_HANDOFF.json、OVERLAP_START_VERIFICATION.json。

V370原模型原第1折SOURCE终点：TOTAL82.496037758876，POR79.979927065565/PERM90.989052707754/SW76.159689118551；最佳55/停止69，源模型RAM最佳预测精确复现，全部144621验证行覆盖一次。本折训练578935行，Q54严格早于最佳55，1670点降权，验证不降权；POR113实际载入本折private descriptor/shared input/own SOURCE三条路径，无历史第五折权重。SOURCE solo墙钟1187.908s含预处理/预热/训练/恢复/缓存写盘。诊断精简回收702.466MiB，保留权重/Q/恢复依赖SHA精确未变。已接管非父子SOURCE进程的OS退出码不可读，不能写exit0；实际终点source_report+fold_report/cache/无Traceback与PID消失已核实。尚无完整8阶段或全OOF成绩。
