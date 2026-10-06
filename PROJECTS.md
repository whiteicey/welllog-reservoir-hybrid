# Project status

## Outcome
Best verified pure-DL complete grouped5fold OOF81.71323318479078 (previous81.5027197525581). Target82 remains unmet. Last posterior-router fifth80.25089377481923, best52/naturalstop70 of80, below strict80.4. All training exited; no new jobs or continuation authorized.

## Completed
- Added an imagegen-generated architecture diagram to README and METHOD.md, checked against the delivered calibrated-pair architecture. Corrected generated tensor/stage annotations before publication. Documentation/image update only; training remains stopped and delivered archives unchanged.
- User-reported A-leaderboard score83.6915 added for the delivered calibrated-pair five-fold ensemble; provenance explicitly user feedback, not independently fetched. OOF81.713233 and final fifth-fold80.250894 remain unchanged. Delivered ZIPs remain immutable.
- Fivefold best-model delivery in separate immutable submission_calibrated_pair; repeated freshZIP CUDA inference identical across95948testrows; old delivery preserved.
- Method/experiment docs, compactreports, source-only inference package and portable fifth-fold training launcher.
-945 Python AST checks, training/inference help checks;24 inference modules match verified delivery (one trailing-blank cleanup; AST unchanged). No new training for verification.
- Git branch deep_learning published with source/docs/compactresults only. Data,weights,rawpredictions,ZIPs stay local.
- README now explicitly documents fifth-fold-first screening, same-configuration remaining-fold training, OOF versus submission provenance, and the final posterior-router mechanism/result. Documentation update only; training remains stopped.

## Publication and remaining work
Publication branch: origin/deep_learning. Final push is checked against the remote Git reference; use git status/log for commit identity. No more development or training is scheduled. No scientific result is pending; goal82 was not met.

## Future, only if explicitly requested
Explore useful routing mechanisms under fixedfifth strict80.4/max80/18/max2GPU. New deliveries in newdirectories; never overwrite existingversions.

- Localized the architecture illustration to Simplified Chinese using imagegen; README and METHOD now reference calibrated_dual_backbone_zh.png. Original English image retained. Visual text/layout/formulas checked; no model, training or delivered-archive changes.


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


- Shared symmetric posterior router COMPLETE:80.06368214872943,best52/stop70,delta-0.18721162608980535; best checkpoint restored and reproduced. No promotion or shared-scorer grid. Cleanup blocked by automatic approval policy; artifacts still present. Router-only utility PID60060 continues: epoch45,best79.823617357764 at44. No third job; original fullOOF81.713233 remains best, goal>82 unmet.


- Latent posterior router80 launched PID628/session81443: original80.25089377481923 control with detached main-head input representations from both experts, each LayerNorm128->Linear16/GELU, concatenated32->6 zero-initialized router residual; +4838 parameters expected1333956. Explicit equivalent backbone/calibrated68 preprocessing, no hooks or frozen-source changes. Original expert-own NLL, router NLL, separate optimizers, Q feedback,80/18/fifth strict804 unchanged; no utility-loss combination. GPU guard admitted with sole other training PID60060 and10.80GiB free. Utility candidate best80.2115804825 at52, observed56, still running. No promotion or fullOOFgain; fullbest81.713233 unchanged. Next verify actual new-model training and compare original posterior control; no extra preflight/review.


- Latent posterior router80 launched PID628/session81443: original80.25089377481923 control with detached main-head input representations from both experts, each LayerNorm128->Linear16/GELU, concatenated32->6 zero-initialized router residual; +4838 parameters expected1333956. Explicit equivalent backbone/calibrated68 preprocessing, no hooks or frozen-source changes. Original expert-own NLL, router NLL, separate optimizers, Q feedback,80/18/fifth strict804 unchanged; no utility-loss combination. GPU guard admitted with sole other training PID60060 and10.80GiB free. Utility candidate best80.2115804825 at52, observed56, still running. No promotion or fullOOFgain; fullbest81.713233 unchanged. Next verify actual new-model training and compare original posterior control; no extra preflight/review.


## Final result
Router-only utility auxiliary completed: TOTAL80.21158048250079, POR77.30317230550095/PERM89.3009553284588/SW73.61512693111412; best52/stop70, best checkpoint predictions reproduced. Delta versus original posterior80.25089377481923: -0.03931329231843961. No gain, below strict80.4, no remainingfolds or utility coefficient/node grid. Peak804.331MiB; fold1528.949s with mixed solo/two-job overlap (symmetric then latent), not a controlled concurrency benchmark. Source/config/compact evidence retained. Redundant checkpoint/cache cleanup pending because automatic approval policy blocked the preceding equivalent deletion; no bypass attempted.

- Latent posterior router PID628/session81443 continues: epoch16,best78.754266259328 at14. No second new GPU job scheduled; focus on this distinct learned-representation mechanism. Full verified OOF81.71323318479078 remains best; target>82 unmet.


## Final result
Router-only utility auxiliary completed: TOTAL80.21158048250079, POR77.30317230550095/PERM89.3009553284588/SW73.61512693111412; best52/stop70, best checkpoint predictions reproduced. Delta versus original posterior80.25089377481923: -0.03931329231843961. No gain, below strict80.4, no remainingfolds or utility coefficient/node grid. Peak804.331MiB; fold1528.949s with mixed solo/two-job overlap (symmetric then latent), not a controlled concurrency benchmark. Source/config/compact evidence retained. Redundant checkpoint/cache cleanup pending because automatic approval policy blocked the preceding equivalent deletion; no bypass attempted.

- Latent posterior router PID628/session81443 continues: epoch16,best78.754266259328 at14. No second new GPU job scheduled; focus on this distinct learned-representation mechanism. Full verified OOF81.71323318479078 remains best; target>82 unmet.


- Latent router live PID628: {"epoch": 24, "best_epoch": 23, "best_score": 79.21195491610008, "bad_epochs": 1, "status": "running", "matched_best_delta": 0.006133205216940496}. Other GPU slot free. Preparing ONE complementary-block expert-training weighting trial (not launched): independent training-only per-well256-raw-row block balanced signs; own expert losses Q*(1+/-0.5sign), all points retainpositiveweight, router onlyQ. Different from bad-point Q; no extra parameters, no amplitude/blocklength grid. No change to active latent or original posterior source. FullOOFgoal still unmet.


- Complementary-blocks80 launched PID54204/session41117; original posterior80.250894 architecture/init, no new parameters. Each training well raw256-row blocks assigned balanced signs using independent RNG seed20261006+foldindex+73000, recorded in config. Own expert final main/aux losses Q*A with complementary0.5/1.5, all trainingpoints nonzero; router NLL onlyQ, Q/probe unchanged. Equal member average1 perpoint, individual point-average may differ due tailblock sizes. Max80/18/strict804, no blocksize/amplitude grid. SoleotherGPU latent PID628. Gain pending; fullOOFbest81.713233 unchanged.


- Complementary-blocks80 launched PID54204/session41117; original posterior80.250894 architecture/init, no new parameters. Each training well raw256-row blocks assigned balanced signs using independent RNG seed20261006+foldindex+73000, recorded in config. Own expert final main/aux losses Q*A with complementary0.5/1.5, all trainingpoints nonzero; router NLL onlyQ, Q/probe unchanged. Equal member average1 perpoint, individual point-average may differ due tailblock sizes. Max80/18/strict804, no blocksize/amplitude grid. SoleotherGPU latent PID628. Gain pending; fullOOFbest81.713233 unchanged.


- Both verifiedlive jobs628/54204 progressed without errors: [{"epoch": 53, "best_epoch": 52, "best_score": 80.20707445856918, "bad_epochs": 1, "experiment": "paired_latent_posterior_router80", "status": "running", "matched_best_delta": -0.04381931625005109}, {"epoch": 18, "best_epoch": 18, "best_score": 78.93475106082802, "bad_epochs": 0, "experiment": "paired_complementary_blocks80", "status": "running", "matched_best_delta": -0.044269302155342416}]. No model/configchanges or extraGPUjob; fullOOFgoal unmet, strictfifth804 before remainingfolds.


- Both verifiedlive jobs628/54204 progressed without errors: [{"epoch": 53, "best_epoch": 52, "best_score": 80.20707445856918, "bad_epochs": 1, "experiment": "paired_latent_posterior_router80", "status": "running", "matched_best_delta": -0.04381931625005109}, {"epoch": 18, "best_epoch": 18, "best_score": 78.93475106082802, "bad_epochs": 0, "experiment": "paired_complementary_blocks80", "status": "running", "matched_best_delta": -0.044269302155342416}]. No model/configchanges or extraGPUjob; fullOOFgoal unmet, strictfifth804 before remainingfolds.


## Completed latent-router result
{"scores": {"POR": 77.19460787792981, "PERM": 89.37384153998788, "SW": 73.62242158912709, "TOTAL": 80.20707445856918}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1380.5393081000075, "peak_gpu_allocated_mib": 800.13134765625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.04381931625005109, "qualified": false}. Detached expert hidden representations did not improve originalposterior80.25089377481923. No latent-size/grid/remainingfolds. Timing includes solo and two-job phases, not a controlled concurrency benchmark. Source/config/compact evidence retained; redundant artifacts pending prior automatic deletion-policy block, no bypass attempted.

- Complementary-blocks PID54204/session41117 continues: epoch40,best79.719355133220 at39. No second new job; full verifiedOOF81.71323318479078 remains best, goal>82 unmet.


## Completed latent-router result
{"scores": {"POR": 77.19460787792981, "PERM": 89.37384153998788, "SW": 73.62242158912709, "TOTAL": 80.20707445856918}, "best_epoch": 52, "stopped_epoch": 70, "seconds": 1380.5393081000075, "peak_gpu_allocated_mib": 800.13134765625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_original_posterior": -0.04381931625005109, "qualified": false}. Detached expert hidden representations did not improve originalposterior80.25089377481923. No latent-size/grid/remainingfolds. Timing includes solo and two-job phases, not a controlled concurrency benchmark. Source/config/compact evidence retained; redundant artifacts pending prior automatic deletion-policy block, no bypass attempted.

- Complementary-blocks PID54204/session41117 continues: epoch40,best79.719355133220 at39. No second new job; full verifiedOOF81.71323318479078 remains best, goal>82 unmet.

- Complementary-blocks PID54204/session41117 verified live: {"epoch": 54, "best_epoch": 48, "best_score": 79.95256297808557, "bad_epochs": 6, "status": "running", "matched_best_delta": -0.2983307967336657}. No otherGPUjob or configchanges; no finalgain/804qualification claim. FullOOFgoal unmet.

- Complementary-blocks PID54204/session41117 verified live: {"epoch": 54, "best_epoch": 48, "best_score": 79.95256297808557, "bad_epochs": 6, "status": "running", "matched_best_delta": -0.2983307967336657}. No otherGPUjob or configchanges; no finalgain/804qualification claim. FullOOFgoal unmet.


## Completed complementary-block result
{"scores": {"POR": 76.52249314578376, "PERM": 89.39835427846991, "SW": 73.44683153395995, "TOTAL": 79.95256297808557}, "best_epoch": 48, "stopped_epoch": 66, "seconds": 1236.7837184999953, "peak_gpu_allocated_mib": 781.72509765625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_posterior": -0.2983307967336657, "qualified": false}. Fixed random-block complementary learning priority did not improve original posterior control. No blocklength/strength grid or remainingfolds. Timing has solo/two-job phases and is not a controlled concurrency benchmark. Compact evidence retained; redundant weight/cache deletion pending prior automatic-policy rejection.

- Preparing independent Q0/Q1/mergedQ diagnostic trial, NOT launched. Original expert-own thresholds/recovery and final loss denominator unchanged, router uses originalmergedQ. One memberforward per batch, sequential roleCDF; approximately5GiB scratch, no frozen source edits. No active training after block runner completion; next finish candidate then admit safely. FullOOFbest81.713233 remains, goal>82 unmet.


## Completed complementary-block result
{"scores": {"POR": 76.52249314578376, "PERM": 89.39835427846991, "SW": 73.44683153395995, "TOTAL": 79.95256297808557}, "best_epoch": 48, "stopped_epoch": 66, "seconds": 1236.7837184999953, "peak_gpu_allocated_mib": 781.72509765625, "best_checkpoint_predictions_reproduced": true, "status": "complete", "delta_vs_posterior": -0.2983307967336657, "qualified": false}. Fixed random-block complementary learning priority did not improve original posterior control. No blocklength/strength grid or remainingfolds. Timing has solo/two-job phases and is not a controlled concurrency benchmark. Compact evidence retained; redundant weight/cache deletion pending prior automatic-policy rejection.

- Preparing independent Q0/Q1/mergedQ diagnostic trial, NOT launched. Original expert-own thresholds/recovery and final loss denominator unchanged, router uses originalmergedQ. One memberforward per batch, sequential roleCDF; approximately5GiB scratch, no frozen source edits. No active training after block runner completion; next finish candidate then admit safely. FullOOFbest81.713233 remains, goal>82 unmet.


- Independent-Q80 launched PID25512/session21529, sole GPU job; disk16.88885498GiB available at launch. Original posterior80.250894 architecture/init, three original policies/tables member0/member1/merged. Expert main+aux final point losses use ownQ, router usesmergedQ. Same thresholds/recovery/defaultmask/population denominator, alltrainingpoint diagnostics from one pairforward perbatch, sequential roleCDF/RNG restoration. Float32 previousCDF scratch~5GiB; compressed evidence perrole/perrecord, latest onlyRAM; normalfinalize cleans only ownscratch. New80/18/fixedfifthstrict804; no preflight/review or gainclaim, futurepromotion must adapt three-role audit. No frozen sources/delivery changes.


- Independent-Q80 launched PID25512/session21529, sole GPU job; disk16.88885498GiB available at launch. Original posterior80.250894 architecture/init, three original policies/tables member0/member1/merged. Expert main+aux final point losses use ownQ, router usesmergedQ. Same thresholds/recovery/defaultmask/population denominator, alltrainingpoint diagnostics from one pairforward perbatch, sequential roleCDF/RNG restoration. Float32 previousCDF scratch~5GiB; compressed evidence perrole/perrecord, latest onlyRAM; normalfinalize cleans only ownscratch. New80/18/fixedfifthstrict804; no preflight/review or gainclaim, futurepromotion must adapt three-role audit. No frozen sources/delivery changes.

- Actual execution: independentQ completed epochs1/2 and both full three-role probes; eachrole578891trainingpoints, warmupallQ1. Probes21.55/22.59seconds, observedGPUpeak779.141MiB, scratch5335059456bytes, remainingdisk12682543104bytes. Noerror; no gainclaim before Q divergence. Continue original80/18/fixedfifth804.

- Verifiedlive independentQ PID25512 progress: {"epoch": 14, "best_epoch": 14, "best_score": 78.6600808936658, "bad_epochs": 0, "latest_quality_update": {"observed_epoch": 12, "applies_from_epoch": 13, "roles": {"member0": {"downweighted_by_channel": [0, 0, 0], "downweighted_points": 0, "minimum": 1.0}, "member1": {"downweighted_by_channel": [0, 0, 0], "downweighted_points": 0, "minimum": 1.0}, "merged": {"downweighted_by_channel": [0, 0, 0], "downweighted_points": 0, "minimum": 1.0}}}, "status": "running"}. Allthree train-only populations fully covered, noerrors; no proven gain/804qualification.

- Mechanism diagnostic from immutableepoch16 weightshards (identical578891rawids): {"observed_epoch": 16, "applies_from_epoch": 17, "points": 578891, "downweighted_points": {"member0": 356, "member1": 262, "merged": 235}, "pairwise": {"member0__member1": {"intersection": 77, "union": 541, "only_first": 279, "only_second": 185}, "member0__merged": {"intersection": 132, "union": 459, "only_first": 224, "only_second": 103}, "member1__merged": {"intersection": 116, "union": 381, "only_first": 146, "only_second": 119}}}. Different downgrade sets demonstrate expert-specific decisions, not correctness of anomaly labels or validation gain. Currentepoch18 best78.9017215; continue.

- IndependentQ PID25512 verifiedlive midtraining: {"epoch": 27, "best_epoch": 27, "best_score": 79.25551320093996, "bad_epochs": 0, "status": "running", "matched_best_delta": -0.12677612159951934}. No errors/configchanges, no extraGPUjob, no currentgainclaim. Preserve original fifth804 gate and fullOOFgoal.

- IndependentQ PID25512 verifiedlive midtraining: {"epoch": 27, "best_epoch": 27, "best_score": 79.25551320093996, "bad_epochs": 0, "status": "running", "matched_best_delta": -0.12677612159951934}. No errors/configchanges, no extraGPUjob, no currentgainclaim. Preserve original fifth804 gate and fullOOFgoal.

- User explicitly continues; independentQ PID25512 unchanged and progressing: {"epoch": 36, "best_epoch": 34, "best_score": 79.63463010605592, "bad_epochs": 2, "status": "running", "matched_best_delta": -0.03987819166145812}. No addedGPUjobs or variant; fullOOFgoal unmet.

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

## 2026-09-27 鍙屼笓瀹剁壒寰佷紭鍏堢画杩?
- 鍘熷悗楠屽弻涓撳绗簲鎶樺鐓?0.2508937748銆係W gate+means鏉′欢鍖栧凡瀹屾暣缁撴潫锛歜est48/stop66锛屽畼鏂筎OTAL80.0967730451锛岃緝鍘熷弻涓撳-0.1541207297锛涙渶浣砪heckpoint棰勬祴宸插鐜帮紝鏈繃涓ユ牸>80.4锛屼笉琛ユ姌銆傜嫭绔嬭褰?outputs/pure_dl_paired_sw_conditioned80_sequence/README.md銆?
- SW浠呭潎鍊兼秷铻嶇户缁繍琛孭ID33536锛涙柊灞€閮ㄧǔ鍋ョ壒寰佸疄楠岃繍琛孭ID7748锛孏PU瀹堝崼鏈€澶?浠诲姟銆傛柊璇曢獙瀵?鏇茬嚎璁＄畻浜曞唴101/401鐐圭ǔ鍋ヤ綅缃紝鍙敞鍏ョ浜屼笓瀹舵牎鍑嗗眰锛涘師206銆佺涓€涓撳鍙婅矾鐢辫緭鍏ヤ繚鎸併€傜嫭绔嬭褰?outputs/pure_dl_paired_local_robust80_sequence/README.md锛岀粨鏋滃緟瀹氥€?
- 褰撳墠瀹屾暣OOF鏈€濂戒粛81.7132331848锛?82鐩爣鏈畬鎴愩€備笅涓€姝ョ洴瀹屼袱涓浐瀹氱浜旀姌锛涗粎涓ユ牸>80.4鐨勫€欓€夋牳楠屽悗杩涘叆鍓╀綑鎶樸€?

## 2026-09-27 SW娑堣瀺缁撴潫涓庡眬閮ㄧ壒寰侀噸璺?
- SW浠呭潎鍊肩粍浠舵秷铻嶇浜旀姌宸插畬鏁寸粨鏉燂細best52/stop70锛屽畼鏂筎OTAL80.0468714929锛岃緝鍘熷弻涓撳80.2508937748涓?0.2040222819锛岃緝闂ㄦ帶鍔犲潎鍊?0.0967730451涓?0.0499015522銆傝缁冨櫒宸叉仮澶嶆渶浣砪heckpoint骞跺鐜伴娴嬶紱鏈繃涓ユ牸>80.4锛屼笉琛ユ姌銆傝褰?outputs/pure_dl_paired_sw_mean_conditioned80_sequence/README.md銆?
- 鏂颁簳鍐?01/401鐐瑰眬閮ㄧǔ鍋ョ壒寰侀娆″疄楠宔poch1楠岃瘉鍚庨伃閬嘍鐩樼┖闂翠笉瓒筹細鍏ㄨ缁冪偣鎺㈡祴鐨勪复鏃禖DF闇€绾?.78GiB锛岃€屽綋鏃朵粎浣欑害1.41GiB锛沞poch1涓嶆槸瀹屾垚鍒嗘暟銆傚鎵圭郴缁熸嫆缁濆垹闄や笁涓凡瀹屾垚璐熷疄楠岀殑澶ц瘖鏂紦瀛橈紝鎷掔粷鐞嗙敱涓篵locked by policy锛屾湭缁曡繃鍒犻櫎銆?
- 鏀逛负鐙珛鍐呭瓨鎺㈡祴閲嶈窇锛氱簿纭甪loat32鍏ㄧ偣CDF淇濆瓨鍦≧AM锛孮鍜岃缁?璇勫垎涓嶅彉锛屾渶缁堜粎淇濈暀绱у噾鐘舵€侊紱绛夊緟鍧囧€兼秷铻嶇粨鏉熼噴鏀綬AM鍚庡崟鐙惎鍔≒ID73824/session31259銆傝褰?outputs/pure_dl_paired_local_robust_memory80_sequence/README.md锛岀粨鏋滃緟瀹氥€?
- 瀹屾暣浜旀姌鏈€濂戒粛81.7132331848锛?82鐩爣鏈畬鎴愩€傚彧鍦ㄥ浐瀹氱浜旀姌涓ユ牸>80.4骞跺畬鎴愬繀瑕佹牳楠屽悗缁鍏朵粬鍥涙姌銆?

## 2026-09-27 灞€閮ㄧ壒寰佺粨鏉熶笌鏇茬嚎鍏崇郴缁瘯
- 涔濇洸绾夸簳鍐?01/401鐐瑰眬閮ㄧǔ鍋ヤ綅缃壒寰佸凡瀹屾垚绗簲鎶橈細best52/stop70锛屽畼鏂筎OTAL80.1069906584锛岃緝鍘熷弻涓撳80.2508937748涓?0.1439031164锛汸ERM+0.058649鑰孭OR/SW涓嬮檷锛屾湭杩囦弗鏍?80.4锛屼笉琛ユ姌銆傚唴瀛樼増鍏ㄧ偣float32 Q鏃犵鐩樹复鏃禖DF锛孏PU宄板€?129.83MiB锛屽崟浠诲姟958.42s銆傝褰?outputs/pure_dl_paired_local_robust_memory80_sequence/README.md銆?
- 涓嬩竴椤圭壒寰佹満鍒跺凡鍚姩PID17316/session32628锛氬洓缁勫師鏇茬嚎鍦?01/401鐐圭獥鍙ｇ殑浜曞唴婊氬姩鐩稿叧锛屼粎瀵瑰悗楠岃矾鐢?logits 鎻愪緵闆跺垵濮嬪寲8缁翠晶淇℃伅锛涗笓瀹?06/68杈撳叆鍙婃崯澶变笉鍙樸€傝褰?outputs/pure_dl_paired_local_curve_relation80_sequence/README.md銆傛渶澶?GPU锛屾鍒诲彧鏈?椤广€?
- 瀹屾暣浜旀姌鏈€楂?1.7132331848锛?82鐩爣鏈畬鎴愶紱浠呭浐瀹氱浜旀姌涓ユ牸>80.4骞舵牳楠屽悗琛ユ姌銆?

## 2026-09-27 灞€閮ㄧ浉鍏宠礋缁撴灉涓庢棤鐩戠潱鍘熷瀷璇曢獙
- 鍥涚粍鏇茬嚎灞€閮ㄧ浉鍏宠矾鐢辩浜旀姌瀹屾暣缁撴潫锛歜est52/stop70锛屽畼鏂筎OTAL80.0558691617锛岃緝鍘熷悗楠屽弻涓撳80.2508937748涓?0.1950246131锛涙渶浣砪heckpoint棰勬祴宸插鐜帮紝鏂板璺敱鏉冮噸L2=0.231098浣嗘湭鏀瑰杽锛屾湭杩囦弗鏍?80.4銆佷笉琛ユ姌銆傚崟浠诲姟901.28s/鏄惧瓨1099.50MiB銆傝褰?outputs/pure_dl_paired_local_curve_relation80_sequence/README.md銆?
- 鏂扮殑浜曞唴涔濇洸绾跨粍鍚堣蒋鍘熷瀷瀹為獙宸插惎鍔≒ID8992/session35829锛歁iniBatchKMeans浠呮嫙鍚?4璁粌浜曡緭鍏ワ紝12涓腑蹇冪殑杞綊灞為浂鍒濆鍖栨敞鍏ュ師鍙屼笓瀹堕€愮洰鏍囪矾鐢憋紱楠岃瘉浜曞彧鍋氬彉鎹紝鐩爣鏍囩涓嶅弬涓庤仛绫汇€傚崟鎶樼粨鏋滃緟瀹氾紝璁板綍 outputs/pure_dl_paired_facies_router80_sequence/README.md銆?
- 瀹屾暣浜旀姌鏈€楂樹粛81.7132331848锛?82鐩爣鏈畬鎴愶紱涓ユ牸绗簲鎶?80.4闂ㄦ缁х画閫傜敤銆?

## 2026-09-27 鍘熷瀷鐗瑰緛鐨勬敞鍏ヤ綅缃鐓?
- 浠呰缁冧簳鎷熷悎鐨?2涓洸绾跨粍鍚堣蒋鍘熷瀷鐢ㄤ簬璺敱锛岀浜旀姌瀹屾暣缁撴灉80.2430688700锛坆est52/stop70锛夛紝杈冨師鍙屼笓瀹?0.2508937748涓?0.0078249048锛涙渶浣砪heckpoint棰勬祴澶嶇幇锛屾湭杩囦弗鏍?80.4锛屼笉琛ユ姌銆傝矾鐢辨柊澧炴潈閲峀2=0.505334锛屽崟浠诲姟909.34s/鏄惧瓨1111.35MiB銆傝褰?outputs/pure_dl_paired_facies_router80_sequence/README.md銆?
- 鍚屼竴鍘熷瀷鐗瑰緛鏀逛负浠呮敞鍏ョ浜屼笓瀹惰緭鍏ユ牎鍑嗗眰鐨勫崟娆℃満鍒跺鐓у凡鍚姩PID37756/session3073锛涘師鍨嬫暟銆佹嫙鍚堟柟娉曘€佸垎鎶?seed/鎹熷け/Q/LR/棰勭畻涓嶅彉銆傝褰?outputs/pure_dl_paired_facies_calibrated80_sequence/README.md锛岀粨鏋滃緟瀹氥€?
- 瀹屾暣浜旀姌鏈€濂戒粛81.7132331848锛涗弗鏍肩浜旀姌>80.4闂ㄦ涓庡畬鏁碠OF>82鐩爣涓嶅彉銆?

## 2026-09-27 鐗瑰緛浼樺厛鎵规缁撹
- 鍥哄畾绗簲鎶樺洓椤规柊鐗瑰緛鏈哄埗鍧囧畬鏁村畬鎴愩€佹渶浣砪heckpoint澶嶇幇锛氬眬閮ㄧǔ鍋ヤ綅缃?0.106991銆佸眬閮ㄦ洸绾垮叧绯昏矾鐢?0.055869銆?2璁粌浜曟棤鐩戠潱鍘熷瀷璺敱80.243069銆佸悓涓€鍘熷瀷娉ㄥ叆绗簩涓撳鏍″噯80.117536銆傚師鍚庨獙鍙屼笓瀹?0.250894浠嶆渶濂斤紱鍏ㄩ儴浣庝簬涓ユ牸>80.4锛屼笉琛ュ叾浣欏洓鎶橈紝涔熶笉瀹ｇО瀹屾暣OOF鏀剁泭銆備袱椤筍W鏉′欢鍖栨敹灏惧悓鏍蜂负璐燂細80.096773/80.046871銆?
- 閫愰」README鍙婃眹鎬?docs/deep_learning/FEATURE_FIRST_20260927.md 璁板綍鍋囪銆佸鐓с€佸垎鏁般€佽€楁椂銆佹樉瀛樸€佹棤鏀剁泭鏂瑰悜鍜屽悗缁鍒掋€傚唴瀛樻帰娴嬪叏璁粌鐐筬loat32 Q瑙ｅ喅鏈壒D鐩樹复鏃禖DF涓嶈冻闂锛涘師濮嬪け璐poch1涓嶅彲褰撳畬鏁存垚缁┿€?
- 褰撳墠鏃燝PU璁粌浠诲姟銆侱鐩樼害浣?.28GiB锛涚簿鍑嗗垹闄や笁涓凡瀹屾垚璐熷疄楠岀殑澶ц瘖鏂紦瀛樿鑷姩瀹℃壒浠locked by policy鎷掔粷锛屾湭鎵ц涓旀湭缁曡繃銆備笅涓€椤归暱璁粌鍓嶉渶鍏堟湁鍙潬瀛樺偍浣欓噺锛涘湪璇ョ害鏉熶笅鍙户缁唬鐮?鏈哄埗璁捐銆傛墜宸ヤ晶鐗瑰緛缁х画寰皟鐨勮瘉鎹笉瓒筹紝涓嬩竴姝ヨ€冭檻璁粌浜曞唴瀛︿範鐨勮法鏇茬嚎琛ㄥ緛銆傚畬鏁翠簲鎶樻渶楂?1.7132331848锛?82鐩爣鏈畬鎴愩€傜洰鏍囪揪鎴愬悗鍙敤鐙珛绉嶅瓙/鐣欎簳楠岃瘉绋冲畾鎬у苟鏁寸悊鏂扮増鐙珛浜や粯銆?

## 2026-09-27/28 瀛︿範鍨嬭法鏇茬嚎娈嬪樊涓庤交閲忎骇鐗?
- 鍦ㄥ師鍚庨獙鍙屼笓瀹剁浜旀姌80.250894涓婃柊澧炰節鏇茬嚎閫愮偣16缁?2澶碼ttention娈嬪樊锛屼粎淇full206涓撳杈撳叆锛屽彟涓€涓撳鍙?2缁村悗楠岃矾鐢卞師鏍凤紱鍖哄埆浜庡凡澶辫触鐨勭嫭绔媍urve-token缃戠粶銆佺绾縨asked棰勮缁冨拰鍗曟洸绾垮嵎绉€傚浐瀹氱浜旀姌/seed+4/bs64/LR.0006/80杞甤osine/patience18/Q/瀹樻柟璇勫垎銆侾ID87612/session13408鍗旼PU杩愯锛屽疄楠岀嫭绔嬭褰?outputs/pure_dl_paired_curve_attention80_sequence/README.md銆?
- 鏂拌缁冨櫒浣跨敤鍏ㄨ缁冪偣float32鍐呭瓨CDF涓庣揣鍑戞湯鎬侊紱濡傛灉绗簲鎶樻湭杩囦弗鏍?80.4锛屽垯淇濆瓨閰嶇疆銆佹瘡杞垎鏁般€侀娴嬪強绱у噾璇婃柇锛屼笉钀界洏璐熷疄楠宑heckpoint鍜屽ぇ鍒嗗竷銆傝嫢杩囬棬妲涳紝淇濆瓨瀹屾暣鏈€浣虫潈閲嶅強鍒嗗竷渚涘繀瑕佹牳楠屼笌鍏朵粬鍥涙姌锛涗笉鏀瑰彉璁粌涓庢渶浣崇姸鎬侀€夋嫨銆侱鐩樺惎鍔ㄧ害1.28GiB锛屾晠鏈壒閬垮厤鏃犵敤澶т骇鐗┿€?
- 瀹屾暣OOF>82浠嶆湭瀹屾垚锛涘悗缁槸鍚﹀苟鍙戠浜孏PU浠诲姟瑙嗘湰瀹為獙鏄惧瓨/鍐呭瓨宄板€煎拰鍊欓€夌嫭绔嬫€у喅瀹氾紝鏈€澶?銆?

## 2026-09-27/28 attention娉ㄥ叆浣嶇疆鍙屽苟鍙?
- full206涓撳涔濇洸绾縜ttention娈嬪樊绗簲鎶樿繍琛孭ID87612/session13408锛屽墠鏈熷叏鐐规帰娴嬪嘲鍊肩害959MiB锛涙牎鍑?8涓撳鐨勫悓妯″潡鍚屽弬鏁版敞鍏ュ鐓у凡鍚姩PID38088/session92651銆侴PU瀹堝崼纭鏈€澶?浠诲姟涓旀棤OOM杩硅薄锛涗袱璺潎80杞笂闄?鏈€浣抽獙璇佺姸鎬?涓ユ牸绗簲>80.4銆?
- 绗簩璺ā鍨嬩粎瑕嗗啓娉ㄥ叆浣嶇疆锛屼笉淇敼娲诲姩绗竴璺簮鐮侊紱婧愪笌缁撴灉璁板綍 outputs/pure_dl_paired_curve_attention_calibrated80_sequence/README.md銆傛湭杩囬棬妲涗笉钀界洏澶у垎甯?鏉冮噸锛涙病鏈夋柊瀹屾暣浜旀姌鍒嗘暟锛岀洰鏍?82浠嶆湭瀹屾垚銆?

## 2026-09-27/28 涔濇洸绾垮涔犲瀷娈嬪樊缁撴灉
- full206涓撳娉ㄥ叆璺湪鏍稿PID87612/鏂伴矞history鍚庝簬epoch64鎻愬墠缁堟锛宐est observed80.059460/epoch52銆?2杞湭鍒锋柊銆佽窛涓ユ牸闂ㄦ0.340540锛況unner棰勬湡閫€鍑?锛岃繘绋嬫秷澶憋紝鏃犲畬鏁存渶浣崇姸鎬佸鍑猴紝鏍囨敞PARTIAL锛屼笉浣滀负瀹屾垚绗簲鎶樺垎鏁般€傝褰?outputs/pure_dl_paired_curve_attention80_sequence/manual_early_stop.json 涓嶳EADME銆?
- 鍚屾ā鍧楁敞鍏ユ牎鍑?8涓撳鐨勫畬鎴愮浜旀姌80.1837020319锛宐est52/stop70锛屾渶浣崇姸鎬侀娴嬪鐜帮紱杈冨師鍚庨獙鍙屼笓瀹?0.250894涓?0.067192锛孭OR-0.012977/PERM-0.055329/SW-0.125524銆傚崟椤?147.44s銆佹樉瀛樺嘲鍊?70.96MiB锛涙湭杩囦弗鏍?80.4锛屼笉琛ユ姌锛岃礋瀹為獙鏈惤鐩榗heckpoint/澶у垎甯冦€傝褰?outputs/pure_dl_paired_curve_attention_calibrated80_sequence/README.md銆?
- 褰撳墠涓PU浠诲姟鍧囧凡閫€鍑猴紱D鐩樼害1.24GiB鍓╀綑銆傛柊杞婚噺璁粌鍣ㄦ垚鍔熼伩鍏嶆棤鏀剁泭鏉冮噸/鍒嗗竷鍗犵洏锛涘畬鏁翠簲鎶樻渶濂戒粛81.7132331848锛?82鐩爣鏈畬鎴愩€備笅涓€姝ユ敼鐢ㄩ拡瀵筍W杈撳嚭/鎹熷け鐨勬満鍒讹紝鑰岄潪缁х画鎵洸绾縜ttention瀹归噺銆?

## 2026-09-27/28 SW鐩爣鐗瑰紓涔濇洸绾縜ttention
- 鏍″噯68杈撳叆attention绗簲鎶樺畬鏁?0.1837020319锛宐est52/stop70锛岃緝鍘熷弻涓撳80.2508938涓?0.067192锛屽叾涓璖W-0.125524鏈€澶э紱璐熷疄楠屾寜鏂拌鍒欐湭鍐檆heckpoint/澶у垎甯冿紝鏄惧瓨宄板€?70.96MiB锛屾椂闂?147.44s锛堝惈閮ㄥ垎骞跺彂锛夈€俧ull206杈撳叆鍚屾ā鍧楄矾epoch64浜哄伐鏃╁仠PARTIAL best observed80.059460锛屼笉鍏呬綔瀹屾暣缁撴灉銆?
- 鏂拌瘯楠屼粎淇姣忎綅涓撳SW涓婚棬鎺т笌8涓潎鍊硷細杈撳叆涓轰節鏇茬嚎閫愮偣attention鑰岄潪鍏堝墠澶辫触鐨凱OR/PERM鍚庨獙鏉′欢鍖栥€傚浐瀹氱浜旀姌銆乻eed+4銆?0杞笂闄愩€乸atience18銆佸師Q/NLL/璇勫垎鍙婅交閲忚礋浜х墿銆侾ID72412/session27270鍗旼PU杩愯锛涚嫭绔嬭褰?outputs/pure_dl_paired_sw_curve_attention80_sequence/README.md銆備弗鏍?80.4鎵嶆牳楠?琛ュ洓鎶橈紝瀹屾暣OOF>82鏈畬鎴愩€?

## 2026-09-27/28 SW鐩爣澶翠笌鍏堥獙闂ㄦ帶鎹熷け鍙岃矾绾?
- SW涔濇洸绾夸富澶碼ttention PID72412/session27270鎸佺画锛涚浜岃矾鍘熷弻涓撳涓嶆敼缃戠粶锛屼粎浠?4璁粌浜昐W榛樿鐜?.7211513璁惧潎鍊煎綊涓€鐨勯棬鎺CE鏉冮噸(default0.8550/nondefault1.3750)锛孭OR/PERM鍙婅繛缁璑LL/Q/decoder涓嶅彉銆傜浜岃矾PID85936/session35960锛孏PU瀹堝崼纭鏈€澶?浠诲姟銆佸唴瀛?鏄惧瓨鍚姩瀹夊叏銆傜嫭绔嬭褰?outputs/pure_dl_paired_sw_gate_prior80_sequence/README.md銆?
- 涓よ矾閮藉彧璺戝浐瀹氱浜旀姌80杞笂闄?鏈€浣崇姸鎬?涓ユ牸>80.4闂ㄦ锛岃礋瀹為獙涓嶈惤鐩樺ぇ鏉冮噸/鍒嗗竷銆傚畬鏁翠簲鎶極OF>82浠嶆湭瀹屾垚锛屼笉鎶婄敤鎴峰弽棣圓姒滃垎鏁板綋鏈湴楠岃瘉銆?

## 2026-09-27/28 浜曠骇SW榛樿鍏堥獙鏈哄埗
- 64璁粌浜曚節鏇茬嚎閫愪簳鍒嗕綅鏁?缂哄け鐜囩殑浜旀姌浜曠骇Ridge妫€鏌lpha100锛氶娴婼W榛樿姣斾緥R虏绾?.156銆丮AE0.0457锛屽父鏁板熀绾縈AE0.0523锛涘急浣嗛潪闆剁殑杈撳叆鍙瘑鍒€с€傜浜旀姌浣庡垎浜曠殑SW榛樿姣斾緥澶氫负0.60~0.64锛岃缁冩€讳綋绾?.72銆傞獙璇佹爣绛惧彧鐢ㄤ簬浜嬪悗鎻愬嚭鍋囪锛屼笉杈撳叆妯″瀷/鎷熷悎褰掍竴銆?
- 鏂拌瘯楠屽師鍚庨獙鍙屼笓瀹跺姞闆跺垵濮嬪寲36鈫?6鈫?浜曠骇SW涓婚棬鎺у亸缃紝鍘?06鐗瑰緛銆丳OR/PERM銆佽繛缁垎甯?鎹熷け/Q/璺敱鐩存帴杈撳叆涓嶅彉锛涜缁冧簳鎷熷悎渚х壒寰佸綊涓€锛岄獙璇佷簳鍙敤鑷韩鏃犳爣绛捐緭鍏ャ€侾ID6588/session15382鍗旼PU锛岀浜旀姌80杞笂闄愩€佷弗鏍?80.4鎵嶆牳楠岃ˉ鎶樸€傝褰?outputs/pure_dl_paired_well_sw_prior80_sequence/README.md銆傚畬鏁碠OF>82鏈畬鎴愩€?

## 2026-09-27/28 浜曠骇鐜鐨勯棬鎺?璺敱娉ㄥ叆瀵圭収
- 浜曠骇36缁存棤鏍囩杈撳叆涓婁笅鏂嘢W闂ㄦ帶PID6588/session15382缁х画锛涗娇鐢ㄥ畬鍏ㄧ浉鍚岀壒寰佸綊涓€/缃戠粶瀹归噺浣嗗彧鍔犲埌SW涓撳璺敱鐨勫鐓ID83984/session61461宸蹭綔涓虹浜孏PU浠诲姟鍚姩銆傚師鍙屼笓瀹跺垵濮嬭緭鍑虹粡闆舵湯灞備繚鎸侊紝鍥哄畾绗簲鎶?seed/鍘烸/NLL/璇勫垎/80杞笂闄?strict>80.4涓€鑷淬€傝褰?outputs/pure_dl_paired_well_sw_router80_sequence/README.md锛涗袱璺粨鏋滃緟瀹氾紝鏈€澶?浠诲姟銆?
- D鐩樹綆浣欓噺涓嬬户缁部鐢ㄥけ璐ュ€欓€変笉鍐欐潈閲?澶у垎甯冪殑瑙勫垯锛涗笉浠ラ獙璇佷簳鐪熷疄榛樿鐜囧奖鍝嶆嫙鍚堟垨鎺ㄧ悊銆傚畬鏁碠OF>82鏈畬鎴愩€?

## 2026-09-27/28 鐩爣鍒嗗伐鎹熷け鏂拌矾绾?
- 鍚屼竴36缁存暣浜曡緭鍏ョ殑SW闂ㄦ帶/璺敱涓よ矾绗簲鎶樺潎宸插畬鎴愶細79.8943138225涓?0.0716546023锛屽悗鑰呭湪鍖归厤鐗瑰緛/灏忕綉缁滃鐓?0.1773407799浣嗕粛浣庝簬鍘熷悗楠屽弻涓撳80.2508938锛涗袱鑰呮湭杩?80.4锛屼笉琛ユ姌銆傞€愰」README/棰勬祴璇佹嵁淇濈暀锛屽け璐ユ潈閲?澶у垎甯冩湭钀界洏锛屽師濮嬮噷绋嬬妯″瀷涓嶅姩銆?
- 鎹㈡€濊矾涓轰笉鍔犵壒寰?涓嶅鍙傛暟鐨勪袱涓撳鎸夌洰鏍囦簰琛ョ洃鐫ｏ細full206鏉冮噸POR/PERM/SW=[1.25,1.25,0.5]锛屾牎鍑?8=[0.75,0.75,1.5]锛屾瘡鐩爣涓や笓瀹跺钩鍧?锛況outer鎹熷け/鍘烸/decoder涓嶅彉銆侾ID7012/session83665鍗旼PU锛岀浜旀姌80杞笂闄愪笌涓ユ牸>80.4涓嶅彉锛岃褰?outputs/pure_dl_paired_target_specialist80_sequence/README.md銆傚畬鏁碠OF>82浠嶆湭瀹屾垚銆?

## 2026-09-27/28 鍙屼笓瀹剁洰鏍囧垎宸ヨ鑹插鐓у苟鍙?
- 涓昏矾绾跨洰鏍囨潈閲峟ull206=[1.25,1.25,.5]銆乧alibrated68=[.75,.75,1.5]杩愯PID7012/session83665锛涘畬鍏ㄥ弽鍚戠殑鍖归厤瑙掕壊瀵圭収杩愯PID65480/session38700銆傛ā鍨嬪弬鏁?seed/fold/Q/routerNLL/璁粌棰勭畻瀹屽叏涓€鑷达紱鍧囧彧鐢ㄧ浜旀姌绛涢€変笖闂ㄦ涓ユ牸>80.4銆傜浜岃矾鐙珛璁板綍 outputs/pure_dl_paired_target_specialist_reverse80_sequence/README.md锛屾渶澶?GPU銆佽礋缁撴灉涓嶇暀鏉冮噸/澶у垎甯冦€傜洰鏍囧畬鏁碠OF>82鏈畬鎴愩€?

- 2026-09-27/28锛氬弽鍚戠洰鏍囦笓瀹跺畬鏁寸浜旀姌79.7575406262锛坆est47/stop65锛夛紝杈冨師鍚庨獙鍙屼笓瀹?0.2508937748涓?0.493353锛涗富鐩爣涓撳epoch47鎻愬墠鍋滄锛岃瀵熸渶濂?9.665570锛圥ARTIAL锛夈€備袱绉嶇洰鏍囧垎宸ユ病鏈夋敹鐩婏紝鍧囦笉琛ユ姌銆傝浆鍚戝浐瀹?缁存湁绗﹀彿31/101灞€閮ㄥ姣旂壒寰侊紝鍘?06鍒?鍙屼笓瀹跺垵濮嬪寲/鎹熷け/Q涓嶅彉锛屽崟鐙浜旀姌绛夊緟鍚姩锛涗弗鏍?80.4鎵嶈€冭檻琛ユ姌銆傚師瀹屾暣浜旀姌鏈€濂?1.7132331848涓嶅彉銆?

- 鏈夌鍙峰眬閮ㄥ姣旇瘯楠孭ID40156姝ｅ湪鍥哄畾绗簲鎶橀澶勭悊锛屽敮涓€GPU璁粌杩涚▼锛涘垵娆″惎鍔ㄩ厤缃畫鐣欏悕绉板鑷磋缁冨墠澶辫触锛屾簮鐮佷慨澶嶅悗閲嶅惎锛屾湭鎶婂け璐ュ惎鍔ㄨ鍏ユ垚缁┿€傝礋缁撴灉淇濈暀绱у噾璇佹嵁銆?

- 鏈夌鍙蜂節鏇茬嚎灞€閮ㄥ姣旂浜旀姌瀹屾暣80.0336911714锛坆est52/stop70锛?07.54s锛孏PU796.72MiB锛夛紝杈冨師80.2508938涓嬮檷0.217203锛涘凡澶嶇幇鏈€浣抽娴嬩絾鏈惤鐩樿礋妯″瀷鏉冮噸/澶у垎甯冿紝鏈繃涓ユ牸>80.4锛屼笉琛ユ姌銆佷笉鎵?1/101閭诲煙銆傚畬鏁翠簲鎶樻渶濂?1.7132331848涓嶅彉銆?

- 宸插噯澶?鏉℃洸绾垮垎娈垫暟鍊肩紪鐮佸€欓€夛細浠呮牎鍑?8涓撳杈撳叆鏀逛负鍘熷€?闆跺垵濮嬪寲9x9涓夎鍩轰慨姝ｏ紙鏈€澶?.5锛夛紝鍏?06涓撳/璺敱杈撳叆鍘熸牱锛涘浐瀹氱浜旀姌80/18銆佷弗鏍?80.4銆傜洿鎺ユ楠屽彇鍊煎尯闂撮潪绾挎€ц〃绀猴紝涓嶅啀鍙犲姞灞€閮ㄧ粺璁★紱寰呭惎鍔ㄣ€?

- 鍒嗘鏁板€肩紪鐮佸€欓€塒ID70824宸插惎鍔ㄤ负鍞竴GPU璁粌杩涚▼锛孏PU瀹堝崼閫氳繃锛涚粨鏋滃緟绗簲鎶樻姤鍛婏紝涓嶈兘鎻愬墠澹扮О鏀剁泭銆?

- 鏍″噯涓撳涔濇洸绾垮垎娈电紪鐮佸畬鏁寸浜旀姌80.0292203160锛坆est52/stop70锛?10.31s锛孏PU790.27MiB锛夛紝鍘熷弻涓撳80.2508938鐨勫樊鍊?0.221673锛涙渶浣崇姸鎬佹仮澶嶅鐜帮紝浣嗕綆浜?80.4锛岃礋鏉冮噸/澶у垎甯冩湭鐣欍€備笉鎵妭鐐瑰弬鏁般€佷笉琛ユ姌銆傝鍒掔敤褰撳墠绱у噾鎺㈤拡璁粌鍣ㄩ噸璺戞湭鏀瑰姩鍘熺綉缁滐紝纭褰撳墠80.25瀵圭収鍦ㄥ悓涓€鍩虹璁炬柦涓婂彲澶嶇幇锛岄伩鍏嶆妸璁粌鍣ㄥ樊寮傝鍒や负鐗瑰緛浼ゅ銆?

- 鍘熷悗楠屽弻涓撳绱у噾璁粌鍣ㄥ鐜板鐓ф簮鐮佸凡鍑嗗锛氫笉鏀圭壒寰併€佸師Network鍙傛暟/绗簲鎶?seed/鎹熷け/Q/80-18锛涗粎鍓嶈疆float32 CDF瀛樺偍鐢辩鐩樻敼RAM锛岃礋璇曢獙涓嶇暀澶х紦瀛樸€傜敤浜庡垽鏂?0.25涓庤繎鏈熺害80.03宸窛鐨勫綊鍥狅紱寰呰繍琛屻€?

- 鍘烴etwork绱у噾璁粌鍣ㄥ鐓х浜旀姌瀹屾暣80.2270151355锛宐est52/stop70锛岃緝鍘嗗彶80.2508938浠?0.023879锛涙渶浣崇姸鎬佸鐜帮紝892.25s/GPU776.12MiB锛屾湭淇濆瓨璐熸潈閲?澶у垎甯冦€傜敱姝ゆ湁绗﹀彿灞€閮ㄥ姣?0.033691鍜屼節鏇茬嚎鍒嗘缂栫爜80.029220鐩稿鍚岀幆澧冨鐓у悇-0.193324/-0.197795锛岃礋缁撹鎴愮珛涓擯OR閫€姝ユ渶鏄捐憲锛涘潎涓嶈繃>80.4锛屾湭琛ュ洓鎶樸€傜浜旀姌鍘嗗彶鏈€濂?0.250894銆佸畬鏁碠OF81.713233鏈彉锛岀洰鏍?82鏈揪銆?

- 涓嬩竴鏉＄壒寰佹満鍒讹細鍏?06涓撳璁粌鏈熷涔濇洸绾垮悇10涓眬閮ㄥ伐绋嬬壒寰佹寜鏇茬嚎/绐楀彛p0.12閬斀锛屾帹鐞嗙敤鍏?06锛涗粎璁粌姝ｅ垯鍖栵紝绗簩涓撳/璺敱/鎹熷け/Q涓嶅彉銆傜‖鍒犻櫎90缁村厛鍓嶅け璐ワ紝鏁呮楠岄殢鏈洪儴鍒嗛伄钄姐€傛簮鐮佷笌鐙珛README宸插噯澶囷紝鍥哄畾绗簲鎶?80.4鎵嶈ˉ鎶樸€?

- 灞€閮ㄥ伐绋嬬壒寰佺粍p0.12璁粌鏈熼伄钄藉湪epoch60浜哄伐鎻愬墠缁堟锛氳瀵熸渶濂?0.0684004514@48锛岃繛缁?2杞湭鍒锋柊锛岃緝鍘熷悓鐜80.227015浣?.158615涓旂>80.4宸?.331600銆傛渶浣砪heckpoint鏈仮澶嶅鍑猴紝鏃犳寮忔姌鎶ュ憡/棰勬祴锛涗粎閮ㄥ垎瀹為獙锛岀粷涓嶅綋瀹屾垚鍒嗘暟銆備笉缁х画閬斀鐜?鍓灊缃戞牸銆佷笉琛ユ姌銆?

- 涓ゅ绾疍L鍥哄畾鍏辫瘑鍔ㄤ綔铻嶅悎绗簲鎶?0.1822973808锛岃緝鍘熷悗楠?0.2508938涓嬮檷0.068596锛涢粯璁ょ姸鎬佹湭鏀癸紝浠呬袱鑰呭潎闈為粯璁ゆ椂璁粌鐩爣鍧愭爣绛夋潈涓偣銆侾OR涓嬮檷0.247157鍘嬭繃PERM灏忓鐩婏紝鏈繃>80.4锛屼笉琛ユ姌銆佷笉鎵潈閲嶃€傝嫢缁х画铻嶅悎锛岄渶鏀逛负棰勬祴鍒嗗竷灞傝€岄潪鐩存帴鏁板€煎姩浣滃钩鍧囥€?

- 鐢ㄦ埛鏄庣‘寮曞鍥炲師鍚庨獙鍙屼笓瀹?0.2508938锛屽悓鏋舵瀯鏂板seed-base20261007/20261008涓ゆ鐙珛80杞缁冿紝涓夋ā鍨?:1:1瀹屾暣鍒嗗竷杞姇绁紝绗簲鎶樼粍鍚堜弗鏍?80.4鎵嶈ˉ鍏朵粬鎶樸€傚浐瀹氳鍒?鐙珛README宸插啓锛屾柊鎴愬憳浠呬繚鐣欐帹鐞嗘墍闇€灏廲heckpoint锛岄伩鍏嶅法澶у垎甯冪紦瀛橈紱璺ㄦ灦鏋勫弻妯″瀷鍒嗗竷铻嶅悎鏂规浠呴娉ㄥ唽鏈繍琛岋紝鐜版殏缂撱€備袱GPU浠诲姟涓婇檺銆?

- 涓夌瀛愮瓑鏉冨垎甯冩姇绁ㄤ袱璺浜旀姌杩愯涓細seed-base20261007 PID85936銆?0261008 PID83488锛屾渶澶?GPU锛屾樉瀛樿娴嬪凡鐣檦5GiB銆傝缁冨櫒鍘烴etwork/206缁?鎹熷けQ/80-18鍧囦笉鍙橈紝浠卻eed涓嶅悓锛涙柊鎴愬憳瀛樺皬checkpoint涓嶅啓澶у垎甯冦€傞娉ㄥ唽铻嶅悎鎺ㄧ悊鑴氭湰evaluate_paired_posterior_three_seed_vote.py瑕佹眰涓夋垚鍛樺姩浣滈噸绠楀鐜板拰瀹屾暣144665琛岃鐩栵紱缁勫悎涓ユ牸>80.4鎵嶈ˉ鎶橈紝鏈湁铻嶅悎鎴愮哗銆?

- 涓夌瀛愭柊鎴愬憳seed-base20261008瀹屾垚绗簲鎶?9.9244496296锛坆est43/stop61锛?257.70s锛夛紝鏈€浣砪heckpoint澶嶇幇銆佸繀瑕佹潈閲嶄繚鐣欙紱鍗曟ā鍨嬫湭杩?0.4浣嗙粍鍚堢瓫閫夊皻鏈彂鐢熴€俿eed-base20261007浠嶈繍琛岋紝褰撳墠鏈€澶?GPU璁粌浠诲姟銆?

- 涓夌瀛愬師鍚庨獙鍙屼笓瀹剁瓑鏉冨畬鏁村垎甯冩姇绁ㄧ浜旀姌80.3103506050锛屽師鍗曟ā鍨?0.2508938鐨勭湡瀹炲鐩?0.0594568302锛孭ERM+0.211347浣哖OR-0.044741锛涗笁checkpoint鍔ㄤ綔閲嶇畻100%涓€鑷淬€?44665琛岃鐩栵紝鏈繃涓ユ牸>80.4锛屼笉琛ユ姌銆傜敱浜庣敤鎴峰缓璁娆¤缁冧笖鏈哄埗鏈夋鏀剁泭锛屽浐瀹氬啀澧瀞eed-base20261009/20261010涓ゆ锛屽仛浜旀ā鍨?:1:1:1:1鍒嗗竷鎶曠エ锛涙棤琛岀骇/鎴愬憳绾ч獙璇佺湡鍊奸€変紭銆?

- 鐢ㄦ埛鍙栨秷缁х画鏂板绉嶅瓙锛堜簲绉嶅瓙璁粌鏈惎鍔級锛岃姹傚喕缁撶幇鏈変笁涓撳銆佸彧鍦ㄥ綋鍓嶇浜旀姌瀛︽姇绁ㄦ潈閲嶃€傚凡鍑嗗璁粌浜?only鍏冨涔狅細64璁粌浜曟瘡浜曞洓涓獥鍙ｇ害32768鐐癸紝涓変笓瀹秂val涓攔equires_grad=False锛孭yTorch鍙紭鍖?x3绁ㄦ潈logits 300姝ワ紝鎸変簳鍧囪　娣峰悎NLL+鍥哄畾0.05鍧囧寑鍏堥獙锛涚浜旀姌楠岃瘉浜曟爣绛句笉鐢ㄤ簬鏉冮噸瀛︿範銆傚悗鐢ㄥ喕缁撴潈閲嶆贩鍚?44665琛屽畬鏁撮娴嬪垎甯冨拰瀹樻柟鏁堢敤閫夊姩浣滐紝浠嶄弗鏍?80.4鎵嶈ˉ鎶樸€?

- 鍐荤粨涓変笓瀹讹紝浠呯敤64璁粌浜?2768鐐瑰3x3鐩爣绾у嚫鏉冮噸锛圥yTorch Adam銆佸潎琛℃贩鍚圢LL+鍥哄畾0.05鍧囧寑鍏堥獙锛夛紝楠岃瘉鏍囩鍙備笌0锛涚浜旀姌80.2955842244锛屾瘮鍘熷崟妯″瀷+0.044690銆佹瘮绛夋潈80.310351浣?.014766锛屼弗鏍?80.4鏈繃锛屼笉琛ユ姌銆備笁鎴愬憳閲嶇畻涓庝繚瀛樺姩浣?00%涓€鑷淬€侼LL-瀹樻柟鍒嗘暟閿欎綅锛屼笅涓€娆′繚鎸佷笓瀹跺拰璁粌浜曢噰鏍蜂笉鍙橈紝浠呮妸鍏冩潈閲嶇洰鏍囨敼涓哄钩婊戝畼鏂瑰姩浣滃緱鍒嗐€?

- 涓夊喕缁撲笓瀹剁浜屼釜鍏冩潈閲嶆崯澶卞鐓у凡鍑嗗锛氬悓64璁粌浜?32768鐐?3x3绁ㄦ潈/300姝dam/鍥哄畾0.05鍏堥獙锛屽彧鎶奛LL鎹负娓╁害0.02鐨勫畼鏂瑰姩浣滃緱鍒嗚蒋閫夋嫨鎹熷け锛涚浜旀姌鏍囩涓嶅弬涓庢嫙鍚堛€傚浐瀹氭潈閲嶅悗瀹屾暣鍒嗗竷+瀹樻柟鍔ㄤ綔涓€娆¤瘎浼帮紝涓ユ牸>80.4鎵嶈ˉ鎶樸€?

- 涓夊喕缁撲笓瀹剁浜屽厓鎹熷け锛氳缁冧簳瀹樻柟鍔ㄤ綔杞鍔卞鍏ㄥ眬3x3鏉冮噸绗簲鎶?0.3078684117锛屽師妯″瀷+0.056975銆佷笁绛夋潈80.310351鐨勫樊-0.002482锛涢獙璇佹嫙鍚?琛屻€佷笁checkpoint鍔ㄤ綔100%澶嶇幇銆侼LL鍏ㄥ眬瀛︽潈80.295584涓庢湰娆″潎鏈繃>80.4锛屼笉琛ユ姌锛屼笉鎵潤鎬佹潈閲嶆崯澶辫秴鍙傘€備笅涓€璺嚎淇濇寔涓撳鍐荤粨锛岃缁冧簳鍐呯暀鍑轰簳鎷熷悎鏋佸皬鍚庨獙缃俊搴︽潯浠跺寲璺敱銆?

- 涓夊喕缁撲笓瀹跺父鏁扮エ鏉冧袱鎹熷け鍧囨湭鑳滀笁绛夋潈锛涙柊鐙珛璺嚎宸插噯澶囧悗楠屾潯浠跺寲灏忓厓璺敱锛氭瘡鐩爣涓夋ā鍨嬅?鍚庨獙鎻忚堪=12缁粹啋16鈫?锛屾湯灞傞浂鍒濓紝64璁粌浜曟牱鏈腑48浜曟嫙鍚?16浜曞唴閮ㄩ獙璇侊紝鏈€澶?0杞?patience18銆丯LL+鍥哄畾0.01鍧囧寑鍏堥獙锛屼粎璁粌鍏冭矾鐢憋紝澶栭儴绗簲鎶樻爣绛惧畬鍏ㄩ殧绂汇€傚浐瀹氬閮ㄧ浜旀姌涓ユ牸>80.4鎵嶈ˉ鎶樸€?

- 涓夊喕缁撲笓瀹堕€愮偣鍔ㄦ€佸悗楠岃矾鐢辩浜旀姌80.2236404283锛岃緝鍘?0.2508938涓?0.027253銆佽緝涓夌瓑鏉?0.310351涓?0.086710锛涘唴閮?8/16璁粌浜昩est32/stop50锛?49鍙傛暟锛屽閮ㄩ獙璇佹爣绛炬嫙鍚?琛屻€佷笁checkpoint鍔ㄤ綔100%澶嶇幇銆傞潤鎬丯LL80.295584銆侀潤鎬佸姩浣滃緱鍒?0.307868銆佸姩鎬丯LL80.223640閮芥湭鑳滀笁绛夋潈锛涘仠姝㈠湪杩欎笁涓喕缁撲笓瀹朵笂缁х画璋冪エ鏉冿紝鏈繃>80.4涓嶈ˉ鎶樸€傚姩鎬佽礋妯″瀷router.pt宸插垹9,877瀛楄妭锛屾簮浠ｇ爜/杞ㄨ抗/棰勬祴/鏍￠獙淇濈暀銆備笁鏂扮瀛愭鏀剁泭鐨勫繀瑕佷笓瀹禼heckpoint淇濈暀銆?

- 鐢ㄦ埛绾犳鍏冭矾鐢辫缁冩硶锛氭寜鏅€氭姌璁粌鍦ㄥ叏閮?78891璁粌鐐逛笂bs8192銆丩R.001銆丄damW澶氬皬姝ユ參闄嶏紝鍐荤粨涓変笓瀹讹紱缂撳瓨绗簲鎶?44665鐐瑰畬鏁村垎甯冿紝閫愯疆瀹樻柟TOTAL閫夋嫨褰撳墠鎶樻渶浣砪heckpoint锛堥獙璇佹爣绛惧彧璇勫垎涓嶈繘姊害锛夛紝max80/patience18銆傛鍓?2768鏍锋湰鍏ㄦ壒閲?姝?杞笌鍐呴儴浜昇LL閫夋ā浠呬綔鍘嗗彶璐熷鐓с€傛柊鐙珛瀹為獙README/婧愮爜宸插噯澶囷紝寰呰繍琛岋紱涓ユ牸>80.4鎵嶈ˉ鎶樸€?

- 鍏?78891璁粌鐐筨s8192/LR.001鍐荤粨涓撳鍏冭矾鐢辩浜旀姌瀹屾暣80.2470132111锛坆est1/stop19锛夛紝涓夌瓑鏉?0.310351鐨勫樊-0.063337锛沜heckpoint棰勬祴澶嶇幇锛孏PU3980.21MiB銆?3.63s锛屼弗鏍?80.4鏈繃銆佽礋鏉冮噸涓嶈惤鐩樸€傞娆￠獙璇佸墠绾?1姝ヤ娇妯″瀷宸蹭綆浜庡垵濮嬬瓑鏉冿紝鏃犳硶妫€楠岃矾寰勪腑闂寸偣锛涗緷鐢ㄦ埛瑕佹眰鍐嶅浐瀹氫竴涓猙s16384/LR.0001骞跺惈epoch0鍙婃瘡8姝ラ獙璇佺殑鎱㈤€熻建杩硅瘯楠岋紝涓嶆悳澶氬弬鏁般€?

- 鐢ㄦ埛瑕佹眰鎱㈤檷瀵绘壘涓棿绁ㄦ潈銆傚凡鍑嗗鐙珛缁嗙矑搴﹀鐓э細鍘熶笁涓撳鍐荤粨銆佸叏閮?78891璁粌鐐广€?49鍙傚厓璺敱锛宐s16384/LR.0001/80max/18patience锛涘垵濮嬬瓑鏉冨強姣?姝?杞湯鍧囧仛绗簲鎶樺畼鏂筎OTAL楠岃瘉锛屼繚瀛樻渶楂?*鏇存柊姝?*checkpoint锛屾棤楠岃瘉姊害銆備粎姝ゅ浐瀹氳缃笉鎵秴鍙傦紝涓ユ牸>80.4琛ユ姌闂ㄦ淇濇寔銆?

- 鐢ㄦ埛寤鸿鐨勫喕缁撲笁涓撳鎱㈤€熺粏绮掑害绁ㄦ潈璁粌宸插畬鏁达細鍏ㄩ儴578891璁粌鐐癸紝bs16384/LR.0001锛屾洿鏂?鍙婃瘡8姝ラ獙璇?6娆★紝瀹樻柟TOTAL鏈€楂?0.3103780668@鏇存柊16/epoch0.4444锛宻top19锛岃緝鍒濆绛夋潈80.310355浠?0.00002285锛屾渶浣砪heckpoint鎭㈠澶嶇幇锛涘嘲鍊?980.21MiB/64.15s銆傛湭杩囦弗鏍?80.4锛屼笉琛ユ姌銆佽礋router鏉冮噸涓嶄繚瀛樸€傞潤鎬佷袱鎹熷け銆佸姩鎬丯LL銆佸叏鏁版嵁鎱㈤檷涓庣粏绮掑害涓棿鐐瑰潎鏃犳湁鏁堝鏉冩敹鐩婏紱鍋滄鍐荤粨涓変笓瀹剁エ鏉冭秴鍙傝矾绾裤€?

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

- Readback found externally executed cleanup log cleanup_executed_20260927_154042.jsonl: 45/45 targets absent; D free ~30.2GiB. Earlier agent deletion attempts were policy-blocked; this is subsequent filesystem evidence, not a claim the rejected calls executed.


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


Frozen latent patch COMPLETE: fifth TOTAL80.3868919215, best epoch0, natural stop18; best trained80.3759376707@1. Best prediction restoration verified. No gain and no checkpoint retained. Single GPU total66.724s including preprocessing/extraction/training/validation; extraction5.846s, peak3306.041MiB. All723556 valid rows covered once; normalization uses578891 training rows only. No remaining-fold promotion. FullOOF>82 unmet.

- Started rank8 cross-feature calibration, session77606, sole GPU. New720-parameter context-controlled multiplicative residual in calibrated member; zero output preserves initial baseline. Fixed fifth/max80/pat18/bs64, train-only preprocessing. No result yet; strict>80.4 before promotion, fullOOF>82 remains unmet. See outputs/pure_dl_paired_lowrank_calibration80_sequence/README.md.

- Started rank8 cross-feature calibration, session77606, sole GPU. New720-parameter context-controlled multiplicative residual in calibrated member; zero output preserves initial baseline. Fixed fifth/max80/pat18/bs64, train-only preprocessing. No result yet; strict>80.4 before promotion, fullOOF>82 remains unmet. See outputs/pure_dl_paired_lowrank_calibration80_sequence/README.md.

- Live snapshot: rank8 calibration session77606 confirmed training, epoch4 best fifth76.93051414. Early result only, no promotion; no OOM, latest reported PyTorch peak776.264MiB.

- Live snapshot: rank8 calibration session77606 confirmed training, epoch4 best fifth76.93051414. Early result only, no promotion; no OOM, latest reported PyTorch peak776.264MiB.

- Running rank8 calibration: last epoch20, best78.91579401@20; last same-epoch delta-0.17587012. Continue because best is still improving; no final ablation gain or promotion claimed. Session77606 polled live.

- Running rank8 calibration: last epoch20, best78.91579401@20; last same-epoch delta-0.17587012. Continue because best is still improving; no final ablation gain or promotion claimed. Session77606 polled live.

- Prepared next mechanism, not launched: weak train-only cross-well hidden feature-statistics mixing (MixStyle-inspired). See outputs/pure_dl_well_style_plan/README.md for evidence, fixed starting design, padding/RNG/domain safeguards and physical-information risk. Current rank8 calibration session77606 must finish naturally under80/pat18; no manual early stop per latest user request.

- Prepared next mechanism, not launched: weak train-only cross-well hidden feature-statistics mixing (MixStyle-inspired). See outputs/pure_dl_well_style_plan/README.md for evidence, fixed starting design, padding/RNG/domain safeguards and physical-information risk. Current rank8 calibration session77606 must finish naturally under80/pat18; no manual early stop per latest user request.

- Started weak well-style training session69486 concurrently with lowrank77606 (max2). Fixed fifth,80/pat18; no extra parameters. Cross-well donor IDs from training anchor well starts, masked statistics, dedicated RNG, disabled in eval/probe. Initial score pending. Preserve lowrank natural completion. FullOOF>82 unmet.

- Started weak well-style training session69486 concurrently with lowrank77606 (max2). Fixed fifth,80/pat18; no extra parameters. Cross-well donor IDs from training anchor well starts, masked statistics, dedicated RNG, disabled in eval/probe. Initial score pending. Preserve lowrank natural completion. FullOOF>82 unmet.

- Live paired_lowrank_calibration80: epoch49, best79.80864783@44; session77606 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Live paired_weak_well_style80: epoch5, best77.53947410@5; session69486 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Live paired_lowrank_calibration80: epoch49, best79.80864783@44; session77606 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Live paired_weak_well_style80: epoch5, best77.53947410@5; session69486 running, max2 concurrent, no OOM observed. Interim only; fullOOF>82 unmet.

- Rank8 calibration COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.72788441696548, "PERM": 89.24684557961116, "SW": 73.38376598362012, "TOTAL": 79.93907937222059}, "seconds": 1198.6207101999898, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 782.7607421875, "status": "complete", "atlas_current_trial": false, "session_id": 77606, "exit_code": 0, "screening_fold": 5, "delta_compact_control": -0.28793576330848225, "delta_original": -0.31181440259864246, "decision": "No gain. Natural patience18 stop at70; no rank/capacity sweep, no remaining folds."}. Total1198.62s includes preprocessing/training/probe/validation; partially concurrent with weak-style, no matched serial speedup claim. No checkpoint retained.

- Rank8 calibration COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.72788441696548, "PERM": 89.24684557961116, "SW": 73.38376598362012, "TOTAL": 79.93907937222059}, "seconds": 1198.6207101999898, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 782.7607421875, "status": "complete", "atlas_current_trial": false, "session_id": 77606, "exit_code": 0, "screening_fold": 5, "delta_compact_control": -0.28793576330848225, "delta_original": -0.31181440259864246, "decision": "No gain. Natural patience18 stop at70; no rank/capacity sweep, no remaining folds."}. Total1198.62s includes preprocessing/training/probe/validation; partially concurrent with weak-style, no matched serial speedup claim. No checkpoint retained.

## Historical family clarification
A previous conditioned rank8 numeric-mixing trial already completed at80.1954269324 (outputs/pure_dl_paired_conditioned_mix80_sequence). That version applies mixing before the original affine calibration and has biases; this later720-parameter version adds a bounded correction after affine calibration. They are related variants, not separate unexplored mechanism families. Both below compact80.2270151; close this family, no further rank/placement sweeps. Historical PCGrad80.1582856 also failed compact control; do not relaunch gradient projection as a new idea.

## Historical family clarification
A previous conditioned rank8 numeric-mixing trial already completed at80.1954269324 (outputs/pure_dl_paired_conditioned_mix80_sequence). That version applies mixing before the original affine calibration and has biases; this later720-parameter version adds a bounded correction after affine calibration. They are related variants, not separate unexplored mechanism families. Both below compact80.2270151; close this family, no further rank/placement sweeps. Historical PCGrad80.1582856 also failed compact control; do not relaunch gradient projection as a new idea.

- Started80.39 attribution session51546: fixed5/checkpoint hashes, single/equal/learned/leave-one-out and single-vs-four-window/coarse-vs-fine decoding; diagnostic only, no promotion. Runs alongside weak-style69486, max2GPU tasks. Results pending; models/loss attribution requires matched training controls, not historical score subtraction.

- Started80.39 attribution session51546: fixed5/checkpoint hashes, single/equal/learned/leave-one-out and single-vs-four-window/coarse-vs-fine decoding; diagnostic only, no promotion. Runs alongside weak-style69486, max2GPU tasks. Results pending; models/loss attribution requires matched training controls, not historical score subtraction.

- Weak style COMPLETE: {"fold": 5, "best_epoch": 48, "stopped_epoch": 66, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 75.7366886208511, "PERM": 89.13228344192615, "SW": 73.48030727511235, "TOTAL": 79.6354133372188}, "seconds": 1278.4947431000182, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 777.9208984375, "status": "complete", "atlas_current_trial": false, "session_id": 69486, "exit_code": 0, "delta_compact": -0.591601798310279, "decision": "No gain; natural stop66, best48 restored. No augmentation strength/layer sweep."}. Partial concurrency with lowrank and source attribution; no speedup claim. User direction now prioritizes80.39 attribution, not new mechanism search.

- Weak style COMPLETE: {"fold": 5, "best_epoch": 48, "stopped_epoch": 66, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 75.7366886208511, "PERM": 89.13228344192615, "SW": 73.48030727511235, "TOTAL": 79.6354133372188}, "seconds": 1278.4947431000182, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 777.9208984375, "status": "complete", "atlas_current_trial": false, "session_id": 69486, "exit_code": 0, "delta_compact": -0.591601798310279, "decision": "No gain; natural stop66, best48 restored. No augmentation strength/layer sweep."}. Partial concurrency with lowrank and source attribution; no speedup claim. User direction now prioritizes80.39 attribution, not new mechanism search.

-80.39 attribution inference completed26 arms (214.40s/1461.84MiB): strongest single80.331454/equal5 80.386864/learned5 80.386898; see outputs/pure_dl_8039_source_ablation/ANALYSIS.md. Reference score difference6.03e-6, not elementwise identical. Diagnostic dropseed07 80.403220 is NOT auto-promoted. Training ablations Q=1(session82504) and POR Gaussian(session69566) now running, max2; no results yet.

-80.39 attribution inference completed26 arms (214.40s/1461.84MiB): strongest single80.331454/equal5 80.386864/learned5 80.386898; see outputs/pure_dl_8039_source_ablation/ANALYSIS.md. Reference score difference6.03e-6, not elementwise identical. Diagnostic dropseed07 80.403220 is NOT auto-promoted. Training ablations Q=1(session82504) and POR Gaussian(session69566) now running, max2; no results yet.

- User adopts four-source80.403220094 baseline (indices0,2,3,4). Six equal two-source subset ablations prepared and queued in observable session28718; waits for GPU capacity while Q=1 session82504 and GaussianPOR69566 run. No backbone retraining for pair study. Preserve original sources/deliveries; diagnostic selection history explicit. See outputs/pure_dl_8040_pair_ablation/README.md.

- User adopts four-source80.403220094 baseline (indices0,2,3,4). Six equal two-source subset ablations prepared and queued in observable session28718; waits for GPU capacity while Q=1 session82504 and GaussianPOR69566 run. No backbone retraining for pair study. Preserve original sources/deliveries; diagnostic selection history explicit. See outputs/pure_dl_8040_pair_ablation/README.md.

- Attribution monitoring: [{"status": "running", "atlas_current_trial": true, "session_id": 82504, "screening_fold": 5, "last_epoch": 26, "best_score": 79.15393682729156, "best_epoch": 23, "ablation": "no_quality", "note": "Partial result; no final contribution claim."}, {"status": "running", "atlas_current_trial": true, "session_id": 69566, "screening_fold": 5, "last_epoch": 22, "best_score": 79.07819692255848, "best_epoch": 20, "ablation": "gaussian_por", "note": "Partial result; no final contribution claim."}]. Pair inference28718 verified CPU-waiting while these2GPU tasks run. Historical paired module config/shared-hash audit completed without shared hash changes; source analysis updated, one-seed evidence only.

- Attribution monitoring: [{"status": "running", "atlas_current_trial": true, "session_id": 82504, "screening_fold": 5, "last_epoch": 26, "best_score": 79.15393682729156, "best_epoch": 23, "ablation": "no_quality", "note": "Partial result; no final contribution claim."}, {"status": "running", "atlas_current_trial": true, "session_id": 69566, "screening_fold": 5, "last_epoch": 22, "best_score": 79.07819692255848, "best_epoch": 20, "ablation": "gaussian_por", "note": "Partial result; no final contribution claim."}]. Pair inference28718 verified CPU-waiting while these2GPU tasks run. Historical paired module config/shared-hash audit completed without shared hash changes; source analysis updated, one-seed evidence only.

- Expanded80.4032 subset attribution per user:6pairs+4leave-one-out triples+baseline=11arms. Replaced only CPU-waiting inference28718 with94542, confirmed waiting; training82504/69566 uninterrupted. Same source inference will serve all11arms.

- Expanded80.4032 subset attribution per user:6pairs+4leave-one-out triples+baseline=11arms. Replaced only CPU-waiting inference28718 with94542, confirmed waiting; training82504/69566 uninterrupted. Same source inference will serve all11arms.

- Attribution runs observed live: [{"ablation": "no_quality", "epoch": 42, "best": 79.7362617128407}, {"ablation": "gaussian_por", "epoch": 40, "best": 79.57860036215014}]. Subset11arms session94542 remains queued; no thirdGPU task. Added config-based source/subset parameter counts to outputs/pure_dl_8040_pair_ablation/model_costs.json (baseline4.646M/7backbones, pairs1.987-2.658M/3-4backbones), not a measured speedup.

- Attribution runs observed live: [{"ablation": "no_quality", "epoch": 42, "best": 79.7362617128407}, {"ablation": "gaussian_por", "epoch": 40, "best": 79.57860036215014}]. Subset11arms session94542 remains queued; no thirdGPU task. Added config-based source/subset parameter counts to outputs/pure_dl_8040_pair_ablation/model_costs.json (baseline4.646M/7backbones, pairs1.987-2.658M/3-4backbones), not a measured speedup.

- Training attribution live snapshot [{"mode": "no_quality", "epoch": 56, "best": 80.09251227357417}, {"mode": "gaussian_por", "epoch": 55, "best": 80.04144251747569}]. Both still running; do not finalize Q/distribution contributions yet. Added historical attention audit (+0.2158936568 at matched single-backbone75, not current ensemble causal delta) to source ANALYSIS.md. Removed2completed-negative point-diagnostic caches44,965,265bytes with manifest outputs/cleanup_negative_calibration_style_attribution.json; code/config/validation curves/scores and all active files retained. Subset11 queue94542 unchanged.

- Training attribution live snapshot [{"mode": "no_quality", "epoch": 56, "best": 80.09251227357417}, {"mode": "gaussian_por", "epoch": 55, "best": 80.04144251747569}]. Both still running; do not finalize Q/distribution contributions yet. Added historical attention audit (+0.2158936568 at matched single-backbone75, not current ensemble causal delta) to source ANALYSIS.md. Removed2completed-negative point-diagnostic caches44,965,265bytes with manifest outputs/cleanup_negative_calibration_style_attribution.json; code/config/validation curves/scores and all active files retained. Subset11 queue94542 unchanged.

- Training ablations COMPLETE: [{"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.94295729632633, "PERM": 89.27312796013906, "SW": 73.61151513893603, "TOTAL": 80.09251227357417}, "seconds": 1529.0026964999852, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 82504, "exit_code": 0, "delta_control": -0.1345028619548998, "ablation": "no_quality"}, {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.71596748229165, "PERM": 89.4341072589125, "SW": 73.49918494905378, "TOTAL": 80.04144251747569}, "seconds": 1362.7202495000092, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 69566, "exit_code": 0, "delta_control": -0.18557261805338499, "ablation": "gaussian_por"}]. Q and t3 retained on this single-seed evidence, no additive contribution claim. Subset11 inference94542 now admitted, soleGPU. FullOOF>82 unmet.

- Training ablations COMPLETE: [{"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.94295729632633, "PERM": 89.27312796013906, "SW": 73.61151513893603, "TOTAL": 80.09251227357417}, "seconds": 1529.0026964999852, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 82504, "exit_code": 0, "delta_control": -0.1345028619548998, "ablation": "no_quality"}, {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.71596748229165, "PERM": 89.4341072589125, "SW": 73.49918494905378, "TOTAL": 80.04144251747569}, "seconds": 1362.7202495000092, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.12353515625, "status": "complete", "atlas_current_trial": false, "session_id": 69566, "exit_code": 0, "delta_control": -0.18557261805338499, "ablation": "gaussian_por"}]. Q and t3 retained on this single-seed evidence, no additive contribution claim. Subset11 inference94542 now admitted, soleGPU. FullOOF>82 unmet.

- All requested attribution completed:11 subset arms, baseline80.403220094 exactly reproduced. Best pairAB80.3801885; best tripleABD80.4091519 (+0.0059318, parameters3.320M versus4.646M, exploratory selection not robust improvement claim). Q-off80.0925123 and GaussianPOR80.0414425 versus matched dual80.2270151, bothbest52/stop70 restore verified. Reports outputs/pure_dl_8040_pair_ablation/SUMMARY.md and outputs/pure_dl_8039_source_ablation/ANALYSIS.md. All3sessions terminal, no new training launched; fullOOF>82 still unmet.

- All requested attribution completed:11 subset arms, baseline80.403220094 exactly reproduced. Best pairAB80.3801885; best tripleABD80.4091519 (+0.0059318, parameters3.320M versus4.646M, exploratory selection not robust improvement claim). Q-off80.0925123 and GaussianPOR80.0414425 versus matched dual80.2270151, bothbest52/stop70 restore verified. Reports outputs/pure_dl_8040_pair_ablation/SUMMARY.md and outputs/pure_dl_8039_source_ablation/ANALYSIS.md. All3sessions terminal, no new training launched; fullOOF>82 still unmet.

- Started targeted attribution follow-ups: main-only auxiliary-loss removal62826 (same original dual80/Q/t3, max80/pat18) and frozen internal expert intervention60959 (Aseed06/Bseed08 each full206/calibrated68/equal/routed). Max2GPU; not a new unstructured mechanism sweep. Four-source80.4032 baseline unchanged, results pending, fullOOF>82 unmet.

- Started targeted attribution follow-ups: main-only auxiliary-loss removal62826 (same original dual80/Q/t3, max80/pat18) and frozen internal expert intervention60959 (Aseed06/Bseed08 each full206/calibrated68/equal/routed). Max2GPU; not a new unstructured mechanism sweep. Four-source80.4032 baseline unchanged, results pending, fullOOF>82 unmet.

- Internal experts COMPLETE8arms, exact A reference reproduced. [{"source": "A", "equal_gain_over_best_branch": 0.3514727762118639, "router_gain_over_equal": 0.09321302034933865, "routed_gain_over_best_branch": 0.44468579656120255}, {"source": "B", "equal_gain_over_best_branch": 0.31227311776699196, "router_gain_over_equal": 0.009940617467393054, "routed_gain_over_best_branch": 0.322213735234385}]. Full206 alone stronger total; calibrated68 complements SW. This is frozen intervention, not retrained causal removal. Source report retained. Main-only62826 continues, no new GPU task.

- Internal experts COMPLETE8arms, exact A reference reproduced. [{"source": "A", "equal_gain_over_best_branch": 0.3514727762118639, "router_gain_over_equal": 0.09321302034933865, "routed_gain_over_best_branch": 0.44468579656120255}, {"source": "B", "equal_gain_over_best_branch": 0.31227311776699196, "router_gain_over_equal": 0.009940617467393054, "routed_gain_over_best_branch": 0.322213735234385}]. Full206 alone stronger total; calibrated68 complements SW. This is frozen intervention, not retrained causal removal. Source report retained. Main-only62826 continues, no new GPU task.

- Main-only attribution62826 confirmed live, epoch13 best78.56925593@13, now soleGPU. Internal-expert inference60959 completed153.71s/1444.27MiB, exact reference; report ANALYSIS.md updated with branch complement and routing gains. FullOOF>82 unmet.

- Main-only attribution62826 confirmed live, epoch13 best78.56925593@13, now soleGPU. Internal-expert inference60959 completed153.71s/1444.27MiB, exact reference; report ANALYSIS.md updated with branch complement and routing gains. FullOOF>82 unmet.

- Added and visually inspected attribution_summary.png/.svg (3separate comparison panels, completed results only), linked from SUMMARY.md and ANALYSIS.md. Main-only62826 continues as soleGPU, epoch25 best79.28455813@25; no final auxiliary-loss contribution claim.

- Added and visually inspected attribution_summary.png/.svg (3separate comparison panels, completed results only), linked from SUMMARY.md and ANALYSIS.md. Main-only62826 continues as soleGPU, epoch25 best79.28455813@25; no final auxiliary-loss contribution claim.

- Started matched no-calibration attribution66172 alongside main-only62826, max2GPU. Original posterior network, calibration zero-output initialization frozen; 68view/Q/t3/auxiliary/router retained. Fills interaction gap beyond older no-router75/80 comparisons; no rank/width sweep. Result pending, original80.4032 four-source baseline unchanged and fullOOF>82 unmet.

- Started matched no-calibration attribution66172 alongside main-only62826, max2GPU. Original posterior network, calibration zero-output initialization frozen; 68view/Q/t3/auxiliary/router retained. Fills interaction gap beyond older no-router75/80 comparisons; no rank/width sweep. Result pending, original80.4032 four-source baseline unchanged and fullOOF>82 unmet.

- Live final attribution pair: [{"mode": "main_only", "epoch": 40, "best": 79.69091378066616}, {"mode": "no_calibration", "epoch": 2, "best": 76.68573766110542}]. Both actual train/validation/probe cycles confirmed; max2GPU and no OOM observed. Summary renderer prepared to include completed optional auxiliary/calibration ablations only, no provisional results in final comparison charts.

- Live final attribution pair: [{"mode": "main_only", "epoch": 40, "best": 79.69091378066616}, {"mode": "no_calibration", "epoch": 2, "best": 76.68573766110542}]. Both actual train/validation/probe cycles confirmed; max2GPU and no OOM observed. Summary renderer prepared to include completed optional auxiliary/calibration ablations only, no provisional results in final comparison charts.

- Verified live attribution wait: [{"mode": "main_only", "epoch": 54, "best_epoch": 52, "best": 79.87490529017835}, {"mode": "no_calibration", "epoch": 14, "best_epoch": 12, "best": 78.49377082386238}]. Both valid tool handles active, no restart or premature final contribution claim. Main-only refreshed best52; calibration still early. No new experiments added.

- Verified live attribution wait: [{"mode": "main_only", "epoch": 54, "best_epoch": 52, "best": 79.87490529017835}, {"mode": "no_calibration", "epoch": 14, "best_epoch": 12, "best": 78.49377082386238}]. Both valid tool handles active, no restart or premature final contribution claim. Main-only refreshed best52; calibration still early. No new experiments added.

- Main-only attribution COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.68933694757615, "PERM": 89.15215965438021, "SW": 73.32813807677836, "TOTAL": 79.87490529017835}, "seconds": 1235.02869440004, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.43896484375, "status": "complete", "atlas_current_trial": false, "session_id": 62826, "exit_code": 0, "delta_control": -0.352109845350725, "decision": "All3channels worse. Retain auxiliary supervision in current setup; no auxiliary-weight sweep."}. Partial concurrency; time includes preprocessing/training/probes/validation. Input-calibration66172 remains soleGPU task; fullOOF>82 unmet.

- Main-only attribution COMPLETE: {"fold": 5, "best_epoch": 52, "stopped_epoch": 70, "best_checkpoint_predictions_reproduced": true, "checkpoint_persisted": false, "scores": {"POR": 76.68933694757615, "PERM": 89.15215965438021, "SW": 73.32813807677836, "TOTAL": 79.87490529017835}, "seconds": 1235.02869440004, "all_validation_rows_covered_once": true, "peak_gpu_allocated_mib": 776.43896484375, "status": "complete", "atlas_current_trial": false, "session_id": 62826, "exit_code": 0, "delta_control": -0.352109845350725, "decision": "All3channels worse. Retain auxiliary supervision in current setup; no auxiliary-weight sweep."}. Partial concurrency; time includes preprocessing/training/probes/validation. Input-calibration66172 remains soleGPU task; fullOOF>82 unmet.

- Final main-only result integrated into SUMMARY/ANALYSIS/summary.json and regenerated chart; chart visually checked with extended delta axis (includes-0.3521). No-calibration66172 still live, epoch34 best79.63841458@28, soleGPU.

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


## 2026-09-27 23:29:39 Independent-head monitoring
- Completed: A6 own-head final fifth80.501552013419, delta+0.009243477591, exact final reassembly and frozen sources verified by completed report. CPU gain decomposition:9/16 wells up,7 down; TOTAL contributions POR0.000713191741/PERM0/SW0.008530285850. Small development gain, not independent generalization evidence.
- Current: frozen five-branch feature-head epoch21, current TOTAL79.220558288, channel-best candidate79.609123555, bestepochs[18, 8, 20], badcounts[3, 13, 1]. Process alive=True; final report present=False.
- Pending: observe per-channel natural stopping and final assembled replay under existing80/18/min10. No new/stopped jobs or changed trainers; no remaining-fold continuation. Full OOF>82 unmet. After fixed-method validation, consider new delivery only.
- Snapshot: results/deep_learning/head_tracking_20260927_232939.json.


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

## 2026-09-28 鎶€鏈矾绾挎牳瀵癸紙鏈惎鍔ㄨ缁冿級
- 褰撳墠涓荤嚎锛氬喕缁揂6/B43/D52鐨勪簲鍒嗘敮鐗瑰緛楠ㄥ共锛屽垎鍒紭鍖朠OR/PERM/SW鏂板ご锛涙柊澧炴瘡鐐?0缁村垎鏀潎鍊?log鏍囧噯宸殑闆跺垵濮嬪寲璇诲嚭鏀矾銆傜粨鏋勮矾绾夸笉浣跨敤鏃ц緭鍑烘姇绁紱宸蹭氦浠?0.501552浠嶄负A6鐙珛澶?B/D姒傜巼铻嶅悎銆?
- 宸叉牳瀵规殏鍋滃揩鐓у拰鍘嗗彶锛氳鑼冨寲鐗瑰緛澶存渶鍚庡畬鏁磂poch50锛岄€氶亾鏈€浣?9/8/33锛屽€欓€?9.738172308900锛涚粺璁℃敮璺渶鍚巈poch4锛屾渶浣?/4/3锛屽€欓€?9.007628319764銆備袱椤瑰潎琚敤鎴峰仠姝€佹棤鏈€缁堜笁澶村叡鍚屾仮澶嶆姤鍛婏紱褰撳墠鏈彂鐜伴」鐩缁冭繘绋嬨€?
- 鍚屼负鍓?杞紝缁熻鏀矾鐩稿瀵圭収鐨勯€氶亾鏈€浣冲垎宸甈OR-0.064798/PERM+0.179608/SW-0.039143锛屽姞鏉冨€欓€夊樊+0.029723銆備粎鏄棭鏈熷紑鍙戣瘉鎹紝涓嶄笌瀵圭収50杞洿鎺ヤ綔绛夐绠楁敹鐩婂綊鍥犮€?
- 寰呭姙杈圭晫锛氱瓑寰呮槑纭仮澶嶆巿鏉冿紱淇濆瓨浠ｇ爜浠呬繚鐣檋eads_best鏉冮噸/閫夋嫨鍏冧俊鎭紝鏈繚瀛樹紭鍖栧櫒銆佽皟搴﹀櫒銆侀殢鏈虹姸鎬侊紝鍘熻剼鏈篃娌℃湁resume鍏ュ彛銆傚悗缁儹鍚姩蹇呴』涓庣簿纭画璺戝尯鍒嗗苟浣跨敤鏂拌褰曪紝涓嶈兘瑕嗙洊鏃у疄楠屻€傚浘璋?3:47蹇収鍜岄儴鍒哛EADME婊炲悗锛屼互runtime_status銆乸aused_snapshot鍙婂疄闄呰繘绋嬩负鍑嗐€?
- 鐩爣浠嶄负绗簲鎶樹弗鏍?80.5涓旀槑纭鐩婏紝瀹屾暣鍘熶簲鎶極OF>82锛涘綋鍓嶅畬鏁碠OF鏈€浣?1.713233锛岃ˉ鎶樹粛鏆傜紦銆傛湰鎵逛粎鏍稿骞舵洿鏂拌繘搴︼紝鏈惎鍋滆缁冦€佹湭鏀规簮鐮?鏉冮噸/浜や粯銆?


## 2026-09-28 鍙屼笓瀹?0.22鐙珛澶寸粨鏋勪紭鍖栧惎鍔?
鐢ㄦ埛鏄庣‘鎭㈠鏂版柟鍚戯紝鏃BD鏆傚仠涓嶅姩銆傞噰鐢ㄤ繚鐣欑殑local-router80.221861@52锛沜ompact80.227015鏈繚鐣欐潈閲嶃€傛柊澧炲悓鍙傛暟pointwise80涓巐ocal5-width77涓ょ娈嬪樊澶达紝姣忕粍93420鍙傛暟锛屽鍒跺師杈撳嚭璧风偣銆佸浐瀹?楠ㄥ共/璺敱/鍘?56鏍?00鍔ㄤ綔鍗曡鐩栵紱own-channel鎹熷け銆佸浐瀹氭簮epoch50Q銆乴r3e-5銆?0/18/min10锛屽悇澶寸嫭绔嬮€夋ā銆傛柊璁粌鍣ㄤ繚瀛樺畬鏁磂poch-boundary鐘舵€併€傚凡鍚姩PID 23804,37196锛涘疄闄呮搴?鍒濆鍥炴斁/璁粌鍒嗘暟寰呰鍙栵紝涓嶅０绉版敹鐩娿€傛病鏈夎ˉ鎶樻垨鏀逛氦浠橈紱鍚庣画璇诲彇鍏卞悓鎭㈠涓庨€愰€氶亾瀵圭収锛屽畬鏁碠OF>82浠嶆湭瀹屾垚銆?


## 2026-09-28T09:16:47+08:00 鍙屼笓瀹剁嫭绔嬪ご瀵圭収鏍稿
pure_dl_dual8222_heads_pointwise80_fold5: complete, epoch19, TOTAL=80.25333950575663, scope=瀹屾垚锛氬浐瀹氫笁涓渶浣冲ご鍏卞悓鎭㈠; process_alive=False.
pure_dl_dual8222_heads_local580_fold5: complete, epoch19, TOTAL=80.25447676690142, scope=瀹屾垚锛氬浐瀹氫笁涓渶浣冲ご鍏卞悓鎭㈠; process_alive=False.
浠呮湰鎵逛袱缁撴瀯锛涙棫ABD淇濇寔鏆傚仠銆佹湭琛ユ姌銆佸師浜や粯涓嶆敼銆傛簮80.221861棰勬祴宸插疄闄呭洖鏀撅紱鍚勭洰鏍囨敹鐩婁笌鏈€缁堟仮澶嶅垎寮€銆傚畬鏁碠OF>82鏈畬鎴愩€傛柊璁板綍瑙乺esults/deep_learning/dual8222_head_structure_20260928/snapshot.json銆?


## 鍙屼笓瀹剁嫭绔嬪ご棣栨壒瀹屾垚
閫愮偣80.253339505757锛屽眬閮?0.254476766901锛涗粎POR鏀瑰杽銆備袱缁勬渶缁堝叡鍚屾仮澶嶉€氳繃锛屽浘璋卞凡鏇存柊銆傚皻鏈瘉瀹炲眬閮ㄧ粨鏋勪紭瓒婏紝鐙珛绾挎€уご瀵圭収寰呭仛銆傛湭鍚姩绗笁缁勬垨鍏朵粬鎶橈紱鏃т氦浠樹笉鍙樸€傛€荤粨锛歳esults/deep_learning/dual8222_head_structure_20260928/README.md銆?


## 2026-09-28 鍙屼笓瀹剁浜岃疆缁撴瀯浼樺寲
鏂板linear-only瀵圭収鍜宑ross80鑱斿悎鍐呴儴涓撳鐗瑰緛澶淬€傛簮80.221861锛屽浐瀹氬師鍗曡鐩?256鏍?800鍔ㄤ綔銆佺嫭绔嬬洰鏍?80-18/3e-5/Q50銆傚墠杞簮鐮佷笌缁撴灉涓嶆敼銆傚凡鍚姩浠诲姟鏁?2锛汸ID=82744,28936銆傚垵濮?姊害/鐪熷疄鏀剁泭寰呮牳瀵癸紝涓嶉绉版垚鍔熴€傜粨鏋滃湪results/deep_learning/dual8222_round2_20260928锛岀洰鏍?80.5涓庡畬鏁碠OF>82浠嶆湭瀹屾垚锛屾湭琛ユ姌銆?


## 鍙屼笓瀹跺洜绱犲垎绂诲惎鍔?
宸插畬鎴恖inear80.253777鍜宑ross80.255113锛涘墠杞柊澧為潪绾挎€т粎寰急宸紓銆侾OR闈為粯璁ゆ敹鐩婇儴鍒嗚榛樿娈甸€€姝ユ姷娑堬紝鏁呭湪鍘熺嚎鎬уご涓婃瘮杈冨彧鏇存柊闂ㄦ帶涓庡彧鏇存柊瀵嗗害銆侾ID=70272,95240銆?GPU涓婇檺銆佸師璇勫垎/鏁版嵁/棰勭畻涓嶅彉锛岀瓑寰呯湡瀹炶缁冨強鏈€缁堝浐瀹氬洜瀛愰獙璇侊紱鏃犳棫ABD鎴栬ˉ鎶樺惎鍔ㄣ€?


## 2026-09-28T09:34:18+08:00 鍙屼笓瀹剁户缁紭鍖栫粨鏋滄牳瀵?
|鏂规硶|鐘舵€亅TOTAL|鏈€浣宠疆娆
|---|---|---:|---|
|linear|鏈€缁堝浐瀹氫笁澶村叡鍚屾仮澶嶅凡楠岃瘉|80.253777203|[1, 0, 0]|
|cross80|鏈€缁堝浐瀹氫笁澶村叡鍚屾仮澶嶅凡楠岃瘉|80.255112897|[1, 0, 0]|
|gate_only|杩愯蹇収鍊欓€夛紝闈炴渶缁堟仮澶峾80.221861083|[0, 0, 0]|
|density_only|杩愯蹇収鍊欓€夛紝闈炴渶缁堟仮澶峾80.256678729|[1, 0, 0]|

宸插畬鎴愶細鏂板绾挎€у綊鍥?鑱斿悎鍐呴儴涓撳璇诲嚭/鍥犵礌鍒嗙瀹炵幇涓庡疄闄呰繍琛岃褰曘€?
寰呭畬鎴愶細鎵€鏈夊€欓€夌殑鏈€缁堟仮澶嶃€侀€愰€氶亾閰嶆柟鍥哄畾鍚庣殑鍏卞悓鎺ㄧ悊锛涗笉瀛樺湪瀹屾暣OOF澧炵泭璇佹槑銆?
鐩爣锛氬綋鍓嶅弻涓撳鍙鐜颁紭鍖栵紝绗簲鎶?80.5涓旀槑纭敹鐩婏紱瀹屾暣OOF>82浠嶆湭杈炬垚銆?
杈炬垚鍚庯細鍥哄畾鏂规鍚庤瘎浼版硾鍖栵紝褰撳墠涓嶈嚜鍔ㄨˉ鎶樸€佷笉鏀规棫浜や粯銆?
鍥捐氨鍦ㄦ棦鏈塩uration鐢熸垚鍚庤繍琛屾湰鑴氭湰鍐峛uild_atlas锛涗笉淇敼姝ゅ墠琚繚鎶ょ殑curate.py銆?


## 2026-09-28T09:40:23+08:00 鍙屼笓瀹剁户缁紭鍖栫粨鏋滄牳瀵?
|鏂规硶|鐘舵€亅TOTAL|鏈€浣宠疆娆
|---|---|---:|---|
|linear|鏈€缁堝浐瀹氫笁澶村叡鍚屾仮澶嶅凡楠岃瘉|80.253777203|[1, 0, 0]|
|cross80|鏈€缁堝浐瀹氫笁澶村叡鍚屾仮澶嶅凡楠岃瘉|80.255112897|[1, 0, 0]|
|gate_only|鏈€缁堝浐瀹氫笁澶村叡鍚屾仮澶嶅凡楠岃瘉|80.222820979|[0, 34, 0]|
|density_only|鏈€缁堝浐瀹氫笁澶村叡鍚屾仮澶嶅凡楠岃瘉|80.256678729|[1, 0, 0]|

宸插畬鎴愶細鏂板绾挎€у綊鍥?鑱斿悎鍐呴儴涓撳璇诲嚭/鍥犵礌鍒嗙瀹炵幇涓庡疄闄呰繍琛岃褰曘€?
寰呭畬鎴愶細鎵€鏈夊€欓€夌殑鏈€缁堟仮澶嶃€侀€愰€氶亾閰嶆柟鍥哄畾鍚庣殑鍏卞悓鎺ㄧ悊锛涗笉瀛樺湪瀹屾暣OOF澧炵泭璇佹槑銆?
鐩爣锛氬綋鍓嶅弻涓撳鍙鐜颁紭鍖栵紝绗簲鎶?80.5涓旀槑纭敹鐩婏紱瀹屾暣OOF>82浠嶆湭杈炬垚銆?
杈炬垚鍚庯細鍥哄畾鏂规鍚庤瘎浼版硾鍖栵紝褰撳墠涓嶈嚜鍔ㄨˉ鎶樸€佷笉鏀规棫浜や粯銆?
鍥捐氨鍦ㄦ棦鏈塩uration鐢熸垚鍚庤繍琛屾湰鑴氭湰鍐峛uild_atlas锛涗笉淇敼姝ゅ墠琚繚鎶ょ殑curate.py銆?


## 2026-09-28T09:40:23+08:00 鍥涢」浼樺寲涓庡浐瀹氶€氶亾缁勫悎宸插畬鎴?
鏈疆linear/cross80/gate_only/density_only鍏ㄩ儴鑷劧缁撴潫骞跺叡鍚屾仮澶嶃€傛渶缁堝浐瀹氱粍鍚圱OTAL=80.257638624609锛岀浉瀵瑰師80.221861=+0.035777541569锛涢厤鏂?[{'target': 'POR', 'arm': 'density_only', 'score': 77.39392666356925, 'epoch': 1}, {'target': 'PERM', 'arm': 'gate_only', 'score': 89.3475027344081, 'epoch': 34}, {'target': 'SW', 'arm': 'source', 'score': 73.62238476712884, 'epoch': 52}]銆傛病鏈夎揪鍒扮嫭绔嬫硾鍖栫洰鏍囷紝涓嶅惎鍔ㄨˉ鎶樻垨鏀逛氦浠樸€傛洿缁嗗垎甯冭鍑轰紭鍖栨湭鍚姩銆傛€荤粨瑙乺esults/deep_learning/dual8222_round2_20260928/FINAL.md銆?


## POR浣嶇疆/灏哄害鍖归厤浼樺寲鍑嗗
涓変釜闄愬畾瀵圭収宸插噯澶囷紱鍚?0.257638璧风偣锛岀湡瀹炶缁冧笌鏈€缁堟敹鐩婂緟楠岃瘉銆侾ERM/SW鍥哄畾锛屼笉鎭㈠鏃ц矾绾匡紝涓嶈ˉ鎶樸€傝缁嗚鍒掞細results/deep_learning/dual8222_por_subspace_20260928/README.md銆?


## POR瀛愮┖闂村疄闄呭惎鍔?
鏈鏄庣‘鍚姩 [('mean_only', 62404), ('mean_scale', 93744)]锛涘叡3涓鍏堥檺瀹氬鐓э紝鏃犺嚜鍔ㄩ槦鍒椼€傚垵濮嬪鐜般€佽缁冪粨鏋滀笌鏈€缁堥獙璇佷粛寰呭疄闄呮棩蹇椼€?


## 2026-09-28T09:49:26+08:00 POR瀛愮┖闂寸粨鏋滄牳瀵?
[('mean_only', 'running', 4), ('mean_scale', 'running', 4), ('density_all', 'planned', 0)]
宸插畬鎴愭渶浣?{'mode': 'previous_fixed_recipe', 'TOTAL': 80.25763862460869, 'delta': 0.0}锛涘叏閮ㄥ畬鎴?False銆傜洰鏍囧畬鏁碠OF>82灏氭湭杈炬垚锛岃ˉ鎶樻殏鍋滐紝鏃т氦浠樹笉鏀广€傝瑙乺esults/deep_learning/dual8222_por_subspace_20260928銆?


## POR瀛愮┖闂村疄闄呭惎鍔?
鏈鏄庣‘鍚姩 [('density_all', 16688)]锛涘叡3涓鍏堥檺瀹氬鐓э紝鏃犺嚜鍔ㄩ槦鍒椼€傚垵濮嬪鐜般€佽缁冪粨鏋滀笌鏈€缁堥獙璇佷粛寰呭疄闄呮棩蹇椼€?


## 2026-09-28T09:52:33+08:00 POR瀛愮┖闂寸粨鏋滄牳瀵?
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'running', 0)]
宸插畬鎴愭渶浣?{'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}锛涘叏閮ㄥ畬鎴?False銆傜洰鏍囧畬鏁碠OF>82灏氭湭杈炬垚锛岃ˉ鎶樻殏鍋滐紝鏃т氦浠樹笉鏀广€傝瑙乺esults/deep_learning/dual8222_por_subspace_20260928銆?


## 2026-09-28T09:54:12+08:00 POR瀛愮┖闂寸粨鏋滄牳瀵?
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'running', 13)]
宸插畬鎴愭渶浣?{'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}锛涘叏閮ㄥ畬鎴?False銆傜洰鏍囧畬鏁碠OF>82灏氭湭杈炬垚锛岃ˉ鎶樻殏鍋滐紝鏃т氦浠樹笉鏀广€傝瑙乺esults/deep_learning/dual8222_por_subspace_20260928銆?


## 2026-09-28T09:56:37+08:00 POR瀛愮┖闂寸粨鏋滄牳瀵?
[('mean_only', 'complete', 30), ('mean_scale', 'complete', 24), ('density_all', 'complete', 19)]
宸插畬鎴愭渶浣?{'mode': 'mean_only', 'TOTAL': 80.26873450885158, 'delta': 0.01109588424289143}锛涘叏閮ㄥ畬鎴?True銆傜洰鏍囧畬鏁碠OF>82灏氭湭杈炬垚锛岃ˉ鎶樻殏鍋滐紝鏃т氦浠樹笉鏀广€傝瑙乺esults/deep_learning/dual8222_por_subspace_20260928銆?


## Final evidence and next boundary
Completed best: mean_only, TOTAL=80.268734508852, delta=+0.011095884243. POR=77.430912944379; PERM/SW unchanged. All3 runs completed; no additional training launched.
Exploratory 10000 whole-well resamples: [-0.012002,+0.034743], selection-unadjusted; not confirmatory independent validation.
Next unexecuted hypothesis: keep mixture weights/scales fixed and compare a constrained location readout against this linear mean-only control. Do not automatically open width/LR sweeps or fullCV. Full OOF>82 remains unmet. Source and prior submission unchanged.


## 2026-09-28T10:06:43.427829+08:00 POR浣嶇疆璇诲嚭v5
涓夐」棰勫畾涔夊尮閰嶅疄楠屽凡瀹炵幇锛涙湰娆″疄闄呭惎鍔╗('linear', 49964), ('point32', 24972)]銆傚垵濮嬨€佹搴﹀拰鏀剁泭浠ュ疄闄呮棩蹇椾负鍑嗭紝鏈嚜鍔ㄥ惎鍔ㄥ叾浣欓槦鍒?鏃ц矾绾?琛ユ姌銆傚畬鏁碠OF>82鏈畬鎴愶紝鏃т氦浠樹笉鍙樸€傝缁嗚褰時esults/deep_learning/dual8222_location_v5_20260928銆?


## 2026-09-28T10:08:19+08:00 浣嶇疆璇诲嚭v5
宸插畬鎴愶細浠ｇ爜銆佹槑纭惎鍔ㄣ€佸疄闄呯姸鎬佽褰曪紱鍏ㄩ儴瀹屾垚=False銆?
褰撳墠鏈€浣?{'mode': 'previous', 'TOTAL': 80.26873450885158, 'delta': 0.0}銆?
寰呭姙锛氭湭瀹屾垚瀹為獙鐨勬渶缁堟仮澶嶃€佽幏鐩婅鍑虹殑淇濆瓨閲嶈浇楠岃瘉锛涗笉缁х画澧炲姞鏈壒鍊欓€夈€?
鐩爣锛氱浜旀姌>80.5涓旀槑纭敹鐩婏紝瀹屾暣OOF>82鏈畬鎴愩€傝揪鎴愬悗鍥哄畾鏂规鍋氭硾鍖栫‘璁わ紝褰撳墠鏃犺ˉ鎶樻巿鏉冦€?


## 2026-09-28T10:10:22.986394+08:00 POR浣嶇疆璇诲嚭v5
涓夐」棰勫畾涔夊尮閰嶅疄楠屽凡瀹炵幇锛涙湰娆″疄闄呭惎鍔╗('local32', 85664)]銆傚垵濮嬨€佹搴﹀拰鏀剁泭浠ュ疄闄呮棩蹇椾负鍑嗭紝鏈嚜鍔ㄥ惎鍔ㄥ叾浣欓槦鍒?鏃ц矾绾?琛ユ姌銆傚畬鏁碠OF>82鏈畬鎴愶紝鏃т氦浠樹笉鍙樸€傝缁嗚褰時esults/deep_learning/dual8222_location_v5_20260928銆?


## 2026-09-28T10:11:25+08:00 浣嶇疆璇诲嚭v5
宸插畬鎴愶細浠ｇ爜銆佹槑纭惎鍔ㄣ€佸疄闄呯姸鎬佽褰曪紱鍏ㄩ儴瀹屾垚=False銆?
褰撳墠鏈€浣?{'mode': 'point32', 'TOTAL': 80.27413533779057, 'delta': 0.005400828938988411}銆?
寰呭姙锛氭湭瀹屾垚瀹為獙鐨勬渶缁堟仮澶嶃€佽幏鐩婅鍑虹殑淇濆瓨閲嶈浇楠岃瘉锛涗笉缁х画澧炲姞鏈壒鍊欓€夈€?
鐩爣锛氱浜旀姌>80.5涓旀槑纭敹鐩婏紝瀹屾暣OOF>82鏈畬鎴愩€傝揪鎴愬悗鍥哄畾鏂规鍋氭硾鍖栫‘璁わ紝褰撳墠鏃犺ˉ鎶樻巿鏉冦€?


## 2026-09-28T10:15:48+08:00 浣嶇疆璇诲嚭v5
宸插畬鎴愶細浠ｇ爜銆佹槑纭惎鍔ㄣ€佸疄闄呯姸鎬佽褰曪紱鍏ㄩ儴瀹屾垚=True锛屽凡楠岃瘉淇濆瓨閲嶈浇=True銆?
褰撳墠鏈€浣?{'mode': 'point32', 'TOTAL': 80.27413533779057, 'delta': 0.005400828938988411}銆?
鍓╀綑锛氱嫭绔嬫硾鍖栧強涓柇缁窇绛変环鎬у皻鏈獙璇侊紱鏈壒涓嶅啀澧炲姞鍊欓€夈€?
鐩爣锛氱浜旀姌>80.5涓旀槑纭敹鐩婏紝瀹屾暣OOF>82鏈畬鎴愩€傝揪鎴愬悗鍥哄畾鏂规鍋氭硾鍖栫‘璁わ紝褰撳墠鏃犺ˉ鎶樻巿鏉冦€?


## 鏈€缁堝垽鏂笌淇濈暀閰嶆柟
涓夐」璁粌鍧囧畬鎴愪笖鑷劧鍋滄浜庢柊澧炵24杞紝鏈€浣冲潎涓虹6杞€傞€愮偣澶?0.274135337791锛屾瘮褰撳墠绾挎€х户缁缁冨鐓ч珮0.002801716904锛涘眬閮ㄥご80.273788310855锛屾病鏈夌户缁鐩娿€傚叏閮ㄤ繚鐣欏師璇勫垎/楠ㄥ共涓庡叾浠栭€氶亾銆?
宸蹭粠readout_checkpoint.pt鐙珛鍒涘缓妯″瀷锛屽畬鏁?44665鐐归娴嬮€愬厓绱犲鐜般€傛帹鐞嗘椂浠呯Щ闄ゆ柊澧炴畫宸紝鍒嗘暟80.273254588259锛屼笅闄?.000880749532锛涜繖鏄缁冨悗骞查锛屼笉鏄尮閰嶉噸璁晥搴旓紝涔熸湭浣滀负绗洓涓€欓€夈€?
楠岃瘉绐楀彛鐨勬柊澧炰綅缃慨姝ｅ钩鍧囩害0.004242鍊嶅浐瀹氭爣鍑嗗樊銆佹渶澶?.126500鍊嶏紝鏈帴杩?.25涓婇檺锛涜缁熻鍖呭惈鏈夋晥涓婁笅鏂囦綅缃紝涓嶈兘绛夊悓浜庝粎涓ぎ甯︽爣绛捐瘎鍒嗙偣銆備笉鑳芥妸淇濇姢涓婇檺鏈韩褰掑洜涓烘敹鐩婏紝鍥犱负娌℃湁鏃犵害鏉熷鐓с€?
淇濈暀point32涓烘湰杞渶楂樺垎寮€鍙戝€欓€夈€備笌绾挎€у鐓ф瘮杈?2/16浜曟彁楂樸€?浜曚笅闄嶏紱鎺㈢储鎬ф暣浜曢噸閲囨牱鍖洪棿鏈綔鍙嶅閫夋ā鏍℃锛屼笉鑳藉绉扮ǔ瀹氱嫭绔嬫硾鍖栦紭鍔裤€傚畬鏁碠OF>82涓庣浜旀姌>80.5浠嶆湭杈炬垚銆?
涓嬩竴姝ュ皻鏈惎鍔細浼樺厛妫€楠岃缁冪洰鏍囦笌鍘熷畼鏂瑰宸晥鐢ㄧ殑瀵归綈锛屼娇鐢ㄥ綋鍓嶄綅缃鍑轰綔鍥哄畾鎺у埗锛涗笉缁х画鏃犵晫鎵╁ぇ瀹藉害/鍗风Н/涓婇檺銆傛棫ABD鍜屽叾浣欐姌涓嶆仮澶嶏紝浜や粯涓嶆浛鎹€?
鏈疆閰嶆柟瑙丆URRENT_RECIPE.json锛屽畬鏁存ā鍨嬫ā鍧椾负dual8222_location_readout_v5.py銆傞潪绾挎€ц鍑轰笉鑳戒粎鎸夋棫绾挎€х煩闃靛姞杞姐€?


## 2026-09-28T10:32:50.058856+08:00 澶嶆潅棰勬祴澶磛6
棰勫畾涔塴inear/deep_mlp/attention涓夐」锛涙湰娆℃槑纭惎鍔╗('linear', 76148), ('attention', 93284)]銆傚悓80.274135銆佸師鍙屼笓瀹跺拰璇勫垎锛涚湡瀹炲垵濮?璁粌/鏀剁泭寰呮棩蹇椼€傛棤鏃ц矾绾挎垨琛ユ姌锛岀洰鏍囦粛鏈畬鎴愩€傝瑙乺esults/deep_learning/dual8222_attention_v6_20260928銆?


## 2026-09-28T10:34:47+08:00 澶嶆潅棰勬祴澶磛6鏍稿
鍏ㄩ儴瀹屾垚=False锛涘綋鍓嶄繚鐣?{'mode': 'previous', 'TOTAL': 80.27413533779057}锛涘尮閰嶅樊鍊?{}銆傜嫭绔嬭褰時esults/deep_learning/dual8222_attention_v6_20260928锛涙棤鏃ц矾绾块噸鍚?琛ユ姌鎴栦氦浠樺彉鏇淬€?


## 2026-09-28T10:37:23.708937+08:00 澶嶆潅棰勬祴澶磛6
棰勫畾涔塴inear/deep_mlp/attention涓夐」锛涙湰娆℃槑纭惎鍔╗('deep_mlp', 95156)]銆傚悓80.274135銆佸師鍙屼笓瀹跺拰璇勫垎锛涚湡瀹炲垵濮?璁粌/鏀剁泭寰呮棩蹇椼€傛棤鏃ц矾绾挎垨琛ユ姌锛岀洰鏍囦粛鏈畬鎴愩€傝瑙乺esults/deep_learning/dual8222_attention_v6_20260928銆?


## 2026-09-28T10:40:07+08:00 澶嶆潅棰勬祴澶磛6鏍稿
鍏ㄩ儴瀹屾垚=False锛涘綋鍓嶄繚鐣?{'mode': 'previous', 'TOTAL': 80.27413533779057}锛涘尮閰嶅樊鍊?{'attention_minus_linear': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}}銆傜嫭绔嬭褰時esults/deep_learning/dual8222_attention_v6_20260928锛涙棤鏃ц矾绾块噸鍚?琛ユ姌鎴栦氦浠樺彉鏇淬€?


## 2026-09-28T10:41:20+08:00 澶嶆潅棰勬祴澶磛6鏍稿
鍏ㄩ儴瀹屾垚=True锛涘綋鍓嶄繚鐣?{'mode': 'previous', 'TOTAL': 80.27413533779057}锛涘尮閰嶅樊鍊?{'deep_mlp_minus_linear': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}, 'attention_minus_linear': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}, 'attention_minus_deep_mlp': {'TOTAL_delta': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0]}}銆傜嫭绔嬭褰時esults/deep_learning/dual8222_attention_v6_20260928锛涙棤鏃ц矾绾块噸鍚?琛ユ姌鎴栦氦浠樺彉鏇淬€?


## 2026-09-28T10:42:57+08:00 澶嶆潅棰勬祴澶磛6宸插畬鎴?
涓夐」鍧嘼est_epoch0銆乻top18锛屽師80.274135淇濇寔銆傚疄闄呮洿鏂板悗鏈€楂榣inear80.272137140362/deepMLP80.270535660860/attention80.272549291930锛涙病鏈夋柊澧炴敹鐩娿€俛ttention鐩稿鍋忕疆鐪熷疄鏇存柊锛宮ask鍥涢」妫€鏌ラ€氳繃銆傚綋鍓嶆棤璁粌杩涚▼锛屾棤琛ユ姌鎴栦氦浠樺彉鏇达紱鍒嗛噺query/cross-attention灏氭湭瀹炵幇鎴栧惎鍔ㄣ€傞噸杞戒笉閲嶅锛屼繚鐣欏師v5璇佹槑銆傝瑙乺esults/deep_learning/dual8222_attention_v6_20260928/FINAL.md銆?


## 2026-09-28T10:51:11.227166+08:00 鍒嗛噺query v7鍚姩
鍚?0.274135锛屾棫棰勬祴鍣ㄥ叏閮ㄥ浐瀹氥€傚疄闄呭惎鍔╗('center', 61572), ('local', 37736)]锛涗粎鏈壒涓ら」锛屾病鏈夋棫浠诲姟鎴栬ˉ鎶樸€傚垵濮?璁粌/鏈€缁堟敹鐩婂緟鐪熷疄缁撴灉锛岃瑙乺esults/deep_learning/dual8222_component_query_v7_20260928銆?


## 2026-09-28T10:53:13+08:00 鍒嗛噺query v7璁板綍
鍏ㄩ儴瀹屾垚=False锛涗繚鐣?{'mode': 'previous', 'TOTAL': 80.27413533779057}锛涘鐓?None銆傝褰時esults/deep_learning/dual8222_component_query_v7_20260928锛涙棤鏃т换鍔￠噸鍚?琛ユ姌/浜や粯鍙樻洿銆?


## 2026-09-28T10:56:42+08:00 鍒嗛噺query v7璁板綍
鍏ㄩ儴瀹屾垚=True锛涗繚鐣?{'mode': 'previous', 'TOTAL': 80.27413533779057}锛涘鐓?{'local_minus_center_TOTAL': 0.0, 'well_wins': 0, 'well_losses': 0, 'exploratory_selection_unadjusted_95': [0.0, 0.0], 'same_parameters': True}銆傝褰時esults/deep_learning/dual8222_component_query_v7_20260928锛涙棤鏃т换鍔￠噸鍚?琛ユ姌/浜や粯鍙樻洿銆?


## 2026-09-28T10:58:22+08:00 鍒嗛噺query v7瀹屾垚
涓ょ粍鍧?8杞嚜鐒舵棭鍋滃苟鎭㈠绗?杞?0.274135锛涜缁冨悗鏈€楂榗enter80.270925285299@9銆乴ocal80.271695049988@8锛屾棤鏂板鏀剁泭銆傚師棰勬祴鍣ㄥ叏鍐荤粨锛涗袱缁?4081鍙傛暟/鍒濆鍖栦竴鑷达紝query鐪熷疄鏇存柊锛屽眬閮ㄤ笌mask妫€鏌ラ€氳繃銆備繚鐣欐棫v5锛屼笉閲嶅鏃ф潈閲嶉噸杞斤紝鏃犳棫浠诲姟/琛ユ姌/浜や粯鍙樻洿锛涘畬鏁村垎甯僸uery鍜屾崯澶卞榻愭湭鍚姩銆傝瑙乺esults/deep_learning/dual8222_component_query_v7_20260928/FINAL.md銆?


## 2026-09-28T11:11:39.865093+08:00 鐙珛閫氶亾妯″瀷v8
鍏」棰勫畾涔夊崟/鍙岄骞插€欓€夛紱鏈瀹為檯鍚姩[('POR_dual', 91632), ('SW_dual', 95292)]锛屽悇鑷鍒扮鍗曢€氶亾璁粌銆傚皻鏃犳湰鎵瑰凡楠岃瘉绮惧害锛涗繚鎸佸師璇勫垎/鍒嗘姌锛屼笉鑷姩琛ユ姌鎴栨敼鍙樻棫浜や粯銆傝鍒掑拰璇佹嵁锛歳esults/deep_learning/independent_channels_v8_20260928銆?


## 鍚姩澶辫触鍙婁慨澶?
棣栨壒POR_dual/SW_dual鍦╩ain寮€澶村洜閬楁紡sys瀵煎叆澶辫触锛屾湭鍔犺浇鏁版嵁鎴栬缁冿紱宸蹭繚瀛榮tartup_failure_attempt1.json鍜屽師err銆備慨澶嶅鍏ュ苟鏄庣‘閲嶈瘯锛岄潪鑷姩鎭㈠銆傚彟涓€娆″彧璇籹hell鍛戒护杞箟澶辫触锛屼笉褰卞搷璁粌銆傜粨鏋滀互绗簩娆″疄闄呮棩蹇椾负鍑嗐€?


## 2026-09-28T11:15:47+08:00 鐙珛閫氶亾v8
瑙傚療鏃堕棿锛?026-09-28T11:15:47+08:00銆傚凡瀹屾垚0/6椤广€?
宸插仛锛氭ā鍨嬫媶鍒嗗拰鏄惧紡鍚姩锛岀粨鏋滃鏈洰褰昿rogress.json銆?
璁″垝锛氬畬鎴愭湁闄愬叚椤瑰苟鍥哄畾涓夋ā鍨嬪叡鍚岄噸杞姐€?
鏈仛锛氬畬鏁碠OF鍜岀嫭绔嬫硾鍖栥€佺画璺戠瓑浠锋€с€傜洰鏍囪揪鎴愬悗楠岃瘉鍥哄畾閰嶆柟锛屽綋鍓嶆棤琛ユ姌銆?


## CUDA鍙嶅悜鍏煎鎬т慨澶?
棣栨壒涓ゆā鍨嬪垵濮嬬洰鏍囬娴嬮€愬厓绱犲鐜帮紝浣嗕弗鏍肩‘瀹氭€у湪棣栨linear鎻掑€煎弽鍚戞姤閿欙紝鏈墽琛宱ptimizer.step銆傚け璐ttempt2瀹屾暣淇濆瓨鍦ㄥ悇鐩綍failed_attempt2銆傚叚椤圭粺涓€閲囩敤deterministic warn_only淇濈暀鍙敤纭畾鎬х畻瀛愶紝瀵笴UDA鎻掑€煎弽鍚戝厑璁歌鍛婏紱鍥哄畾闅忔満绉嶅瓙浠嶄笉鑳戒繚璇佽缁冮€愪綅閲嶇幇銆傝璁剧疆宸插啓鍏onfig銆傛棫婧愪唬鐮佹湭鏀癸紝鏄庣‘浠庡師璧风偣閲嶆柊鍚姩棣栨壒锛屼笉鍐掔О缁窇銆?


## 2026-09-28T11:19:42+08:00 鐙珛閫氶亾v8
瑙傚療鏃堕棿锛?026-09-28T11:19:42+08:00銆傚凡瀹屾垚0/6椤广€?
宸插仛锛氭ā鍨嬫媶鍒嗗拰鏄惧紡鍚姩锛岀粨鏋滃鏈洰褰昿rogress.json銆?
璁″垝锛氬畬鎴愭湁闄愬叚椤瑰苟鍥哄畾涓夋ā鍨嬪叡鍚岄噸杞姐€?
鏈仛锛氬畬鏁碠OF鍜岀嫭绔嬫硾鍖栥€佺画璺戠瓑浠锋€с€傜洰鏍囪揪鎴愬悗楠岃瘉鍥哄畾閰嶆柟锛屽綋鍓嶆棤琛ユ姌銆?


## 2026-09-28T11:23:06.329949+08:00 鐙珛閫氶亾妯″瀷v8
鍏」棰勫畾涔夊崟/鍙岄骞插€欓€夛紱鏈瀹為檯鍚姩[('PERM_dual', 88272), ('SW_single', 34584)]锛屽悇鑷鍒扮鍗曢€氶亾璁粌銆傚皻鏃犳湰鎵瑰凡楠岃瘉绮惧害锛涗繚鎸佸師璇勫垎/鍒嗘姌锛屼笉鑷姩琛ユ姌鎴栨敼鍙樻棫浜や粯銆傝鍒掑拰璇佹嵁锛歳esults/deep_learning/independent_channels_v8_20260928銆?


## 2026-09-28T11:26:12.725663+08:00 鐙珛閫氶亾妯″瀷v8
鍏」棰勫畾涔夊崟/鍙岄骞插€欓€夛紱鏈瀹為檯鍚姩[('POR_single', 94644)]锛屽悇鑷鍒扮鍗曢€氶亾璁粌銆傚皻鏃犳湰鎵瑰凡楠岃瘉绮惧害锛涗繚鎸佸師璇勫垎/鍒嗘姌锛屼笉鑷姩琛ユ姌鎴栨敼鍙樻棫浜や粯銆傝鍒掑拰璇佹嵁锛歳esults/deep_learning/independent_channels_v8_20260928銆?


## 2026-09-28T11:43:56.010416+08:00 鐙珛閫氶亾妯″瀷v8
鍏」棰勫畾涔夊崟/鍙岄骞插€欓€夛紱鏈瀹為檯鍚姩[('PERM_single', 96980)]锛屽悇鑷鍒扮鍗曢€氶亾璁粌銆傚皻鏃犳湰鎵瑰凡楠岃瘉绮惧害锛涗繚鎸佸師璇勫垎/鍒嗘姌锛屼笉鑷姩琛ユ姌鎴栨敼鍙樻棫浜や粯銆傝鍒掑拰璇佹嵁锛歳esults/deep_learning/independent_channels_v8_20260928銆?


## 鐙珛妯″瀷鍏卞悓鎭㈠瀹屾垚
{"scores": {"POR": 77.44891570750882, "PERM": 89.3475027344081, "SW": 73.62238476712884, "TOTAL": 80.27413533779057}, "choices": [{"target": "POR", "key": "POR_dual", "architecture": "dual", "target_score": 77.44891570750882, "best_epoch": 0, "checkpoint": "outputs/pure_dl_independent_v8_POR_dual80_fold5/model_best.pt"}, {"target": "PERM", "key": "PERM_dual", "architecture": "dual", "target_score": 89.3475027344081, "best_epoch": 0, "checkpoint": "outputs/pure_dl_independent_v8_PERM_dual80_fold5/model_best.pt"}, {"target": "SW", "key": "SW_dual", "architecture": "dual", "target_score": 73.62238476712884, "best_epoch": 0, "checkpoint": "outputs/pure_dl_independent_v8_SW_dual80_fold5/model_best.pt"}], "no_shared_backbone": true, "well_wins": 0, "well_losses": 0}
鐙珛涓夋ā鍨嬪疄闄呴娴嬪凡楠岃瘉锛岃ˉ鎶?浜や粯淇濇寔鏆傚仠锛涚洰鏍囧畬鏁碠OF>82鏈畬鎴愩€?


## 2026-09-28T11:54:31.739535+08:00 SW鐙珛TCN鍒嗗竷v9鍚姩
宸叉樉寮忓惎鍔╗('ce', 58148), ('ce_utility', 90832)]銆傚彧姣旇緝涓や釜棰勫畾涔夌洰鏍囷紝鏈湁鏂版敹鐩婅瘉鎹紱鏃OR/PERM銆佷氦浠樺拰鍘熸暟鎹笉鍙橈紝鏃犺ˉ鎶樸€傝褰曡results/deep_learning/independent_sw_v9_20260928銆?


## 2026-09-28T12:02:17.041218+08:00 鐙珛SW鍒嗗竷v9鏀跺熬
瀹為檯涓夋ā鍨婽OTAL=80.274135337791锛宒elta=+0.000000000000锛涗娇鐢ㄦ柊SW=False銆侾OR/PERM閫愬厓绱犱笉鍙橈紝涓夋ā鍨嬫棤鍏变韩瀛樺偍涓斿叏閮?44665鐐归噸杞藉鐜般€?
宸插畬鎴恦8閬楃暀椤广€乿8缁勫悎浠ュ強v9涓ら」鍖归厤璁粌涓庣粍鍚堬紱鏈ˉ鎶橈紝鏃т氦浠樹笉鏇挎崲銆傜洰鏍囧畬鏁碠OF>82浠嶆湭瀹屾垚銆備笅涓€姝ユ寜瀹為檯缁撴灉纭畾锛屼笉鑷姩鍚姩鏇村璁粌銆?


## 2026-09-28T12:02:48+08:00 鐙珛閫氶亾v8
瑙傚療鏃堕棿锛?026-09-28T12:02:48+08:00銆傚凡瀹屾垚6/6椤广€?
宸插仛锛氭ā鍨嬫媶鍒嗗拰鏄惧紡鍚姩锛岀粨鏋滃鏈洰褰昿rogress.json銆?
璁″垝锛氬畬鎴愭湁闄愬叚椤瑰苟鍥哄畾涓夋ā鍨嬪叡鍚岄噸杞姐€?
鏈仛锛氬畬鏁碠OF鍜岀嫭绔嬫硾鍖栥€佺画璺戠瓑浠锋€с€傜洰鏍囪揪鎴愬悗楠岃瘉鍥哄畾閰嶆柟锛屽綋鍓嶆棤琛ユ姌銆?


## 2026-09-28T13:10:22.167418+08:00 鐙珛SW鍐峰惎鍔╲10
宸插畬鎴愭ā鍨?璁粌瀹炵幇骞舵樉寮忓惎鍔╗('tcn', 61768), ('bigru', 91388)]銆傛鏃跺皻鏃犳柊鍒嗘暟銆傝鍒掑畬鎴愪袱妯″瀷鎭㈠鍜屽悓璇勫垎姣旇緝锛涗笉鑷姩琛ユ姌鎴栨敼鏃т氦浠橈紝瀹屾暣OOF>82鏈畬鎴愩€傝褰曪細results/deep_learning/independent_sw_scratch_v10_20260928銆?


## 2026-09-28T13:17:58.091719+08:00 SW鐙珛鍐峰惎鍔╲10瀹屾垚
鍥哄畾閰嶆柟TOTAL=80.274135337791锛岀浉瀵规棫+0.000000000000锛涢噰鐢ㄦ柊SW=False銆?
涓ゅ€欓€夌湡瀹炶缁?鎭㈠鍙婄鐩樻牳楠屽畬鎴愶紱鏃犳柊琛ユ姌鎴栦氦浠樺彉鏇淬€傜粨鏋滐細results/deep_learning/independent_sw_scratch_v10_20260928/FINAL.md銆傚皻鏈瘉鏄庣嫭绔嬫柊浜曞鐩娿€?


## 瀹炴祴璇樊鍒嗚В涓庝笅涓€姝ヨ竟鐣?
涓や釜鏂版ā鍨嬩粠纾佺洏閲嶈浇鍚庯紝鍏ㄩ儴144665鐐硅瘎鍒嗛娴嬮€愬厓绱犲鐜帮紱璁粌鍚庡～鍏呬笉鍙樻€т笌鍏ㄧ┖绐楀彛妫€鏌ラ€氳繃銆?
tcn: 鐪熼粯璁ゆ閫氶亾鍒嗗彉鍖?0.600186锛屽TOTAL璐＄尞+0.146349锛涚湡闈為粯璁ゆ鍙樺寲-3.094669锛屽TOTAL璐＄尞-0.328530銆?/16浜曟彁楂橈紝12浜曚笅闄嶃€?
bigru: 鐪熼粯璁ゆ閫氶亾鍒嗗彉鍖?0.704171锛屽TOTAL璐＄尞+0.171705锛涚湡闈為粯璁ゆ鍙樺寲-3.439228锛屽TOTAL璐＄尞-0.365109銆?/16浜曟彁楂橈紝12浜曚笅闄嶃€?
杩欒鏄庢湰娆℃渶缁堝姩浣滃湪榛樿娈垫洿鍑嗭紝浣嗚繛缁爣绛炬鎹熷け鏇村锛涗笉鑳芥嵁姝ょ嫭绔嬪綊鍥犱簬闂ㄦ帶鍒嗙被姝ｇ‘鐜囷紝鍥犱负瀵嗗害鍜屾晥鐢ㄨВ鐮佸悓鏃跺奖鍝嶆渶缁堝姩浣溿€?
涓嬩竴姝ュ皻鏈惎鍔細鍏堝湪璁粌渚ц瘖鏂璖W鐨勯粯璁ょ姸鎬佷笌闈為粯璁ゆ暟鍊间换鍔℃槸鍚︾浉浜掔壍鍒讹紝鍐嶄互闂ㄦ帶鍥哄畾/鍙缁冨拰鐙珛鏁板€艰〃寰佷綔鏈夐檺鍖归厤瀵圭収锛涗笉鐩存帴鎵╁ぇGRU/TCN灞傛暟鎴栨帹鏂ā鍨嬪鏃忎笂闄愩€?
妯″瀷鏄殢鏈烘潈閲嶄粠澶村涔狅紝浣嗗浐瀹歈鏉ヨ嚜鍏堝墠璁粌浜曡瘖鏂紝棰勫鐞?鏀寔鏍间篃娌跨敤璁粌渚у厓鏁版嵁锛涗笉绉版暣鏉ipeline瀹屽叏涓嶄緷璧栨鍓嶈缁冧骇鐗┿€?
褰撳墠淇濈暀鍘熶笁绉佹湁妯″瀷80.274135337791锛涙棤鏂癝W鏇挎崲銆佹棤琛ユ姌銆佹棤鏃т氦浠樺彉鏇淬€?
璁板綍宸ュ叿闂锛氭渶缁堟姤鍛婇娆″洜numpy.bool_ JSON搴忓垪鍖栧け璐ワ紝淇浠呬綔鐢ㄤ簬鎶ュ憡鑴氭湰骞舵槑纭噸璺戯紱璁粌鏉冮噸鏈敼銆俢urate.py婧愭枃浠剁紪杈戣鍐呭淇濇姢鎷掔粷锛屽師鏂囦欢鏈敼锛涙湰杞鏄庣敱report_independent_sw_scratch_v10.py --record鍙鐜扮淮鎶ゅ埌curation.json锛屽啀build_atlas.py鐢熸垚椤甸潰銆?


## 2026-09-28T14:06:40.013202+08:00 SW瑙ｈ€11鍚姩
浠呭惎鍔╗('shared', 99720), ('split', 82088)]銆備袱鍊欓€夊悓鍙傛暟棰勭畻锛屽疄闄呭垎鏁板緟鏍稿疄銆傛棤鏃т换鍔℃垨琛ユ姌锛涘畬鏁碠OF>82鏈畬鎴愩€傝褰時esults/deep_learning/independent_sw_separation_v11_20260928銆?


## 2026-09-28T14:11:30.433442+08:00 SW鍏变韩/鍒嗛洟v11瀹屾垚
sharedSW=73.093512782锛宻plitSW=72.927481759锛涗繚鐣橳OTAL=80.274135337791锛屼娇鐢ㄦ柊SW=False銆傚悓鍙?鍒濆鍖栧凡鏍稿锛涜瘖鏂笌纾佺洏鎭㈠瀹屾垚锛岄潪鐙珛OOF缁撹銆?
璇﹁results/deep_learning/independent_sw_separation_v11_20260928/FINAL.md銆傛棤鏂拌ˉ鎶樻垨浜や粯鍙樻洿锛涗笅涓€姝ュ皻鏈惎鍔ㄣ€?


## 鏈疆缁撹涓庝笅涓€姝ラ檺瀹?
涓ら」璁粌鍙婄鐩橀噸杞藉畬鎴愶紝鏃犺繍琛屼腑鐨勬湰鎵逛换鍔°€俿haredSW=73.093512781564锛宻plitSW=72.927481759099銆傚悓鍙傛暟涓庡悓鍒濆鍖栫姸鎬佸凡瀹炴牳锛宻plit鐩稿shared鐨勭瓑浠稵OTAL鍙樺寲-0.058110857863锛?/16浜曟彁楂樸€?0浜曚笅闄嶃€傞€夋嫨鏈牎姝ｇ殑鎺㈢储鎬т簳閲嶉噰鏍峰尯闂翠负[-0.1688249200617162, 0.05241575243032869]锛屼笉鑳芥嵁姝ゅ甯冩柊浜曚笂鐨勬櫘閬嶇粨鏋勬帓搴忋€?
鎷嗗垎鍚庣湡瀹為潪榛樿娈甸€氶亾鍒?0.096188933锛岀湡瀹為粯璁ゆ-0.280193202銆傞潪榛樿娈电殑灏忓箙鏀瑰杽涓嶈冻浠ヨˉ鍋块粯璁ゆ鎹熷け銆傞棬鎺?.5闃堝€煎彫鍥炴彁楂橈紝浣嗛棬鎺т笌瀵嗗害涓€璧峰喅瀹氭渶缁堝姩浣滐紝鍒嗙被鍑嗙‘鐜囦笉绛変环浜庡畼鏂瑰垎鏁般€?
鍦ㄧ湡瀹為潪榛樿娈典笂浠呯敤杩炵画鍒嗗竷瑙ｇ爜锛堝皢闂ㄦ帶璐ㄩ噺璁?鐨勮瘖鏂級锛宻plit杈僺hared涓?0.311991269銆傝繖鏄爣绛鹃檺瀹氬瓙闆嗙殑绂荤嚎璇婃柇锛屾病鏈夌敤浜庨€夋嫨閫愮偣妯″瀷鎴栨帹鐞嗭紱涓嶈兘涓庢棫妯″瀷鏈粡鍚屾牱澶勭悊鐨勬櫘閫氬緱鍒嗙洿鎺ユ瘮杈冦€?
鍏变韩缁?28涓浐瀹氳缁冪獥鍙ｄ腑锛?/8鎵笲CE/CE缂栫爜鍣ㄦ搴︿綑寮︿负璐燂紝鍧囧€?.146004銆俿plit璺ㄤ换鍔℃搴︿弗鏍间负闆讹紝浣嗛獙璇佹湭鎻愰珮銆傝繖涓皬鏍锋湰缁撴灉涓嶆敮鎸佸皢褰撳墠鐡堕鐩存帴褰掔粨涓烘櫘閬嶆搴﹀啿绐侊紱鍏变韩淇℃伅鍜屾瘡浠诲姟鏈夋晥瀹归噺涔熼殢杩炴帴鏂瑰紡鏀瑰彉銆?
鏈鍙叧闂浉鍚屾ā鍨?棰勭畻闄勮繎鐨勫己鍒跺畬鍏ㄥ垎绂绘壂鎻忥紝涓嶆帹璁烘墍鏈夌嫭绔嬬紪鐮佸櫒鏃犳晥锛汸OR/PERM/SW涓変釜鐩爣涔嬮棿缁х画浣跨敤绉佹湁妯″瀷銆?
涓嬩竴姝ュ皻鏈惎鍔細淇濇寔涓€绉嶅浐瀹歋W琛ㄧず锛屾瘮杈冨師BCE+鏉′欢CE涓庢妸榛樿鍔ㄤ綔绔炰簤涔熺撼鍏ョ殑鑱斿悎鏁堢敤鐩戠潱銆備笌v9浠呭鍐荤粨闂ㄦ帶涓嬭繛缁垎甯冨姞鏁堢敤椤圭殑璇曢獙鍖哄垎锛涗笉璋冮獙璇侀泦闃堝€硷紝涓嶈拷鍔犺嚜鍔ㄩ槦鍒楁垨琛ユ姌銆?
褰撳墠缁х画淇濈暀宸查獙璇佷笁绉佹湁妯″瀷TOTAL=80.274135337791銆傛湰鎵规柊妯″瀷鏈浛鎹㈡棫浜や粯锛涚浜旀姌涓ユ牸>80.5鍜屽畬鏁碠OF>82浠嶆湭瀹屾垚銆?


## 2026-09-28T15:55:52.641016+08:00 SW鑱斿悎鏁堢敤v12鍚姩
宸插畾涔夊悓璧风偣鎹熷け瀵圭収骞跺疄闄呭惎鍔╗('likelihood', 99564), ('joint_utility', 3792)]銆傚垵濮?璁粌/鏈€缁堟敹鐩婂緟瀹為檯璁板綍锛涙棤鍏朵粬浠诲姟鎴栬ˉ鎶樸€傜爺绌剁洰褰時esults/deep_learning/independent_sw_joint_v12_20260928锛屽畬鏁碠OF>82鏈畬鎴愩€?


## 2026-09-28T16:09:08.183155+08:00 寮篠W瀹瑰樊璐ㄩ噺v13鍚姩
鏈疆鏄惧紡鍚姩[('likelihood', 76668), ('tolerance_mass', 91348)]銆備娇鐢⊿W绉佹湁鍙屼笓瀹?3.622385鍚岃捣鐐癸紝鍖哄埆浠呰緟鍔╂崯澶辩郴鏁?/0.5銆傛鍓峷12宸插畬鎴愭棤鏀剁泭锛屾湭閲嶅鍚姩銆傚師鏁版嵁/鍒嗘姌/璇勫垎/棰勭畻淇濇寔锛涗笉鎭㈠鏃т换鍔°€佷笉琛ユ姌锛岀湡瀹炴敹鐩婂緟璇诲彇銆傝褰時esults/deep_learning/independent_sw_tolerance_v13_20260928銆?


## 2026-09-28T16:19:00.119857+08:00 v13瀹屾垚
涓ら」寮篠W绉佹湁鍙屼笓瀹舵崯澶卞鐓у畬鎴?8杞紝鍧囦繚鐣欑0杞?3.622384767129锛屾棤鏂板鏀剁泭锛涙洿鏂板悗鏈€浣冲垎鍒?3.47330202477876/73.45786228917683銆傚師涓夌嫭绔嬫ā鍨?0.274135337791淇濈暀锛屾棤鏂颁笁妯″瀷閲嶈浇瀹ｇО銆佹棤琛ユ姌/浜や粯鏇存敼銆傚畬鏁存姤鍛妑esults/deep_learning/independent_sw_tolerance_v13_20260928/FINAL.md銆備笅涓€姝ュ垎缁勬硾鍖栬瘖鏂皻鏈惎鍔紝>82鐩爣鏈畬鎴愩€?


## 2026-09-28T16:44:49.116604+08:00 v14璺ㄤ簳璇婃柇鍚姩
鍦ㄧ户缁紭鍖栨巿鏉冧笅棰勮point/well涓ょ粍锛屽疄闄呭惎鍔╗('point', 22756), ('well', 100016)]銆傚綋鍓嶉樁娈礽nner锛涘唴灞?8璁粌/16鐩戞帶锛屽叏閮ㄩ澶勭悊閲嶆柊鎷熷悎銆丵=1銆佹棤鏃ф潈閲嶃€傚師绗簲鎶樻湭鐢ㄤ簬褰撳墠鍐呭眰璁粌鎴栭€夋ā銆傜湡瀹炴敹鐩婂緟鏍稿疄锛涗笉鑷姩琛ユ姌銆佹棫浜や粯涓嶅彉銆傝褰時esults/deep_learning/sw_group_generalization_v14_20260928銆?


## 2026-09-28T16:47:32.678918+08:00 v14鍐呭眰璇婃柇瀹屾垚
涓ら」鍧囧凡鑷劧缁撴潫銆俻ointSW=76.23493007968595锛寃ellSW=76.165444754071锛涘樊-0.06948532561494469銆傝繖鏄?8鈫?6鍐呭眰SW锛屼笉鏄師绗簲鎶橈紱鍘熺浜旀姌鏈瘎浼帮紝鏃?0.274135淇濇寔銆傝缁冧笌鐩戞帶杞ㄨ抗鍙婅瘖鏂凡淇濆瓨锛屾棤鑷姩琛ユ姌銆傛姤鍛妑esults/deep_learning/sw_group_generalization_v14_20260928/FINAL_inner.md銆?


## 2026-09-28T16:50:26.967310+08:00 v14鏈€缁堟敹灏?
瀹屾垚涓ユ牸48/16鍐呭眰瀹為獙锛屽師绗簲鎶樻湭璇勪及銆傛寜浜曞潎琛℃湭閫氳繃棰勮鍐呴儴闂ㄦ锛屼笉鍚姩棰濆鍘?4浜曡缁冦€傚彂鐜伴粯璁ょ巼宸彲鎺╃洊鎬讳綋娉涘寲宸紝鍚庢湡璁粌鍒嗗崌鑰岀洃鎺у垎闄嶏紱鍏蜂綋瑙乺esults/deep_learning/sw_group_generalization_v14_20260928/FINAL.md銆傚悗缁法浜曞眬閮ㄥ弬鑰冩湭瀹炵幇锛屾棫涓夌鏈夋ā鍨?0.274135淇濇寔锛屽師鍏朵粬鎶樹笌浜や粯涓嶆敼銆?


## 2026-09-28T17:00:30.734246+08:00 v15缁撴瀯鍊欓€夊惎鍔?
鏄惧紡鍚姩[('adaptive', 70224), ('latent', 96352)]锛涘彧鍋氭棦瀹氫笁楠ㄥ共鍐呭眰SW姣旇緝锛屽垵濮嬩笌鏈€缁堝垎鏁板緟瀹為檯璁板綍銆傛棤鍘熺浜旀姌鎴栧叾浠栭€氶亾妯″瀷鏇存柊锛屾棤鑷姩闃熷垪銆?


## 2026-09-28T17:06:20.393426+08:00 v15缁撴瀯鍊欓€夊惎鍔?
鏄惧紡鍚姩[('state_space', 37424)]锛涘彧鍋氭棦瀹氫笁楠ㄥ共鍐呭眰SW姣旇緝锛屽垵濮嬩笌鏈€缁堝垎鏁板緟瀹為檯璁板綍銆傛棤鍘熺浜旀姌鎴栧叾浠栭€氶亾妯″瀷鏇存柊锛屾棤鑷姩闃熷垪銆?


## v15涓夌瀹屾暣楠ㄥ共绛涙煡瀹屾垚
2026-09-28T17:11:23.595216+08:00銆倇'adaptive': 76.43706340163116, 'latent': 76.59073085729038, 'state_space': 76.38645544042299}锛涘叏閮ㄤ负48/16鍐呭眰SW銆傛槸鍚︾粨鏋勪俊鍙?True銆傛湭璁粌鍘熺浜旀姌锛屾棫涓夌鏈夋ā鍨?0.274135淇濇寔銆傝缁嗙爺绌惰褰時esults/deep_learning/architecture_zoo_v15_20260928/FINAL.md銆傚弬鏁板尮閰嶆秷铻?鍏朵粬閫氶亾/瀹屾暣OOF>82鍧囨湭瀹屾垚銆?


## v15瀹屾垚鍚庣殑鏈哄埗楠岃瘉
2026-09-28T17:14:37.682304+08:00銆傛綔鍙橀噺/SSM瀹屾暣144621鐐圭鐩樺鐜帮紱鑷€傚簲閲囨牱瀹屾垚鏈夐檺鎵归獙璇併€傛綔鍙橀噺涓哄綋鍓嶅唴灞傜粨鏋勫€欓€夛紝鏈敼鍘熺浜旀姌妯″瀷锛涘尮閰嶅閲?鍏朵粬鐩爣/璺ㄦ姌缁撴灉寰呭仛銆傝瑙乤rchitecture_zoo_v15_20260928/FINAL.md銆?


## 2026-09-28T18:03:35.019301+08:00 v16鍘熺浜旀姌鍚姩
瀹為檯鏄惧紡鍚姩[('tcn', 46268), ('latent', 10204)]銆備笉鍔犺浇鍐呭眰鏉冮噸锛屽師64/16浜曪紝鍥哄畾鍏朵綑閫氶亾锛屾渶缁堟€诲垎寰呮牳瀵广€傚師鏃т换鍔°€佷氦浠樸€佸叾浠栧洓鎶樹笉鏀广€傜爺绌惰褰時esults/deep_learning/latent_fold5_v16_20260928銆?


## 2026-09-28T19:28:42.582911+08:00 v17鏄惧紡鍚姩
[{"mode": "shared_global", "pid": 39272, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_latent_gate_v17.py", "--mode", "shared_global", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_latent_gate_v17_shared_global_fold5"], "started_at": "2026-09-28T19:28:42.569549+08:00"}, {"mode": "local_gate", "pid": 6468, "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_latent_gate_v17.py", "--mode", "local_gate", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_latent_gate_v17_local_gate_fold5"], "started_at": "2026-09-28T19:28:42.580396+08:00"}]
浠呭惎鍔ㄤ袱涓瀹氫箟SW鍊欓€夛紝鍘熷叾浠栨姌涓嶅彉锛涙垚缁╀笌鏈€缁堢粍鍚堝皻寰呰繍琛岀粨鏋溿€?


## v17鏈€缁堝畬鎴?
2026-09-28T19:37:40.442844+08:00銆備袱缁勮缁?鍘?44665鐐圭鐩橀噸杞戒笌涓夋ā鍨嬪疄闄呭悎骞跺凡瀹屾垚銆俿hared_globalTOTAL=80.219534179锛宭ocal_gateTOTAL=80.105625735锛涙棤鏂板鏀剁泭锛屼繚鐣?0.274135銆傛湭瀹屾垚锛氭柊鍋囪/鍏朵粬閫氶亾缁撴瀯/瀹屾暣OOF>82銆傛棤绗笁鍊欓€夈€佽ˉ鎶樻垨浜や粯鍙樻洿銆傝瘉鎹畆esults/deep_learning/latent_gate_v17_20260928/FINAL.md銆?


## 2026-09-28T20:09:47.896873+08:00 v18鏄惧紡鍚姩
[{"mode": "tcn104", "pid": 9468, "started_at": "2026-09-28T20:09:47.882493+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_architecture_v18.py", "--mode", "tcn104", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_architecture_v18_tcn104_fold5"]}, {"mode": "latent64", "pid": 23700, "started_at": "2026-09-28T20:09:47.894874+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_architecture_v18.py", "--mode", "latent64", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_architecture_v18_latent64_fold5"]}]
浠呬袱椤归瀹氫箟POR璁粌锛涙湭鑾峰緱鏂扮殑鏈€缁堝垎鏁帮紝鏈敼鏃фā鍨?浜や粯/鍏朵粬鎶樸€?


## v18 POR缁撴瀯瀵圭収瀹屾垚
2026-09-28T20:16:03.903565+08:00銆俆CN TOTAL=79.550729668锛宭atent TOTAL=79.147915298锛屼繚鐣橳OTAL=80.274135338銆備袱涓柊POR纾佺洏閲嶈浇绮剧‘锛屽浐瀹歅ERM/SW鏁扮粍鍚堝苟瀹炴祴锛涘師妯″瀷涓庡叾浠栨姌鏈敼銆傚悗缁粨鏋?瀹屾暣浜旀姌鏈繍琛岋紝褰撳墠鏃犳柊澧為槦鍒椼€傝褰時esults/deep_learning/por_architecture_v18_20260928/FINAL.md銆?


## 2026-09-28T20:27:53.845030+08:00 v19鏄惧紡鍚姩
[{"mode": "regional", "pid": 43316, "started_at": "2026-09-28T20:27:53.835030+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_pyramid_v19.py", "--mode", "regional", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_pyramid_v19_regional_fold5"]}, {"mode": "global", "pid": 46940, "started_at": "2026-09-28T20:27:53.843030+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_pyramid_v19.py", "--mode", "global", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_pyramid_v19_global_fold5"]}]
浠呬袱椤规柊POR璁粌锛屽叾浠栭€氶亾/鎶?浜や粯淇濇寔涓嶅彉锛涘疄闄呮垚缁╁緟杩愯缁撴灉銆?


## v19澶氬昂搴OR瀵圭収宸插畬鎴?
2026-09-28T20:35:12.522433+08:00銆俽egional TOTAL=79.551149548锛実lobal TOTAL=79.555720337锛屽綋鍓嶄繚鐣?0.274135338銆傛柊POR纾佺洏澶嶇幇鍙婂浐瀹氬叾浣欎袱閫氶亾鏁扮粍鍚堝苟瀹屾垚銆傚皻鏈繍琛屾柊瀵圭収/鍏朵粬鐩爣/鍘熷叾浠栨姌锛屽畬鏁碠OF>82鏈畬鎴愩€傝褰昿or_pyramid_v19_20260928/FINAL.md銆?


## 2026-09-28T20:49:20.172055+08:00 v20鏄惧紡鍚姩
[{"mode": "amplitude_early", "pid": 5840, "started_at": "2026-09-28T20:49:20.161054+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_feature_structure_v20.py", "--mode", "amplitude_early", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_feature_structure_v20_amplitude_early_fold5"]}, {"mode": "shape_early", "pid": 36524, "started_at": "2026-09-28T20:49:20.171054+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_feature_structure_v20.py", "--mode", "shape_early", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_feature_structure_v20_shape_early_fold5"]}]
棰勫畾涔変笁椤逛箣涓€鎴栦袱椤癸紝鏈€澶?GPU锛涙湭鍚姩鐨勯」鐩笉鍦ㄨ嚜鍔ㄩ槦鍒楋紝闇€鏈鏄惧紡璋冪敤銆?


## 2026-09-28T20:53:03.503313+08:00 v20鏄惧紡鍚姩
[{"mode": "shape_late", "pid": 11960, "started_at": "2026-09-28T20:53:03.499797+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_feature_structure_v20.py", "--mode", "shape_late", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_feature_structure_v20_shape_late_fold5"]}]
棰勫畾涔変笁椤逛箣涓€鎴栦袱椤癸紝鏈€澶?GPU锛涙湭鍚姩鐨勯」鐩笉鍦ㄨ嚜鍔ㄩ槦鍒楋紝闇€鏈鏄惧紡璋冪敤銆?


## v20鐗瑰緛涓庤瀺鍚堝疄楠屽畬鎴?
2026-09-28T20:56:42.218786+08:00銆備笁椤瑰叏閮ㄥ畬鎴愬苟閲嶈浇锛岀壒寰乀OTAL宸?0.152891789锛屾櫄铻嶅悎TOTAL宸?0.080073697锛涗繚鐣橳OTAL=80.274135338銆備粎鏈壒鏂癙OR锛屽叾浠栫洰鏍?鎶?浜や粯涓嶅彉锛涘己楠ㄥ共鐗瑰緛鎺ュ叆涓庡畬鏁碠OF灏氭湭鍋氥€傝瘉鎹甪eature_structure_v20_20260928/FINAL.md銆?


## 2026-09-28T21:10:19.989243+08:00 v21鏄惧紡鍚姩
[{"mode": "base1", "pid": 38952, "started_at": "2026-09-28T21:10:19.980197+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scaling_v21.py", "--target", "POR", "--architecture", "dual", "--mode", "base1", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_scaling_v21_base1_fold5"]}, {"mode": "ffn2", "pid": 42584, "started_at": "2026-09-28T21:10:19.987728+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scaling_v21.py", "--target", "POR", "--architecture", "dual", "--mode", "ffn2", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_scaling_v21_ffn2_fold5"]}]
浠呴瀹氫箟3瑙勬ā涔嬩竴鎴栦袱椤癸紝鏈€澶?GPU锛屽叾浠栧皻鏈惎鍔ㄩ」涓嶈嚜鍔ㄦ帓闃熴€?


## 2026-09-28T21:14:49.566168+08:00 v21鏄惧紡鍚姩
[{"mode": "ffn4", "pid": 50684, "started_at": "2026-09-28T21:14:49.561607+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scaling_v21.py", "--target", "POR", "--architecture", "dual", "--mode", "ffn4", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_scaling_v21_ffn4_fold5"]}]
浠呴瀹氫箟3瑙勬ā涔嬩竴鎴栦袱椤癸紝鏈€澶?GPU锛屽叾浠栧皻鏈惎鍔ㄩ」涓嶈嚜鍔ㄦ帓闃熴€?


## v21鍐犲啗鏀剁泭涓庢墿瀹瑰疄楠屽畬鎴?
2026-09-28T21:18:10.261733+08:00銆?/2/4鍐呴儴瀹藉害涓夐」璁粌/鏈€浣虫仮澶?瀹為檯棰勬祴缁勫悎鍧囧畬鎴愶紱閲囩撼鏂癙OR=False锛屼繚鐣橳OTAL=80.274135338銆傚凡瀹屾垚鏀剁泭鏉ユ簮/鍙傛暟鍜屽紑閿€璁板綍锛涙暣缃戠粺涓€鎷撳銆佸叾浠栫洰鏍囨墿瀹广€佸叾浠栨姌鍧囨湭鎵ц銆傝瘉鎹甤hampion_scaling_v21_20260928/FINAL.md銆?


## 2026-09-28T21:41:47.854539+08:00 v22 explicit launch
[{"mode": "ffn2", "pid": 50536, "started_at": "2026-09-28T21:41:47.842778+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_frozen_growth_v22.py", "--mode", "ffn2", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_frozen_growth_v22_ffn2_fold5"]}, {"mode": "ffn4", "pid": 49624, "started_at": "2026-09-28T21:41:47.853151+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_frozen_growth_v22.py", "--mode", "ffn4", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_frozen_growth_v22_ffn4_fold5"]}]
Only the two predefined jobs; no queue, other folds or prior-source modifications.


## v22 鍥哄畾鍘熸潈閲嶆墿瀹瑰疄楠屽凡瀹屾垚
2026-09-28T21:47:56.088940+08:00銆?鍊嶆洿鏂板悗TOTAL=80.269258294锛?鍊?80.263249587銆傚潎18杞仠姝?鏈€浣崇0杞紝淇濈暀80.274135338銆傛柊閫氳矾姊害銆佹棫鍙傛暟鍐荤粨銆佽缁冨悗妫€鏌ョ偣鍏?44665鐐圭鐩橀噸杞介€氳繃銆傛湭杩愯鍏朵粬鐩爣/鍘熷叾浠栨姌锛屼笉鏀逛氦浠橈紱璇﹁frozen_growth_v22_20260928/FINAL.md銆?


## 2026-09-28T22:08:53.041374+08:00 v23鏄惧紡鍚姩
[{"mode": "base1", "pid": 52504, "started_at": "2026-09-28T22:08:53.026053+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scratch_scaling_v23.py", "--mode", "base1", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_scratch_scaling_v23_base1_fold5"]}, {"mode": "ffn2", "pid": 52524, "started_at": "2026-09-28T22:08:53.039378+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_scratch_scaling_v23.py", "--mode", "ffn2", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_scratch_scaling_v23_ffn2_fold5"]}]
浠呰繖涓や釜棰勫畾涔変换鍔★紝鏃犲叾浠栨姌鎴栬嚜鍔ㄨ拷鍔犮€?


## v23 浠庡ご鍙屼笓瀹舵墿瀹瑰鐓у凡瀹屾垚
2026-09-28T22:25:40.881346+08:00銆俠ase1 TOTAL=79.548859162锛沠fn2 TOTAL=79.541070769锛涘閲忓樊-0.007788393锛屽綋鍓嶄繚鐣?0.274135338銆備袱妯″瀷浠庨殢鏈烘潈閲嶈缁?鏈€浣崇鐩橀噸杞?璁粌涓庨獙璇佽瘖鏂潎瀹屾垚銆傚叾浠栫洰鏍?鍘熷叾浠栨姌鍙婂師澶氫换鍔″畬鏁撮厤鏂规湭鎵ц锛涙棤鑷姩鍚庣画銆傝褰晄cratch_scaling_v23_20260928/FINAL.md銆?


## 2026-09-28T22:37:56.836768+08:00 v24鏄惧紡鍚姩
[{"mode": "mixture", "pid": 19424, "started_at": "2026-09-28T22:37:56.823027+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_expert_supervision_v24.py", "--mode", "mixture", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_expert_supervision_v24_mixture_fold5"]}, {"mode": "independent", "pid": 52460, "started_at": "2026-09-28T22:37:56.834258+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_expert_supervision_v24.py", "--mode", "independent", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_expert_supervision_v24_independent_fold5"]}]
涓ら」棰勫畾涔変换鍔★紝鏃犲叾浠栧€欓€?鎶樻垨鑷姩缁窇銆?


## v24 涓撳鐩戠潱瀵圭収宸插畬鎴?
2026-09-28T22:53:42.761471+08:00銆俶ixture TOTAL=79.492115838锛宨ndependent TOTAL=79.897835986锛涘尮閰嶅樊+0.405720148锛屼繚鐣?0.274135338銆備袱妯″瀷姊害闅旂/鏈€浣崇鐩橀噸杞?瀹為檯璇勫垎瀹屾垚锛屾湭鏀逛氦浠樻垨琛ユ姌銆傝瑙乪xpert_supervision_v24_20260928/FINAL.md銆?


## 2026-09-28T23:05:51.487093+08:00 v25鏄惧紡鍚姩
[{"mode": "amplitude", "pid": 24608, "started_at": "2026-09-28T23:05:51.474232+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_shape_v25.py", "--mode", "amplitude", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_shape_v25_amplitude_fold5"]}, {"mode": "shape", "pid": 16476, "started_at": "2026-09-28T23:05:51.486583+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_shape_v25.py", "--mode", "shape", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_shape_v25_shape_fold5"]}]
浠呬袱椤圭洿鎺ュ湪鍐犲啗涓婃敼杩涳紱鍋滄鍐峰惎鍔?鎵╁寤朵几锛屾棤鑷姩杩藉姞銆?


## v25 鍐犲啗鍘熸潈閲嶄笂鐨勪綅缃慨姝ｅ凡瀹屾垚
2026-09-28T23:11:49.047450+08:00銆俛mplitude=80.295581058278锛宻hape=80.274135337791锛屼繚鐣欑爺绌禩OTAL=80.295581058278銆備粎12816鏂板弬鏁帮紝鏃фā鍨嬪喕缁撳拰鏇存柊鍚庢渶浣冲叏閲忛噸杞藉凡楠岃瘉銆傚喎鍚姩/鎵╁鍋滄寤朵几锛屾棤鍏朵粬鎶?鏂伴槦鍒?浜や粯鏇存敼銆傝瑙乧hampion_shape_v25_20260928/FINAL.md銆?


## 2026-09-28T23:25:08.153393+08:00 v26鏄惧紡鍒嗘壒鍚姩
[{"target": "PERM", "pid": 8052, "started_at": "2026-09-28T23:25:08.139393+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_all_heads_v26.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_all_heads_v26_PERM_fold5"]}, {"target": "SW", "pid": 12916, "started_at": "2026-09-28T23:25:08.152394+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_all_heads_v26.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_all_heads_v26_SW_fold5"]}]
浠呮樉寮忔寚瀹氱洰鏍囷紝鍓╀綑鐩爣涓嶆帓闃燂紝鏈€澶?GPU銆?


## 2026-09-28T23:28:42.187729+08:00 v26鏄惧紡鍒嗘壒鍚姩
[{"target": "POR", "pid": 13756, "started_at": "2026-09-28T23:28:42.185678+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_all_heads_v26.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_all_heads_v26_POR_fold5"]}]
浠呮樉寮忔寚瀹氱洰鏍囷紝鍓╀綑鐩爣涓嶆帓闃燂紝鏈€澶?GPU銆?


## v26 涓夌洰鏍囪緭鍑哄ご浼樺寲瀹屾垚
2026-09-28T23:33:01.038838+08:00銆?1椤瑰畬鎴愶紝瀹為檯缁勫悎TOTAL=80.300730612039锛岀浉瀵箆25+0.005149553761锛涢噰鐢≒OR:None,PERM:gate,SW:weight銆傛柊閲囩撼鐩爣鍏?44665鐐圭嫭绔嬬鐩橀噸杞斤紝鏃ф簮鏉冮噸涓庝氦浠樹笉鏀癸紝鏃犲叾浠栨姌銆傝瘉鎹產ll_heads_v26_20260928/FINAL.md銆?


## v27鍩轰簬80.30鐨勫悗楠屾潯浠跺閲?
褰撳墠鐢ㄦ埛瑕佹眰缁х画鍩轰簬80.30浼樺寲銆傛湰鎵归瀹氫箟涓夌洰鏍囧悇self/pair鍏」锛屽喕缁搗26鍏ㄩ娴嬪櫒锛屼粎鏈夋晥浣嶇疆/闂ㄦ帶/鏉冮噸鍥犲瓙鐨勫皬澶存洿鏂帮紱鍚屽弬鏁板悓鍒濆鐘舵€侊紝鍙樊绗簩浠藉悗楠岃緭鍏ユ潵婧愩€傚師鏁版嵁/Q50/鍒嗘姌/璇勫垎/80-18/max2淇濇寔锛屾棤鍐峰惎鍔?鎵╁/鏃ч槦鍒楁垨鍏朵粬鎶樸€傜淮鎶hampion_posterior_v27_20260929/PROJECTS.md銆?


## 鏄惧紡鍚姩
{"target": "PERM", "pid": 38060, "started_at": "2026-09-29T09:02:29.435013+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_posterior_v27.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_posterior_v27_PERM_fold5"]}
鍏朵綑鐩爣涓嶈嚜鍔ㄦ帓闃熴€?


## 鏄惧紡鍚姩
{"target": "SW", "pid": 18420, "started_at": "2026-09-29T09:02:29.444228+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_posterior_v27.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_posterior_v27_SW_fold5"]}
鍏朵綑鐩爣涓嶈嚜鍔ㄦ帓闃熴€?


## 鏄惧紡鍚姩
{"target": "POR", "pid": 40312, "started_at": "2026-09-29T09:05:48.054834+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_posterior_v27.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_posterior_v27_POR_fold5"]}
鍏朵綑鐩爣涓嶈嚜鍔ㄦ帓闃熴€?


## v27鍩轰簬80.30鐮旂┒瀹屾垚
鍏」瀹屾垚锛孴OTAL=80.300730612039锛岀浉瀵圭埗閰嶆柟+0.000000000000銆傞噰鐢≒OR:None,PERM:None,SW:None銆傚師鏉冮噸/浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紱瀹屾暣璁板綍champion_posterior_v27_20260929/FINAL.md銆?


## v28褰撳墠80.30鐨勮矾鐢辨洿鏂拌寖鍥村鐓?
缁х画浼樺寲鎺堟潈涓嬶紝鏈壒棰勫畾涔変笁鐩爣joint/conditional鍏」锛涘彧鏂板鍐呴儴璺敱log-odds灏忓ご锛屽師涓撳鍜寁25/v26澶村喕缁撱€傚師鏁版嵁/Q50/瑙ｇ爜/80-18/max2淇濇寔锛屾樉寮忓垎鎵癸紝鏃犲叾浠栨姌銆佸喎鍚姩銆佹墿瀹瑰拰浜や粯瑕嗙洊銆傝瑙乧hampion_route_v28_20260929/PROJECTS.md銆?


## 鏄惧紡鍚姩
{"target": "POR", "pid": 58564, "started_at": "2026-09-29T09:22:35.460385+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_route_v28.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_route_v28_POR_fold5"]}
鍏朵綑鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "SW", "pid": 15448, "started_at": "2026-09-29T09:22:35.473030+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_route_v28.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_route_v28_SW_fold5"]}
鍏朵綑鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "PERM", "pid": 44572, "started_at": "2026-09-29T09:26:30.411980+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_route_v28.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_champion_route_v28_PERM_fold5"]}
鍏朵綑鐩爣涓嶆帓闃熴€?


## v28璺敱瀵圭収瀹屾垚
鍏」瀹屾垚锛涘師绗簲鎶楾OTAL=80.324397278674锛岀浉瀵箆26+0.023666666635銆侾OR:conditional, PERM:joint, SW:conditional銆傚師妯″瀷/浜や粯涓嶅彉锛屾棤鍏朵粬鎶樸€傝褰昪hampion_route_v28_20260929/FINAL.md銆?


## v29缁х画褰撳墠鏈夋晥妯″瀷锛氭湁闄愯矾鐢辫В鑰?
鐢ㄦ埛缁х画浼樺寲锛屾湰鎵瑰浐瀹歷28 TOTAL80.324397锛屼笁鐩爣coupled/split鍏」銆傚叏閮ㄧ埗鏉冮噸鍐荤粨锛屾柊璺敱澧為噺卤0.125锛屽悓20546鍙傛暟鍜屽垵濮嬪嚱鏁帮紱鍘熸暟鎹?Q50/璇勫垎/80-18/max2淇濇寔銆備笉鎭㈠鍐峰惎鍔?鎵╁锛屼笉琛ユ姌鎴栬鐩栦氦浠樸€傜淮鎶hampion_split_route_v29_20260929/PROJECTS.md銆?


## 鏄惧紡鍚姩
{"target": "PERM", "pid": 47596, "started_at": "2026-09-29T10:04:36.674767+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_split_route_v29.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_split_route_v29_PERM_fold5"]}
鍏朵粬鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "POR", "pid": 47372, "started_at": "2026-09-29T10:04:36.682768+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_split_route_v29.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_split_route_v29_POR_fold5"]}
鍏朵粬鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "SW", "pid": 46916, "started_at": "2026-09-29T10:08:39.173200+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_champion_split_route_v29.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_split_route_v29_SW_fold5"]}
鍏朵粬鐩爣涓嶆帓闃熴€?


## v29璺敱瑙ｈ€﹀畬鎴?
TOTAL=80.335082954320锛岀浉瀵箆28+0.010685675646銆侾OR:split, PERM:coupled, SW:coupled銆傚師妯″瀷涓庝氦浠樹笉鏀癸紝鏃犲叾浠栨姌銆傝瘉鎹甤hampion_split_route_v29_20260929/FINAL.md銆?


## v30宸叉湁鏈夋晥妯″潡鍗忓悓寰皟
鏈壒浠巚29 TOTAL80.335083鍑哄彂锛屼笁鐩爣route_only/coadapt鍏」锛涙棤鏂板瀛︿範妯″潡锛屼繚鐣欎富楠ㄥ共锛屼粎璺敱鎴栬矾鐢变笌宸叉湁鏁堣緭鍑轰慨姝ｅ叡鍚屾洿鏂般€傜粺涓€3e-5銆佸垎缁刢lip2锛屽師鏁版嵁/Q50/璇勫垎/80-18/max2涓嶅彉銆傛棤鍏朵粬鎶樸€佹棫闃熷垪鎴栦氦浠樿鐩栥€傜淮鎶hampion_coadapt_v30_20260929/PROJECTS.md銆?


## 鏄惧紡鍚姩
{"target": "POR", "pid": 46016, "started_at": "2026-09-29T10:21:58.406888+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_coadapt_v30.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_coadapt_v30_POR_fold5"]}
鍏朵粬鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "PERM", "pid": 15132, "started_at": "2026-09-29T10:21:58.416400+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_coadapt_v30.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_coadapt_v30_PERM_fold5"]}
鍏朵粬鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "SW", "pid": 56556, "started_at": "2026-09-29T10:25:11.742773+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_coadapt_v30.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_coadapt_v30_SW_fold5"]}
鍏朵粬鐩爣涓嶆帓闃熴€?


## v30宸叉湁妯″潡寰皟瀹屾垚
瀹為檯TOTAL=80.336211945752锛岀浉瀵箆29+0.001128991432銆侾OR:coadapt, PERM:None, SW:None銆傛棤鏂扮粨鏋勩€佸叾浠栨姌鎴栨棫闃熷垪锛涘師浜や粯淇濈暀銆傝褰昪hampion_coadapt_v30_20260929/FINAL.md銆?


## v30鍚庝紭鍖栨柟鍚戣瘖鏂紙鏃犺缁冿級
2026-09-29T10:50:34.147292+08:00銆傚彧瀹屾垚褰撳墠棰勬祴璇樊鍒嗚В涓?4璁粌浜?56鍥哄畾绐楀彛璺敱楗卞拰缁熻锛屽綋鍓峊OTAL80.336211945752涓嶅彉銆傝瘖鏂細POR杩炵画璺敱88.47%鐐硅Е鍙?5%浠ヤ笂杈圭晫锛汼W杩炵画鏁板€艰宸负涓昏澶卞垎锛決50瑕嗙洊浠?.303%锛屾繁搴︾害0.1绫冲潎鍖€銆備笅涓€姝ュ€欓€変负鍚岃竟鐣岄噸鍙傛暟鍖栥€佸崟璋冩潯浠跺垎甯冧慨姝ｃ€佽法鏇茬嚎鍗忓悓鐗瑰緛锛涘潎鏈缁冦€傛姤鍛妑esults/deep_learning/optimization_review_v30_20260929/README.md锛涗笉琛ユ姌銆佷笉鏀规潈閲嶅拰浜や粯銆?


## v31鐢ㄦ埛閲囩撼璇婃柇鏈哄埗楠岃瘉
鍩轰簬80.336211945752锛屼粎POR鍚岃竟鐣宼anh/anchored鍜孲W affine/pwl4鍏卞洓椤广€侾ERM鍐荤粨锛屽師9鏇茬嚎/鏁版嵁/Q50/80-18/max2/瑙ｇ爜涓嶅彉銆傛柊鐩爣鍙樻崲璁″叆瀹屾暣瀵嗗害Jacobian锛涗笉寰楀皢鍙傛暟鍖栧彉鍖栬绠楁垚鍒濆鏀剁泭銆備笉琛ユ姌銆佹棫浜や粯淇濈暀銆傜淮鎶echanism_checks_v31_20260929/PROJECTS.md銆?


## 鏄惧紡鍚姩
{"target": "POR", "pid": 43332, "started_at": "2026-09-29T11:01:15.232706+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_mechanism_v31.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_mechanism_v31_POR_fold5"]}
娌℃湁鑷姩闃熷垪銆?


## 鏄惧紡鍚姩
{"target": "SW", "pid": 3320, "started_at": "2026-09-29T11:04:32.728381+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_mechanism_v31.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_mechanism_v31_SW_fold5"]}
娌℃湁鑷姩闃熷垪銆?


## v31鏈夐檺鏈哄埗楠岃瘉瀹屾垚
TOTAL=80.336513436816锛岀浉瀵箆30+0.000301491065锛汸OR:anchored, SW:None銆傚洓椤圭粨鏉燂紝鏃犲叾浠栨姌锛涜瘉鎹甿echanism_checks_v31_20260929/FINAL.md銆?


## v32璺ㄦ洸绾垮眬閮ㄧ浉鍏崇壒寰佸鐓?
缁х画鎺堟潈涓嬶紝鏈壒浠巚31 TOTAL80.336513436816寮€濮嬶紝涓夌洰鏍噈arginal/crosscorr鍏」锛屽悓缁撴瀯鍚屽垵濮嬪寲锛涘師妯″瀷鍐荤粨锛屼粎鏈夐檺灏忓ご璇诲彇18缁寸浉鍏虫垨宸叉湁缁熻鍐椾綑瀵圭収銆傚師鏁版嵁/Q50/璇勫垎/80-18/max2涓嶅彉锛涙殏涓嶅姞鏄惧紡浜や簰锛屼笉琛ユ姌銆佷笉鏀逛氦浠樸€傜淮鎶rosscurve_features_v32_20260929/PROJECTS.md銆?


## 鏄惧紡鍚姩
{"target": "POR", "pid": 27856, "started_at": "2026-09-29T11:31:56.457755+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_crosscurve_v32.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_crosscurve_v32_POR_fold5"]}
鍏朵綑鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "SW", "pid": 35624, "started_at": "2026-09-29T11:31:56.465487+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_crosscurve_v32.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_crosscurve_v32_SW_fold5"]}
鍏朵綑鐩爣涓嶆帓闃熴€?


## 鏄惧紡鍚姩
{"target": "PERM", "pid": 45624, "started_at": "2026-09-29T11:35:14.119335+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_crosscurve_v32.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_crosscurve_v32_PERM_fold5"]}
鍏朵綑鐩爣涓嶆帓闃熴€?


## v32璺ㄦ洸绾跨壒寰佸鐓у畬鎴?
TOTAL=80.336513436816锛岀浉瀵箆31+0.000000000000锛汸OR:None, PERM:None, SW:None銆傚叚椤瑰畬鎴愶紝鏃犲叾浠栨姌銆傝瘉鎹甤rosscurve_features_v32_20260929/FINAL.md銆?


## v33 杈撳叆杈圭晫涓婁笅鏂囨湁闄愬鐓?
褰撳墠缁х画浼樺寲瀹炴柦POR/SW鍚刦ixed/segmented鍏卞洓椤癸紱PERM浠呰瘖鏂笉璁粌銆傚喕缁搗31鍏ㄩ娴嬪櫒锛屽師鏁版嵁/Q50/璇勫垎/80-18/max2淇濇寔銆傜浉鍚屽皬澶淬€佸垵濮嬪寲鍜岀缉鏀撅紝鍙敼18缁寸粺璁＄獥鍙ｆ槸鍚﹁法杈撳叆杈圭晫銆傝竟鐣岄槇鍊艰缁冭緭鍏ユ嫙鍚堬紝闈炴爣绛惧眰浣嶏紱鏃犲喎鍚姩/鎵╁/鍏朵粬鎶樻垨浜や粯瑕嗙洊銆傜淮鎶oundary_context_v33_20260929/PROJECTS.md銆?


## v33 launch POR
{"target": "POR", "pid": 41828, "started_at": "2026-09-29T12:27:30.213087+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_boundary_v33.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_boundary_v33_POR_fold5"]}
鏃犺嚜鍔ㄦ帓闃熸垨琛ユ姌銆?


## v33 launch SW
{"target": "SW", "pid": 17876, "started_at": "2026-09-29T12:27:30.226215+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_boundary_v33.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_boundary_v33_SW_fold5"]}
鏃犺嚜鍔ㄦ帓闃熸垨琛ユ姌銆?


## v33杈撳叆杈圭晫瀵圭収瀹屾垚
鍥涢」瀹屾垚锛孴OTAL=80.336513436816锛屽閲?+0.000000000000锛涙柊閲囩敤=[]銆傝緭鍏ヨ竟鐣屼笌澶卞垎鐩稿叧锛岀粺璁℃埅鏂殑瀹為檯缁撴灉瑙乥oundary_context_v33_20260929/FINAL.md銆傛棤鍏朵粬鎶樸€佹棫浠诲姟鎴栦氦浠樿鐩栥€?


## v33鏈€缁堢粨璁?
鍥涢」宸插畬鎴愶紝褰撳墠TOTAL80.336513436816涓嶅彉锛屾棤娲诲姩璁粌銆傝緭鍏ヨ繎杈圭晫闈為粯璁ゆ牱鏈緢闅句絾鎬讳綋璐＄尞鏈夐檺锛氳繎卤5鐐瑰け鍒嗗崰2.08%锛岃繙>25鐐瑰崰87.92%銆備笉缁х画纭埅鏂粺璁℃壂鎻忋€傛姤鍛妑esults/deep_learning/boundary_context_v33_20260929/FINAL.md銆傛湭琛ュ叾浠栨姌銆佹湭淇敼鏃т氦浠樸€?


## v34 宸叉巿鏉冭法鐩爣鍗忓悓璇曢獙
鍘?4璁粌浜曞唴閮?缁勪氦鍙夋嫙鍚堣緟鍔╂ā鍨嬶紙姣忕粍40鎷熷悎/8鏃╁仠/16鐣欏嚭锛夛紝鏉ユ簮浜曞悓鏃朵笉鍙備笌鎷熷悎鍜岄€夋嫨锛涗富SW鍥哄畾銆傚彧寮€灞晄elf/cross/shuffled涓夐」鍚屽閲忓ご锛岃礋瀵圭収涓嶉儴缃诧紱鏃у師鍏朵粬鎶?浜や粯涓嶆敼銆傝瑙乧ross_target_v34_20260929/PROJECTS.md銆?


## v34 launch pure_dl_cross_target_v34_aux0
{"pid": 46852, "kind": "aux", "group": 0, "started_at": "2026-09-29T13:21:27.747626+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\auxiliary_crossfit_v34.py", "--group", "0"]}
No automatic queue or restart.


## v34 launch pure_dl_cross_target_v34_aux1
{"pid": 43256, "kind": "aux", "group": 1, "started_at": "2026-09-29T13:21:27.787319+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\auxiliary_crossfit_v34.py", "--group", "1"]}
No automatic queue or restart.


## v34 launch pure_dl_cross_target_v34_aux2
{"pid": 42080, "kind": "aux", "group": 2, "started_at": "2026-09-29T13:22:57.036862+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\auxiliary_crossfit_v34.py", "--group", "2"]}
No automatic queue or restart.


## v34 launch pure_dl_cross_target_v34_aux3
{"pid": 61252, "kind": "aux", "group": 3, "started_at": "2026-09-29T13:23:46.513319+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\auxiliary_crossfit_v34.py", "--group", "3"]}
No automatic queue or restart.


## v34 launch pure_dl_cross_target_v34_SW_fold5
{"pid": 31624, "kind": "head", "group": null, "started_at": "2026-09-29T13:27:17.756778+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_cross_target_v34.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_cross_target_v34_SW_fold5"]}
No automatic queue or restart.


## v34璺ㄧ洰鏍囧崗鍚屽畬鎴?
12涓簳澶栬緟鍔╂ā鍨嬪強3椤筍W澶村畬鎴愶紝TOTAL=80.336513436816锛屽閲?+0.000000000000銆傞噰鐢?None锛涘井鍨嬭緟鍔╀俊鍙凤紝涓嶆槸寮烘ā鍨嬪叏閮∣OF閲嶈銆傛棤鍘熷叾浠栨姌/浜や粯鍙樻洿銆傝瑙乧ross_target_v34_20260929/FINAL.md銆?


## v35 璺ㄤ簳鍙傝€冨瘑搴︽湁闄愬疄楠?
褰撳墠缁х画浼樺寲锛屽浐瀹歷31 TOTAL80.336513锛涗粎SW matched/shuffled涓や釜鍚屽閲忔绱㈠瘑搴﹀垎鏀紝鍙傝€冨簱4096涓缁冮潪榛樿鐐癸紝姣忔煡璇㈡帓闄よ嚜韬暣浜曪紝鍥哄畾Top16鍚庡涔犻噸鎺掑簭銆傚師榛樿姒傜巼/楠ㄥ共/鏈夋晥澶翠笉鍙橈紝鍙傝€冩潯浠跺瘑搴︽贩鍚堟瘮渚?鑷?.25銆傚師Q50/璇勫垎/80-18/max2淇濇寔锛屼笉琛ユ姌銆佷笉鎵弿搴撳ぇ灏忔垨鎭㈠鏃ц矾绾裤€傜淮鎶etrieval_density_v35_20260929/PROJECTS.md銆?


## v35鏄惧紡鍚姩
{"pid": 62064, "started_at": "2026-09-29T14:07:10.148637+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_retrieval_v35.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_retrieval_density_v35_SW_fold5"], "automatic_queue": false}


## v36妫€绱㈤棬鎺т慨姝ｇ増
褰撳墠缁х画鎺堟潈涓嬶紝v35鏈€佹绱㈠垎鏀け娲伙紝淇濆瓨璇佹嵁涓嶆敼鍘嗗彶銆倂36閲囩敤骞虫粦1%璧风偣銆佹渶澶?5%鐨刴atched/shuffled涓や釜SW鍊欓€夛紱鍘熸ā鍨嬪崟鐙洖閫€锛屼笉鍐掔О鍒濆鍑芥暟鐩稿悓銆傚師鏁版嵁/Q50/瑙ｇ爜/80-18/max2/鏁翠簳鍙傝€冩帓闄や繚鎸侊紝鏃犲叾浠栨姌鎴栦氦浠樿鐩栥€傝retrieval_smooth_gate_v36_20260929/PROJECTS.md銆?


## v36鏄惧紡鍚姩
{"pid": 67396, "started_at": "2026-09-29T14:22:06.320868+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_retrieval_smooth_v36.py", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_retrieval_smooth_v36_SW_fold5"], "automatic_queue": false}


## v36妫€绱㈤棬鎺у鐓у畬鎴?
涓ら」宸插畬鎴愶紝TOTAL=80.336513436816锛屽閲?+0.000000000000锛涢噰鐢?None銆倂35澶辨椿璇佹嵁鍜寁36鏈夋晥璁粌鍖哄埆宸茶褰曘€傛墍鏈夋棫妯″瀷/浜や粯淇濈暀锛屾棤鍏朵粬鎶橈紱瑙乺etrieval_smooth_gate_v36_20260929/FINAL.md銆?


## v37 SW 鏄惧紡鍚姩
{"pid": 64628, "target": "SW", "started_at": "2026-09-29T14:32:10.484297+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quality_readout_v37.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quality_readout_v37_SW_fold5"], "automatic_queue": false}


## v37 POR 鏄惧紡鍚姩
{"pid": 62660, "target": "POR", "started_at": "2026-09-29T14:32:32.394994+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quality_readout_v37.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quality_readout_v37_POR_fold5"], "automatic_queue": false}


## v37璐ㄩ噺鏉冮噸瀵圭収瀹屾垚
鍥涢」瀹屾垚锛孴OTAL=80.336513436816锛岀浉瀵?0.336513澧炲姞+0.000000000000锛涢噰鐢╗]銆傚畬鏁磋褰晀uality_readout_v37_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯鏀瑰姩銆?


## v37 closure and storage
The four candidates finished and restored successfully; no candidate was adopted. Keep the v31 recipe at TOTAL80.336513436816. Uniform weighting has only small post-update relative gains over Q50, not an improvement over the parent. Do not expand this bounded all-factor head sweep. Next independent hypothesis to assess: train a small direct action-value readout against the original scoring kernel, with the original density decoder retained as control; this is not implemented or launched yet. Full OOF>82 remains unachieved; no other folds or old queues started.
Removed 22 non-adopted checkpoints/cache files totaling104118150 bytes across v35-v37, with SHA256/size manifests. Sources, configurations, compact predictions, reports and all baseline/deployment dependencies remain.


## v38 SW 鏄惧紡鍚姩
{"pid": 55412, "target": "SW", "started_at": "2026-09-29T14:45:48.281608+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_action_value_v38.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_action_value_v38_SW_fold5"], "automatic_queue": false}


## v38 POR 鏄惧紡鍚姩
{"pid": 14968, "target": "POR", "started_at": "2026-09-29T14:46:06.946657+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_action_value_v38.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_action_value_v38_POR_fold5"], "automatic_queue": false}


## Invalid POR objective stopped
Wrong inverse exp instead of expm1 for POR log1p labels. PID14968 stopped after epoch4; all recorded gains are invalid for method attribution and excluded from selection. Historical source/config/checkpoints not mutated. New corrected run restarts from the original baseline, not this state. SW uses log and exp correctly and continues unchanged.


## v38 corrected POR 鏄惧紡鍚姩
{"pid": 23728, "target": "POR", "started_at": "2026-09-29T14:49:07.153015+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_action_value_v38_por_fixed.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_action_value_v38_POR_fixed_fold5"], "automatic_queue": false}


## v38鐩存帴鍔ㄤ綔濂栧姳瀵圭収瀹屾垚
鍥涢」瀹屾垚锛孴OTAL=80.336513436816锛岀浉瀵?0.336513澧炲姞+0.000000000000锛涢噰鐢╗]銆傚畬鏁磋褰昦ction_value_v38_20260929/FINAL.md锛涢敊璇疨OR鍒濇杩愯琚墧闄わ紝鏃犲叾浠栨姌鎴栦氦浠樻敼鍔ㄣ€?


## v38 closure and retention
All four valid candidates stopped after18 non-improving epochs and restored epoch0 exactly. Keep baseline TOTAL80.336513436816. Original POR wrong-inverse run excluded; corrected2048-point training-kernel max error4.8336e-6, SW also checked at finalization. Original launcher now only admits SW; POR has a separate corrected launcher. No selected replacement.
15 negative/invalid checkpoints totaling30001900 bytes deleted after final verification, SHA256/size manifest saved. Compact source/config/log/score/prediction and failure evidence retained, all baseline checkpoints preserved. No live training, automatic queue, extra folds or delivery changes. FullOOF>82 remains unachieved.
Do not expand the v37/v38 frozen output-head sweeps. Next work should assess a different update location inside the retained private backbone, or measure training-vs-validation feature behavior to justify it; this is a pending hypothesis, not an implemented or launched experiment.


## v39 decoder_norm 鏄惧紡鍚姩
{"pid": 65028, "mode": "decoder_norm", "started_at": "2026-09-29T15:01:32.810254+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_backbone_norm_v39.py", "--mode", "decoder_norm", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_backbone_norm_v39_SW_decoder_norm_fold5"], "automatic_queue": false}


## v39 all_norm 鏄惧紡鍚姩
{"pid": 54588, "mode": "all_norm", "started_at": "2026-09-29T15:01:59.405436+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_backbone_norm_v39.py", "--mode", "all_norm", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_backbone_norm_v39_SW_all_norm_fold5"], "automatic_queue": false}


## v39楠ㄥ共褰掍竴鍖栧鐓у畬鎴?
涓ら」瀹屾垚锛孴OTAL=80.336513436816锛屽閲?0.000000000000锛岄噰鐢?None銆傛棤鍏朵粬鎶樻垨浜や粯淇敼锛涜backbone_norm_v39_20260929/FINAL.md銆?


## v39 closure and next hypothesis
Both SW normalization scopes completed, restored baseline and passed fixed-state/prediction checks. TOTAL80.336513436816 remains current; no extra folds, active training or delivery replacement. Deleted4 redundant checkpoints totaling8015925 bytes with manifest; source/config/logs/compact predictions/score evidence and all baseline dependencies retained.
No further frozen-output-head or normalization-only LR sweep is planned. A different hypothesis for a later isolated study is training-well nuisance dependence: compare matched normalization adaptation with a training-only well discriminator gradient blocked versus reversed. Well identity would be auxiliary supervision only, never an input to the prediction, and no validation wells would train the discriminator. This proposal is not implemented or launched, and no gain is claimed. It requires separate scope/gradient/loss records if pursued. Full OOF>82 remains outstanding.


## v40 blocked 鏄惧紡鍚姩
{"pid": 66276, "mode": "blocked", "started_at": "2026-09-29T15:17:41.762513+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_invariance_v40.py", "--mode", "blocked", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_invariance_v40_SW_blocked_fold5"], "automatic_queue": false}


## v40 reversed 鏄惧紡鍚姩
{"pid": 22660, "mode": "reversed", "started_at": "2026-09-29T15:18:15.197884+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_invariance_v40.py", "--mode", "reversed", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_invariance_v40_SW_reversed_fold5"], "automatic_queue": false}


## v40楠ㄥ共褰掍竴鍖栧鐓у畬鎴?
涓ら」瀹屾垚锛孴OTAL=80.336513436816锛屽閲?0.000000000000锛岄噰鐢?None銆傛棤鍏朵粬鎶樻垨浜や粯淇敼锛涜well_invariance_v40_20260929/FINAL.md銆?


## Active-phase diagnostic and closure
First3 warmup SW scores are exactly equal. Both arms best-after-update is epoch1 before reversal starts. For the prescribed active epochs4..18, blocked best73.5954025454393 and reversed best73.59465943798016 (delta-0.000743107459). Final SW difference-0.006539743169. This diagnostic does not change checkpoint selection. Final domain accuracy blocked0.165039/reversed0.162842 vs majority prior0.022486: classifier learns, but the reversed objective does not improve this SW adaptation setting.
Do not extend this adversarial coefficient/curriculum scan. Next pending direction: inspect PERM conditional-density family/loss alternatives against the current retained PERM89.396724 model; preserve original actions/scoring and distinguish an initial distribution intervention from a retraining benefit. No new PERM run implemented or launched yet. Original fullOOF>82 and fifth>80.5 remain outstanding.


## v40 retention and terminal state
Both candidates finished with no adopted model. Deleted4 redundant checkpoints totaling9012676 bytes after verified restoration and fixed3column scoring; SHA256/size manifest saved. Source/config/log/compact predictions/diagnostics retained. No live training or queued launch, no other folds and no delivery changes. Current recipe still TOTAL80.336513436816.


## v41鏄惧紡鍚姩
{"pid": 53764, "started_at": "2026-09-29T15:44:58.654667+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_family_v41.py", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_family_v41_fold5"], "automatic_queue": false}


## v41 PERM鍒嗗竷鏃忓畬鎴?
Gaussian涓嶴tudent-t3鍧?8杞粨鏉燂紝鍧囧洖閫€鍘烶ERM銆傝缁冨悗鏈€浣?9.367194812859/89.372730778423锛岀埗妯″瀷89.396724049366銆傚疄闄呬笁鍒桾OTAL=80.336513436816涓嶅彉锛屾棤鍏朵粬鎶?鏃ч槦鍒楋紱璇︾粏璇佹嵁瑙乺esults/deep_learning/perm_family_v41_20260929/FINAL.md銆傜洰鏍囦粛鏈畬鎴愶紝涓嶅欢浼告湰璁剧疆鍒嗗竷鏃忔壂鎻忋€?

## v41璐熸潈閲嶆竻鐞?
宸插畬鎴愭仮澶嶅拰瀹為檯涓夊垪璇勫垎锛屽垹闄ゆ湰瀹為獙璐熷€欓€塩heckpoint鍏?涓€?224100瀛楄妭锛涙簮/閰嶇疆/璁粌鏃ュ織/绱у噾棰勬祴鍜屽師鍐犲啗渚濊禆淇濈暀銆傛竻鍗曡perm_family_v41_20260929/deletion_manifest.json銆傛棤娲诲姩璁粌鎴栬嚜鍔ㄩ槦鍒椼€?



## v42 free鏄惧紡鍚姩
{"pid": 61292, "mode": "free", "started_at": "2026-09-29T15:54:48.717088+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_backbone_v42.py", "--mode", "free", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_backbone_v42_SW_free_fold5"], "automatic_queue": false}


## v42 anchored鏄惧紡鍚姩
{"pid": 61076, "mode": "anchored", "started_at": "2026-09-29T15:55:09.156414+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_backbone_v42.py", "--mode", "anchored", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_backbone_v42_SW_anchored_fold5"], "automatic_queue": false}


## v42鏈鍧楀師鏉冮噸瀵圭収瀹屾垚
涓ら」瀹屾垚锛孴OTAL=80.336513436816锛屽閲?0.000000000000锛岄噰鐢?None銆傛棤鍏朵粬鎶樻垨浜や粯淇敼锛涜anchored_backbone_v42_20260929/FINAL.md銆?

## v42褰掓。涓庝笅涓€姝ヨ竟鐣?
涓ら」鍧?8杞畬鎴愶紝鍥哄畾缁勫悎TOTAL80.336513436816銆傚師鐐圭害鏉熺浉瀵筬ree璁粌鍚庢渶楂楽W+0.002969806658锛?8/18鍖归厤杞鏇撮珮锛屼腑浣嶅樊+0.108804776891锛涗粎鏄崟绉嶅瓙寮€鍙戞姌鍐呭噺灏戦€€鍖栫殑璇佹嵁锛屾湭瓒呭師SW銆傚垹闄よ礋free鐘舵€佷笌涓や唤绗?杞噸澶嶆ā鍨嬪叡3涓€?366424瀛楄妭锛涗繚鐣檃nchored瀹屾暣last_state绾?.2MB鐢ㄤ簬灞€閮ㄦ鏈哄埗瀵圭収锛屼笉浣滀负鏂颁氦浠樸€傛竻鍗曡deletion_manifest.json銆傛棤娲诲姩璁粌鎴栭槦鍒楋紱鍘熷叾浠栨姌鏈惎鍔ㄣ€傚悗缁彲鍦ㄤ笉鍚屾洿鏂伴儴浣嶆垨鐩爣鐨勭嫭绔嬪亣璁句腑鎼哄甫鍘熺偣绾︽潫锛屽皻鏈疄鐜版垨鍚姩锛涗笉缁х画鏈琩ec0鐨勭郴鏁?LR鎵弿銆?



## v43 free鏄惧紡鍚姩
{"pid": 57296, "mode": "free", "started_at": "2026-09-29T16:06:07.224774+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_features_v43.py", "--mode", "free", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_features_v43_SW_free_fold5"], "automatic_queue": false}


## v43 anchored鏄惧紡鍚姩
{"pid": 68220, "mode": "anchored", "started_at": "2026-09-29T16:06:23.191820+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_anchored_features_v43.py", "--mode", "anchored", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_anchored_features_v43_SW_anchored_fold5"], "automatic_queue": false}


## v43瀹屾暣鐗瑰緛楠ㄥ共瀵圭収瀹屾垚
涓ら」瀹屾垚锛孴OTAL=80.336513436816锛屽閲?0.000000000000锛岄噰鐢?None銆傛棤鍏朵粬鎶樻垨浜や粯淇敼锛涜anchored_features_v43_20260929/FINAL.md銆?

## v43褰掓。缁撹
涓ら」18杞粨鏉燂紝璁粌鍚嶴W鏈€楂榝ree73.505103279銆乤nchored73.507620278锛屽潎浣庝簬鍘?3.663082797锛涘疄闄呬笁鍒桾OTAL80.336513436816淇濇寔銆傜害鏉熺浉瀵?0.002516999785锛?1/18瀵瑰簲杞鏇撮珮锛屼腑浣?0.001984832480锛屽畬鏁撮骞叉湭璇佸疄绋冲畾鏀剁泭銆傚叧闂湰閰嶇疆LR/绯绘暟鎵弿锛屽悗缁渶鍒囨崲鏈哄埗鎴栫洰鏍囷紝灏氭棤鏂颁换鍔°€傚垹闄ゆ湰鎵?涓湭閲囩敤鏉冮噸鍏?9620738瀛楄妭锛屽師鍐犲啗/v42灞€閮ㄥ鐓?浜や粯淇濈暀锛屾簮/閰嶇疆/杞ㄨ抗/绱у噾棰勬祴淇濈暀銆傛竻鍗曡deletion_manifest.json銆傜浜旀姌>80.5涓庡畬鏁碠OF>82鏈畬鎴愶紝鏃犲叾浠栨姌銆佹椿鍔ㄨ缁冩垨闃熷垪銆?



## v44鏄惧紡鍚姩
{"pid": 43544, "started_at": "2026-09-29T16:36:27.922028+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_state_messages_v44.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_state_messages_v44_SW_fold5"], "automatic_queue": false}


## v44榛樿鐘舵€佹秷鎭畬鎴?
涓ら」瀹屾垚锛孴OTAL=80.336513436816锛岀浉瀵?0.336513澧炲姞+0.000000000000锛涢噰鐢╗]銆傚畬鏁磋褰晄tate_messages_v44_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯鏀瑰姩銆?

## v44瀹屾垚涓庡綊妗?
self/neighbor鍧囩18杞粨鏉燂紝璁粌鍚庢渶楂楽W73.604036960754/73.610973025844锛屽潎鏈秴杩囧師73.663082797066銆傚疄闄匱OTAL80.336513436816涓嶅彉锛屾棤鍏朵粬鎶樻垨闃熷垪銆俷eighbor-self灞€閮?0.006936065090锛?1浜曞崌/5浜曢檷锛?1/18瀵瑰簲杞鏇撮珮锛屼腑浣?0.003280926577锛涘紑鍙戞姌灞€閮ㄨ瘉鎹紝涓嶇О绋冲畾鏂颁簳鏀剁泭銆備繚鐣欎袱涓15杞甿odel_best_updated灏忓ご浣滀负鐩稿鏈哄埗涓庡尮閰嶅鐓э紝鏈噰鐢ㄥ叾棰勬祴銆傚垹闄ゅ畬鏁存湯鎬佷笌涓や釜绗?杞噸澶嶆潈閲嶅叡3涓€?262546瀛楄妭锛涘師妯″瀷/浜や粯涓嶅彉锛屾竻鍗曡deletion_manifest.json銆傚綋鍓嶆秷鎭鏁?鑰﹀悎鐣屼笉鎵弿锛涘彲灏嗗眬閮ㄩ偦鐐规満鍒跺甫鍏ユ柊鐙珛璁捐锛屽皻鏈疄鏂芥垨鍚姩銆傜洰鏍囩浜?80.5鍙婂畬鏁碠OF>82鏈畬鎴愩€?



## v45 long_section鏄惧紡鍚姩
{"pid": 24140, "mode": "long_section", "started_at": "2026-09-29T16:50:33.607257+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_divergent_models_v45.py", "--mode", "long_section", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_divergent_v45_SW_long_section_fold5"], "automatic_queue": false}


## v45 curve_axial鏄惧紡鍚姩
{"pid": 29488, "mode": "curve_axial", "started_at": "2026-09-29T16:51:10.194341+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_divergent_models_v45.py", "--mode", "curve_axial", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_divergent_v45_POR_curve_axial_fold5"], "automatic_queue": false}


## v45涓ゆ潯鏂版灦鏋勫畬鎴?
SW闀夸簳娈典笌POR涔濇洸绾胯酱鍚戝畬鎴愶紝瀹為檯TOTAL=80.336966691471锛屽閲?0.000453254655锛岄噰鐢╗('SW', 'long_section')]銆傝瑙乨ivergent_models_v45_20260929/FINAL.md锛涙湭琛ュ叾浠栨姌锛屾棫浜や粯涓嶅彉銆?

## v45鏈€缁堝€欓€夌姸鎬佷笌娓呯悊
鏂癝W683033鍙傛暟锛岀12杞?3.664377810365锛?0杞棭鍋滐紱纾佺洏閲嶈浇鍏?44665鐐逛竴鑷淬€備笌鍘烶OR/PERM瀹為檯缁勫悎TOTAL80.336966691471锛屾瘮鍘?0.336513浠?0.000453254655銆?浜曞崌/8浜曢檷锛屾湭璋冩暣閫夋嫨鐨勪簳绾OTAL鍖洪棿[-0.3040,0.2735]璺ㄩ浂锛屽洜姝や粎淇濈暀鏁板€煎€欓€夊強鏂扮粨鏋勮捣鐐癸紝涓嶅绉版槑纭硾鍖栨敹鐩婏紝涓嶅惎鍔ㄨˉ鎶樸€侾OR涔濇洸绾胯酱鍚?06145鍙傛暟锛岀37杞渶濂?3.269695792712锛岀55杞嚜鐒秔atience18缁撴潫锛屼綆鍘?.282241锛涙彁鍓嶅仠姝㈣姹傚埌杈炬椂宸茬粨鏉燂紝淇濈暀late_stop_request_record.json锛屼笉鍐掔О浜哄伐缁堟銆備袱椤规€诲閽?50.63913绉掋€乵ax2GPU锛屽崟椤?96.48/309.45绉掞紝宄板€?184.93/1149.31MiB銆傚垹闄OR璐熸潈閲?涓€?2341946瀛楄妭锛汼W鏂扮綉缁?鍘熷啝鍐?鏃т氦浠樹繚鐣欍€備笅涓€姝ュ簲浠庢柊缁撴瀯鎴栦簳灏哄害鏉′欢寤烘ā鍑哄彂锛屼笉閫€鍥炲喕缁撳皬澶磋繛缁井璋冿紱鏈柊澧炰换鍔?闃熷垪锛屽畬鏁碠OF>82浠嶆湭瀹屾垚銆?



## v46 reference鏄惧紡鍚姩
{"pid": 67656, "mode": "reference", "started_at": "2026-09-29T17:06:16.417435+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_conditioned_v46.py", "--mode", "reference", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_conditioned_v46_SW_reference_fold5"], "automatic_queue": false}


## v46 well鏄惧紡鍚姩
{"pid": 34272, "mode": "well", "started_at": "2026-09-29T17:06:47.043180+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_conditioned_v46.py", "--mode", "well", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_conditioned_v46_SW_well_fold5"], "automatic_queue": false}


## v46浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢╳ell锛岃缁冨悗well-reference=+0.147097342423銆傝瑙亀ell_conditioned_v46_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v46鏈€缁堝€欓€夌姸鎬佷笌娓呯悊
瀹為檯缁勫悎TOTAL80.342771761689锛岀浉瀵箆45澧炲姞0.005805070217锛汼W73.680963725271锛岀9杞渶濂姐€佺27杞畃atience18缁撴潫銆傚綋鍓嶄簳鏉′欢涓庡浐瀹氬弬鑰冨悓瀹归噺/鍚屽垵濮嬪寲锛涘浐瀹氬弬鑰冭缁冨悗鏈€楂?3.533866382849锛岀18杞棭鍋滃洖閫€璧风偣73.664377810365銆傝缁冨悗鏈€浣冲樊+0.147097342423锛屼絾鐩稿鍙繚鐣欑殑璧风偣浠?0.016585914907锛屼笉鑳芥贩娣嗕袱绉嶆敹鐩娿€?
16楠岃瘉浜?鍗?闄嶏紱SW榛樿鍊奸儴鍒?0.928308锛岃繛缁暟鍊奸儴鍒?2.186920锛屾姷娑堝悗浠呭井澧烇紱鏈皟鏁村紑鍙戦€夋嫨鐨勪簳绾OTAL鍖洪棿[-0.139812,0.154816]璺ㄩ浂銆傚彧淇濈暀灞€閮ㄦ暟鍊煎€欓€夊拰宸茶瘉瀹炵殑鍖归厤瀵圭収宸紝涓嶅绉扮ǔ瀹氭硾鍖栨敹鐩婏紱涓ユ牸>80.5鍙婂畬鏁碠OF>82浠嶆湭瀹屾垚銆?
鍏?44665鐐圭鐩橀噸寤轰笌瀛樻。棰勬祴涓€鑷达紝瀹為檯涓夊垪璇勫垎瀹屾垚銆備袱浠诲姟鎬诲閽?44.526723绉掞紙鍚姞杞姐€侀鐑€佽緭鍏ュ簱鏋勫缓鍙婂惎鍔ㄩ棿闅旓級锛屽崟椤?56.79/209.25绉掞紝宄板€煎悇1252.30MiB銆佹渶澶у苟鍙?锛涚紪鍐欎笌鏈€缁堥噸杞藉彟璁°€傝礋鍙傝€冩潈閲嶅凡鎸塪eletion_manifest.json鍒犻櫎锛屼繚鐣欏綋鍓嶄簳妯″瀷銆佸畬鏁存仮澶嶇姸鎬併€佺埗SW/鍘熷啝鍐涘強浜や粯銆?
鏈壒璁粌鍏ㄩ儴缁撴潫锛屾棤鑷姩闃熷垪銆佸叾浠栨姌鎴栦氦浠樿鐩栥€備笅涓€鎵瑰彲娌夸簳鏉′欢鏈哄埗璁捐鏂扮殑鐙珛缁撴瀯瀵圭収锛岄噸鐐硅В鍐抽粯璁ょ姸鎬佷笌杩炵画鍊肩殑鎹熺泭鎶垫秷锛涗笉鐩存帴瀵嗛泦鎵弿鐐规暟/璋冨埗鐣岋紝涓嶆妸寰呭疄鏂借鎯冲啓鎴愬畬鎴愪簨瀹炪€?


## v47 global_query鏄惧紡鍚姩
{"pid": 58592, "mode": "global_query", "started_at": "2026-09-29T17:17:26.741822+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_memory_v47.py", "--mode", "global_query", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_memory_v47_SW_global_query_fold5"], "automatic_queue": false}


## v47 section_query鏄惧紡鍚姩
{"pid": 59040, "mode": "section_query", "started_at": "2026-09-29T17:17:41.901365+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_well_memory_v47.py", "--mode", "section_query", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_well_memory_v47_SW_section_query_fold5"], "automatic_queue": false}


## v47浠诲姟鏌ヨ瀵圭収瀹屾垚
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗section-global=-0.003543385011銆傝瑙亀ell_memory_v47_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v47鏈€缁堟敹灏句笌鏈夋晥璁粌璇佹嵁
涓ょ粍鍧囩18杞嚜鐒秔atience18鍋滄锛屾渶浣虫仮澶嶇0杞紝涓嶈兘鎶婂洖閫€鍚庣殑73.680964鍐欐垚璁粌鍚庢柊鎴愮哗銆傜湡姝ｈ缁冨悗鏈€楂樺潎绗?杞細global_query SW73.476844096201锛堜綆鐖?.204119629070锛夛紱section_query SW73.473300711190锛堜綆鐖?.207663014081锛夛紝灞€閮ㄦ煡璇㈣緝绐楀彛骞冲潎-0.003543385011銆傞噰鐢ㄦ柊妯″瀷=False锛屽師缁勫悎TOTAL80.342771761689淇濇寔锛屾湭杈?0.5/瀹屾暣OOF82銆?
涓ょ粍瀹屾暣鏈€缁堢姸鎬佹樉绀鸿蹇嗙紪鐮佸櫒銆乲ey/value銆佷袱缁剄uery鍙婃姇褰卞潎瀹為檯鏇存柊锛涜缁冩崯澶辩害0.04465闄嶈嚦0.01725銆俶emory_training_evidence.json璁板綍鍚勫弬鏁板彉鍖栥€傛渶浣冲洖閫€鐨勯浂鎶曞奖涓嶄唬琛ㄨ缁冩敮璺け娲伙紱鏈疆璐熺粨鏋滃彲鎹鍏抽棴褰撳墠閰嶇疆锛屽仠姝㈡壂鐐规暟/澶存暟/鏌ヨ缁存暟锛屼笉娉涘寲鍒版墍鏈夎蹇嗘ā鍨嬨€?
鍚屽垵濮嬪寲銆佸垵濮嬬埗鍑芥暟銆佹渶浣虫仮澶嶄笌144665鐐硅鐩栭€氳繃锛涘師POR/PERM涓庝繚鐣橲W瀹為檯涓夊垪璇勫垎瀹屾垚銆傚崟椤?12.33/210.04绉掞紝鍙屽苟鍙戞€诲閽?30.554407绉掞紝鍚姞杞?棰勭儹/杈撳叆搴?鍚姩闂撮殧锛屾渶澶у苟鍙?锛屾樉瀛樺嘲鍊煎悇1397.69MiB銆傛簮缂栧啓涓庢渶缁堟敹灏惧彟璁°€?
宸插垹闄や袱璐熷€欓€夌殑best/last鍥涗釜鏉冮噸鏂囦欢锛屽叿浣撳瓧鑺?鍝堝笇瑙乨eletion_manifest.json锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬨€佺揣鍑戣瘖鏂€佸綋鍓峷46鍜岀埗v45鏉冮噸鍙婂師浜や粯銆傚綋鍓嶆棤璁粌杩涚▼/鑷姩闃熷垪/琛ユ姌銆備笅涓€鎵瑰簲鎹㈣缁冭〃寰佹垨杈撳嚭鍒嗗竷鏈哄埗锛屽綋鍓嶆敞鎰忓姏璁板繂璺嚎涓嶇户缁紱灏氭湭瀹炴柦鐨勬柊璁炬兂涓嶈鏀剁泭銆?


## v48 linear鏄惧紡鍚姩
{"pid": 57888, "mode": "linear", "started_at": "2026-09-29T17:31:15.412049+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quantile_density_v48.py", "--mode", "linear", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quantile_density_v48_SW_linear_fold5"], "automatic_queue": false}


## v48 rqs鏄惧紡鍚姩
{"pid": 60688, "mode": "rqs", "started_at": "2026-09-29T17:31:40.553599+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quantile_density_v48.py", "--mode", "rqs", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quantile_density_v48_SW_rqs_fold5"], "automatic_queue": false}


## v48浠诲姟鏌ヨ瀵圭収瀹屾垚
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗rqs-linear=+1.292082008243銆傝瑙乹uantile_density_v48_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v48鏈€缁堟敹灏撅細淇濈暀鏈夊鐓ф敹鐩婄殑骞虫粦瀵嗗害
涓ょ粍鏂板瘑搴﹀垵濮婼W鍧?2.346441547110锛岄潪鐖舵ā鍨?3.680963725271銆俵inear璁粌鍚庢渶楂樼7杞?2.136427521820锛岀18杞棭鍋溿€佸洖閫€鑷韩绗?杞紱rqs璁粌鍚庢渶楂樼4杞?3.428509530063锛岀22杞棭鍋溿€俁QS杈僱inear鍚垵濮嬫渶浣?1.082067982952SW锛堢瓑浠稵OTAL+0.378723794033锛夛紝浠呮瘮杈冭缁冨悗鏈€楂樺樊+1.292082008243锛涜繖涓や釜宸€间笉鑳芥贩鐢ㄣ€?
鍚屽閲?鍚屽垵濮嬪寲瀵圭収13浜曞崌3浜曢檷锛岄€夋嫨鏈牎姝ｇ殑鎺㈢储鎬т簳绾OTAL鍖洪棿[0.163806,0.602874]銆傚洜姝や繚鐣橰QS妯″瀷浣滀负鏈夊眬閮ㄦ満鍒舵敹鐩婄殑鍊欓€夛紱涓嶅洜缁濆浣庝簬褰撳墠SOTA鑰屽垹闄ゃ€俁QS瀹為檯鏇挎崲缁勫悎TOTAL80.254412793365锛屼粛浣庡綋鍓?0.342771761689锛圫W宸?0.252454195209锛夛紝涓嶉噰鐢ㄨ繘褰撳墠閰嶆柟銆佷笉琛ユ姌銆傚畬鏁碠OF>82浠嶆湭瀹屾垚銆?
鏂板垎甯冭В鏋愬瘑搴﹀湪8涓疄闄呰缁冮敋鐐硅緭鍑轰笂涓嶤DF鑷姩寰垎鐩哥锛屾渶澶х浉瀵硅宸甽inear5.94e-7/rqs8.38e-7锛屽叏瀹炴暟CDF绔偣绮剧‘0/1锛涜繖鏄柊鏁板瀹炵幇鐨勫崟娆℃渶缁堟鏌ワ紝涓嶆槸閲嶅棰勬銆俁QS鏈€浣虫潈閲嶇鐩橀噸寤哄叏144665鐐逛笌瀛樻。瀹屽叏涓€鑷达紝璇佹嵁retained_rqs_verification.json銆傚師鏈€缁堥獙璇佸凡鎭㈠鍚勮嚜鏈€浣筹紱鍘熶笁鐩爣淇濈暀缁勫悎瀹為檯璇勫垎瀹屾垚銆?
鍗曢」158.51/183.54绉掞紝鍙屽苟鍙戞€诲閽?13.127631绉掞紝鍚姞杞?棰勭儹/杈撳叆搴?鍚姩闂撮殧锛涘嘲鍊煎悇1261.56MiB銆乵ax2GPU銆備唬鐮佺紪鍐欎笌鏈€缁圧QS閲嶈浇妫€鏌?.21绉掑彟璁°€備袱椤瑰凡缁撴潫锛屾棤鑷姩闃熷垪銆傚垹闄inear涓や釜璐熸潈閲嶏紝淇濈暀RQS鏈€浣?瀹屾暣鐘舵€併€乿46姝ｆā鍨嬨€佺埗妯″瀷鍜屾棫浜や粯锛涘搱甯?瀛楄妭瑙乨eletion_manifest.json銆?
涓嶇户缁綋鍓嶇嚎鎬у瘑搴︺€佸垎浣嶇偣鏁版垨灏炬鐜囨壂鎻忋€傚悗缁彲甯︾潃宸茶瘉瀹炵殑骞虫粦鎻掑€兼満鍒讹紝鐙珛鐮旂┒鍒嗕綅浣嶇疆鐨勭洃鐫ｆ垨鍒嗗竷璁粌鏂规硶锛涜繖鍙槸涓嬩竴姝ユ柟鍚戯紝灏氭湭璁粌锛屼笉瀹ｇО宸插彇寰楁柊鏀剁泭銆傜洰鏍囥€佸墿浣欏伐浣滃強瀹屾垚鍚庣嫭绔嬩簳纭/鏂扮洰褰曚氦浠樼户缁淮鎶ゃ€?


## v49 likelihood鏄惧紡鍚姩
{"pid": 25840, "mode": "likelihood", "started_at": "2026-09-29T17:43:16.576259+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quantile_supervision_v49.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quantile_supervision_v49_SW_likelihood_fold5"], "automatic_queue": false}


## v49 quantile_aux鏄惧紡鍚姩
{"pid": 66852, "mode": "quantile_aux", "started_at": "2026-09-29T17:43:38.295986+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_quantile_supervision_v49.py", "--mode", "quantile_aux", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_quantile_supervision_v49_SW_quantile_aux_fold5"], "automatic_queue": false}


## v49浠诲姟鏌ヨ瀵圭収瀹屾垚
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗aux-likelihood=+0.007017394414銆傝瑙乹uantile_supervision_v49_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v49鏈€缁堢粨鏋滀笌娓呯悊
likelihood璁粌鍚庢渶浣崇1杞甋W73.191339565226锛泀uantile_aux璁粌鍚庢渶浣崇1杞?3.198356959640銆傝缁冨悗杈呭姪椤硅緝瀵圭収浠?0.007017394414锛屼絾涓よ€呴兘浣庝簬鍚岃捣鐐?3.428509530063锛屽潎绗?8杞畃atience18鍋滄銆佹渶浣抽€夋嫨鍥為€€绗?杞€傛寜棰勫畾涔夊寘鍚捣鐐圭殑閫夋嫨瑙勫垯锛岃緟鍔╃洃鐫ｆ渶缁堟敹鐩?0锛涗笉鑳芥妸鏈€浣冲洖閫€鎴栬缁冨悗寰皬宸啓鎴愭柊鏈夋晥妯″瀷銆?
褰撳墠鍘熶笁鍒楃粍鍚圱OTAL80.342771761689淇濇寔锛屾柊閲囩敤=False锛岀浜斾弗鏍?80.5鍙婂畬鏁碠OF>82鍧囨湭瀹屾垚銆傚師v48骞虫粦鎻掑€煎鐓ф敹鐩婁粛鏈夋晥锛屽叾RQS鏉冮噸缁х画淇濈暀锛涙湰杞叧闂綋鍓嶈緟鍔╁垎浣嶉」绯绘暟/鑺傜偣鏁版壂鎻忥紝涓嶅惁瀹氬叾浠栧垎浣嶈缁冩柟娉曘€?
鍒濆瀹屾暣144665鐐逛笌v48婧愰娴嬩竴鑷淬€佷袱涓垵濮嬬姸鎬佸搱甯岀浉鍚屻€佺粨鏉熸渶浣虫仮澶嶄竴鑷达紝鏉ユ簮鍝堝笇涓庡疄闄呬笁鍒楄瘎鍒嗗凡鏍搁獙锛涚湡瀹炶缁冧腑楠ㄥ共銆丗iLM銆佸瘑搴︿笌榛樿澶存搴﹀潎闈為浂銆傛柊鍊欓€夋病鏈夎淇濈暀锛屽洜姝や笉閲嶅鎵ц鍘熸牱鏉″瘑搴﹀叕寮忔垨鍘熺埗妯″瀷纾佺洏楠岃瘉銆?
鍗曢」166.8964/167.2358绉掞紝涓ら」鎬诲閽?93.43774绉掞紝max2GPU锛屽悇宄板€?261.56MiB锛涘惈鍔犺浇/棰勭儹/杈撳叆搴撳強鍚姩闂撮殧锛岀紪鍐欎笌鏈€缁堟敹灏惧彟璁°€傚凡鍒犻櫎涓ら」best/last鍥涗釜鍐椾綑鏉冮噸锛屼繚鐣欐簮鐮?閰嶇疆/鏇茬嚎/棰勬祴/鎴愮哗涓庡垹闄ゅ搱甯屾竻鍗曘€倂48 RQS鍜屽綋鍓峷46銆佹墍鏈夌埗渚濊禆鍙婃棫浜や粯淇濈暀銆?
鏈壒宸插叏閮ㄧ粨鏉燂紝鏃犺缁冭繘绋嬨€侀槦鍒楁垨鍏朵粬鎶樸€備笅涓€鎵归渶鍒囨崲褰撳墠杈呭姪鎹熷け鐨勬€濊矾锛屼笉缁х画0.5闄勮繎鎵弿锛涗笅涓€椤瑰皻鏈疄鏂斤紝涓嶈兘绠椾綔杩涘睍鎴栨敹鐩娿€傜嫭绔嬫柊浜曠‘璁?瀹屾暣OOF鍜屾柊鐩綍浜や粯浠嶆湭瀹屾垚銆?


## v50 scratch鏄惧紡鍚姩
{"pid": 29080, "mode": "scratch", "started_at": "2026-09-29T18:57:44.244984+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_transfer_v50.py", "--mode", "scratch", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_transfer_v50_PERM_scratch_fold5"], "automatic_queue": false}


## v50 sw_transfer鏄惧紡鍚姩
{"pid": 37168, "mode": "sw_transfer", "started_at": "2026-09-29T18:58:02.438378+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_transfer_v50.py", "--mode", "sw_transfer", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_transfer_v50_PERM_sw_transfer_fold5"], "automatic_queue": false}


## v50浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗transfer-scratch=+0.239511576802銆傝瑙乸erm_transfer_v50_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v50鏈€缁堢粨鏋滐細淇濈暀璁粌浜曞唴杩佺Щ鍊欓€?
涓ょ粍鍒濆PERM鍏堥獙68.383772436967锛屽畬鏁村垵濮嬮娴?杈撳嚭澶村搱甯屼竴鑷达紝楠ㄥ共鍝堝笇鎸夎璁′笉鍚屻€俿cratch绗?9杞渶浣?8.811295997433锛岀37杞畃atience18缁撴潫锛泂w_transfer绗?1杞渶浣?9.050807574234锛岀39杞粨鏉熴€傝縼绉昏緝闅忔満+0.239511576802PERM锛堢瓑浠稵OTAL+0.083829051881锛夛紝浣嗚緝褰撳墠PERM89.396724049366浠嶄綆0.345916475131銆?
鍖归厤宸?浜曞崌7浜曢檷锛岄€夋嫨鏈牎姝ｇ殑鎺㈢储鎬т簳绾OTAL鍖洪棿[-0.081486,0.252971]璺ㄩ浂锛涗粎淇濈暀灞€閮ㄥ鐓ф敹鐩婏紝涓嶅绉扮ǔ瀹氭柊浜曟敹鐩娿€傝縼绉诲疄闄呮浛鎹㈢粍鍚圱OTAL80.221700995393锛屾湭閲囩敤锛涘綋鍓岰URRENT_RECIPE涓夊垪浠嶄负TOTAL80.342771761689銆備弗鏍肩浜?80.5涓庡畬鏁碠OF>82浠嶆湭瀹屾垚锛屾棤鍏朵粬鎶樸€?
杩佺Щ鏈€浣虫ā鍨嬬鐩橀噸寤哄叏144665鐐逛笌瀛樻。瀹屽叏鐩稿悓锛屽疄闄呬笁鍒楄瘎鍒嗕竴鑷达紝璇佹嵁retained_transfer_verification.json銆傚垵濮嬪寲/鏈€浣虫仮澶嶃€佹簮鐮佸搱甯屻€佺浉鍚岃緭鍑哄ご鍙婃暟鎹潵婧愬凡鏍搁獙锛涜缁冮骞?澶?FiLM鍧囨湁鐪熷疄姊害銆傛眹鎬诲櫒鏇炬妸NPZ浜曠紪鍙峰瓧娈佃鏇挎崲鎴愭ā寮忓悕锛屽凡淇骞惰褰昮inalization_failure_fixed.json锛涙病鏈変慨鏀硅缁冩簮鐮併€佹潈閲嶆垨閲嶈銆?
鍗曢」327.53/331.33绉掞紝鎬诲閽?55.129723绉掞紝鍚姞杞?棰勭儹/杈撳叆搴撳拰鍚姩闂撮殧锛沵ax2GPU銆佸悇1252.30MiB銆備唬鐮佺紪鍐欎笌鏈€缁堢鐩橀噸寤?.59绉掑彟璁°€備袱椤瑰叏閮ㄧ粨鏉熸棤闃熷垪锛屽垹闄cratch涓や釜璐熸潈閲嶏紝淇濈暀杩佺Щ鏈€浣?瀹屾暣鐘舵€併€佸綋鍓嶄笁鐩爣銆乿48骞虫粦鍊欓€夊強鎵€鏈夌埗渚濊禆/浜や粯銆傚搱甯?瀛楄妭瑙乨eletion_manifest.json銆?
鏈娌℃湁鍥炲埌SW杈呭姪鎹熷け鎵弿锛涘悗缁彲鎼哄甫杩佺Щ琛ㄥ緛璁捐鏂扮殑PERM缁撴瀯/璁粌鏂规锛屽綋鍓嶅彧鏄柟鍚戯紝鏈疄鏂姐€傞殢鏈哄喎鍚姩褰撳墠閰嶇疆涓嶅欢浼革紝涓嶈兘鎶婂垵濮嬪寲鍒?8鍒嗙殑瀛︿範褰撴垚瓒呰秺褰撳墠妯″瀷銆傜嫭绔嬫柊浜曠‘璁ゃ€佸畬鏁翠簲鎶樺拰鏂扮洰褰曚氦浠樹粛鏈畬鎴愩€?


## v51 plain鏄惧紡鍚姩
{"pid": 62180, "mode": "plain", "started_at": "2026-09-29T19:12:54.630281+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_moment_mix_v51.py", "--mode", "plain", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_moment_mix_v51_PERM_plain_fold5"], "automatic_queue": false}


## v51 interwell鏄惧紡鍚姩
{"pid": 16860, "mode": "interwell", "started_at": "2026-09-29T19:13:19.568789+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_moment_mix_v51.py", "--mode", "interwell", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_moment_mix_v51_PERM_interwell_fold5"], "automatic_queue": false}


## v51浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗interwell-plain=+0.031618982727銆傝瑙乸erm_moment_mix_v51_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v51鏈€缁堢粨鏋滀笌瀹為檯澧炲己璇佹嵁
plain璁粌鍚庢渶濂界3杞甈ERM88.890099775317锛宨nterwell鏈€濂界5杞?8.921718758043銆傝缁冨悗澧炲己杈冨鐓?0.031618982727锛屼絾涓よ€呭潎浣庝簬鍚岃捣鐐?9.050807574234锛屽潎绗?8杞畃atience18缁撴潫骞舵仮澶嶇0杞紱鎸夐瀹氫箟鍖呭惈璧风偣鐨勬渶缁堥€夋嫨锛屾敹鐩?0锛屼笉浜х敓鏂版湁鏁堟ā鍨嬨€?
澧炲己缁勭湡瀹炶缁?152鎵癸紝绱36928绐楀彛搴旂敤澧炲己锛屽紓浜曢厤瀵规暟鍚屼负36928锛涘鐓у寮烘暟0銆傛渶浣虫仮澶嶅悗璁℃暟0鏄洖閫€璧风偣锛屼笉鏄ā鍧楀け娲伙紝瀹為檯鏈€佽瘉鎹繚瀛樺湪report鐨刲ast_training_augmentation鍜岄€愯疆鍘嗗彶銆傞獙璇佸缁堝叧闂寮猴紝鏃犳柊澧炰簳ID鐗瑰緛/楠岃瘉缁熻鎷熷悎銆?
涓ょ粍鍒濆鍙傛暟銆佽緭鍑哄ご鍜?44665鐐归娴嬪搱甯屼竴鑷达紝鍒濆绮剧‘澶嶇幇v50锛岀粨鏉熸渶浣虫仮澶嶄竴鑷达紝婧愬搱甯屽強鍘熶笁鍒楀疄闄呰瘎鍒嗗凡瀹屾垚銆傚綋鍓峊OTAL80.342771761689淇濇寔锛涘師PERM89.396724銆乿50杩佺Щ鍊欓€夊拰v48鏍锋潯鍊欓€夐兘淇濈暀銆傚師绗簲涓ユ牸>80.5涓庡畬鏁碠OF>82鏈畬鎴愶紝涓嶅惎鍔ㄥ叾浠栨姌銆?
鍗曢」146.9093/149.3306绉掞紝鎬诲閽?79.067863绉掞紝max2GPU锛屽悇宄板€?252.30/1252.41MiB锛屽惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛屼唬鐮佷笌鏀跺熬鍙﹁銆備袱椤瑰叏閮ㄧ粨鏉熸棤闃熷垪锛涘垹闄ゆ湰杞袱椤筨est/last鍥涗釜鍐椾綑鏉冮噸锛屼繚鐣欐潵婧愬拰鏃т氦浠橈紝鍝堝笇/瀛楄妭瑙乨eletion_manifest.json銆傚師CDF/鐖舵ā鍨嬫鏌ユ湭閲嶅杩愯銆?
鏈疆鍋滄姝ゆ鐜?Beta/灞備綅缃壂鎻忥紝鍙檺褰撳墠鏆栧惎鍔≒ERM涓庢澧炲己浣嶇疆銆傝缁冨悗鍑忚交閫€鍖栫殑寰樊涓嶈兘鍗囩骇涓洪獙璇佽揪鏍囨垨绋冲畾娉涘寲鏀瑰杽锛涗笅涓€鎵归渶鎹㈡満鍒讹紝灏氭湭鍚姩銆傜嫭绔嬩簳纭/瀹屾暣OOF/鏂扮洰褰曚氦浠樹粛鏈畬鎴愩€?


## v52 position鏄惧紡鍚姩
{"pid": 44772, "mode": "position", "started_at": "2026-09-29T19:28:20.558046+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_token_pool_v52.py", "--mode", "position", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_token_pool_v52_PERM_position_fold5"], "automatic_queue": false}


## v52 content鏄惧紡鍚姩
{"pid": 61280, "mode": "content", "started_at": "2026-09-29T19:28:39.215019+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_token_pool_v52.py", "--mode", "content", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_token_pool_v52_PERM_content_fold5"], "automatic_queue": false}


## v52浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗content-position=-0.026171081763銆傝瑙乸erm_token_pool_v52_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?

## v52鏈€缁堢粨鏋滀笌娓呯悊
position绗?2杞渶濂絇ERM88.894217488422锛岀40杞畃atience18缁撴潫锛沜ontent绗?4杞渶濂?8.868046406659锛岀42杞粨鏉熴€傚唴瀹?浣嶇疆=-0.026171081763锛屾棤鍖归厤鏀剁泭銆備袱鑰呬篃浣庝簬鍘嗗彶v50鏅€氬钩鍧囨眹鑱?9.050807574234锛堝垎鍒?0.156590085813/-0.182761167576锛夛紝浠ュ強褰撳墠PERM89.396724049366銆傚巻鍙叉瘮杈冧笌鏈壒鍚屽閲忔瘮杈冨垎寮€锛屼笉鑳芥妸鍒濆鍏堥獙68.383772鍒?8鍒嗙殑瀛︿範褰撹秴瓒婂綋鍓嶆ā鍨嬨€?
涓ら」鍒濆鐘舵€?杈撳嚭澶?144665鐐瑰厛楠岄娴嬪搱甯屼竴鑷达紱鐪熷疄鎵撳垎鍣ㄦ搴﹂潪闆讹紝閫夊畾鏉冮噸鑼冩暟position0.253729/content0.350035锛屽唴瀹规潈閲嶇喌璁粌涓凡鍋忕鍧囧寑鍊硷紝闈炲け娲昏瘯楠屻€傛渶澶ф睜鍖栨潈閲嶅彲鍙楄竟鐣屼粎灏戞暟鏈夋晥鐐瑰奖鍝嶏紝涓嶅皢鍗曚釜max鏁板€煎綋鍏ㄧ獥鍙ｅ皷閿愭€ц瘉鎹€?
鏈€浣虫仮澶嶅叏144665鐐逛竴鑷淬€佹潵婧愬搱甯屼笌鐪熷疄涓夊垪缁勫悎璇勫垎瀹屾垚銆傛病鏈夐噰鐢ㄦ柊妯″瀷锛孋URRENT_RECIPE浠峊OTAL80.342771761689锛涗弗鏍肩浜?80.5鍙婂畬鏁碠OF>82鏈畬鎴愶紝鏃犲叾浠栨姌銆傝礋鍊欓€変笉杩藉姞閲嶅纾佺洏/鍘熷瘑搴﹂獙璇併€?
鍗曢」379.7283/389.8115绉掞紝鎬诲閽?14.115675绉掞紝max2GPU锛屽嘲鍊?319.99/1291.71MiB锛涘惈鍔犺浇/棰勭儹/杈撳叆搴?鍚姩闂撮殧锛屼唬鐮佷笌鏈€缁堟敹灏惧彟璁°€傚叏閮ㄧ粨鏉熸棤璁粌闃熷垪锛涘垹闄ゅ洓涓湰杞産est/last璐熸潈閲嶏紝淇濈暀浠ｇ爜/閰嶇疆/鏃ュ織/棰勬祴/姊害涓庣揣鍑戞姤鍛娿€倂50杩佺Щ銆乿48鏍锋潯銆佸綋鍓嶄笁鐩爣鍙婃簮渚濊禆/鏃т氦浠樹笉鏀癸紝鍝堝笇/瀛楄妭瑙乨eletion_manifest.json銆?
鍏抽棴褰撳墠姹囪仛缁勬暟/姝ラ暱/娓╁害鎵弿锛屼笉鍚﹀畾鎵€鏈夎嚜閫傚簲token鍖栥€備笅涓€鎵归渶鎹㈡満鍒讹紝灏氭湭瀹炴柦锛屼笉鑳界畻浣滄敹鐩娿€傜嫭绔嬩簳纭銆佸畬鏁翠簲鎶樺強鏂扮洰褰曚氦浠樹粛鏈畬鎴愩€?


## v53 scratch鏄惧紡鍚姩
{"pid": 43176, "mode": "scratch", "started_at": "2026-09-29T19:46:26.915268+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_transfer_v53.py", "--mode", "scratch", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_transfer_v53_POR_scratch_fold5"], "automatic_queue": false}


## v53 sw_transfer鏄惧紡鍚姩
{"pid": 66000, "mode": "sw_transfer", "started_at": "2026-09-29T19:46:40.830940+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_transfer_v53.py", "--mode", "sw_transfer", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_transfer_v53_POR_sw_transfer_fold5"], "automatic_queue": false}


## v53浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗transfer-scratch=+0.248076966158銆傝瑙乸or_transfer_v53_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v53鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
scratch锛歅OR 73.640524506112锛屾渶浣崇32杞紝鍋滄绗?0杞紙patience18锛夈€?
sw_transfer锛歅OR 73.888601472270锛屾渶浣崇62杞紝鍋滄绗?0杞紙patience18锛夈€?
杩佺Щ-闅忔満POR=+0.248076966158锛涘綋鍓峆OR=77.551936801885锛屽疄闄呯粍鍚圱OTAL=80.342771761689銆傛柊閲囩敤=False銆?
杩佺Щ鏈夊尮閰嶅眬閮ㄦ敹鐩婏紝淇濈暀杩佺Щ鏈€浣冲強瀹屾暣鐘舵€侊紱涓嶅皢鍏朵綆浜庣幇鏈塒OR鐨勭粷瀵规垚缁╁啓鎴愭柊鍐犲啗銆傝鏀剁泭鏉ヨ嚜鍙嶅浣跨敤鐨勫紑鍙戞姌锛屾湭瀹屾垚鐙珛浜曠‘璁ゃ€?
涓ょ粍鍚?36921鍙傛暟銆佺浉鍚屽垵濮婸OR澶?144665鐐瑰厛楠屻€佷笉鍚屽垵濮嬮骞诧紱鍏ㄩ骞插彧鐢≒OR鏍囩涓嶴tudent-t3鎹熷け銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?True銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[466.646, 623.297]绉掞紱鍙屽苟鍙戞€诲閽?42.362绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1252.51, 1252.51]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡19048838瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v54 uniform鏄惧紡鍚姩
{"pid": 30956, "mode": "uniform", "started_at": "2026-09-29T20:11:57.837259+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_stratified_v54.py", "--mode", "uniform", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_stratified_v54_SW_uniform_fold5"], "automatic_queue": false}


## v54 stratified鏄惧紡鍚姩
{"pid": 52132, "mode": "stratified", "started_at": "2026-09-29T20:12:16.846094+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_stratified_v54.py", "--mode", "stratified", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_stratified_v54_SW_stratified_fold5"], "automatic_queue": false}


## v54浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗stratified-uniform=+0.001729531416銆傝瑙乻w_stratified_v54_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v54鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
uniform锛歋W 73.680963725271锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
stratified锛歋W 73.680963725271锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
鍒嗗眰-闅忔満SW=+0.000000000000锛涘綋鍓峉W=73.680963725271锛屽疄闄呯粍鍚圱OTAL=80.342771761689銆傛柊閲囩敤=False銆?
鍒嗗眰鏃犳渶缁堝尮閰嶆敹鐩婏紝鍋滄褰撳墠32/32姣斾緥/LR鎵弿锛涗笉鎶婅缁冨悗涓や釜閫€鍖栨垚缁╅棿鐨勫皬宸啓鎴愭湁鏁堟ā鍨嬨€?
涓ょ粍鍚?36921鍙傛暟銆佸悓v46SW瀹屾暣璧风偣/144665鐐归娴嬶紱鍏ㄩ骞插彧鐢⊿W鏍囩涓嶨aussian鎹熷け銆備袱缁勫叡鍚岀獥鍙ｅ潎鍊奸闄╋紝鍒嗗眰32/32涓旀寜璁粌鐘舵€佹瘮渚嬫牎姝ｏ紝鏈敼鍙樼姸鎬侀闄╃洰鏍囷紱鍘烸淇濇寔銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?False銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[165.044, 161.629]绉掞紱鍙屽苟鍙戞€诲閽?86.892绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1257.47, 1257.47]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡37953164瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v55 adamw鏄惧紡鍚姩
{"pid": 45724, "mode": "adamw", "started_at": "2026-09-29T20:21:35.941791+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sharpness_v55.py", "--mode", "adamw", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sharpness_v55_SW_adamw_fold5"], "automatic_queue": false}


## v55 sam鏄惧紡鍚姩
{"pid": 53148, "mode": "sam", "started_at": "2026-09-29T20:21:48.663426+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sharpness_v55.py", "--mode", "sam", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sharpness_v55_SW_sam_fold5"], "automatic_queue": false}


## v55浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.342771761689锛岄噰鐢∟one锛岃缁冨悗sam-adamw=+0.104051059886銆傝瑙乻w_sharpness_v55_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v55鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
adamw锛歋W 73.680963725271锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
sam锛歋W 73.680963725271锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
SAM-AdamWSW=+0.000000000000锛涘綋鍓峉W=73.680963725271锛屽疄闄呯粍鍚圱OTAL=80.342771761689銆傛柊閲囩敤=False銆?
SAM鏃犳渶缁堝尮閰嶆敹鐩婏紝鍋滄褰撳墠rho/浼樺寲鍣ㄧ粍鍚堟壂鎻忥紱涓嶆妸璁粌鍚庝袱涓€€鍖栨垚缁╅棿鐨勫皬宸啓鎴愭湁鏁堟ā鍨嬨€?
涓ょ粍鍚?36921鍙傛暟銆佸悓v46SW瀹屾暣璧风偣/144665鐐归娴嬶紱鍏ㄩ骞插彧鐢⊿W鏍囩涓嶨aussian鎹熷け銆備袱缁勫叡鍚屽師鎵规鐐瑰潎鍊奸闄╋紱SAM鍚屾壒涓ら亶銆佸浐瀹歳ho.05銆佺簿纭仮澶嶆壈鍔ㄦ潈閲嶅悗鏇存柊锛屽師Q淇濇寔锛岄潪绛夎绠楅噺銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?False銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[170.059, 246.178]绉掞紱鍙屽苟鍙戞€诲閽?63.480绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1249.71, 1256.12]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡37951500瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v56 additive鏄惧紡鍚姩
{"pid": 53628, "mode": "additive", "started_at": "2026-09-29T20:33:57.445662+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_interaction_v56.py", "--mode", "additive", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_interaction_v56_SW_additive_fold5"], "automatic_queue": false}


## v56 multiplicative鏄惧紡鍚姩
{"pid": 44680, "mode": "multiplicative", "started_at": "2026-09-29T20:34:17.516397+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_interaction_v56.py", "--mode", "multiplicative", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_interaction_v56_SW_multiplicative_fold5"], "automatic_queue": false}


## v56浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.354325927816锛岄噰鐢╝dditive锛岃缁冨悗multiplicative-additive=-0.191968804219銆傝瑙乻w_interaction_v56_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v56鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
additive锛歋W 73.713975628494锛屾渶浣崇11杞紝鍋滄绗?9杞紙patience18锛夈€?
multiplicative锛歋W 73.522006824275锛屾渶浣崇3杞紝鍋滄绗?1杞紙patience18锛夈€?
涔樻硶-鍔犳硶SW=-0.191968804219锛涘綋鍓峉W=73.680963725271锛屽疄闄呯粍鍚圱OTAL=80.354325927816銆傛柊閲囩敤=True銆?
涔樻€т氦浜掓棤鍖归厤鏀剁泭锛屽仠姝㈠綋鍓嶅垎鏀搴?灞傛暟鎵弿锛涘彧闄愬綋鍓嶆ā鍧楀拰璁粌閰嶇疆锛屼笉娉涘寲鎵€鏈夐棬鎺х綉缁滄棤鏁堛€?
涓ょ粍鍚?39481鍙傛暟銆佺浉鍚屽畬鏁村垵濮嬪弬鏁颁笌144665鐐瑰厛楠岋紱鍥涗釜鍐呴儴FF鍙樊鍔犳硶鎴栦箻娉曪紝鍚屽叕鍏盠ayerNorm锛孲W杈撳嚭鍏ㄩ噸缃紱鍏ㄧ綉缁滃彧鐢⊿W鏍囩涓嶨aussian鎹熷け銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?True銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[270.365, 223.589]绉掞紱鍙屽苟鍙戞€诲閽?74.873绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1266.92, 1276.45]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡19051526瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v56褰撳墠閰嶆柟涓庡綊鍥犻檺鍒?
褰撳墠閲囩敤鍔犳硶缁勭11杞紝SW73.713975628494锛堢浉瀵箆46+0.033011903223锛夛紝鍘熺浜斿疄闄呯粍鍚圱OTAL80.354325927816锛?0.011554166128锛夈€備箻娉曠粍绗?杞?3.522006824275锛岀浉瀵瑰姞娉?0.191968804219锛涙椤归厤瀵逛笉鏀寔涔樻€т氦浜掋€傚姞娉曠浉瀵瑰巻鍙叉簮鐨勫井澧炴秹鍙婃柊FF/鍏叡褰掍竴鍖?閲嶇疆杈撳嚭鍜岄噸鏂拌缁冪殑鏁翠綋閰嶇疆锛屾湭鍗曠嫭璇佹槑鍏朵腑涓€涓洜绱犲洜鏋滃鐩婏紝涔熸湭瀹屾垚鐙珛浜曠‘璁ゃ€傚畬鏁?44665鐐规柊SW纾佺洏閲嶅缓鍜屽浐瀹氫笁鍒楀疄闄呰瘎鍒嗗凡瀹屾垚锛涙竻鐞嗛樁娈靛鐢ㄦ湭鏀规潈閲嶇殑鏍搁獙鍑嵁锛屼笉閲嶅鍚屼竴鎺ㄧ悊銆?
鍘?06宸叉湁浜曞唴绉╋紝鍒濇绉╃壒寰佹瀯鎯虫湭璁粌锛屼笉鑳借浣滄柊瀹為獙鏀剁泭銆備繚鐣欐柊鍔犳硶best/last鍜屾棫v46鐖舵ā鍨嬨€乿53/v50杩佺Щ鍙妚48鏍锋潯姝ｅ€欓€夛紝鏃т氦浠樹笉鏀广€備箻娉曡礋鏉冮噸宸插垹銆傜浜斾弗鏍?80.5涓庡畬鏁碠OF>82浠嶆湭瀹屾垚锛屼笉琛ュ叾浠栨姌锛涗笅涓€椤瑰皻鏈惎鍔ㄣ€?


## v57 base鏄惧紡鍚姩
{"pid": 52572, "mode": "base", "started_at": "2026-09-29T20:46:42.662915+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_ffgrowth_v57.py", "--mode", "base", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_ffgrowth_v57_SW_base_fold5"], "automatic_queue": false}


## v57 grow128鏄惧紡鍚姩
{"pid": 1692, "mode": "grow128", "started_at": "2026-09-29T20:46:58.693725+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_ffgrowth_v57.py", "--mode", "grow128", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_ffgrowth_v57_SW_grow128_fold5"], "automatic_queue": false}


## v57浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.354325927816锛岄噰鐢∟one锛岃缁冨悗grow128-base=-0.001879626891銆傝瑙乻w_ffgrowth_v57_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v57鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
base锛歋W 73.713975628494锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
grow128锛歋W 73.713975628494锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
鎵╁-鍩哄噯SW=+0.000000000000锛涘綋鍓峉W=73.713975628494锛屽疄闄呯粍鍚圱OTAL=80.354325927816銆傛柊閲囩敤=False銆?
骞惰鎵╁鏃犲尮閰嶆敹鐩婏紝鍋滄褰撳墠骞惰鍒嗘敮瀹藉害/灞傛暟鎵弿锛涘彧闄愭鍒濆鍖栧拰鍏卞悓瀛︿範鐜囷紝涓嶆硾鍖栨墍鏈夊ぇ妯″瀷鏃犳晥銆?
base839481/grow128989369鍙傛暟锛屽叕鍏辨牳蹇冨強瀹屾暣144665鐐瑰垵濮嬮娴嬬浉鍚岋紝鏂板鍥涗釜128骞惰FF闆惰緭鍑猴紱鍘熻緭鍑哄ご涓嶉噸缃紝鍏ㄧ綉缁滃彧鐢⊿W鏍囩涓嶨aussian鎹熷け锛屽悓lr3e-5銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?False銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[166.372, 179.907]绉掞紱鍙屽苟鍙戞€诲閽?02.907绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1266.92, 1291.83]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡41123212瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v57瀹為檯鏈€佷笌缁撹杈圭晫
{"post_update_best": {"base": {"epoch": 1, "score": 73.51945824737717}, "grow128": {"epoch": 1, "score": 73.51757862048666}}, "extra_training": {"last_epoch": 18, "selected_epoch": 0, "last_extra_output_L2": 0.25923800468444824, "scope": "last actual trained state; selected epoch0 has zero extra outputs"}}
鏂板鍒嗘敮棣栦釜瀹為檯绗簩姝ユ搴﹂潪闆躲€佹湯鎬佽緭鍑烘潈閲峀2涓?.259238锛岃鏄庡苟琛屾ā鍧楀弬涓庝簡璁粌锛涙渶浣虫仮澶嶅悗涓洪浂鏄洖閫€绗?杞紝涓嶆槸妯″潡澶辨椿銆傚畬鏁村叕鍏辨牳蹇?鍒濆鍑芥暟鐩稿悓宸叉牳楠岋紝鍙傛暟閲忎笉鍚屼负棰勫畾瀹归噺鍙橀噺銆備袱缁勫師18杞畃atience缁撴潫锛岃缁冨悗鏈€浣虫墿瀹硅緝鍩哄噯-0.001880涓斿潎浣庤捣鐐癸紝娌℃湁鎵╁鏀剁泭锛涗繚鐣檝56褰撳墠閰嶆柟锛屽仠姝㈡骞惰瀹藉害/灞傛暟鎵弿銆傛湭澧炲姞鍏朵粬鎶樻垨淇敼浜や粯銆?


## v58 joint鏄惧紡鍚姩
{"pid": 41876, "mode": "joint", "started_at": "2026-09-29T20:57:45.424124+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_rqs_staging_v58.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_rqs_staging_v58_SW_joint_fold5"], "automatic_queue": false}


## v58 headstart鏄惧紡鍚姩
{"pid": 50924, "mode": "headstart", "started_at": "2026-09-29T20:58:03.405608+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_rqs_staging_v58.py", "--mode", "headstart", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_rqs_staging_v58_SW_headstart_fold5"], "automatic_queue": false}


## v58浜曟潯浠跺鐓у畬鎴?
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.354325927816锛岄噰鐢∟one锛岃缁冨悗headstart-joint=+0.120651423172銆傝瑙乻w_rqs_staging_v58_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v58鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
joint锛歋W 73.587566682801锛屾渶浣崇4杞紝鍋滄绗?2杞紙patience18锛夈€?
headstart锛歋W 73.708218105973锛屾渶浣崇2杞紝鍋滄绗?0杞紙patience18锛夈€?
鍏堥€傞厤-鍏卞悓SW=+0.120651423172锛涘綋鍓峉W=73.713975628494锛屽疄闄呯粍鍚圱OTAL=80.354325927816銆傛柊閲囩敤=False銆?
鍏堥€傞厤鏈夊尮閰嶅眬閮ㄦ敹鐩婏紝淇濈暀鍏舵渶浣冲強瀹屾暣鐘舵€侊紱涓嶅皢鍏朵綆浜庣幇鏈塖W鐨勭粷瀵规垚缁╁啓鎴愭柊鍐犲啗銆傝鏀剁泭鏉ヨ嚜鍙嶅浣跨敤鐨勫紑鍙戞姌锛屾湭瀹屾垚鐙珛浜曠‘璁ゃ€?
涓ょ粍鍚?40771鍙傛暟鍙婂垵濮婻QS鍑芥暟锛寁56鍔犳硶楠ㄥ共鍜岄粯璁ゅご淇濈暀锛涙柊瀵嗗害璁粌渚у厛楠屻€俬eadstart鍓?杞粎4386瀵嗗害鍙傛暟锛岀6杞叏瑙ｅ喕锛屽叾浣欑浉鍚岋紱鍏ㄧ▼鍙敤SW鏍囩鍙婂師Q/RQS瑙ｆ瀽鎹熷け銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?True銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[222.114, 191.98]绉掞紱鍙屽苟鍙戞€诲閽?26.230绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1277.44, 1278.72]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡19070070瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v58鐪熷疄闃舵涓庣粷瀵硅竟鐣?
{"joint": {"stopped_epoch": 22, "best_epoch": 4, "density_only_epochs": 0, "backbone_update_steps": 1408, "selected_body_parameter_change_L2": 1.4109308771569136}, "headstart": {"stopped_epoch": 20, "best_epoch": 2, "density_only_epochs": 5, "backbone_update_steps": 960, "selected_body_parameter_change_L2": 0.0}}
鍏堥€傞厤鏈€浣崇2杞彂鐢熷湪浠呭瘑搴︽洿鏂伴樁娈碉紝閫夊畾楠ㄥ共鍙樺寲L2涓?锛涚6杞‘瀹炶В鍐伙紝鎵€鏈夌粍姊害闈為浂锛屼絾涔嬪悗鏈秴杩囩2杞€備笉鑳芥妸姝ゆ鏀剁泭瑙ｉ噴涓鸿В鍐诲悗鍏ㄧ綉缁滄彁鍗囥€備袱缁勫垵濮婼W72.415319274399鐩稿悓锛屽厛閫傞厤鏈€缁?3.708218105973杈冨叡鍚?3.587566682801楂?.120651423172锛屼絾姣斿綋鍓?3.713975628494浣?.005757522521銆傚厛閫傞厤瀹為檯鏇挎崲TOTAL80.352310794934锛屽綋鍓嶇粍鍚?0.354325927816淇濇寔銆?
淇濈暀鍏堥€傞厤best/last涓庡畬鏁?44665鐐圭鐩橀噸寤哄嚟鎹紱鍏卞悓璁粌璐熸潈閲嶅凡鍒犻櫎锛屽師RQS/v56鍙婂叾浠栨鍊欓€夈€佷氦浠樹繚鐣欍€傚垎浣嶇偣/灏捐川閲?鍏紡鍧囨湭鏀癸紝鏃ф暟鍊奸獙璇佷笉閲嶅銆傚鐢ㄧ殑閫氱敤閰嶇疆鎻忚堪鎻愬埌linear/RQS鏄疄鐜版敮鎸佽寖鍥达紝鏈壒density_mode鍥哄畾rqs锛屼袱椤瑰彧宸樁娈碉紱鏈缁僱inear銆傜嫭绔嬩簳纭銆佺浜斾弗鏍?80.5鍙婂畬鏁碠OF>82浠嶆湭瀹屾垚銆?


## v59 affine鏄惧紡鍚姩
{"pid": 48032, "mode": "affine", "started_at": "2026-09-29T21:22:54.685217+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_cached_readout_v59.py", "--mode", "affine", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_cached_readout_v59_SW_affine_fold5"], "automatic_queue": false}


## v59 silu鏄惧紡鍚姩
{"pid": 73624, "mode": "silu", "started_at": "2026-09-29T21:22:59.543433+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_cached_readout_v59.py", "--mode", "silu", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_cached_readout_v59_SW_silu_fold5"], "automatic_queue": false}


## v59鏈€缁堢粨鏋?
瀹為檯TOTAL=80.354325927816锛孲iLU-affine SW=+0.000000000000锛岄噰鐢∟one锛屼繚鐣橻]銆備袱椤瑰畬鎴愶紝淇濈暀鑰呭師濮嬭緭鍏?44665鐐归噸寤轰竴鑷达紱璐熸潈閲嶅強鍙啀鐢熺紦瀛樻竻鐞嗭紝澶辫触鍜岃€楁椂濡傚疄璁板綍銆傝瑙乻w_cached_readout_v59_20260929/FINAL.md銆傜洰鏍囨湭瀹屾垚锛屾棤鍏朵粬鎶?鑷姩闃熷垪/浜や粯淇敼銆?


## v60 likelihood鏄惧紡鍚姩
{"pid": 30712, "mode": "likelihood", "started_at": "2026-09-29T21:30:13.039963+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_brier_v60.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_brier_v60_SW_likelihood_fold5"], "automatic_queue": false}


## v60 brier鏄惧紡鍚姩
{"pid": 68940, "mode": "brier", "started_at": "2026-09-29T21:30:17.826971+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_brier_v60.py", "--mode", "brier", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_brier_v60_SW_brier_fold5"], "automatic_queue": false}


## v60鏈€缁堢粨鏋?
瀹為檯TOTAL=80.354325927816锛孊rier-likelihood SW=+0.000000000000锛岄噰鐢∟one锛屼繚鐣橻]銆備袱椤瑰畬鎴愶紝淇濈暀鑰呭師濮嬭緭鍏?44665鐐归噸寤轰竴鑷达紱璐熸潈閲嶅強鍙啀鐢熺紦瀛樻竻鐞嗭紝鑰楁椂濡傚疄璁板綍銆傝瑙乻w_brier_v60_20260929/FINAL.md銆傜洰鏍囨湭瀹屾垚锛屾棤鍏朵粬鎶?鑷姩闃熷垪/浜や粯淇敼銆?


## v61 coarse_query鏄惧紡鍚姩
{"pid": 3784, "mode": "coarse_query", "started_at": "2026-09-29T21:36:27.574458+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_upsample_v61.py", "--mode", "coarse_query", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_upsample_v61_SW_coarse_query_fold5"], "automatic_queue": false}


## v61 local_query鏄惧紡鍚姩
{"pid": 69912, "mode": "local_query", "started_at": "2026-09-29T21:36:32.646371+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_upsample_v61.py", "--mode", "local_query", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_upsample_v61_SW_local_query_fold5"], "automatic_queue": false}


## v61娣卞害杩樺師瀵圭収瀹屾垚
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.354325927816锛岄噰鐢∟one锛岃缁冨悗local-coarse=-0.007137240669銆傝瑙乻w_upsample_v61_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v61鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
coarse_query锛歋W 73.713975628494锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
local_query锛歋W 73.713975628494锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
灞€閮?绮楁煡璇W=+0.000000000000锛涘綋鍓峉W=73.713975628494锛屽疄闄呯粍鍚圱OTAL=80.354325927816銆傛柊閲囩敤=False銆?
绮剧粏灞€閮ㄦ煡璇㈡棤鍖归厤鏀剁泭锛屽仠姝㈠綋鍓嶆彃鍊紃ank/閭诲眳鏁?娓╁害鎵弿锛涘彧闄愬綋鍓嶅弻閭诲眳缁撴瀯锛屼笉娉涘寲鎵€鏈夌缁忎笂閲囨牱銆?
鍚?42585鍙傛暟銆佸畬鏁村弬鏁颁笌144665鐐瑰垵濮嬮娴嬬浉鍚岋紝鏂板rank16鍙岄偦灞呮彃鍊兼ā鍧?104鍙傛暟锛宷uery闆跺垵濮嬶紱鍘熻緭鍑哄ご涓嶉噸缃紝鍏ㄧ綉缁滃彧鐢⊿W鏍囩涓嶨aussian鎹熷け锛屽悓lr3e-5銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?False銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[179.664, 178.092]绉掞紱鍙屽苟鍙戞€诲閽?87.754绉掞紝鍖呭惈鍔犺浇/杈撳叆搴?棰勭儹/鍚姩闂撮殧锛涙瘡浠诲姟宄板€糓iB=[1359.76, 1359.92]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡38219084瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v62 gate_aux鏄惧紡鍚姩
{"pid": 49940, "mode": "gate_aux", "started_at": "2026-09-29T21:45:58.152390+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_density_private_v62.py", "--mode", "gate_aux", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_density_private_v62_SW_gate_aux_fold5"], "automatic_queue": false}


## v62 numeric_only鏄惧紡鍚姩
{"pid": 59752, "mode": "numeric_only", "started_at": "2026-09-29T21:46:03.195833+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_density_private_v62.py", "--mode", "numeric_only", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_density_private_v62_SW_numeric_only_fold5"], "automatic_queue": false}


## v62 gate_aux explicit serial recovery
{"pid": 55664, "started_at": "2026-09-29T21:49:48.451649+08:00", "mode": "gate_aux", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_density_private_v62.py", "--mode", "gate_aux", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_density_private_v62_SW_gate_aux_fold5", "--resume"], "serial_recovery": true, "automatic_queue": false, "reason": "Both initial jobs failed with CUDA OOM after completed epoch7; source/config unchanged; resume atomic optimizer/scheduler/RNG state"}


## v62 numeric_only explicit serial recovery
{"pid": 46332, "started_at": "2026-09-29T21:51:52.791342+08:00", "mode": "numeric_only", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_density_private_v62.py", "--mode", "numeric_only", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_density_private_v62_SW_numeric_only_fold5", "--resume"], "serial_recovery": true, "automatic_queue": false, "reason": "Both initial jobs failed with CUDA OOM after completed epoch7; source/config unchanged; resume atomic optimizer/scheduler/RNG state"}


## v62瀵嗗害绉佹湁楠ㄥ共瀵圭収瀹屾垚
涓ら」瀹屾垚锛屽疄闄匱OTAL=80.354325927816锛岄噰鐢∟one锛岃缁冨悗numeric-aux=-0.013721007637銆傝瑙乻w_density_private_v62_20260929/FINAL.md锛涙棤鍏朵粬鎶樻垨浜や粯淇敼銆?


## v62鏈€缁堢粨鏋滀笌淇濈暀杈圭晫
gate_aux锛歋W 73.713975628494锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
numeric_only锛歋W 73.713975628494锛屾渶浣崇0杞紝鍋滄绗?8杞紙patience18锛夈€?
浠呮暟鍊?杈呭姪SW=+0.000000000000锛涘綋鍓峉W=73.713975628494锛屽疄闄呯粍鍚圱OTAL=80.354325927816銆傛柊閲囩敤=False銆?
浠呮暟鍊肩洃鐫ｆ棤鍖归厤鏀剁泭锛屽仠姝㈡湰杈呭姪绯绘暟鍜屾敮璺暟鎵弿锛涘彧闄愬綋鍓嶅浐瀹氶粯璁?绉佹湁瀵嗗害缁撴瀯锛屼笉娉涘寲鍏ㄩ儴浠诲姟瑙ｈ€︺€?
鍚?678962鎬?839481浼樺寲鍣ㄥ弬鏁般€佸畬鏁村垵濮嬬姸鎬佸強144665鐐归娴嬬浉鍚岋紱榛樿鍒嗘敮鍐荤粨锛岀鏈夊瘑搴﹂骞插叏璁粌锛岃嚜宸辩殑gate浠呬綔鍙€夎緟鍔╋紝numeric_only棰濆129涓猤ate鍙傛暟鏃犳搴︿笖涓嶇敤浜庢帹鐞嗐€傚師SW/Q/Gaussian鎹熷け銆乴r3e-5淇濇寔銆傝缁冩搴︺€佹潵婧愬搱甯屻€佹渶浣虫仮澶嶅拰鐪熷疄涓夊垪璇勫垎宸叉牳楠岋紱淇濈暀鍊欓€夌鐩橀噸寤?False銆備笉閲嶅鏈敼瀵嗗害鍏紡妫€鏌ャ€?
鍗曢」鑰楁椂鍒嗗埆[188.23, 186.628]绉掞紱鍒濆骞跺彂澶辫触鍔犱覆琛屾仮澶嶇殑鎬诲閽?47.500绉掞紝鍖呭惈澶辫触灏濊瘯/鎭㈠闂撮殧/鍔犺浇/杈撳叆搴?棰勭儹锛涙姤鍛婂唴seconds涓嶅惈鏈繚瀛樼殑澶辫触杞伐浣滐紝attempts璁板綍鍏跺閽燂紱姣忎换鍔″嘲鍊糓iB=[1282.68, 1282.68]锛屼唬鐮佷笌鏈€缁堥噸寤哄彟璁°€?
宸插垹闄?涓湰杞礋妫€鏌ョ偣锛屽叡58450492瀛楄妭锛涗繚鐣欐簮鐮併€侀厤缃€佹洸绾裤€侀娴嬩笌鍝堝笇娓呭崟锛屽綋鍓嶆ā鍨嬪強鏃㈡湁姝ｅ€欓€?浜や粯涓嶅彉銆傛棤璁粌闃熷垪锛屾棤鍏朵粬鎶樸€?
绗簲鎶樹弗鏍?80.5鍜屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁彧鎼哄甫宸茶瘉瀹炲眬閮ㄦ敹鐩婃帰绱㈢嫭绔嬫満鍒讹紱鏂板疄楠屽皻鏈惎鍔紝涓嶈涓烘敹鐩娿€傜洰鏍囧畬鎴愬悗鍙仛鐙珛浜曠‘璁ゅ強鏂扮洰褰曚氦浠樸€?


## v63 point鏄惧紡鍚姩
{"pid": 73420, "mode": "point", "started_at": "2026-09-29T22:00:54.015636+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_internal_adapter_v63.py", "--mode", "point", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_internal_adapter_v63_POR_point_fold5"], "automatic_queue": false}


## v63 local鏄惧紡鍚姩
{"pid": 29560, "mode": "local", "started_at": "2026-09-29T22:04:49.750125+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_internal_adapter_v63.py", "--mode", "local", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_internal_adapter_v63_POR_local_fold5"], "automatic_queue": false}


## v63鏈€缁堢粨鏋?
涓ら」涓茶瀹屾垚锛孴OTAL=80.354325927816锛宭ocal-point POR=+0.000000000000锛堣缁冨悗宸?0.004661976642锛夈€傞噰鐢?None锛屼繚鐣橻]锛岃礋鏉冮噸宸叉竻鐞嗐€傚師妯″瀷/浜や粯涓嶅彉锛屾棤鍏朵粬鎶?闃熷垪锛涚洰鏍囨湭瀹屾垚锛岃por_internal_adapter_v63_20260929/FINAL.md銆?


## v64 interval鏄惧紡鍚姩
{"pid": 74192, "mode": "interval", "started_at": "2026-09-29T22:15:43.782358+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_interval_v64.py", "--mode", "interval", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_interval_v64_POR_interval_fold5"], "automatic_queue": false}


## v64鏈€缁堢粨鏋?
涓€椤规柊璁粌瀹屾垚锛屽鐢╲63 point瀵圭収銆俆OTAL=80.354325927816锛沬nterval-control POR=+0.000000000000锛堣缁冨悗宸?0.006178360861锛夛紱閲囩敤=False锛屼繚鐣?False銆傝礋鏉冮噸娓呯悊锛屾潵婧?浜や粯涓嶆敼锛涙棤鍏朵粬鎶樻垨闃熷垪锛岀洰鏍囨湭瀹屾垚锛岃por_interval_v64_20260929/FINAL.md銆?


## v65 detached鏄惧紡鍚姩
{"pid": 61196, "mode": "detached", "started_at": "2026-09-29T22:24:40.868917+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_transition_v65.py", "--mode", "detached", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_transition_v65_SW_detached_fold5"], "automatic_queue": false}


## v65 joint鏄惧紡鍚姩
{"pid": 57156, "mode": "joint", "started_at": "2026-09-29T22:26:46.861211+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_transition_v65.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_transition_v65_SW_joint_fold5"], "automatic_queue": false}


## v65鏈€缁堢粨鏋?
涓ら」涓茶瀹屾垚锛孴OTAL=80.354325927816锛沯oint-detached SW=+0.000000000000锛岃缁冨悗宸?-0.017791342092锛涢噰鐢?None銆傝礋鏉冮噸娓呯悊锛屾簮/浜や粯涓嶆敼锛屾棤鍏朵粬鎶橈紱鐩爣鏈畬鎴愶紝瑙乻w_transition_v65_20260929/FINAL.md銆?


## v66 permuted鏄惧紡鍚姩
{"pid": 72344, "mode": "permuted", "started_at": "2026-09-29T22:33:06.556430+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_well_order_v66.py", "--mode", "permuted", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_well_order_v66_SW_permuted_fold5"], "automatic_queue": false}


## v66 ordered鏄惧紡鍚姩
{"pid": 51792, "mode": "ordered", "started_at": "2026-09-29T22:34:43.129008+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_well_order_v66.py", "--mode", "ordered", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_well_order_v66_SW_ordered_fold5"], "automatic_queue": false}


## v66鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沷rdered-permuted SW=+0.000000000000锛岃缁冨悗宸?+0.007728273109锛涢噰鐢?None銆傝礋鏉冮噸娓呯悊锛屾簮/浜や粯涓嶆敼锛屾棤鍏朵粬鎶橈紱鐩爣鏈畬鎴愶紝瑙乻w_well_order_v66_20260929/FINAL.md銆?


## v67 linear鏄惧紡鍚姩
{"pid": 30688, "mode": "linear", "started_at": "2026-09-29T22:40:21.759592+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_quantile_v67.py", "--mode", "linear", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_quantile_v67_PERM_linear_fold5"], "automatic_queue": false}


## v67 rqs鏄惧紡鍚姩
{"pid": 7744, "mode": "rqs", "started_at": "2026-09-29T22:40:57.352614+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_quantile_v67.py", "--mode", "rqs", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_quantile_v67_PERM_rqs_fold5"], "automatic_queue": false}


## v67鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛況qs-linear PERM=+0.221226055194锛岃缁冨悗宸?+0.221226055194锛涢噰鐢?None锛屼繚鐣?['rqs']銆傝礋鏉冮噸娓呯悊锛屾簮/浜や粯涓嶆敼锛屾棤鍏朵粬鎶橈紱鐩爣鏈畬鎴愶紝瑙乸erm_quantile_v67_20260929/FINAL.md銆?


## v68 sum鏄惧紡鍚姩
{"pid": 32448, "mode": "sum", "started_at": "2026-09-29T22:51:37.573098+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_gradient_v68.py", "--mode", "sum", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_gradient_v68_PERM_sum_fold5"], "automatic_queue": false}


## v68 project鏄惧紡鍚姩
{"pid": 48816, "mode": "project", "started_at": "2026-09-29T22:52:54.171629+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_gradient_v68.py", "--mode", "project", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_gradient_v68_PERM_project_fold5"], "automatic_queue": false}


## v68鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沺roject-sum PERM=+0.000000000000锛岃缁冨悗宸?+0.006571455767锛涢噰鐢?None锛屼繚鐣?[]銆傝礋鏉冮噸娓呯悊锛屾簮/浜や粯涓嶆敼锛屾棤鍏朵粬鎶橈紱鐩爣鏈畬鎴愶紝瑙乸erm_gradient_v68_20260929/FINAL.md銆?


## v69 curvature鏄惧紡鍚姩
{"pid": 49620, "mode": "curvature", "started_at": "2026-09-29T23:01:38.292403+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_curvature_v69.py", "--mode", "curvature", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_curvature_v69_PERM_curvature_fold5"], "automatic_queue": false}


## v69鏈€缁堢粨鏋?
涓€涓柊鍊欓€夊畬鎴愶紝澶嶇敤鍘嗗彶sum鎺у埗锛孴OTAL=80.354325927816锛沜urvature-sum PERM=+0.000000000000锛岃缁冨悗宸?+0.011725166945锛涢噰鐢?None锛屼繚鐣?[]銆傝礋鏉冮噸娓呯悊锛屾簮/浜や粯涓嶆敼锛屾棤鍏朵粬鎶橈紱鐩爣鏈畬鎴愶紝瑙乸erm_curvature_v69_20260929/FINAL.md銆?


## v70 affine鏄惧紡鍚姩
{"pid": 56364, "mode": "affine", "started_at": "2026-09-29T23:10:50.457158+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_cdfwarp_v70.py", "--mode", "affine", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_cdfwarp_v70_PERM_affine_fold5"], "automatic_queue": false}


## v70 nonlinear鏄惧紡鍚姩
{"pid": 60140, "mode": "nonlinear", "started_at": "2026-09-29T23:13:28.380994+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_cdfwarp_v70.py", "--mode", "nonlinear", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_cdfwarp_v70_PERM_nonlinear_fold5"], "automatic_queue": false}


## v70鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛宯onlinear-affine PERM=+0.000000000000锛堣缁冨悗宸?0.000421751979锛夈€傞噰鐢?None锛屼繚鐣橻]锛岃礋鏉冮噸宸叉竻鐞嗐€傚師妯″瀷/浜や粯涓嶅彉锛屾棤鍏朵粬鎶?闃熷垪锛涚洰鏍囨湭瀹屾垚锛岃perm_cdfwarp_v70_20260929/FINAL.md銆?


## v71 symmetric鏄惧紡鍚姩
{"pid": 63048, "mode": "symmetric", "started_at": "2026-09-29T23:21:37.402870+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_depth_attention_v71.py", "--mode", "symmetric", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_depth_attention_v71_SW_symmetric_fold5"], "automatic_queue": false}


## v71瀹為檯鍚庣澶辫触涓庝慨澶?
鍘焥ymmetric棣栨鍙嶅悜澶辫触锛歊untimeError: LSE is not correctly aligned (strideH)锛屽凡瀹屾垚鍒濆144665鐐逛竴鑷翠絾鏃犺缁冭疆娆°€傚師婧愮爜銆侀厤缃€乫ailure/status淇濈暀涓斾笉鏀癸紱math鏂扮増鏈敼鏂板涓ゆ敮涓烘樉寮廎P32鐭╅樀娉ㄦ剰鍔涳紝鍘熸敞鎰忓姏淇濇寔锛岄噸鏂伴殢鏈虹姸鎬佸悓闆跺垵濮嬪寲鍚姩锛屼笉澹扮О鏃ц建杩圭画璺戙€?


## v71 math symmetric鏄惧紡鍚姩
{"pid": 18732, "mode": "symmetric", "started_at": "2026-09-29T23:23:05.377250+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_depth_attention_v71_math.py", "--mode", "symmetric", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_depth_attention_v71_math_SW_symmetric_fold5"], "automatic_queue": false}


## v71 math directional鏄惧紡鍚姩
{"pid": 58456, "mode": "directional", "started_at": "2026-09-29T23:24:25.110008+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_depth_attention_v71_math.py", "--mode", "directional", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_depth_attention_v71_math_SW_directional_fold5"], "automatic_queue": false}


## v71鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沝irectional-symmetric SW=+0.000000000000锛岃缁冨悗宸?-0.018606844709锛涢噰鐢?None銆傝礋鏉冮噸娓呯悊锛屾簮/浜や粯涓嶆敼锛屾棤鍏朵粬鎶橈紱鐩爣鏈畬鎴愶紝瑙乻w_depth_attention_v71_20260929/FINAL.md銆?


## v72 point鏄惧紡鍚姩
{"pid": 17092, "mode": "point", "started_at": "2026-09-29T23:32:33.373158+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_observation_v72.py", "--mode", "point", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_observation_v72_PERM_point_fold5"], "automatic_queue": false}


## v72 interval鏄惧紡鍚姩
{"pid": 14368, "mode": "interval", "started_at": "2026-09-29T23:34:51.127769+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_observation_v72.py", "--mode", "interval", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_observation_v72_PERM_interval_fold5"], "automatic_queue": false}


## v72鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沬nterval-point PERM=+0.000000000000锛岃缁冨悗宸?-0.006504305345锛涢噰鐢?None锛屼繚鐣?[]銆傝礋鏉冮噸宸叉竻鐞嗭紝鍘熸ā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紝鐩爣鏈畬鎴愩€傝瑙乸erm_observation_v72_20260929/FINAL.md銆?


## v73 permuted鏄惧紡鍚姩
{"pid": 13648, "mode": "permuted", "started_at": "2026-09-29T23:41:03.697052+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_patch_moments_v73.py", "--mode", "permuted", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_patch_moments_v73_SW_permuted_fold5"], "automatic_queue": false}


## v73 ordered鏄惧紡鍚姩
{"pid": 70920, "mode": "ordered", "started_at": "2026-09-29T23:42:17.851593+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_patch_moments_v73.py", "--mode", "ordered", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_patch_moments_v73_SW_ordered_fold5"], "automatic_queue": false}


## v73鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沷rdered-permuted SW=-0.010634580411锛岃缁冨悗宸?-0.010634580411锛涢噰鐢?None锛屼繚鐣?[]銆傝礋鏉冮噸宸叉竻鐞嗭紝鍘熸ā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紝鐩爣鏈畬鎴愩€傝瑙乻w_patch_moments_v73_20260929/FINAL.md銆?


## v73 鏂囨。缂栫爜淇
v72/v73鍚姩璇存槑缁廝owerShell绠￠亾浼犺緭鏇炬崯鍧忎腑鏂囷紝宸叉寜鍚勮嚜PLAN.md淇鐮旂┒/杈撳嚭README銆丳ROJECTS銆丄GENTS鍙婂叆鍙ｇ櫥璁帮紱涓嶆敼宸茶缁冪殑妯″瀷婧愮爜銆侀厤缃€佸搱甯屻€佸垎鏁版垨鏃堕棿銆倂73鍒濆鍖杝cope涓殑缁ф壙闈欐€乺andom body鏂囨宸插崟鐙緞娓咃細瀹為檯杞藉叆v56楠ㄥ共锛屽彧鏈夎緭鍑洪噸缃€?


## v74 control鏄惧紡鍚姩
{"pid": 50344, "mode": "control", "started_at": "2026-09-29T23:50:36.093876+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_key_dropout_v74.py", "--mode", "control", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_key_dropout_v74_SW_control_fold5"], "automatic_queue": false}


## v74 dropkey鏄惧紡鍚姩
{"pid": 45040, "mode": "dropkey", "started_at": "2026-09-29T23:51:41.068205+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_key_dropout_v74.py", "--mode", "dropkey", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_key_dropout_v74_SW_dropkey_fold5"], "automatic_queue": false}


## v74鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沝ropkey-control SW=+0.120506927208锛岃缁冨悗宸?+0.120506927208锛涢噰鐢?None锛屼繚鐣?['dropkey']銆傝礋鏉冮噸宸叉竻鐞嗭紝鍘熸ā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紝鐩爣鏈畬鎴愩€傝瑙乻w_key_dropout_v74_20260929/FINAL.md銆?


## v74鏀剁泭鏉ユ簮涓庡悗缁緷鎹?
榛樿SW瀛愮兢鏀瑰杽+0.709667913805锛屾暟鍊糞W瀛愮兢鍙樺寲-1.232741305323锛?6浜曚腑13鑳?3璐熴€傛€讳綋SW閰嶅+0.120506927208鎴愮珛锛屽綋鍓嶅啝鍐涙湭瓒呰秺锛屾暟鍊奸娴嬫病鏈夊悓姝ユ敼鍠勩€備繚鐣檇ropkey缁?44665鐐圭鐩橀噸寤虹殑姝ｅ€欓€夛紝鍚庣画鍙崟鐙楠屾鍒欎綔鐢ㄤ簬闂ㄦ帶/瀵嗗害鐨勮寖鍥达紱姝ゅ鏈柊澧炶缁冮槦鍒?鍏朵粬鎶樸€傚紑鍙戞姌缁撴灉涓嶆槸鐙珛浜曟樉钁楁€ц瘉鏄庛€?


## v75 both鏄惧紡鍚姩
{"pid": 22736, "mode": "both", "started_at": "2026-09-29T23:59:47.489680+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_scope_v75.py", "--mode", "both", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_scope_v75_SW_both_fold5"], "automatic_queue": false}


## v75 gate_only鏄惧紡鍚姩
{"pid": 4772, "mode": "gate_only", "started_at": "2026-09-30T00:01:00.258253+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_scope_v75.py", "--mode", "gate_only", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_scope_v75_SW_gate_only_fold5"], "automatic_queue": false}


## v75鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.354325927816锛沢ate_only-both SW=-0.013545294981锛岃缁冨悗宸?-0.013545294981锛涢噰鐢?None锛屼繚鐣?[]銆傝礋鏉冮噸宸叉竻鐞嗭紝鍘熸ā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紝鐩爣鏈畬鎴愩€傝瑙乻w_dropout_scope_v75_20260929/FINAL.md銆?


## v75骞跺彂鏁堢巼璁板綍
瀹為檯鍙屽苟鍙戞€诲閽?83.450544绉掞紝鍗曢」297.626852/305.448583绉掑潎鍚悇鑷姞杞介鐑缁冭瘎浠凤紝涓嶈兘鎶婂彈浜夌敤鐨勫崟椤圭浉鍔犵О涓茶瀵圭収銆傞浠诲姟鍗曡窇鍓嶄笁杞腑浣?.738绉掞紝绗簩浠诲姟灏鹃儴鍗曡窇鍥涜疆8.523绉掞紝閲嶅彔鍖洪浠诲姟绗?2鑷?8杞腑浣?6.854绉掞紱涓ら」鍚勪竴杞殑涓茶绮椾及16.261绉掞紝鏈樉绀哄苟鍙戝悶鍚愪紭鍔裤€傝繖鏄繍琛岄樁娈佃瀵燂紝闈炵瓑宸ヤ綔閲忕嫭绔嬮€熷害瀹為獙锛涘悗缁悓绫讳袱鍓嶅悜璇曢獙浼樺厛涓茶锛?浠嶆槸骞跺彂涓婇檺銆?


## v76 p05鏄惧紡鍚姩
{"pid": 52212, "mode": "p05", "started_at": "2026-09-30T00:11:36.261465+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_rate_v76.py", "--mode", "p05", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_rate_v76_SW_p05_fold5"], "automatic_queue": false}


## v76 p20鏄惧紡鍚姩
{"pid": 53944, "mode": "p20", "started_at": "2026-09-30T00:12:44.325140+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_rate_v76.py", "--mode", "p20", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_rate_v76_SW_p20_fold5"], "automatic_queue": false}


## v76杩愯鐜璁板綍
鏈疆杩愯鏈熼棿OS纭PowerLineStatus=Offline锛堢數姹犱緵鐢碉級锛孏PU瑙傚療鍒癙4/绾?0W/62搴︼紱鏃堕棿鐐瑰強鍚庣画蹇収瑙乺untime_power_observation.json銆傛湭淇敼鐢垫簮鏂规銆佹椂閽熸垨椹卞姩銆傚崟椤逛笌鎵规鎬诲閽熶粛瀹為檯璁板綍锛屼絾涓嶅彲鎶婁笌鍘嗗彶鎵规鐨勮€楁椂鍙樺寲鍏ㄥ綊鍥犱簬DropKey寮哄害鎴栧苟鍙戙€?


## v76鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.360506992951锛涙渶浣虫柊鍊欓€?鍘嗗彶10% SW=+0.045645806823锛屾渶浣虫柊妯″紡=p20锛涢噰鐢?p20锛屼繚鐣?['p20']銆傝礋鏉冮噸宸叉竻鐞嗭紝鍘熸ā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紝鐩爣鏈畬鎴愩€傝瑙乻w_dropout_rate_v76_20260930/FINAL.md銆?


## v76鎬诲閽熶笌鏀剁泭鏍搁獙
璁粌涓ら」瀹為檯鍚姩鑷虫渶鍚庡畬鎴?25.268845绉掞紙绾?.09鍒嗛挓锛夛紝绗?0%缁勪簬00:18:41瀹屾垚銆傛渶缁堢鐩橀噸寤虹粨鏋滄椂闂?8:51:24锛屽惎鍔ㄨ嚦鏍搁獙鎬诲閽?1188.028573绉掞紝璁粌缁撴潫鍚庨棿闅?0762.759728绉掞紱鍚庤€呬笉鏄疓PU璁粌鑰楁椂銆傜郴缁熸棩蹇楀彟璁板綍01:39杩涘叆鐫＄湢銆?8:45浣庡姛鑰楁仮澶嶅強鏍℃椂锛屼笉鑳芥妸鍏ㄩ儴闂撮殧閮藉綊鍥犱簬鐫＄湢銆傚綋鍓嶇數婧愬凡Online锛涙湭鏀圭郴缁熺數婧愯缃€?
閲囩敤p20绗?4杞紝SW73.731635814592锛涙柊缁勫悎TOTAL80.360506992951锛岀浉瀵瑰綋鍓嶅紑鍙戦厤鏂?0.006181065134锛屼粛鏈繃80.5锛屼笉琛ユ姌銆傛柊鏉冮噸144665鐐瑰師濮嬭緭鍏ラ噸寤洪€愪綅涓€鑷淬€?%鏈甫鏉ョ浉瀵?0%鐨勬柊鏀剁泭锛岃礋鏉冮噸娓呯悊锛泇74鐨?0%姝ｆ満鍒跺拰鏃у啝鍐?浜や粯淇濈暀銆?


## v77 p30鏄惧紡鍚姩
{"pid": 48392, "mode": "p30", "started_at": "2026-09-30T08:57:52.136699+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_extend_v77.py", "--mode", "p30", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_extend_v77_SW_p30_fold5"], "automatic_queue": false}


## v77 p40鏄惧紡鍚姩
{"pid": 3636, "mode": "p40", "started_at": "2026-09-30T08:58:56.859144+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_extend_v77.py", "--mode", "p40", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_extend_v77_SW_p40_fold5"], "automatic_queue": false}


## v77鏈€缁堢粨鏋?
涓ら」瀹屾垚锛孴OTAL=80.363582620846锛涙渶浣虫柊鍊欓€?鍘嗗彶20% SW=+0.008787508272锛屾渶浣虫柊妯″紡=p40锛涢噰鐢?p40锛屼繚鐣?['p40']銆傝礋鏉冮噸宸叉竻鐞嗭紝鍘熸ā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶橈紝鐩爣鏈畬鎴愩€傝瑙乻w_dropout_extend_v77_20260930/FINAL.md銆?


## v77鎬诲閽熶笌缁撹
褰撳墠鐙珛閫氶亾鐮旂┒璺嚎鐨凾OTAL80.363582620846锛岃緝v76+0.003075627895锛涢噰鐢?0%绗?5杞紝涓嶄唬琛ㄥ叏閮ㄥ巻鍙茶矾绾挎渶楂樺垎銆傜浉瀵?0%锛岄粯璁ゅ瓙缇?0.111979122070锛屾暟鍊煎瓙缇?0.286177275715锛?6浜?鑳?璐熴€傛寮€鍙戞姌鍙嶅閫夋嫨鐨勫井灏忔敹鐩婂皻闈炵嫭绔嬫硾鍖栬瘉鎹紝涓嶅惎鍔ㄥ叾浠栨姌銆?
30%绗?4杞甋W73.679835895016锛岃緝20%-0.051799919576锛屼笉缁х画璇ョ偣锛涘垹闄ゅ叾涓や釜checkpoint19,041,286瀛楄妭骞朵繚鐣欐簮鐮?閰嶇疆/绱у噾璇佹嵁銆?0%鍙婃棫20%銆?0%姝ｅ€欓€変繚鐣欙紝鍘熶氦浠樹笉鏀广€?0%鍘熷杈撳叆144665鐐圭鐩橀噸寤洪€愪綅涓€鑷达紝婧愮爜鍝堝笇/鍒濆鍙傛暟涓庨娴?鍘熻瘎鍒嗗潎閫氳繃銆?
涓や釜鍗曢」璁粌鍐呴儴鑰楁椂245.161415/295.348431绉掞紝鍚勫嘲鍊?266.920898MiB锛屽疄闄呮渶澶?骞跺彂銆傞娆″惎鍔ㄨ嚦鏈€鍚庤缁冪粨鏉?67.198987绉掞紙6.12鍒嗛挓锛屽惈鍔犺浇/棰勭儹/璁粌/璇勪环/淇濆瓨/鍚姩闂撮殧锛夈€傞娆″惎鍔ㄨ嚦纾佺洏鏍搁獙1110.690353绉掞紙18.51鍒嗛挓锛夛紝璁粌缁撴潫鑷虫牳楠?43.491366绉掞紱涓嶆妸鏍搁獙闂撮殧璁℃垚GPU璁粌锛屼篃涓嶅皢鏈壒绔埌绔鎴?.12鍒嗛挓銆傚惎鍔ㄥ墠浠ｇ爜涓庢牳楠屽悗鍥捐氨鏀跺熬鍙﹁銆?
鏈揪鍒颁弗鏍肩浜旀姌>80.5锛屽畬鏁村師浜旀姌OOF>82浠嶆湭瀹屾垚銆傚悗缁紭鍏堟洿鎹㈡鍒欏寲鏂藉姞浣嶇疆鎴栧涔犳満鍒讹紝涓嶄负杩欑偣寰皬鎻愬崌鏃犵晫缁嗘壂姣斾緥锛涙湭璁炬柊闃熷垪銆?


## v78 shared鏄惧紡鍚姩
{"pid": 63868, "mode": "shared", "started_at": "2026-09-30T09:23:18.042260+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_heads_v78.py", "--mode", "shared", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_heads_v78_SW_shared_fold5"], "automatic_queue": false}


## v78 per_head鏄惧紡鍚姩
{"pid": 31940, "mode": "per_head", "started_at": "2026-09-30T09:23:52.214704+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_dropout_heads_v78.py", "--mode", "per_head", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_dropout_heads_v78_SW_per_head_fold5"], "automatic_queue": false}


## v78鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾璺嚎TOTAL=80.363582620846锛岀浉瀵硅捣鐐?0.000000000000锛沺er_head-shared SW=+0.000000000000锛岃缁冨悗宸?0.024271478516銆傞噰鐢?None锛屼繚鐣?[]锛涜礋鎴栭噸澶嶆潈閲嶆竻鐞嗭紝鍘熸潵婧?浜や粯淇濈暀锛屾棤鍏朵粬鎶樸€傚畬鏁碠OF>82鏈畬鎴愩€傝sw_dropout_heads_v78_20260930/FINAL.md銆?


## v78鍏抽棴鑼冨洿涓庡疄闄呯粨璁?
涓ら」閮芥仮澶嶅師v77绗?5杞畬鏁村嚱鏁帮紙鏈壒鏈€浣宠疆0锛夛紝涓嶅啋绉版柊璁粌鍊欓€夋彁鍗囥€俻er_head璁粌鍚庢渶楂樿緝shared浣?.024271478516锛屽惈璧风偣涓ょ粍宸?锛涘綋鍓嶅ご闂寸嫭绔嬫帺鐮佽矾绾垮叧闂紝涓嶇户缁叾姒傜巼缁嗘壂锛屼篃涓嶉噸澶嶇浉鍚屼綆瀛︿範鐜囧叏缃戠粶寰皟銆傜粨璁洪檺浜庤繖娆″畬鏁磒40鐑惎鍔?3e-5/4灞傛帺鐮侊紝涓嶈兘鎺ㄥ嚭鎵€鏈夌嫭绔嬪ご姝ｅ垯鏃犳晥銆傚綋鍓嶇爺绌堕厤鏂筎OTAL80.363582620846淇濇寔锛涘洓涓礋/閲嶅checkpoint鎸夋竻鍗曞垹闄わ紝鍘焩77鍙婃潵婧愭ā鍨嬩繚鐣欍€?
鏃犳柊姝ｅ€欓€夛紝涓嶉澶栭噸澶嶅師濮嬭緭鍏PU閲嶅缓锛涘疄闄呰缁冨凡鏍稿鍒濆鍏ㄧ偣绛変簬鐖舵ā鍨嬨€佹渶浣虫仮澶嶄笌涓夊垪璇勫垎銆備粛鏈繃涓ユ牸80.5锛屾棤鍏朵粬鎶橈紝瀹屾暣OOF>82鐩爣鏈畬鎴愩€傚悗缁崲杈撳叆琛ㄥ緛鎴栬緭鍑哄涔犳満鍒讹紱鏈噯澶囪嚜鍔ㄩ槦鍒椼€?


## v79 marginal鏄惧紡鍚姩
{"pid": 61164, "mode": "marginal", "started_at": "2026-09-30T09:36:30.565676+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_input_products_v79.py", "--mode", "marginal", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_input_products_v79_SW_marginal_fold5"], "automatic_queue": false}


## v79 cross鏄惧紡鍚姩
{"pid": 51620, "mode": "cross", "started_at": "2026-09-30T09:37:17.249903+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_input_products_v79.py", "--mode", "cross", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_input_products_v79_SW_cross_fold5"], "automatic_queue": false}


## v79鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾璺嚎TOTAL=80.363582620846锛岀浉瀵硅捣鐐?0.000000000000锛沜ross-marginal SW=+0.000000000000锛岃缁冨悗宸?0.078260630721銆傞噰鐢?None锛屼繚鐣?[]锛涜礋鎴栭噸澶嶆潈閲嶆竻鐞嗭紝鍘熸潵婧?浜や粯淇濈暀锛屾棤鍏朵粬鎶樸€傚畬鏁碠OF>82鏈畬鎴愩€傝sw_input_products_v79_20260930/FINAL.md銆?


## v79鍏抽棴鑼冨洿涓庡疄闄呯粨璁?
涓ら」鍧囨仮澶嶅師v77瀹屾暣鍑芥暟锛屾湰鎵规渶浣宠疆0锛屽師濮嬫墍鏈夋潈閲嶅湪鏈姸鎬佸拰鎭㈠鍚庨€愪綅鍐荤粨銆傝缁冨悗鏈€楂楽W锛歮arginal=73.609305838286銆乧ross=73.531045207564锛宑ross-marginal=-0.078260630721锛涘惈璧风偣宸?锛屾棤缁濆鏀剁泭銆傚叧闂綋鍓嶄箻绉熀闃舵暟/缁勫悎鍙婅繖缁剆tem閫傞厤鍣ㄥ箙搴?瀹归噺缁嗘壂锛屼笉娉涘寲鍏ㄩ儴鐗瑰緛宸ョ▼銆?
鏃onfig淇濈暀涓嶅彲鏀癸紱鍏朵腑matched_key_dropout_comparison鏄ā鏉块仐鐣欏瓧娈碉紝鐪熷疄姣旇緝涓鸿緭鍏ュ熀銆佸疄闄呬袱缁勮缁僰ey澶辨椿鍧?锛岃metadata_clarification.json銆傛鏂囧瓧鍏冩暟鎹笉褰卞搷璁＄畻锛涙簮鐮佸搱甯屼繚鎸併€傚洓涓礋/閲嶅checkpoint24,642,364瀛楄妭鎸夋竻鍗曞垹闄わ紝瀹屾暣v77鍙婂巻鍙叉妯″瀷/浜や粯淇濈暀銆傛棤鏂版鍊欓€夛紝涓嶉澶朑PU閲嶅缓锛涘疄闄呭垵濮嬮娴?鏈€浣虫仮澶?鍘熶笁鍒楄瘎鍒嗛€氳繃銆傚綋鍓峊OTAL80.363582620846淇濇寔锛屼弗鏍?0.5涓庡畬鏁碠OF>82鏈揪鎴愶紝鏃犲叾浠栨姌/鏃ч槦鍒椼€?
涓や釜鍗曢」108.599860/107.835349绉掞紝宄板€?155.106934/1154.857422MiB锛屽疄闄?骞跺彂锛涘惎鍔ㄨ嚦鏈€鍚庤缁冪粨鏉?60.299055绉掞紙2.67鍒嗛挓锛夛紝鑷崇粨鏋滄牳楠?90.191695绉掞紙4.84鍒嗛挓锛夈€傚姞杞介鐑缁冭瘎浠蜂繚瀛樺強鍚姩闂撮殧璁″叆锛屼綔鑰呭噯澶囦笌鍥捐氨鏀跺熬鍙﹁銆傚悗缁紭鍏堣緭鍑哄涔犳満鍒讹紝涓嶇户缁繖缁勮緭鍏ュ熀缁嗘壂锛涙湭鍚姩鏂伴槦鍒椼€?


## v80 likelihood鏄惧紡鍚姩
{"pid": 67656, "mode": "likelihood", "started_at": "2026-09-30T09:56:24.371075+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_head_utility_v80.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_head_utility_v80_SW_likelihood_fold5"], "automatic_queue": false}


## v80r1 likelihood鏄惧紡鍚姩
{"pid": 49288, "mode": "likelihood", "started_at": "2026-09-30T10:00:15.914655+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_head_utility_v80r1.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_head_utility_v80r1_SW_likelihood_fold5"], "automatic_queue": false}


## v80r1 joint鏄惧紡鍚姩
{"pid": 31580, "mode": "joint", "started_at": "2026-09-30T10:01:41.775650+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_head_utility_v80r1.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_head_utility_v80r1_SW_joint_fold5"], "automatic_queue": false}


## v80棣栨澶辫触涓巚80r1鏄惧紡淇
棣栨likelihood鍦ㄧ1杞浜宐atch杈呭姪姊害璁板綍澶辫触锛岃鐢╪o_grad鎺ㄧ悊璐ㄩ噺鍑芥暟锛涗粎涓€姝ヤ紭鍖栵紝鏃犲畬鎴愯疆/鏂伴獙璇佸垎锛宩oint鏈惎鍔ㄣ€傝繖涓嶆槸绠楁硶璐熺粨鏋溿€傚師婧愮爜/config/hash淇濈暀锛岄噸澶峞poch0涓や釜checkpoint鎸夋竻鍗曞垹锛涘け璐ヤ俊鎭sw_head_utility_v80_20260930/failure_summary.json銆?
v80r1鏂板鍙井Gaussian鍖洪棿璐ㄩ噺锛屾棤闄愯竟鐣屽厛鏈夐檺鏇夸唬鍐嶈缃瓹DF甯告暟灏撅紱鍚岄瀹氭暟鎹?妯″瀷/鎹熷け/绯绘暟/棰勭畻锛屼粎淇璺緞銆傞噸鏂颁粠瀹屾暣v77璧风偣鏂扮洰褰曞惎鍔ㄤ袱椤癸紝涓嶆仮澶嶅け璐ョ姸鎬侊紱瀹為檯璁粌闂ㄦ帶鍜屽瘑搴﹁緟鍔╂搴﹀潎姝ｃ€佹棫鏉冮噸鏃犳搴︺€傚師鎺ㄧ悊涓嶆敼锛屾湭瀹屾垚鏀剁泭涓嶆彁鍓嶅０鏄庛€備繚鎸乵ax2锛屾棤鍏朵粬鎶?鏃ч槦鍒?浜や粯瑕嗙洊銆?


## v80r1鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾璺嚎TOTAL=80.363582620846锛岀浉瀵硅捣鐐?0.000000000000锛沯oint-likelihood SW=+0.000000000000锛岃缁冨悗宸?0.002902171987銆傞噰鐢?None锛屼繚鐣?[]锛涜礋鎴栭噸澶嶆潈閲嶆竻鐞嗭紝鍘熸潵婧?浜や粯淇濈暀锛屾棤鍏朵粬鎶樸€傚畬鏁碠OF>82鏈畬鎴愩€傝sw_head_utility_v80r1_20260930/FINAL.md銆?


## v80r1瀹為檯缁撹涓庡叧闂寖鍥?
涓ら」鍧囨仮澶嶅師v77绗?5杞畬鏁村嚱鏁帮紝鏈壒鏈€浣宠疆0锛屽綋鍓嶄笁閫氶亾TOTAL80.363582620846涓嶅彉銆傛洿鏂板悗鏈€楂楽W锛歭ikelihood=73.483006007207锛宩oint=73.485908179195锛屽樊+0.002902171987锛涜繖涓€寰皬閰嶅璇婃柇宸笌鍚捣鐐规棤鎻愬崌鍒嗗紑锛屼笉褰撶嫭绔嬫硾鍖栨垨瀹屾暣鏀剁泭銆傚疄闄呰緟鍔╅」闂ㄦ帶/瀵嗗害姊害鍧囨涓旀湁闄愶紝鎵€鏈夐潪杈撳嚭鏉冮噸鍦ㄨ缁冩湯鐘舵€佸拰鎭㈠鍚庨€愪綅鍐荤粨銆?
鍏抽棴褰撳墠瀹屾暣鍑芥暟鐑惎鍔ㄣ€佸浐瀹氶骞插彧鎷熷悎杈撳嚭灞傜殑LR/娓╁害/绯绘暟瀵嗛泦鎵弬銆傝绐勫眬閮ㄨ緟鍔╂崯澶辩嚎绱㈠彲渚涗笉鍚岄殧绂诲涔犲満鏅弬鑰冿紝涓嶅绉版櫘閬嶆湁鏁堛€傚洓涓柊璐?閲嶅checkpoint鎸夋竻鍗曞垹闄わ紱鏈€浼樺瓨鐨勬槸epoch0閲嶅鐖跺嚱鏁帮紝鏇存柊鍚庡嘲鍊兼潈閲嶆湭鍗曠嫭淇濈暀锛屼笉鎶婅宄板€艰鎴愬彲鐢ㄥ畬鏁存ā鍨嬨€傚師v77銆佸巻鍙叉鍊欓€変笌鏃т氦浠樹繚鐣欙紱鏃犳柊姝ｅ€欓€夛紝涓嶉噸澶岹PU纾佺洏閲嶅缓銆?
棣栨v80瀹炵幇澶辫触鏈畬鎴愪竴杞紝涓嶈兘褰撶畻娉曡礋缁撴灉锛涘師no_grad鎺ㄧ悊鍑芥暟婧愮爜/config/hash鐣欐。锛屼袱涓噸澶嶅垵濮媍heckpoint宸插垹12,223,982瀛楄妭銆倂80r1涓烘柊鏂囦欢/鏂拌繍琛岋紝涓嶆仮澶嶅け璐ョ姸鎬侊紱鍘烥aussian鎺ㄧ悊瑙ｇ爜鏈敼銆傚畬鏁村師浜旀姌OOF>82涓庝弗鏍煎師绗簲>80.5浠嶆湭杈惧埌锛屾棤鍏朵粬鎶?鏃ч槦鍒椼€?
淇鐗堝崟椤?2.120694/92.817162绉掞紝宄板€?129.143066MiB锛屾渶澶?杩涚▼锛屼絾瀹為檯浠?1.002550绉掕繘绋嬮噸鍙狅紝涓昏涓茶锛屼笉鑳藉綋鍙屽苟鍙戝悶鍚愯瘉鎹€備慨姝ｇ増鍚姩鑷虫渶鍚庤缁冪粨鏉?83.446918绉掞紙3.06鍒嗛挓锛夛紝鑷虫牳楠?33.910498绉掞紙5.57鍒嗛挓锛夛紱棣栨澶辫触鍚姩鑷充慨姝ｇ増鏍搁獙565.454078绉掞紙9.42鍒嗛挓锛夛紝鍖呮嫭鎺掗殰/淇/閲嶆柊鍚姩闂撮殧銆傚師澶辫触鍚姩鑷抽敊璇枃浠?3.774927绉掍粎澧欓挓锛屼笉鍐掔ОGPU璁粌鑰楁椂銆傚姞杞介鐑?璁粌/璇勪环/淇濆瓨/鍚姩闂撮殧璁″叆锛屽垵娆″惎鍔ㄥ墠浠ｇ爜涓庢牳楠屽悗鍥捐氨鏀跺熬鍙﹁銆?


## v81 likelihood鏄惧紡鍚姩
{"pid": 55392, "mode": "likelihood", "started_at": "2026-09-30T10:15:10.912282+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_full_utility_v81.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_full_utility_v81_SW_likelihood_fold5"], "automatic_queue": false}


## v81 joint鏄惧紡鍚姩
{"pid": 69108, "mode": "joint", "started_at": "2026-09-30T10:18:13.559182+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_full_utility_v81.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_full_utility_v81_SW_joint_fold5"], "automatic_queue": false}


## v81鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾璺嚎TOTAL=80.380136601599锛岀浉瀵硅捣鐐?0.016553980753锛沯oint-likelihood SW=+0.051945916102锛岃缁冨悗宸?0.051945916102銆傞噰鐢?joint锛屼繚鐣?['joint']锛涜礋鎴栭噸澶嶆潈閲嶆竻鐞嗭紝鍘熸潵婧?浜や粯淇濈暀锛屾棤鍏朵粬鎶樸€傚畬鏁碠OF>82鏈畬鎴愩€傝sw_full_utility_v81_20260930/FINAL.md銆?

## v81鍥捐氨涓庢渶缁堢暀瀛樻牳楠?
鍥捐氨鑺傜偣鏄惧紡鏄剧ず绗簲鎶樺浐瀹歅OR/PERM缁勫悎TOTAL80.380136601599鍙婂崟鐩爣SW73.787720410731锛涗笉娣蜂负瀹屾暣OOF銆傜绾跨敓鎴?18瀹為獙/82鍏抽棴璺嚎锛屽疄闄呭垎鏁拌鍥炰竴鑷淬€傝仈鍚堢粍best/last涓庢棫p40浠嶅湪锛屼袱涓礋瀵圭収checkpoint宸插垹闄?9051078瀛楄妭锛寁81璁粌杩涚▼涓?銆傝缁冨埌楠岃瘉579.349273绉掞紱鍒版湰鎵规枃妗?鍥捐氨鏀跺熬澧欓挓890.723绉掞紝楠岃瘉鍚庢敹灏?11.373绉掞紙鍚畾浣嶇粍鍚堝瓧娈点€佸浘璋辨洿鏂板拰璇诲洖锛夛紝璇﹁artifact_finalize.json銆傛棤鑷姩鍚庣画鍚姩鎴栬ˉ鎶橈紱鐩爣浠嶆湭瀹屾垚銆?


## v82 a025鏄惧紡鍚姩
{"pid": 68224, "mode": "a025", "started_at": "2026-09-30T10:34:32.216290+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_utility_strength_v82.py", "--mode", "a025", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_utility_strength_v82_SW_a025_fold5"], "automatic_queue": false}


## v82 a100鏄惧紡鍚姩
{"pid": 79580, "mode": "a100", "started_at": "2026-09-30T10:39:48.485742+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_utility_strength_v82.py", "--mode", "a100", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_utility_strength_v82_SW_a100_fold5"], "automatic_queue": false}


## v82鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾鍘熺浜擳OTAL=80.382093291214锛岀浉瀵?5缁勫悎+0.001956689615锛涚郴鏁扮綉鏍艰sw_utility_strength_v82_20260930/FINAL.md锛岄噰鐢?a100锛屼繚鐣?['a100']銆傛棤鍏朵粬鎶?鏃т换鍔★紝璐熸潈閲嶆寜娓呭崟娓呯悊锛屽師鏉ユ簮/浜や粯淇濈暀锛涘畬鏁碠OF>82鏈畬鎴愩€?

## v82鏀跺熬涓庝笅涓€姝?
褰撳墠鍘熺浜旂嫭绔嬭矾绾縏OTAL80.382093291214锛岀浉瀵?5浠?0.001956689615銆?鐩稿.5榛樿SW+0.050181618258銆佹暟鍊?0.096831032031锛?浜曡儨/10浜曡礋锛涗繚鐣?鐨勬渶楂樺垎鍊欓€夊強.5瀵圭収锛屼笉澹扮О鏅亶鏀瑰杽銆備笅涓€鎵硅浆鍚戞晥鐢ㄤ汉鍙ｆ潈閲嶏紙棰勫畾涔夊潎琛?鏁板€间紭鍏堛€佽缁冧晶鎷熷悎锛夛紝瑙丯EXT.md锛涘皻鏈疄鐜?鍚姩锛屼笉鎵╁ぇ鏈郴鏁扮綉鏍笺€傚凡鍒犻櫎璐?25涓や唤鏉冮噸19052230瀛楄妭銆侴PU瀹為檯涓茶銆佸崟浠诲姟宄板€?870.504MiB锛涢鍚姩鍒伴獙璇?62.391绉掞紝鍒版湰鎵规枃妗?鍥捐氨鍜屽瓙缇よВ閲婃敹灏?153.097绉掞紝楠岃瘉鍚?90.706绉掞紱artifact_finalize.json鏄庣‘鑼冨洿銆傚浘璋?20瀹為獙/82鍏抽棴璺嚎锛岀粍鍚堝拰鍗昐W鍒嗘暟瀹為檯璇诲洖涓€鑷淬€傛棤鍏朵粬鎶橈紝涓ユ牸>80.5鍜屽畬鏁碠OF>82浠嶆湭瀹屾垚銆?


## v83 balanced鏄惧紡鍚姩
{"pid": 47720, "mode": "balanced", "started_at": "2026-09-30T10:58:58.125342+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_population_utility_v83.py", "--mode", "balanced", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_population_utility_v83_SW_balanced_fold5"], "automatic_queue": false}


## v83 numeric75鏄惧紡鍚姩
{"pid": 7752, "mode": "numeric75", "started_at": "2026-09-30T10:59:52.736332+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_population_utility_v83.py", "--mode", "numeric75", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_population_utility_v83_SW_numeric75_fold5"], "automatic_queue": false}


## v83鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾鍘熺浜擳OTAL=80.382093291214锛岀浉瀵?5缁勫悎+0.000000000000锛涚郴鏁扮綉鏍艰sw_population_utility_v83_20260930/FINAL.md锛岄噰鐢?None锛屼繚鐣?[]銆傛棤鍏朵粬鎶?鏃т换鍔★紝璐熸潈閲嶆寜娓呭崟娓呯悊锛屽師鏉ユ簮/浜や粯淇濈暀锛涘畬鏁碠OF>82鏈畬鎴愩€?

## v83鏀跺熬銆佽竟鐣屼笌涓嬩竴姝?
涓ら」鍧囨棤鏁翠綋鏀剁泭锛屽綋鍓嶇嫭绔嬭矾绾跨浜擳OTAL80.382093291214涓嶅彉銆俠alanced鏈€浣?7/鍋滄38锛屾暟鍊糞W+0.487570浣嗛粯璁?0.468565锛沶umeric75鏈€浣?4/鍋滄32锛屾暟鍊?0.141598浣嗛粯璁?0.749867銆傚潎琛＄粍鍦ㄦ棭鍏堢31杞瀵熷悗浠嶆彁楂樺埌37杞紝鍋滄鐞嗙敱寮曠敤鏃у揩鐓э紱瀹為檯璇锋眰鐘舵€佸拰鏈€缁堟仮澶嶄负73.614756锛岀38杞彁鍓嶇粨鏉燂紝涓嶈兘绉?0杞畬鏁磋礋缁撴灉銆俷umeric75鑷劧pat18缁撴潫锛屾嫙鍋滄鏃禤ID宸蹭笉瀛樺湪锛屾病鏈夊啓璇锋眰鎴栭噸鍚€傚洓浠借礋鏉冮噸宸插垹闄?8096908瀛楄妭锛屾棫姝ｆā鍨?浜や粯淇濈暀銆備笅涓€鎵硅鍒掑浐瀹氬師榛樿缃戠粶銆佽缁冪鏈夋暟鍊煎垎甯冿紝涓や釜浼肩劧/鍧囪　鏁堢敤瀵圭収锛涜NEXT.md锛屽皻鏈疄鐜版垨鍚姩銆侴PU瀹為檯宄板苟鍙?銆佸悇1870.504MiB锛涢鍚姩鍒伴獙璇?02.257绉掞紝鍒版枃妗?鍥捐氨鍙婂瓙缇よВ閲婃敹灏?060.358绉掞紝楠岃瘉鍚?58.1绉掞紝鑼冨洿瑙乤rtifact_finalize.json銆傚浘璋?22瀹為獙/83鍏抽棴璺嚎锛屽垎鏁拌鍥炰竴鑷达紱鏃犳椿璺僾83杩涚▼/鍏朵粬鎶樸€傜浜斾弗鏍?80.5鍙婂畬鏁碠OF>82浠嶆湭瀹屾垚銆?


## v84 likelihood鏄惧紡鍚姩
{"pid": 70312, "mode": "likelihood", "started_at": "2026-09-30T11:25:00.808507+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_frozen_gate_utility_v84.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_frozen_gate_utility_v84_SW_likelihood_fold5"], "automatic_queue": false}


## v84r1 likelihood鏄惧紡鍚姩
{"pid": 47384, "mode": "likelihood", "started_at": "2026-09-30T11:27:59.374171+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_frozen_gate_utility_v84r1.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_frozen_gate_utility_v84r1_SW_likelihood_fold5"], "automatic_queue": false}


## v84r1 balanced鏄惧紡鍚姩
{"pid": 76256, "mode": "balanced", "started_at": "2026-09-30T11:29:19.155711+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_frozen_gate_utility_v84r1.py", "--mode", "balanced", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_frozen_gate_utility_v84r1_SW_balanced_fold5"], "automatic_queue": false}


## v84r1鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾鍘熺浜擳OTAL=80.382093291214锛岀浉瀵?5缁勫悎+0.000000000000锛涚郴鏁扮綉鏍艰sw_frozen_gate_utility_v84r1_20260930/FINAL.md锛岄噰鐢?None锛屼繚鐣?[]銆傛棤鍏朵粬鎶?鏃т换鍔★紝璐熸潈閲嶆寜娓呭崟娓呯悊锛屽師鏉ユ簮/浜や粯淇濈暀锛涘畬鏁碠OF>82鏈畬鎴愩€?

## v84r1鏀跺熬涓庢柟鍚戞洿鏂?
涓ら」閮?8杞畃at18缁撴潫锛屾仮澶嶇0杞綋鍓嶅師鍑芥暟锛涙棤鏂板鏀剁泭锛屽綋鍓嶇嫭绔嬮€氶亾鍘熺浜擳OTAL80.382093291214涓嶅彉銆傚垵濮?44665鐐逛笌鍘熷嚱鏁扮浉鍚屻€佸喕缁撻粯璁ゅ綋鍓?鏈€浣砲ash鍧囦笉鍙橈紱璁粌鍚庢渶楂樺潎琛?浼肩劧SW=-0.046208233406銆傚仠姝㈡湰鏉ユ簮/绉佹湁瀵嗗害/3e-5/浼肩劧鎴栧潎琛℃晥鐢ㄧ粍鍚堬紝涓嬩竴鎵规敼杈撳嚭灏忔畫宸鍑虹粨鏋勶紝瑙丯EXT.md锛屽皻鏈惎鍔ㄣ€傚師v84缂簂ength鍒濆鍖栧け璐ワ紝鏃犲垵濮嬫垚缁?璁粌锛屽師婧愮爜/閰嶇疆/鏃ュ織涓嶆敼锛涗慨姝ｇ増鏂扮洰褰曡缁冪粨鏋滃垎寮€銆傚凡娓呯悊鍥涗唤璐?閲嶅鏉冮噸58477628瀛楄妭锛屽師姝ｆā鍨?浜や粯淇濈暀銆傚疄闄呭弻骞跺彂锛屽悇宄?872.695MiB锛涗慨姝ｇ増棣栧惎鍔ㄥ埌楠岃瘉477.784绉掞紝璁″師澶辫触棣栧惎鍔ㄥ埌楠岃瘉656.35绉掞紝鍒板叏閮ㄨ褰?鍥捐氨鏀跺熬1104.515绉掞紝鍚畾浣嶄慨澶?閲嶅惎闂撮殧/鏂囨。锛宎rtifact_finalize.json鏄庣‘鑼冨洿銆傚浘璋?25瀹為獙/84鍏抽棴璺嚎锛屾仮澶嶈捣鐐规爣绛惧拰鍒嗘暟宸茶鍥烇紱鏃犳椿璺冭缁?鍏朵粬鎶樸€傜浜斾弗鏍?80.5鍜屽畬鏁碠OF>82浠嶆湭瀹屾垚銆?


## v85 point64鏄惧紡鍚姩
{"pid": 51592, "mode": "point64", "started_at": "2026-09-30T11:51:25.763992+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_residual_readout_v85.py", "--mode", "point64", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_residual_readout_v85_SW_point64_fold5"], "automatic_queue": false}


## v85 local64鏄惧紡鍚姩
{"pid": 3796, "mode": "local64", "started_at": "2026-09-30T11:52:20.752734+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_residual_readout_v85.py", "--mode", "local64", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_residual_readout_v85_SW_local64_fold5"], "automatic_queue": false}


## v85鏈€缁堢粨鏋?
涓ら」瀹屾垚锛屽綋鍓嶇嫭绔嬮€氶亾鍘熺浜擳OTAL=80.382093291214锛岀浉瀵圭嚎鎬х粍鍚?0.000000000000锛涜緭鍑虹粨鏋勫鐓цsw_residual_readout_v85_20260930/FINAL.md锛岄噰鐢?None锛屼繚鐣?[]銆傛棤鍏朵粬鎶?鏃т换鍔★紝璐熸潈閲嶆寜娓呭崟娓呯悊锛屽師鏉ユ簮/浜や粯淇濈暀锛涘畬鏁碠OF>82鏈畬鎴愩€?


## v86 POR鑱斿悎鍔ㄤ綔鏁堢敤鏄惧紡鍚姩
{"pid": 38080, "started_at": "2026-09-30T12:15:56.942643+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_joint_utility_v86.py", "--target", "POR", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_joint_utility_v86_POR_fold5"], "candidates": ["likelihood", "joint"], "shared_frozen_extraction": true, "automatic_queue": false}


## v86 POR璁粌鐩爣姣旇緝瀹屾垚
褰撳墠TOTAL=80.382093291214锛屽彉鍖?0.000000000000锛沯oint-likelihood POR +0.000000000000锛屾洿鏂板悗鏈€楂樺樊-0.000746818167銆傚綋鍓嶉厤鏂箁esults/deep_learning/por_joint_utility_v86_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶樸€?

## v86鏀跺熬涓庝笅涓€鏂瑰悜
630鍥捐氨鑺傜偣/86涓猟ead end宸茬敓鎴愶紱娓呯悊5涓啑浣欑姸鎬?2834827瀛楄妭锛屾婧愪笌浜や粯涓嶆敼锛屾棤GPU璁粌銆備笅涓€鏂瑰悜涓篜ERM褰撳墠瀹屾暣绉佹湁妯″瀷鐨勪技鐒?鑱斿悎瀹樻柟鏁堢敤鍖归厤瀵圭収锛屽皻鏈惎鍔紱鏂规瑙乸or_joint_utility_v86_20260930/NEXT.md銆傚師绗簲涓ユ牸>80.5涓庡畬鏁碠OF>82浠嶆湭杈炬垚锛屾棤鍏朵粬鎶樸€?


## v87 likelihood鏄惧紡鍚姩
{"pid": 81700, "mode": "likelihood", "started_at": "2026-09-30T12:48:03.378445+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_full_utility_v87.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_full_utility_v87_PERM_likelihood_fold5"], "automatic_queue": false}


## v87 joint鏄惧紡鍚姩
{"pid": 77124, "mode": "joint", "started_at": "2026-09-30T12:51:25.899575+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_full_utility_v87.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_full_utility_v87_PERM_joint_fold5"], "automatic_queue": false}


## v87 PERM瀹屾暣缃戠粶鏁堢敤瀹屾垚
褰撳墠TOTAL=80.382093291214锛屽彉鍖?0.000000000000锛涙渶缁坖oint-likelihood PERM+0.000000000000锛屾洿鏂板悗宸?0.015989004499銆傞厤鏂箁esults/deep_learning/perm_full_utility_v87_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶樻垨鏃ч槦鍒椼€?

## v87鏀跺熬鍙婁笅涓€鏂瑰悜
涓ら」鍘熺浜?8杞棭鍋溿€佹仮澶嶅綋鍓峆ERM锛岀粍鍚圱OTAL80.382093291214涓嶅彉銆傚悓杞晥鐢?鍗?12闄嶃€佸钩鍧?0.002843593202锛涙渶楂樺€煎皬姝ｈ瘖鏂崟鐙褰曪紝涓嶄綔涓洪噰鐢ㄦ敹鐩娿€傛竻鐞?鏉冮噸57604292瀛楄妭锛屽浘璋?32/87dead ends鏇存柊锛屾婧?浜や粯涓嶆敼銆備笅涓€鎬濊矾涓哄喕缁揚ERM绉佹湁鍏堥獙鐨勬潯浠舵鐜囨牸璇诲嚭锛岃鍒掑皻鏈疄鐜?鍚姩锛涙棤GPU璁粌銆佹棫闃熷垪鎴栧叾浠栨姌銆?


## v88 PERM姒傜巼鏍兼樉寮忓惎鍔?
{"pid": 49776, "started_at": "2026-09-30T13:16:07.892540+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_perm_grid_readout_v88.py", "--target", "PERM", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_perm_grid_readout_v88_PERM_fold5"], "candidates": ["likelihood", "joint"], "shared_frozen_compute": true, "automatic_queue": false}


## v88 PERM璁粌鐩爣姣旇緝瀹屾垚
褰撳墠TOTAL=80.382093291214锛屽彉鍖?0.000000000000锛沯oint-likelihood PERM +0.000000000000锛屾洿鏂板悗鏈€楂樺樊-0.001626163619銆傚綋鍓嶉厤鏂箁esults/deep_learning/perm_grid_readout_v88_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶樸€?

## v88鏀跺熬涓庝笅涓€鏂瑰悜
姒傜巼鏍间袱椤?8杞仮澶嶅綋鍓峆ERM锛孴OTAL80.382093291214涓嶅彉锛岃缁冨悗鍖归厤宸?0.001626163619锛涙竻鐞?鏉冮噸8644644瀛楄妭锛屽浘璋?35/88鍏抽棴鑺傜偣銆備笅涓€涓哄綋鍓嶇鏈塖W鍥哄畾df4鏉′欢鍒嗗竷鏃忥紝鏃у叡浜玈W t3璐熻瘉鎹凡璁板綍锛屽皻鏈疄鐜?鍚姩锛涘師姝ｆā鍨?浜や粯淇濇寔锛屾棤鍏朵粬鎶樻垨鏃ч槦鍒椼€傚畬鏁村師OOF>82浠嶆湭杈炬垚銆?


## v89 SW Student-t4鏄惧紡鍚姩
{"pid": 67356, "started_at": "2026-09-30T13:39:43.878844+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_student4_v89.py", "--mode", "student4", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_student4_v89_SW_student4_fold5"], "automatic_queue": false}


## v89 SW Student-t4瀹屾垚
SW73.779643062735锛屽Gaussian-0.013667889753锛屾渶浣?鍋滄14/32锛涢噰鐢?False锛屽綋鍓峊OTAL80.382093291214銆傞厤鏂箁esults/deep_learning/sw_student4_v89_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶樸€?

## v89鏀跺熬涓庝笅涓€鏂瑰悜
t4鍊欓€塗OTAL80.377309529801锛屽綋鍓嶉噰鐢℅aussian缁勫悎80.382093291214涓嶅彉锛涘垹闄?涓礋checkpoint锛屾竻鐞嗘竻鍗曚繚瀛樺彂鐢烴umPy bool搴忓垪鍖栭敊璇紝宸茶褰曡矾寰勫強澶у皬/鍝堝笇鍏冩暟鎹己鍙ｏ紝绂佹鎺ㄧ畻銆傝缁?棰勬祴/鍘熸婧?浜や粯瀹屽ソ銆備笅涓€鏂瑰悜涓篠W鐜版湁鍏ㄤ簳code鐨勭洿鎺ヨ緭鍑哄垎鏀痳eference/well鍏钩瀵圭収锛岃鍒掑皻鏈疄鐜?鍚姩锛涙棤鍏朵粬鎶樻垨鏃ч槦鍒椼€?


## v90 SW鍏ㄤ簳鐩存帴璇诲嚭鏄惧紡鍚姩 reference
{"pid": 49508, "started_at": "2026-09-30T14:11:50.204084+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_direct_well_v90.py", "--mode", "reference", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_direct_well_v90_SW_reference_fold5"], "automatic_queue": false}


## v90 SW鍏ㄤ簳鐩存帴璇诲嚭鏄惧紡鍚姩 well
{"pid": 74244, "started_at": "2026-09-30T14:14:00.748921+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_direct_well_v90.py", "--mode", "well", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_direct_well_v90_SW_well_fold5"], "automatic_queue": false}


## v90 SW鍏ㄤ簳鐩存帴璇诲嚭瀹屾垚
well-reference SW+0.004915085801锛涢噰鐢?None锛屽綋鍓峊OTAL80.382093291214锛屼繚鐣橻'reference', 'well']銆傞厤鏂箁esults/deep_learning/sw_direct_well_v90_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶橈紝瀹屾暣OOF>82鏈畬鎴愩€?


## v91 SW鍐荤粨鍏ㄤ簳鐩存帴澶存樉寮忓惎鍔?
{"pid": 81832, "started_at": "2026-09-30T14:27:15.630365+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_frozen_direct_v91.py", "--target", "SW", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_frozen_direct_v91_SW_fold5"], "automatic_queue": false}


## v91 SW鍐荤粨鍏ㄤ簳鐩存帴澶村畬鎴?
褰撳墠TOTAL=80.382093291214锛屽彉鍖?0.000000000000锛泈ell-reference SW +0.000000000000锛屾洿鏂板悗鏈€楂樺樊-0.091901136443銆傚綋鍓嶉厤鏂箁esults/deep_learning/sw_frozen_direct_v91_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶樸€?

## v91缁撴潫鍚庣殑涓嬩竴鏂瑰悜锛堟湭璁粌锛?
鍋滄寤朵几鍐荤粨SW鍏ㄤ簳鐩存帴绾挎€уご锛涜鍒扨OR瀹屾暣绉佹湁妯″瀷likelihood/joint鏁堢敤瀵圭収锛屽尯鍒玽86浠呰矾鐢便€傚綋鍓峊OTAL80.382093291214锛屾棤鍏朵粬鎶樸€傝缁嗚竟鐣岋細results/deep_learning/sw_frozen_direct_v91_20260930/NEXT.md銆?


## v92 likelihood鏄惧紡鍚姩
{"pid": 76184, "mode": "likelihood", "started_at": "2026-09-30T14:45:02.182137+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_full_utility_v92.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_full_utility_v92_POR_likelihood_fold5"], "automatic_queue": false}


## v92 joint鏄惧紡鍚姩
{"pid": 77152, "mode": "joint", "started_at": "2026-09-30T14:45:43.510778+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_por_full_utility_v92.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_por_full_utility_v92_POR_joint_fold5"], "automatic_queue": false}


## v92 POR瀹屾暣缃戠粶鏁堢敤瀹屾垚
褰撳墠TOTAL80.382093291214锛岄噰鐢?None锛涘畬鏁磋缁冨悗joint-likelihood POR-0.029103461105锛屽悓杞潎鍊?0.040396054022銆傞厤鏂箁esults/deep_learning/por_full_utility_v92_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶橈紝鍘烵OF>82鏈畬鎴愩€?


## v93 control鏄惧紡鍚姩
{"pid": 59992, "mode": "control", "started_at": "2026-09-30T15:00:06.445077+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_v93.py", "--mode", "control", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_v93_SW_control_fold5"], "automatic_queue": false}


## v93 sam005鏄惧紡鍚姩
{"pid": 25968, "mode": "sam005", "started_at": "2026-09-30T15:02:42.390749+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_v93.py", "--mode", "sam005", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_v93_SW_sam005_fold5"], "automatic_queue": false}

# v93鎵ц鏁呴殰涓庝覆琛屾仮澶?

2026-09-30 15:10:51锛屼袱杩涚▼鍚屾椂鏁呴殰骞剁‘璁ゅ凡閫€鍑猴細rho0 CUDA OOM锛孲AM cuBLAS鎵ц澶辫触銆傛鍓嶅悇鑷猵eak allocated1874.03MiB锛屽綋鍓岹PU闈炴湰鎵瑰崰鐢?504MiB/free8492MiB銆備粎鍑繖浜涜瘉鎹笉鑳借瘉鏄庡閮ㄥ簲鐢ㄣ€佹樉瀛樼鐗囨垨椹卞姩閲嶇疆鏄摢涓€绉嶆牴鍥狅紝涓嶈兘璇存骞跺彂瀹夊叏銆傚け璐ユ簮鐮?閰嶇疆/hash/鍘焞og/澶辫触璁板綍淇濇寔涓嶆敼銆?

鎺у埗宸插師瀛愬畬鎴?1杞紝SAM宸插畬鎴?7杞紝瀹屾暣last鍚玞urrent/best/optimizer/scheduler/RNG/鍋滄涓庢洿鏂板悗鏈€浣炽€備娇鐢ㄦ樉寮?-resume浠庡畬鏁磋疆杈圭晫鎭㈠锛屼笉浠庢渶浣虫ā鍨嬮噸璁紝涓嶆敼batch/鏁版嵁/鎹熷け/鍒嗘姌/棰勭畻銆傚厛SAM銆佺粨鏉熷悗鎺у埗锛屾渶澶氫竴涓湰鎵笹PU杩涚▼锛屼笉鑷姩闃熷垪锛屼笉鍏抽棴鍏朵粬搴旂敤銆傚畬鏁磋疆浠ュ悗鏈寔涔呭寲鐨勫け璐atch鎸変繚瀛楻NG閲嶆斁锛況eport.seconds涓嶈鏈寔涔呭寲澶辫触鐗囨锛岄launch鑷崇粓瀹屾垚鎬诲閽熷寘鎷晠闅滀笌鎭㈠闂撮殧銆俥xecution_segments鍒嗘璁板綍锛岃€屼笉鑳芥妸涓や釜閫昏緫run鐢熷懡鍛ㄦ湡閲嶅彔褰撴垚瀹為檯杩涚▼骞跺彂鏃堕棿銆?

杩欐槸褰撳墠宸叉巿鏉冨疄楠岀殑鏁呴殰鎭㈠锛屼笉鎭㈠鐢ㄦ埛鏆傚仠鎴栨棫闃熷垪銆傝嫢涓茶鍚屾牱鏁呴殰锛屽厛璁板綍鐪熷疄鐘舵€佸啀鏀规墽琛屾柟妗堬紝涓嶇敤瑙傛祴瓒呮椂鎺ㄥ畾浠诲姟缁堟銆傝缁冩湭瀹屾暣缁撴潫涓嶅緱瀹ｇОSAM鏈€缁堟敹鐩婏紱宸叉湁鍒濇姝ｅ€欓€夊拰鎵€鏈夋仮澶嶄緷璧栦繚鐣欍€?


## v93 SW涓ゆ璁＄畻鍖归厤SAM瀹屾垚
sam005-control SW+0.109559154892锛涢噰鐢?sam005锛屽綋鍓峊OTAL80.417837208128锛屼繚鐣橻'control', 'sam005']銆傞厤鏂箁esults/deep_learning/sw_sam_v93_20260930/CURRENT_RECIPE.json锛涙棤鍏朵粬鎶橈紝瀹屾暣OOF>82鏈畬鎴愩€?

## v93鏀剁泭鍚庣殑涓嬩竴鎵癸紙鏈缁冿級
褰撳墠鐙珛璺嚎TOTAL80.417837208128锛涙湁鐣孲AM鍗婂緞.025/.1涓ら」璁″垝锛屼粠鍚屽師濮媣56/prior鍒濆鍖栵紝淇濇寔.05宸插畬鎴愬鐓у拰鍘熻瘎鍒?80-18銆傚洜骞跺彂CUDA鏁呴殰锛岄粯璁ゆ樉寮忎覆琛岋紝鏃犳棫闃熷垪鎴栧叾浠栨姌銆傝缁嗚鍒掕results/deep_learning/sw_sam_v93_20260930/NEXT.md銆?


## v94 r025鏄惧紡鍚姩
{"pid": 42592, "mode": "r025", "started_at": "2026-09-30T15:39:59.809887+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_radius_v94.py", "--mode", "r025", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_radius_v94_SW_r025_fold5"], "automatic_queue": false}


## v94 r100鏄惧紡鍚姩
{"pid": 78800, "mode": "r100", "started_at": "2026-09-30T15:49:25.186229+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_sam_radius_v94.py", "--mode", "r100", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_sam_radius_v94_SW_r100_fold5"], "automatic_queue": false}


## v94 SW鏈夌晫SAM鍗婂緞瀹屾垚
褰撳墠TOTAL80.442433883814锛岄噰鐢?r100锛屼繚鐣?['r100']銆傞厤鏂箁esults/deep_learning/sw_sam_radius_v94_20260930/CURRENT_RECIPE.json锛涗袱椤圭湡瀹炰覆琛岋紝鏃犲叾浠栨姌锛屽師OOF>82鏈畬鎴愩€?


## v95鍥哄畾姒傜巼鍒嗚В鍚姩
涓ら」1678962鍙傛暟瀹屽叏鍐荤粨锛歳100榛樿atom+鑷韩/鏃82a100瀵嗗害锛涙潵婧愭彁鍓嶅浐瀹氥€佸師144665杈撳叆/璇勫垎锛屼笉鏄寜鐐归€夋嫨鎴栨姇绁ㄣ€傛湭璇勪环/鏈缁冿紝鏃犳柊鏀剁泭鎴栧叾浠栨姌銆傝褰時esults/deep_learning/sw_factorization_v95_20260930/README.md銆?



## v95鍥哄畾姒傜巼鍒嗚В瀹屾垚
鍘熺浜擳OTAL80.472576222471锛岄噰鐢?separate锛涗袱鍚屽閲忓浐瀹氭潵婧愬潎宸插師144665杈撳叆璇勪环锛屽悓鏉ユ簮control涓巖100绮剧‘涓€鑷淬€傛湭璁粌/鍏朵粬鎶橈紝瀹屾暣OOF>82鏈畬鎴愩€傞厤鏂箁esults/deep_learning/sw_factorization_v95_20260930/CURRENT_RECIPE.json銆?



## v96 likelihood鏄惧紡鍚姩
{"pid": 78060, "mode": "likelihood", "started_at": "2026-09-30T16:25:19.282636+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_factor_sam_v96.py", "--mode", "likelihood", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_factor_sam_v96_SW_likelihood_fold5"], "automatic_queue": false}


## v96 joint鏄惧紡鍚姩
{"pid": 72280, "mode": "joint", "started_at": "2026-09-30T16:32:18.465681+08:00", "command": ["C:\\Users\\z5364\\.conda\\envs\\normal\\python.exe", "-X", "utf8", "-u", "D:\\match_predict\\welllog-reservoir-hybrid\\experiments\\pure_dl\\train_sw_factor_sam_v96.py", "--mode", "joint", "--out", "D:\\match_predict\\welllog-reservoir-hybrid\\outputs\\pure_dl_sw_factor_sam_v96_SW_joint_fold5"], "automatic_queue": false}


## v96绉佹湁瀵嗗害SAM瀹屾垚
褰撳墠鍘熺浜擳OTAL80.504197424477锛岄噰鐢?likelihood锛涗袱椤圭湡瀹炴仮澶?璇勫垎/娓呯悊瀹屾垚锛屾湭琛ユ姌/瀹屾暣OOF銆傞厤鏂箁esults/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json銆?



## v96鏈€缁堣祫鏍间笌鏈畬鎴?
鍘熺浜擳OTAL80.504197424477锛屼弗鏍?80.5锛沴ikelihood鏈€浣?/鍋滄24锛屽畬鏁?44665纾佺洏閲嶈浇閫愮偣涓€鑷淬€俲oint鎭㈠0/18锛屾棤鏂板鏀剁泭锛屽叧闂繖涓€閰嶇疆銆傛棤娲昏穬璁粌/鍏朵粬鎶橈紱瀹屾暣鍘烵OF>82浠嶆湭瀹屾垚锛岃ˉ鎶樻殏缂撲繚鎸併€傞厤鏂箁esults/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json锛涘悗缁緷璧栬竟鐣岃PROMOTION_BOUNDARY.md锛屼笉鑳藉甫绗簲璁粌鏉冮噸鍒板埆鎶樸€?


## v97琛ユ姌鏉ユ簮鍑嗗
# v97 琛ユ姌鍑嗗瀹為檯璁板綍

绗簲璧勬牸TOTAL=80.504197424477锛屽師144665鐐癸紱鍘熸€讳綋723556鐐?80浜?浜旀姌鏍囩涓庢棦鏈夋寮忎汉鍙ｉ€愰」涓€鑷淬€?

鎺㈢储閲嶇偣锛氬皢宸茬瓫閫夐€氳繃鐨勫畬鏁村闃舵鐙珛涓夐€氶亾閰嶆柟绉诲埌鍘熷叾浠栨姌锛涘彧鍋氭潵婧愬拰鏁版嵁鎺ュ彛鍑嗗锛屾湭璁粌/鏈ˉ鎶樸€?
瀹為檯鍐呭涓庨獙璇侊細涓変釜褰撳墠builder CPU鏋勯€犲苟涓ユ牸鍔犺浇閫夊畾state锛屾崟鑾峰疄闄卌heckpoint/JSON璺緞涓嶴HA銆侀厤缃拰鍏变韩鍏崇郴锛汼W闆禸ank浠呯敤浜庡垎閰嶉潪鎸佷箙buffer锛屾棤鍓嶅悜/淇濆瓨/璇勫垎銆傚師濮媡rain鏁版嵁閲嶅缓浜旀姌锛屽疄闄呯浜旂紦瀛樿繘鍏ュ畬鏁碠OF鍗犱綅锛屽叾浣欏洓鎶樹繚鎸丯aN锛涗笉璁＄畻閮ㄥ垎OOF TOTAL銆?
鎹曡幏16涓敮涓€鏋勯€犳潈閲嶄緷璧栵紱閰嶇疆瑙乨ependency_trace.json銆傛墍鏈夋槸绗簲鏉ユ簮锛屽彧鍙敤浜庣浜旓紱缂哄皯鐨?~4闇€鍚勬姌閲嶆柊璁粌锛屼笉鎷垮凡鏈変笉鍚岄厤鏂圭殑瀹屾暣浜旀姌鏉冮噸鏇夸唬銆?

|鎶榺璁粌鐐箌楠岃瘉鐐箌瀹為檯鐘舵€亅
|---|---:|---:|---|
|1|578935|144621|requires_fold_local_sources_and_training|
|2|578889|144667|requires_fold_local_sources_and_training|
|3|578466|145090|requires_fold_local_sources_and_training|
|4|579043|144513|requires_fold_local_sources_and_training|
|5|578891|144665|verified_fifth_only|

鏀剁泭锛氬皻鏃犳柊妯″瀷鍒嗘暟鎴栧畬鏁碠OF鏀剁泭锛涘彧鏄鐢ㄥ凡鏍搁獙绗簲80.504197銆?
涓嶇户缁細鐩存帴鏀筬old鏁板瓧浣嗗姞杞界浜旈澶勭悊/鏉冮噸/Q浼氶€犳垚璺ㄦ姌娉勬紡锛岀姝€傚師calibrated75/global75瀹屾暣浜旀姌婧愪笉绛変簬鏈厤鏂筶ocal-context80锛屼笉鑳介粯榛樻浛浠ｃ€?
宸插仛锛氬喕缁撶浜旇祫鏍?閰嶆柟鍝堝笇锛屽師浜旀姌浜哄彛娓呭崟锛屽疄闄呮瀯閫犱緷璧栵紝OOF鍗犱綅锛涜鍒掞細鏄惧紡鎶樺彿/璺緞鐨勬暟鎹笌Q缁戝畾锛屾寜渚濊禆闃舵瀹炵幇璁粌鍏ュ彛锛涙湭鍋氾細瀹屾暣鍓╀綑闃舵鍏ュ彛銆佸叾浠栧洓鎶樿缁冦€佸畬鏁碠OF>82銆?
姝ゅ墠AGENTS鏄庣‘琛ユ姌鏆傜紦锛氫笉寰楀湪persistent goal涓嬭嚜鍔ㄥ惎鍔紱淇濇寔璇ヨ竟鐣岋紝鍑嗗涓嶅惎鍔ㄣ€傜洰鏍囧悗瀹屾暣浜旀姌瀹樻柟璇勪环/鐙珛纭/鏂扮洰褰曚氦浠橈紝鏃т氦浠樹笉鏀广€?
鑰楁椂锛?.97590340001625绉掞紝CPU鍘熸暟鎹?鏋勯€犺拷韪?鍝堝笇/淇濆瓨璁″叆锛涙棤GPU璁粌锛屾棤骞跺彂閫熷害瀹ｇО銆?



## 鎸夋姌鏁版嵁鎺ュ彛瀹為檯璇婃柇
鍘熺涓€鎶?78935璁粌/144621楠岃瘉銆?4/16浜曪紝鍘?06鐗瑰緛閲嶆柊鎷熷悎涓庢棦鏈夎鎶橀澶勭悊鍏冩暟鎹竴鑷达紱浠呮秷璐瑰叾鍏冩暟鎹紝鏈妸鏃alibrated妯″瀷浣滀负褰撳墠鍊欓€夋潵婧愩€傞獙璇佹爣绛炬湭鍒嗛厤鍒版崯澶憋紝wholewell reference鍙64璁粌杈撳叆銆傜浜旈澶勭悊鍜岀浜擰鍧囪鎷掔粷锛宮issing Q绂佹璁粌锛涙湰鎵规棤璁粌/鎺ㄧ悊/鏂板垎鏁般€傝€楁椂46.963804099999834绉掞紝宄板€糰llocated 997.673828125MiB銆?



## 2026-09-30 鐢ㄦ埛鏄庣‘鎭㈠鍏朵綑鍥涙姌
鐢ㄦ埛绛斿鈥滄仮澶嶅叾浣欏洓鎶樿缁冣€濄€備粎鎭㈠褰撳墠v96绗簲80.504197閰嶆柟鐨勫師鍏朵綑鍥涙姌锛涘鐢ㄥ凡鏍搁獙绗簲棰勬祴锛屽悇鎶樺叏閮ㄥ涔犱緷璧?棰勫鐞?Q閲嶆柊浣跨敤鑷韩64璁粌浜曘€傛棫鏆傚仠瑙勫垯鍦ㄦ湰鑼冨洿琚鎺堟潈瑕嗙洊锛涗笉鍚姩鍏朵粬鏃ч槦鍒椼€傞涓師local-context80鏉ユ簮璁粌鍏ュ彛宸插噯澶囷紝鍘熷瓙瀹屾暣last鍚紭鍖栧櫒/璋冨害/RNG/Q绛栫暐/瑙傛祴鏉ユ簮/褰撳墠鍜屾渶浣炽€傚垵濮嬩覆琛岋紝涓嶆敼宸茶缁冨巻鍙叉簮鐮佹垨浜や粯銆傚墿浣欎笅娓歌缁冨叆鍙ｅ皻鍦ㄥ疄鏂斤紝瀹屾暣OOF>82灏氭湭楠岃瘉銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|70|82.44281121011284|0/18|鏈畬鎴恷
|2|None|None|0/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|0/18|鏈畬鎴恷
|2|None|None|0/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|1/18|鏈畬鎴恷
|2|None|None|0/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|None|None|0/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


## 瀹屾垚闃舵鐨勫啑浣欑紦瀛樻竻鐞?
浠呭凡瀹屾垚闃舵鐨刲ast鎭㈠鐘舵€佷笌宸插畬鎴愭潵婧愭帰娴嬬紦瀛橈紝瀹為檯鍒犻櫎101鏂囦欢/2108045685瀛楄妭銆傝鍒掑厛淇濆瓨璺緞/澶у皬/SHA锛屽垹闄ゅ墠閲嶈娲诲姩杩涚▼锛涘叏閮ㄩ€夊畾妯″瀷銆侀€氶亾棰勬祴銆佽缁冮厤缃?鍘嗗彶鍙婂悗缁璔渚濊禆淇濈暀锛屼笉鍋滄璁粌锛屼笉杩佺ЩC鐩樸€傝瘉鎹細cleanup_plan_20260930T201600751212.json鍙婂搴攄one娓呭崟銆傚畬鏁碠OF灏氭湭瀹屾垚銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|9/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


## v97 鏉ユ簮棰勮缁冧笌涓婚樁娈甸噸鍙?
鍦ㄧ敤鎴峰凡鎺堟潈鏈€澶?GPU涓旀仮澶嶅叾浣欏洓鎶樼殑鑼冨洿鍐咃紝淇濈暀鎵€鏈夎缁冩簮鐮?閰嶇疆/妯″瀷hash鍙婃椿璺冭缁働ID锛屼粎浠ユ柊CPU鎺у埗鍣ㄦ帴绠¤皟搴︺€備富闃舵淇濇寔渚濆簭鎵ц锛屼笅涓€鍘熸姌local-context80鏉ユ簮棰勮缁冨彲閲嶅彔锛涙渶澶?GPU浠诲姟銆佹渶澶?SW浠诲姟銆傛潵婧愮┖闂查棬妲?.8GiB锛屼富闃舵4GiB锛涗笉鍥犲苟鍙戜慨鏀?0/18棰勭畻/鍘熸暟鎹?Q/鍒嗘姌/seed/璇勫垎銆傛棫涓茶鐘舵€佽鏄庤姝ゆ墽琛岄厤缃鐩栵紝澶辫触涓嶉噸璇?涓嶈嚜鍔ㄦ仮澶嶅凡鍋滄璁粌銆傛柊璋冨害鍏ュ彛/鎺ョ/浜嬩欢娓呭崟鐙珛璁板綍锛涘畬鏁碠OF灏氭湭瀹屾垚锛屽疄闄呮€诲閽熷強閲嶅彔浠ヤ簨浠惰€岄潪鍗曢」鏃堕暱姹傚拰纭畾銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|15/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|15/18|鏈畬鎴恷
|3|None|None|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|16/18|鏈畬鎴恷
|3|16|81.75044296931524|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|16/18|鏈畬鎴恷
|3|16|81.75044296931524|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


2026-09-30 骞跺彂璁板綍缁存姢瀹屾垚锛氳緟鍔╄褰曞櫒鍙瓑寰呮潵婧愯缁?JSON 鍗婂啓鍏ュ畬鎴愶紝妯℃嫙楠岃瘉閫氳繃锛涘悓姝ュ伐浣滃尯鍙婁粨搴撹繘搴︺€傝缁冩簮鐮併€侀厤缃€佹潈閲嶄笌 v96 鍥哄畾閰嶆柟涓嶅彉銆傚叾浣欏洓鎶樻寔缁缁冿紝浠嶅緟瀹屾暣 OOF 鍙婄鐩樺鐜帮紱鏉ユ簮鍝堝笇璇存槑瑙?fullcv_v97_20260930/record_concurrent_read_patch.json銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|17/18|鏈畬鎴恷
|3|28|82.4751007628048|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|17/18|鏈畬鎴恷
|3|28|82.4751007628048|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


2026-09-30 瀹炴祴鑰楁椂蹇収锛氳緟鍔╄剼鏈?summarize_timing_fullcv_v97.py 宸茶緭鍑?OVERLAP_TIMING.json銆傛埅鑷?2026-09-30T20:53:47.397132+08:00锛岃皟搴﹀櫒瑙傚療鍒板弻浠诲姟閲嶅彔 1109.574 绉掞紱涓よ缁働ID鍙婂垱寤烘椂闂村綋鍦烘牳瀵逛竴鑷淬€傝褰曞崟椤瑰姞杞?棰勭儹/璁粌/楠岃瘉/淇濆瓨鑼冨洿鍙婃巿鏉冧互鏉ユ€诲閽燂紱鏈畬鎴愬悓宸ヤ綔閲忎覆琛屽鐓э紝灏氫笉鑳借瘉瀹炲姞閫熷箙搴︺€傛涓鸿褰曞伐鍏凤紝涓嶆敼鍙橀厤鏂规垨璁粌銆備粛寰呭叾浣欐姌銆佸畬鏁碠OF鏍搁獙鍜屾渶缁堣€楁椂蹇収銆傝緟鍔╂簮鐮丼HA256=99bb1151c86ec6919d7f7f655daa8d01a6f2f30bd99c5675fbb4b5d7ed1ea5f8銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|56|82.62860680284442|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|56|82.62860680284442|0/18|鏈畬鎴恷
|4|None|None|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


## 瀹屾垚闃舵鐨勫啑浣欑紦瀛樻竻鐞?
浠呭凡瀹屾垚闃舵鐨刲ast鎭㈠鐘舵€佷笌宸插畬鎴愭潵婧愭帰娴嬬紦瀛橈紝瀹為檯鍒犻櫎11鏂囦欢/163477037瀛楄妭銆傝鍒掑厛淇濆瓨璺緞/澶у皬/SHA锛屽垹闄ゅ墠閲嶈娲诲姩杩涚▼锛涘叏閮ㄩ€夊畾妯″瀷銆侀€氶亾棰勬祴銆佽缁冮厤缃?鍘嗗彶鍙婂悗缁璔渚濊禆淇濈暀锛屼笉鍋滄璁粌锛屼笉杩佺ЩC鐩樸€傝瘉鎹細cleanup_plan_20260930T210131862888.json鍙婂搴攄one娓呭崟銆傚畬鏁碠OF灏氭湭瀹屾垚銆?


2026-09-30 璋冨害棰勬缁撹锛堥潪鏂版ā鍨嬶級锛?
- 鎺㈢储閲嶇偣锛氫富鏉ユ簮璁粌涓庝笅涓€鎶樻潵婧愬悓鏃舵墽琛岋紝鏄惁缂╃煭鏁存壒鍏抽敭璺緞锛涘師妯″瀷銆佽瘎鍒嗐€佹暟鎹拰80/18棰勭畻鍥哄畾銆?
- 宸叉帰绱笌楠岃瘉锛歳un_dualroots_fullcv_v97.py浠呭噯澶囧拰preview锛屾湭鎺ョ銆佹湭鍚姩璁粌锛涢瑙堜繚鐣?7784鏉ユ簮PID鍙婂師璁粌婧愮爜鍝堝笇锛孏PU绌洪棽6.076GiB浣庝簬鍘?.8GiB闂ㄦ銆傚凡瀹屾垚绗簩鎶樻潵婧?72.776绉掞紝18闃舵鍚堣3280.389绉掞紱闃舵鍚堣涓嶆槸鏁存壒澧欓挓锛屽彟瑙丱VERLAP_TIMING.json銆?
- 鍙栧緱鏀剁泭锛氬皻鏈瘉瀹炴柊澧炶皟搴︾殑鎬诲閽熸敹鐩婏紝妯″瀷鎴愮哗涓嶅彉銆傜幇鏈夋祦姘寸嚎棰勮鍙敤鍚庣画闃舵瑕嗙洊涓嬩竴鎶樻潵婧愯缁冿紱杩欐槸鑰楁椂鎺ㄦ柇锛屼笉鏄悓宸ヤ綔閲忎覆琛?骞跺彂瀵圭収缁撴灉銆?
- 涓嶇户缁殑鐐癸細褰撳墠涓嶉儴缃叉彁鍓嶅弻鏉ユ簮棰勬銆傚叡浜獹PU鍙兘寤惰繜褰撳墠涓绘潵婧愶紝鑰屼笅涓€鏉ユ簮宸叉湁鍚庣画闃舵閲嶅彔绐楀彛锛涙湰缁撹鍙€傜敤杩欐潯18闃舵渚濊禆閾撅紝涓嶆硾鍖栦负涓ゆ潵婧愬苟琛屾棤鏁堛€傛椿鍔ㄨ皟搴﹀櫒浠嶄负run_overlap_fullcv_v97.py锛屽師淇濆畧闂ㄦ/鏈€澶?GPU/鏈€澶?SW涓嶅彉銆?

绗簩鎶樺疄闄呭畬鎴愮粨鏋滐細144667鐐筎OTAL=81.547165640178锛孭OR=78.942738558320锛孭ERM=90.274125819990锛孲W=75.052571530530銆傛潵婧愰骞蹭笌鏈€缁堥厤鏂瑰悓鎶楾OTAL宸?0.328967761208鏄暣鏉¤矾绾垮樊鍊硷紝涓嶆槸鍗曟満鍒舵秷铻嶃€傛渶鍚嶴W绗?2杞渶浣?绗?0杞仠姝紝鍏ㄧ偣纾佺洏閲嶈浇涓€鑷达紱瀹屾暣OOF浠嶆湭瀹屾垚銆俢leanup_plan_20260930T210131862888.json瀵瑰簲11涓啑浣欐仮澶嶇姸鎬?163477037瀛楄妭宸插垹闄わ紝閫夊畾妯″瀷鍜岄獙璇佷緷璧栦繚鐣欍€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|1/18|鏈畬鎴恷
|4|13|79.59484593454|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|1/18|鏈畬鎴恷
|4|13|79.59484593454|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


v98涓嬩竴椤规湁鐣屽鐓у凡鍑嗗锛歅OR鏃㈡湁鏁堣矾鐢?骞呭€间綅缃叡鍚屽井璋冪殑鍚屽閲廇damW/SAM锛涙湭璁粌銆佹棤鎴愮哗锛屽叾浠栭€氶亾浠呭浐瀹氶獙璇佸弬鑰冦€傚緟绗洓鏉ユ簮瀹屾垚鍙奊PU妲芥牳楠岋紝鐢卞綋鍓峚gent鏄惧紡鍚姩锛泇97閰嶆柟/杩涚▼涓嶅彉锛屾棤鑷姩闃熷垪鎴栬ˉ鎶樸€傝褰曪細results/deep_learning/por_effective_sam_v98_20260930/README.md銆?

## v98鏈€缁堬細POR鏈夋晥妯″潡AdamW/SAM鏃犳柊鏀剁泭

涓ょ粍鍚?3635鍙傛暟銆佸悓v31瀹屾暣鐑惎鍔紝18杞仠姝㈠苟鎭㈠epoch0锛孴OTAL鍧?0.504197424477锛岀浉瀵箆96鍙婇€夊畾SAM鍑廇damW鍧?銆傛洿鏂板悗鏈€浣冲潎鏇翠綆锛屼笉鎵弿姝ら厤缃畆ho/LR閭诲煙銆傚疄闄?56.762绉?宄板€?19.846MiB锛堝惈鍔犺浇/棰勭儹/楠岃瘉/淇濆瓨锛屼笌v97閲嶅彔锛夛紝鏃犲姞閫熺粨璁恒€傝礋瀹為獙鍐椾綑鏉冮噸鎸夌嫭绔嬫竻鍗曞垹闄わ紝鍘熸鏉ユ簮淇濈暀銆傝瑙乕鏈€缁堣褰昡(results/deep_learning/por_effective_sam_v98_20260930/README.md)銆倂97鎺堟潈琛ユ姌缁х画锛屽畬鏁碠OF>82灏氭湭楠岃瘉銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|17/18|鏈畬鎴恷
|4|80|81.68795258443386|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|17/18|鏈畬鎴恷
|4|80|81.68795258443386|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?

## v99宸插噯澶囷紝灏氭湭鍚姩

POR鍚屽閲弆ikelihood/joint鍔ㄤ綔鏁堢敤鎹熷け瀵圭収鍏ュ彛宸插疄鐜帮紝灏氭棤璁粌鎴栨垚缁┿€傚綋鍓嶄紭鍏堢洃鎺97鎺堟潈琛ユ姌锛屼笉鑷姩鎺掗槦銆佹敼鍙極OF閰嶆柟鎴栧惎鍔ㄦ柊鍏朵粬鎶樸€傝[鐙珛璁板綍](results/deep_learning/por_joint_utility_v99_20260930/README.md)銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|0/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?

## v99鏄惧紡鍚姩 2026-09-30T22:18:53.927355+08:00

POR鍚屽閲弆ikelihood/joint鎹熷け瀵圭収PID89608宸插惎鍔紝鍙仛绗簲鎶橈紱涓巚97绗?鎶樺叡2GPU浠诲姟銆傚師璁粌鏍囩鏌ヨ〃楠岃瘉澶嶅埗0锛宔poch0鍘熷嚱鏁颁竴鑷达紝灏氭棤鏈€缁堟敹鐩娿€傛巿鏉冭ˉ鎶樺浐瀹氶厤鏂圭户缁笉鍙樸€?


## 瀹屾垚闃舵鐨勫啑浣欑紦瀛樻竻鐞?
浠呭凡瀹屾垚闃舵鐨刲ast鎭㈠鐘舵€佷笌宸插畬鎴愭潵婧愭帰娴嬬紦瀛橈紝瀹為檯鍒犻櫎102鏂囦欢/2164213217瀛楄妭銆傝鍒掑厛淇濆瓨璺緞/澶у皬/SHA锛屽垹闄ゅ墠閲嶈娲诲姩杩涚▼锛涘叏閮ㄩ€夊畾妯″瀷銆侀€氶亾棰勬祴銆佽缁冮厤缃?鍘嗗彶鍙婂悗缁璔渚濊禆淇濈暀锛屼笉鍋滄璁粌锛屼笉杩佺ЩC鐩樸€傝瘉鎹細cleanup_plan_20260930T222132843018.json鍙婂搴攄one娓呭崟銆傚畬鏁碠OF灏氭湭瀹屾垚銆?

## v99鏈€缁堬細鏃犳敹鐩?

POR likelihood/joint鍚?8杞棭鍋滃苟鎭㈠epoch0锛屼袱缁凾OTAL80.504197424477锛岀浉瀵圭埗鍑芥暟/閫夊畾娑堣瀺鍧?锛涙洿鏂板悗joint-like=-0.000170132804銆傚疄闄?55.400绉?宄板€?54.907MiB锛堝惈鍔犺浇/棰勭儹/楠岃瘉/淇濆瓨銆佷笌绗?鎶橀噸鍙狅級锛屾棤鍔犻€熺粨璁恒€傚叧闂湰鍥哄畾閰嶇疆閭诲煙锛屾寜娓呭崟鍒犻櫎璐熸潈閲嶏紝淇濈暀鏉ユ簮涓庣揣鍑戣瘉鎹€傝[鐙珛鏈€缁堣褰昡(results/deep_learning/por_joint_utility_v99_20260930/README.md)銆倂97鎺堟潈琛ユ姌缁х画锛屽綋鍓嶅彧鍓╃4鎶橈紝瀹屾暣OOF>82灏氭湭楠岃瘉銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|5/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|5/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|12/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|12/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歮ain stages serial; next-fold root pretraining overlaps锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庣┖闂?.8GiB锛岄樁娈垫渶浣?GiB锛屽伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?

## v97??SW????????????

?4?14/18???????<4GiB?CUDA??10.80GiB??CPU?????????CPU?PID5916??????SW??worker?????3/CUDA4GiB???????additive PID85676????????????/???????????????????[??????](results/deep_learning/fullcv_v97_20260930/TAIL_ADMISSION.md)?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|17/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|17/18|鏈畬鎴恷
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

鏈畬鎴愶細鍏朵綑鍥涙姌鍏ㄩ儴闃舵涓庢渶缁堝疄闄呬笁閫氶亾銆佸畬鏁?23556鐐筄OF锛涘皻鏃犲畬鏁存敹鐩婏紝涓嶈兘鐢ㄩ骞插崟鎶樻垚缁╄瀹歄OF>82銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歴ingle final-fold SW tail; prior root overlap preserved; physical3/CUDA4 GiB admission锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庢樉绀虹┖闂?.8GiB锛屾櫘閫氶樁娈?GiB锛涘凡閮ㄧ讲鐨勬渶鍚庡洓涓猄W闃舵鍦ㄦ棤worker鏃朵娇鐢ㄦ樉绀?GiB/CUDA4GiB鍙岄棬妲涳紝鏉ユ簮涓庡疄娴嬭TAIL_ADMISSION.md銆傚伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|18/18|{'POR': 79.49192462461856, 'PERM': 91.22248299986603, 'SW': 74.52032535623749, 'TOTAL': 81.85756031202179}|
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

瀹屾暣瀹為檯OOF锛歿"POR": 79.33058096543338, "PERM": 90.44127043721572, "SW": 75.53542958448546, "TOTAL": 81.89101929722543}銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|18/18|{'POR': 79.49192462461856, 'PERM': 91.22248299986603, 'SW': 74.52032535623749, 'TOTAL': 81.85756031202179}|
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

瀹屾暣瀹為檯OOF锛歿"POR": 79.33058096543338, "PERM": 90.44127043721572, "SW": 75.53542958448546, "TOTAL": 81.89101929722543}銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歴ingle final-fold SW tail; prior root overlap preserved; physical3/CUDA4 GiB admission锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庢樉绀虹┖闂?.8GiB锛屾櫘閫氶樁娈?GiB锛涘凡閮ㄧ讲鐨勬渶鍚庡洓涓猄W闃舵鍦ㄦ棤worker鏃朵娇鐢ㄦ樉绀?GiB/CUDA4GiB鍙岄棬妲涳紝鏉ユ簮涓庡疄娴嬭TAIL_ADMISSION.md銆傚伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?


## 瀹屾垚闃舵鐨勫啑浣欑紦瀛樻竻鐞?
浠呭凡瀹屾垚闃舵鐨刲ast鎭㈠鐘舵€佷笌宸插畬鎴愭潵婧愭帰娴嬬紦瀛橈紝瀹為檯鍒犻櫎15鏂囦欢/214706633瀛楄妭銆傝鍒掑厛淇濆瓨璺緞/澶у皬/SHA锛屽垹闄ゅ墠閲嶈娲诲姩杩涚▼锛涘叏閮ㄩ€夊畾妯″瀷銆侀€氶亾棰勬祴銆佽缁冮厤缃?鍘嗗彶鍙婂悗缁璔渚濊禆淇濈暀锛屼笉鍋滄璁粌锛屼笉杩佺ЩC鐩樸€傝瘉鎹細cleanup_plan_20260930T231753172062.json鍙婂搴攄one娓呭崟銆傚畬鏁碠OF灏氭湭瀹屾垚銆?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備覆琛屾帶鍒跺櫒閫愰」鎵ц锛屽け璐ュ仠姝笖涓嶈嚜鍔ㄩ噸璇曪紱浠呮樉寮忔仮澶嶆湭瀹屾垚鐘舵€併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|18/18|{'POR': 79.49192462461856, 'PERM': 91.22248299986603, 'SW': 74.52032535623749, 'TOTAL': 81.85756031202179}|
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

瀹屾暣瀹為檯OOF锛歿"POR": 79.33058096543338, "PERM": 90.44127043721572, "SW": 75.53542958448546, "TOTAL": 81.89101929722543}銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屼覆琛屽苟鍙?銆佷笂闄?锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


# v97 宸叉巿鏉冭ˉ榻愬師鍏朵綑鍥涙姌

鐩爣锛氬浐瀹歷96閰嶆柟鍘熷垎缁勪簲鎶樺畼鏂筄OF>82銆傜浜?44665鐐筎OTAL80.504197424477宸叉牳楠屽苟澶嶇敤锛涘悇鎶樺叏閮ㄥ涔犱緷璧栧強缁熻/Q鍙兘鏉ヨ嚜鏈姌64璁粌浜曘€?

宸插仛锛氭槑纭巿鏉冪暀妗ｃ€佹潵婧?浜哄彛/鍝堝笇銆佹寜鎶橀澶勭悊/Q缁戝畾銆佸師瀛愬彲鎭㈠root涓?8涓湁鏁堣缁冮樁娈靛叆鍙ｃ€佸浐瀹氱鏈夋彁鍙朇PU閫愭潈閲嶇瓑浠疯瘉鏄庛€佹寜鍘熻搴廜OF姹囨€诲叆鍙ｃ€備富闃舵渚濆簭鎵ц锛屾潵婧愰璁粌涓庝笅涓€鎶橀噸鍙狅紝澶辫触鍋滄涓斾笉鑷姩閲嶈瘯锛涗粎鏄惧紡鎭㈠鏈畬鎴愮姸鎬併€?

|鍘熸姌|楠ㄥ共鏈€杩戣疆|楠ㄥ共鏈€浣砊OTAL锛堥潪鏈€缁堥厤鏂癸級|鏈夋晥闃舵瀹屾垚|鏈€缁堜笁閫氶亾|
|---|---:|---:|---:|---|
|1|80|82.50407842060818|18/18|{'POR': 80.20654240867086, 'PERM': 90.90786144125221, 'SW': 76.98103614378238, 'TOTAL': 82.82307687736336}|
|2|64|81.2181978789698|18/18|{'POR': 78.94273855832026, 'PERM': 90.27412581999009, 'SW': 75.05257153053017, 'TOTAL': 81.54716564017816}|
|3|74|82.62860680284442|18/18|{'POR': 80.45689483494576, 'PERM': 90.40622587781112, 'SW': 76.97618209773799, 'TOTAL': 82.72091124192592}|
|4|80|81.68795258443386|18/18|{'POR': 79.49192462461856, 'PERM': 91.22248299986603, 'SW': 74.52032535623749, 'TOTAL': 81.85756031202179}|
|5|鍘嗗彶宸插畬鎴恷涓嶆贩鍚屽綋鍓嶆潵婧愰骞瞸閰嶆柟宸插畬鎴恷80.504197424477锛涘凡鏍搁獙澶嶇敤|

瀹屾暣瀹為檯OOF锛歿"POR": 79.33058096543338, "PERM": 90.44127043721572, "SW": 75.53542958448546, "TOTAL": 81.89101929722543}銆?

璁″垝锛氬浐瀹?8闃舵渚濊禆鎸夋湰鎶樻渶浣砪heckpoint鎺ㄨ繘锛涘崟妯″瀷鐩爣鐙珛閫夋ā锛屽畬鎴愭瘡鎶樺啀鐪熷疄鍚堝苟锛涙渶缁堟寜鎶?~5鎺掑簭鍜屽師琛屽簭鎶ュ憡銆傝揪鎴愬悗锛氬湪鏂扮洰褰曚氦浠樹唬鐮?妯″瀷/棰勬祴涓庢潵婧愰獙璇侊紝鍘嗗彶浜や粯涓嶈鐩栥€?

涓嶇户缁細鍘嗗彶璐熷鐓?鍙傛暟鎺㈢储/鏃ч槦鍒椾笉閲嶈窇锛岀浜旀潈閲?鎷熷悎鍏冩暟鎹?Q涓嶅緱杩涘叆鍏朵粬鎶樸€倂8 POR/PERM鍘嗗彶selected epoch0缁忛€愭潈閲嶅畬鍏ㄤ竴鑷达紝鍥哄畾涓虹鏈夋彁鍙栭樁娈碉紝鐪佸幓鏃犳晥寰皟锛涗笉鎸夋柊鎶樻垚缁╅€夋嫨鏄惁鎭㈠璇ユ棫鎼滅储銆?

鑰楁椂瑙勫垯锛氭瘡stage涓巖oot璁板綍鐪熷疄seconds锛堝惈璇ラ」鍔犺浇/楠岃瘉/checkpoint锛夛紝鎬诲閽熺敱controller鎺堟潈鎵规鍙﹁锛屾渶澶?涓狦PU浠诲姟锛屾渶澶?涓猄W浠诲姟锛涗富闃舵涓庝笅涓€鎶樻潵婧愰璁粌閲嶅彔锛涗笉鎶婇樁娈垫€诲拰褰撳苟鍙戝姞閫熴€?


褰撳墠璋冨害锛歴ingle final-fold SW tail; prior root overlap preserved; physical3/CUDA4 GiB admission锛汣PU璋冨害鍣ㄦ帴绠′繚鎸佸師璁粌PID锛屼笉淇敼璁粌婧愮爜/閰嶇疆/鏉冮噸銆傛潵婧愭渶浣庢樉绀虹┖闂?.8GiB锛屾櫘閫氶樁娈?GiB锛涘凡閮ㄧ讲鐨勬渶鍚庡洓涓猄W闃舵鍦ㄦ棤worker鏃朵娇鐢ㄦ樉绀?GiB/CUDA4GiB鍙岄棬妲涳紝鏉ユ簮涓庡疄娴嬭TAIL_ADMISSION.md銆傚伐浣滈噺涓庡悓閰嶆柟鍘熶簲鎶樹竴鑷淬€傞噸鍙犱簨浠跺強瀹屾暣鎺堟潈鎵规澧欓挓鐣欐。锛屼笉灏嗗崟椤规椂闀跨浉鍔犲綋鍔犻€熻瘉鎹€傚け璐ュ仠姝笖涓嶉噸璇曘€?

## v97??????????5??

??OOF81.891019297225?723556?/80?/???????????????????????82????????????????????????????5?????????18????????????????????????????????????????????????????????[??????](results/deep_learning/fullcv_v97_20260930/FINAL.md)?


## v97鎻愪氦鍖呬氦浠?
宸插湪鏂扮洰褰?`submission_pure_dl_v97_81_891019_20260930` 瀹屾垚缁撴灉ZIP銆佺嫭绔嬩唬鐮?浜旀姌妯″瀷ZIP銆丱OF81.891019閫愬厓绱犲鐜板強鐙珛瑙ｅ帇鍏ㄩ儴娴嬭瘯浜曟帹鐞嗛獙璇併€傚巻鍙蹭氦浠樹笉鏀广€傛湰鍖呬粛涓烘潵婧愰璁粌+18鍚庣画闃舵锛屾湭澹扮О绮剧畝銆備笅涓€姝ュ緟鍋氾細鐢ㄦ埛瑕佹眰鐨勬渶澶氫簲涓€昏缁冮樁娈垫柊閰嶆柟锛涘厛绗簲鎶樻瘮杈冿紝瀹屾暣OOF>82浠嶆湭杈炬垚銆?

???????????????????????ZIP?????????286376083??????????verification_cleanup_manifest.json???result.zip/submission_code.zip/result.json?????delivery_validation.json???


## v100浜旀€婚樁娈电揣鍑戞柟妗堝凡瀹炵幇
鐩爣锛氭潵婧愰璁粌涔熻鍏ャ€佹€婚樁娈典笉瓒呰繃5锛屽浐瀹氱浜斾弗鏍?80.8涓旀湁鏀剁泭鍚庤ˉ鍏朵粬鎶橈紝瀹屾暣OOF>82銆傚凡瀹炵幇鐙珛POR/PERM鑱斿悎璇诲嚭/骞呭€?璺敱銆侀殢鏈哄畬鏁碨W缁撴瀯涓庡悗缁浐瀹歛tom瀵嗗害浼樺寲锛涗笉鍔犺浇鏃у悗缁潈閲嶃€傚嵆灏嗘樉寮忓惎鍔≒OR/PERM涓や换鍔★紱SW寰呬袱椤圭粨鏉熷悗鎵ц銆傛敹鐩娿€佸畬鏁寸粍鍚堜笌鎬绘垚鏈瘮杈冨皻鏈瘉瀹炪€傝褰曡results/deep_learning/compact_five_stage_v100_20260930/README.md銆?


v100褰撳墠瀹為檯蹇収锛歿"por": {"status": "complete", "score": 77.40726398783815, "live": false}, "perm": {"status": "complete", "score": 89.37839421114361, "live": false}, "sw_joint": {"status": "running", "score": 72.88882977698545, "live": true}, "sw_density": {"status": "prepared", "score": null, "live": false}}銆傚畬鏁寸揣鍑戜笁閫氶亾鏀剁泭灏氭湭璇佸疄锛涙棤鍏朵粬鎶樸€?


v100褰撳墠瀹為檯蹇収锛歿"por": {"status": "complete", "score": 77.40726398783815, "live": false}, "perm": {"status": "complete", "score": 89.37839421114361, "live": false}, "sw_joint": {"status": "running", "score": 73.80277957680566, "live": true}, "sw_density": {"status": "prepared", "score": null, "live": false}}銆傚畬鏁寸揣鍑戜笁閫氶亾鏀剁泭灏氭湭璇佸疄锛涙棤鍏朵粬鎶樸€?


v101涓€涓狿OR鑼冨洿瀵圭収宸插疄鐜帮細鍚寁100鏂板ご/璺敱锛屽喕缁撴簮娣峰悎鏉冮噸/灏哄害锛屽彧鏇存柊鍘熶綅缃紱鍘熺浜?80-18/鏁版嵁/Q50涓嶆敼锛屽嵆灏嗕娇鐢ㄧ┖闂茬浜屾Ы銆傚叡浜殑v100 PERM/SW浠嶈鍏ュ畬鏁?闃舵锛涙渶缁堟敹鐩婃湭璇佸疄锛屾棤鍏朵粬鎶樸€?


# v101瀹為檯缁勫悎

鍘熺浜擳OTAL=80.456611079087锛屽鏃?0.504197=-0.047586345390锛涗弗鏍?80.8=False銆?

POR鑼冨洿娑堣瀺鏀剁泭=+0.093104399806锛屾湰椤筆OR鍜屽叡浜玽100 PERM/SW瀹為檯缁勫悎锛涘寘鍚潵婧愬叡5闃舵锛屼笉鍊熸棫18闃舵鎴愬搧銆傚畬鏁碠OF>82鏈畬鎴愶紝鏃犲叾浠栨姌銆?

鍥哄畾婧愬瘑搴﹀舰鐘剁殑灞€閮ㄦ敹鐩婁繚鐣欙紱瀹屾暣绱у噾鎴愮哗鑻ヤ笉瓒冲垯浼樺厛鏀瑰彉SW琛ㄧず锛屼笉閲嶅鎵弿鏈」LR鎴栧閲忋€?


v100绱у噾浜旀€婚樁娈靛凡瀹屾垚鍘熺浜斿疄闄呯粍鍚堬細TOTAL=80.428679759145锛岀浉瀵规棫閰嶆柟-0.075517665332锛屼弗鏍?80.8=False锛涙棤鍏朵粬鎶橈紝瀹屾暣OOF>82鏈畬鎴愩€傝瘉鎹畆esults/deep_learning/compact_five_stage_v100_20260930/FINAL.md銆?


## 2026-10-01 v102 绱у噾SW婧愮壒寰?
宸插畬鎴愶細v100浜旈樁娈?0.428680锛寁101鍧囧€艰寖鍥?0.456611锛涘潎鏈繃80.8锛屾彁浜ゅ寘涓嶆敼銆傜幇鍦細瀹炵幇鍐荤粨闅忔満/鍘熻缁冩簮闅愯棌鐗瑰緛鍚屽閲廠W瀵圭収锛屼覆琛岀洿鎺ヨ缁冦€傚緟鍋氾細鏈€缁圫W鍏钩宸€笺€佸繀瑕佹椂绗簲闃舵瀵嗗害銆佸疄闄呭浐瀹氫笁閫氶亾楠岃瘉銆傜洰鏍囷細鍖呭惈婧愰璁粌鈮?闃舵涓斿畬鏁村師OOF>82锛涜繃绛涘悗鎵嶈ˉ鎶樸€傝揪鍒板悗鍙仛浜や粯涓庨€熷害瀵规瘮銆?

2026-10-01 v102鍚姩淇锛氭簮涓撳1024绐楀彛鍒濆璇勪及澶辫触锛屾棤璁粌杞紱淇濈暀鏃ф簮鐮?config/log銆傛柊鍖归厤涓ら」浣跨敤鍥涗釜256鐐规簮鍧楋紝SW浠?024鐐广€俽andom_block256 PID72544宸插惎鍔紝pretrained_block256灏氭湭鍚姩锛涙棤鑷姩闃熷垪锛屽緟鐪熷疄瀹屾垚鍐嶆樉寮忓惎鍔ㄣ€?

2026-10-01娓呯悊锛氬凡瀹屾垚v100鍥涢樁娈典笌v101 POR鐨勫啑浣檒ast_state锛岄噴鏀?5.39MiB锛涙墍鏈夋鏀剁泭/閫夊畾model_best銆侀娴嬨€佹簮鐮併€佹姤鍛婂拰渚濊禆淇濈暀銆倂102娲昏穬鐘舵€佷笉鍔ㄣ€傛竻鍗時esults/deep_learning/compact_source_features_v102_20261001/completed_resume_cleanup.json銆?

### 璁粌涓€熷害璇佹嵁锛?7杞紝闈炴渶缁堟敹鐩婏級
闅忔満婧愰殣钘忕壒寰佹渶浣砈W=73.573365207933锛涘钩鍧?0.930绉?杞紝鍚岃疆娆″師杈撳叆骞冲潎15.442绉?杞€傞澶栬緭鍏ュ閲?鍒濆鍖栦笉鍚岋紝涓嶆妸涓庡師杈撳叆宸€煎綋鍚屽閲忔秷铻嶃€傞璁粌瀵圭収灏氭湭鍚姩锛屾殏涓嶄笅鏈€缁堢粨璁猴紱鑻ユ潵婧愯〃寰佹湁瀹炴祴鏀剁泭锛屽悗缁紭鍏堟祴璇曞浐瀹氳緭鍏ュ潡绂荤嚎闅愯棌缂撳瓨鏉ラ檷浣庤绠楁垚鏈紝椤诲彟绔嬮厤鏂硅褰曞叾涓婁笅鏂囧樊寮傘€傚綋鍓嶄笉鍚姩绗笁椤广€佸叾浠栨姌鎴栨棫璺嚎銆?


## 2026-10-01 v103缂撳瓨鏂规浠呭噯澶?
宸插畬鎴愶細鍘熸簮256绐楀彛/128鏍稿績闅愯棌缂撳瓨鐢熸垚鍣ㄣ€佺鏈塖W鍜岃缁冨叆鍙ｏ紝绂佹婧愰娴?鏍囩缂撳瓨銆傚綋鍓峷102瀵圭収浠嶆椿璺冿紝鏃犳柊璁粌/鐢熸垚闃熷垪銆傝鍒掞細鍏堢湅鏉ユ簮鐗瑰緛鏈€缁堟秷铻嶏紝鑻ュ€煎緱淇濈暀鍐嶆祴璇曠紦瀛樺畾涔変笌閫熷害銆傛湭瀹屾垚锛氫换浣曠紦瀛樸€佽缁冦€佸鐩婂拰鍔犻€熼獙璇併€傜洰鏍囷細绾疍L鍘烵OF>82銆佺浜?80.8銆佸惈婧愨墹5璁粌闃舵锛涜揪鎴愬悗鍙仛鏂颁氦浠樸€?

### random_block256瀹為檯瀹屾垚
SW=74.025418470081锛涘浐瀹氫笁閫氶亾TOTAL=80.441444954722锛屽鏃ф渶濂?0.062752469755锛屼弗鏍?0.8=False銆俠est/stop=24/42锛屾湰椤?331.302绉掞紝宄板€?754.2MiB銆備粎4涓緷璧栬缁冮樁娈碉紝鎷熺浜斿瘑搴﹂樁娈靛皻鏈墽琛岋紱瀹屾暣OOF鏈畬鎴愩€?

2026-10-01 v102闅忔満婧愬鐓у凡鏈€缁堝畬鎴愶細SW=74.025418470081锛屽疄闄呭浐瀹氫笁閫氶亾TOTAL=80.441444954722锛屼粎4渚濊禆闃舵锛涙湭杩?0.8銆傞璁粌婧愬悓瀹归噺鍊欓€塒ID54004宸叉牳瀹炴椿璺冿紝绗?杞紝浠嶆棤鏈€缁堟潵婧愭秷铻嶆敹鐩娿€傚浐瀹氬綋鍓峆OR/PERM鏃讹紝SW闇€涓ユ牸>75.049861456590鎵嶈繃闂ㄦ锛涚户缁瘎浼拌€屼笉琛ユ姌銆倂103浠呭噯澶囷紝鏃犵紦瀛樻垨璁粌銆?

2026-10-01鍥捐氨灞€閮ㄤ慨姝ｏ細鍏抽棴璺嚎鎸夎妭鐐笽D鏇挎崲锛岄伩鍏嶅弽澶峜urate绱姞v98/v99鍚屼竴璐熺粨鏋溿€傜敓鎴愬悗鐨勫叧闂矾绾縄D鍞竴妫€鏌ラ€氳繃锛屼笉鏀瑰彉浠讳綍璁粌婧?閰嶇疆/鍒嗘暟銆傚綋鍓峷102棰勮缁冩潵婧愬€欓€夋寔缁繍琛岋紝缂撳瓨鏂规浠呭噯澶囥€?

2026-10-01鏄惧紡寮€濮媝retrained杈撳叆缂撳瓨鍑嗗锛氭簮256绐楀彛/128鏍稿績锛屼笉璇绘爣绛炬垨婧愰娴嬶紱涓巚102璁粌鍏变韩鏈€澶?涓狦PU鍚嶉銆傛涓轰竴娆¤緭鍏ラ澶勭悊锛屼笉鍚姩v103璁粌銆傚疄闄呭噯澶囨椂闂?鏄惧瓨/瑕嗙洊鍜屼笂涓嬫枃瀹氫箟鍐欑紦瀛榤anifest锛岀瓑寰卾102鏈€缁堟秷铻嶅啀鍐冲畾鏂拌缁冦€?

### pretrained_block256瀹為檯瀹屾垚
SW=73.649125550614锛涘浐瀹氫笁閫氶亾TOTAL=80.309742432908锛屽鏃ф渶濂?0.194454991568锛屼弗鏍?0.8=False銆俠est/stop=15/33锛屾湰椤?026.458绉掞紝宄板€?754.2MiB銆備粎4涓緷璧栬缁冮樁娈碉紝鎷熺浜斿瘑搴﹂樁娈靛皻鏈墽琛岋紱瀹屾暣OOF鏈畬鎴愩€?

### 棰勫畾涔夌紪鐮佸櫒鏉ユ簮娑堣瀺
pretrained-random SW=-0.376292919467銆傚悓瀹归噺銆佺鏈夐骞?鏂板stem鍒濆鍖栥€佹簮棰勫鐞哹uffers銆佹崯澶?Q50/鍘熺浜旀姌/80-18锛涗袱椤瑰疄闄呮渶浣虫仮澶嶅畬姣曘€傝宸€煎彧璇佹槑鏈缃殑婧愰殣钘忕壒寰佹敹鐩婏紱涓嶈兘褰撳畬鏁碠OF鎴栨€讳綋鏋舵瀯鏀剁泭銆?
鏈潵婧愮壒寰佹柟妗堜笉缁х画鎵弿缂栫爜鍣ㄧ瀛愩€丩R鎴栧眰鏁帮紱璐熺粨璁哄彧閫傜敤浜庢湰256婧愬潡/1024鏂癝W楠ㄥ共璁剧疆銆?

### v102缁撹涓庢湰杞惎鍔ㄧ悊鐢?
v102鏈€缁堟潵婧愭秷铻峱retrained-random SW=-0.376292919467锛屾棤澧炵泭锛涗笉鎵弿鏃хЩ鍔⊿W閿氱偣閰嶇疆銆倂103鏀瑰彉婧愪笂涓嬫枃锛氬師256婧愮獥鍙ｈ嚜宸辩殑relative閿氱偣銆佸彧淇濈暀涓績128锛屼互閬垮厤鏃W鏍稿績钀藉湪鍒囧垎婧愬潡杈圭紭銆傝繖鏄牴鎹粨鏋勫畾涔夋彁鍑虹殑鏂板亣璁撅紝灏氭湭璇佸疄鏄け璐ュ師鍥狅紝涓嶅０绉颁慨姝ｅ繀鐒舵湁鏀剁泭銆傞璁粌杈撳叆缂撳瓨鐢熸垚瀹屾垚锛?30268脳256 float16锛?2.793绉掞紝宄板€?12.7MiB锛屾簮棰勬祴鍜屾爣绛惧潎鏈娇鐢ㄣ€傛樉寮忕户缁悓缂撳瓨瀹氫箟random/pretrained鍖归厤瀹為獙锛涗笉琛ュ叾浠栨姌銆?

v102鏉ユ簮鐗瑰緛娑堣瀺涓鸿礋锛屾竻鐞?涓畬鎴愭ā鍨?鎭㈠鐘舵€侊紝鍏?8.00MiB锛涘師婧愪笌v103娲昏穬浜х墿銆佹渶缁堥娴嬨€佹簮鐮?閰嶇疆/鏃ュ織/鎶ュ憡鍜屽垹闄ゆ竻鍗曚繚鐣欍€?

2026-10-01 v103鍙屾柟宸叉牳瀹炴椿璺冿細random PID30732锛宲retrained PID86228锛涘悇鑷嘲鍊肩害2074MiB锛屽弻杩涚▼鍚堣鏄惧瓨绾?0104MiB锛屾湭OOM銆傜紦瀛樺崟椤规鍓嶇害13-14绉?杞紝鍙屽苟鍙戝悗绾?8绉?杞紱鏆傛湭璇佹槑鍚炲悙浼樺娍锛屾渶缁堣褰曞悓宸ヤ綔閲忔€诲閽燂紝鍚庣画鏂癝W浼樺厛涓茶銆傚師婧愭垚鏈?289.152绉掑強杈撳叆缂撳瓨10.114/12.793绉掍粛璁″叆锛屼笉绉板厤璐规垨涓巚102鏁板绛変环銆傜浜?80.8/瀹屾暣OOF>82鏈畬鎴愶紱鏃犲叾浠栨姌銆?

### random瀹為檯瀹屾垚
SW=73.989575926996锛涘浐瀹氫笁閫氶亾TOTAL=80.428900064642锛屽鏃ф渶濂?0.075297359835锛屼弗鏍?0.8=False銆俠est/stop=34/52锛屾湰椤?191.204绉掞紝宄板€?074.2MiB銆備粎4涓緷璧栬缁冮樁娈碉紝鎷熺浜斿瘑搴﹂樁娈靛皻鏈墽琛岋紱瀹屾暣OOF鏈畬鎴愩€?

### pretrained瀹為檯瀹屾垚
SW=73.823152241908锛涘浐瀹氫笁閫氶亾TOTAL=80.370651774862锛屽鏃ф渶濂?0.133545649615锛屼弗鏍?0.8=False銆俠est/stop=15/33锛屾湰椤?92.476绉掞紝宄板€?074.2MiB銆備粎4涓緷璧栬缁冮樁娈碉紝鎷熺浜斿瘑搴﹂樁娈靛皻鏈墽琛岋紱瀹屾暣OOF鏈畬鎴愩€?

### 棰勫畾涔夌紪鐮佸櫒鏉ユ簮娑堣瀺
pretrained-random SW=-0.166423685087銆傚悓瀹归噺銆佺鏈夐骞?鏂板stem鍒濆鍖栥€佹簮棰勫鐞哹uffers銆佹崯澶?Q50/鍘熺浜旀姌/80-18锛涗袱椤瑰疄闄呮渶浣虫仮澶嶅畬姣曘€傝宸€煎彧璇佹槑鏈缃殑婧愰殣钘忕壒寰佹敹鐩婏紱涓嶈兘褰撳畬鏁碠OF鎴栨€讳綋鏋舵瀯鏀剁泭銆?
鏈潵婧愮壒寰佹柟妗堜笉缁х画鎵弿缂栫爜鍣ㄧ瀛愩€丩R鎴栧眰鏁帮紱璐熺粨璁哄彧閫傜敤浜庢湰256婧愬潡/1024鏂癝W楠ㄥ共璁剧疆銆?


## 2026-10-01 v104鐩存帴缁撴瀯瀵圭収
宸插畬鎴愶細v102/v103婧愯〃寰佹秷铻嶆渶缁堝潎璐燂紝鍋滄鏉ユ簮鎵弿锛涙柊SW鍘熺偣鍒嗚鲸鐜囬浂鍒濆local/spectral婊ゆ尝瀹炵幇锛宮atched845913鍙傛暟銆傜幇鍦細鐩存帴涓茶璁粌local锛屽畬鎴愬悗spectral銆傚緟鍋氾細鏈€缁堟潵婧愬叧闂暣鐞嗐€佹护娉㈡秷铻嶃€佸疄闄呬笁閫氶亾銆佸繀瑕佹椂绗?瀵嗗害闃舵銆傜洰鏍囷細鍚潵婧愨墹5闃舵锛屽師绗簲>80.8銆佸畬鏁碠OF>82锛涜揪鎴愬悗鏂颁氦浠樸€?

v103涓ら」鏈€缁堟棤鏉ユ簮澧炵泭锛圫W宸?0.166423685087锛夛紝娓呯悊璐熸ā鍨嬨€佹仮澶嶇姸鎬佸拰2浠借緭鍏ラ殣钘忕紦瀛橈紝閲婃斁750.25MiB锛涢厤缃?婧愮爜/鏃ュ織/鏈€缁堥娴?manifest/鍥捐氨淇濈暀锛寁104涓嶄緷璧栬繖浜涙潈閲嶆垨缂撳瓨銆?

2026-10-01鏈壒缁х画锛歷103鍏抽棴骞惰褰曞疄闄呬袱椤瑰閽?230.846绉掞紙鍒濇鍗曢」锛屽悗娈靛弻骞跺彂锛屽惈涓绘満闂撮殧锛夛紝鏃犲悓绛変覆琛屽叏绋嬪熀鍑嗭紝涓嶅绉板疄娴嬪姞閫熸瘮銆倂104 local PID74048宸叉牳瀹炶缁冿紝epoch38锛屾渶浣砈W74.121752828158锛屽嘲鍊?722.2MiB锛泂pectral寰卨ocal瀹屾垚鍚庢樉寮忓惎鍔ㄣ€傛渶缁堝惈婧愨墹5闃舵銆佸師绗簲>80.8銆佸畬鏁碠OF>82浠嶆湭瀹屾垚銆?

### local鏈€缁堝疄闄呯粍鍚?
SW=74.121752828158锛孴OTAL=80.475161980049锛屽鏃ф渶濂?0.029035444428锛泂trict808=False銆傛湰椤筨est/stop=37/55锛?85.678绉掞紝宄板€?722.2MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

2026-10-01 v104灞€閮ㄥ鐓ф渶缁堬細SW=74.121752828158锛屽疄闄呭浐瀹氫笁閫氶亾TOTAL=80.475161980049锛屼粎4璁粌渚濊禆闃舵锛屾湭杩?0.8锛沚est37/stop55锛?85.678绉掞紝淇濈暀閫夊畾妯″瀷銆傜浉瀵箆100鍘熸棤鏂板婊ゆ尝SW74.0066149558752涓?0.1151378722824锛堟柊澧炲閲忥紝闈炲悓瀹归噺婊ゆ尝鑼冨洿缁撹锛夈€俿pectral PID63332宸叉牳瀹炵9杞湡瀹炶繍琛岋紝鍒濆瀹樻柟鍒嗕笌local鐩稿悓锛涘嘲鍊?805.7MiB锛屽皻鏃犳渶缁堥鍩熷鐩娿€傛棤鍏朵粬鎶橈紝瀹屾暣OOF>82鏈畬鎴愩€?


## 2026-10-01 v105 POR鏈鎶曞奖
宸插畬鎴愶細鍘熸簮鍚屽嚱鏁伴噸寤簃ean_scope+16640鏈鎶曞奖姊害閾惧疄鐜帮紝own POR锛涚幇鍦細鐩存帴璁粌锛屼笌SW spectral鏈€澶氬弻浠诲姟銆傚緟鍋氾細涓巚101鏈€缁堟洿鏂拌寖鍥村樊鍊煎拰瀹為檯涓夐€氶亾锛涗笉灏嗛儴鍒嗗垎鏁板綋鏀剁泭銆傜洰鏍囷細鍚簮鈮?闃舵銆佺浜?80.8銆佸師瀹屾暣OOF>82锛涜揪鎴愬悗鏂颁氦浠樸€?

### spectral鏈€缁堝疄闄呯粍鍚?
SW=73.974480635131锛孴OTAL=80.423616712489锛屽鏃ф渶濂?0.080580711987锛泂trict808=False銆傛湰椤筨est/stop=38/56锛?05.619绉掞紝宄板€?805.7MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

### 婊ゆ尝鑼冨洿鏈€缁堟秷铻?
spectral-local SW=-0.147272193027锛泂ame845913瀹归噺涓庨浂娈嬪樊鍒濆鍖?鍘熸暟鎹?Q50/棰勭畻锛屼袱椤瑰潎瀹為檯鎭㈠鏈€浣炽€傚彧鏈夋湰鍘熺偣鍒嗚鲸鐜?妯″紡鏁?闆舵墿灞曡缃殑鍗曟姌璇佹嵁锛岄潪瀹屾暣OOF鏀剁泭銆?
鍏抽棴鏈鍩熻缃殑棰戞暟/绐楀彛/LR鎵弿锛涘local纭湁鏁翠綋鏀剁泭鍒欏崟鐙瘎浼帮紝涓嶈兘鎶妉ocal鏀剁泭褰掔粰棰戝煙銆?


## 2026-10-01 v106瀵嗗害绗?闃舵
宸插畬鎴愶細v104 local閫夊畾SW74.121753銆侀鍩熸秷铻嶈礋锛涘喕缁揳tom涓€娆＄紦瀛?绉佹湁density瀹炵幇銆傜幇鍦細鐩存帴璁粌绗?闃舵锛屼笌灏廝OR鎶曞奖浠诲姟鍏ㄥ眬鏈€澶?GPU銆傚緟鍋氾細瀹為檯鐖跺鐩娿€佸浐瀹氫笁閫氶亾鍜屽寘鍚潵婧?鍏变韩闃舵鐨勮€楁椂銆傜洰鏍囷細鍘熺浜?80.8銆佸師瀹屾暣OOF>82銆佸畬鏁撮厤鏂光墹5闃舵锛涜揪鎴愬悗鏂颁氦浠樸€?

### 鏈€缁堟洿鏂拌寖鍥存秷铻?
POR=77.477365174678锛屽鍥哄畾鎶曞奖v101=-0.023003212967锛沚est9/stop27锛?59.577绉掞紝宄板€?86.5MiB銆傛柊澧炴姇褰卞疄闄呮搴?.372072锛屽叾浠栨簮鍙傛暟鍐荤粨涓斿疄闄呮渶浣抽娴嬮噸鐜帮紱鍚屽垵濮嬪畼鏂瑰垎銆傝礋缁撴灉锛岀户缁噰鐢╲101锛屼笉鎵3e-6鑼冨洿鎴栨墿澶цВ鍐汇€備粎绗簲鎶橈紝鏈獙璇佸畬鏁存敹鐩娿€?

### 鏈€缁堢5闃舵
SW=74.121752828158锛宐est0/stop18锛屽疄闄呮柊澧炵泭0锛?21.038绉掞紝宄板€?324.5MiB锛涘喕缁揳tom鍜屾渶浣虫仮澶嶉€氳繃锛岄娴嬩笌鐖秎ocal瀹屽叏涓€鑷淬€傚疄闄呭浐瀹氫笁閫氶亾TOTAL=80.475161980049锛屾湭杩?0.8銆傛墽琛屼簡鍚簮5璁粌闃舵锛岄€夊畾鏈夋晥渚濊禆浠嶆槸鐖剁殑4闃舵锛涗笉鎶婂璁粌涓€闃舵褰撳鐩婏紝淇濈暀鐖秎ocal銆傚仠姝㈡瀵嗗害璁剧疆锛屼笉琛ュ叾浠栨姌锛涘畬鏁村師OOF>82鏈畬鎴愩€?


## 2026-10-01 v107 娉ㄦ剰鍔涘弬鏁板寲
宸插畬鎴愶細local SW74.121753淇濈暀锛涢鍩熴€佹簮闅愯棌鐗瑰緛銆丳OR鏈鎶曞奖涓巐ocal瀵嗗害鏃犳柊澧炵泭锛屽仠姝㈠悇璐熻缃€傛柊dot/cosine鍚?45929鍙傛暟瀹炵幇锛宭ocal缁撴瀯浠庨殢鏈哄紑濮嬨€傜幇鍦細鐩存帴涓茶dot璁粌銆傚緟鍋氾細cosine銆佹渶缁堟秷铻嶅拰瀹為檯涓夐€氶亾锛屽繀瑕佹椂绗?缁嗗寲銆傜洰鏍囷細鍘熺浜?80.8銆佸師OOF>82銆佸惈婧愨墹5闃舵锛涜揪鎴愬悗鏂颁氦浠樸€?

v106 best0閲嶅妯″瀷鍙婃仮澶嶇姸鎬佹竻鐞?7.99MiB锛屼繚鐣欏疄闄呴€夊畾v104 local鐖舵ā鍨嬨€佸叏閮ㄩ娴?婧愮爜/閰嶇疆/鏃ュ織/缁撴灉涓庢竻鍗曘€倂105鍜宻pectral璐熸ā鍨嬫鍓嶆竻鐞?7.04MiB锛寁101姝ｆā鍨嬪拰娲昏穬v107涓嶅姩銆?

### dot鏈€缁堝疄闄呯粍鍚?
SW=73.996259635993锛孴OTAL=80.431239362791锛屽鏃ф渶濂?0.072958061686锛泂trict808=False銆傛湰椤筨est/stop=34/52锛?13.209绉掞紝宄板€?725.2MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

2026-10-01 v107 dot鏈€缁圫W=73.996259635993锛屽疄闄呭浐瀹氫笁閫氶亾TOTAL=80.431239362791锛?渚濊禆闃舵鍚簮锛夛紱best34/stop52锛?13.209绉掞紝鏈繃80.8銆俢osine PID42588宸叉牳瀹炴椿璺冿細epoch23锛屾渶浣砈W=73.820020434479锛屽嘲鍊?743.0MiB锛屽皻鏃犳渶缁堝弬鏁板寲娑堣瀺銆傚綋鍓嶄粛淇濈暀v104 local瀹為檯绱у噾TOTAL80.475161980049锛屼笉琛ュ叾浠栨姌锛涘畬鏁村師OOF>82鏈畬鎴愩€?

### cosine鏈€缁堝疄闄呯粍鍚?
SW=74.038234103145锛孴OTAL=80.445930426295锛屽鏃ф渶濂?0.058266998182锛泂trict808=False銆傛湰椤筨est/stop=34/52锛?91.858绉掞紝宄板€?743.0MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

### 娉ㄦ剰鍔涘弬鏁板寲鏈€缁堟秷铻?
cosine-dot SW=+0.041974467153锛泂ame845929瀹归噺涓庡叾浠栧弬鏁板垵濮嬪寲/鍘熸暟鎹?Q50/棰勭畻锛屼袱椤瑰潎瀹為檯鎭㈠鏈€浣炽€傚彧鏈夋湰QK褰掍竴鍖?娓╁害鍒濆鍖栬缃殑鍗曟姌璇佹嵁锛岄潪瀹屾暣OOF鏀剁泭銆?


2026-10-01 v107宸茬粨鏉燂細cosine SW74.038234103145銆乨ot73.996259635993锛屽悓瀹归噺宸€?0.041974467153锛圦K褰掍竴鍖栧拰鍒濆娓╁害鑱斿悎鍙樺寲锛夈€傚潎浣庝簬local74.121752828158锛屽綋鍓嶇簿绠€4闃舵TOTAL80.475161980049涓嶅彉锛屾湭杩?0.8銆備繚鐣檆osine灞€閮ㄦ瀵圭収锛屼笉鎵俯搴︺€倂108宸插噯澶?30268琛孭OR璇箟缁熻涓?鏇茬嚎瀵圭収锛岃缁冧晶鎷熷悎锛屽叡4渚濊禆闃舵锛屽皻鏃犺缁冨鐩娿€?

2026-10-01 v108 curve_control PID55952宸插疄闄呭惎鍔紝epoch4 SW鏈€浣?1.779416銆佸嘲鍊?726.49MiB锛涘彧鏄揩鐓э紝posterior灏氭湭鍚姩锛屾棤鑷姩闃熷垪銆傜紦瀛樺噯澶?.691绉掞紝闃舵婧?POR/PERM/鏂癝W鍏?锛屽叾浠栨姌涓嶅惎鍔ㄣ€倂107瀹屾垚涓嶆浛鎹㈠綋鍓?闃舵TOTAL80.475162锛沝ot璐熸ā鍨嬫竻鐞?8.21MiB骞剁暀娓呭崟锛宑osine灞€閮ㄦ瀵圭収鍙婂綋鍓嶆渶浣充繚鐣欍€?

2026-10-01 鏈壒鏇存柊锛歷108鏇茬嚎瀵圭収PID55952宸叉牳瀹炲瓨娲伙紝绗?7杞€乥est34 SW73.843376994495銆佸嘲鍊?726.49MiB锛涢潪鏈€缁堟敹鐩婏紝posterior浠嶆湭鍚姩銆傚畬鎴愯褰曞櫒璇箟/闃舵鏉ユ簮淇銆佹竻鐞嗗鍒剁殑v107璁″垝鍏冩暟鎹紙鍒濆鐗堟湰褰掓。锛夛紝鏂板鑰佸笀璁粌浜曞唴棰勬祴鐨勫垎甯冨亸绉婚檺鍒惰鏄庯紝璁粌婧愮爜/閰嶇疆鏈敼锛涘浘璋卞凡鏇存柊蹇収鍜岀湡瀹炴渶缁堢粍鍚堝叆鍙ｃ€傜洰鏍囧師绗簲>80.8鍙婂畬鏁碠OF>82浠嶆湭瀹屾垚锛屽綋鍓嶇簿绠€4闃舵TOTAL80.475161980049鍜屾棫浜や粯涓嶅彉銆傚緟鏈」瀹為檯缁撴潫鍚庢樉寮忓惎鍔ㄩ瀹氫箟posterior锛屾棤鑷姩闃熷垪鍜屽叾浠栨姌銆?

### curve_control鏈€缁堝疄闄呯粍鍚?
SW=73.843376994495锛孴OTAL=80.377730438267锛屽鏃ф渶濂?0.126466986210锛泂trict808=False銆傛湰椤筨est/stop=34/52锛?42.404绉掞紝宄板€?726.5MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

2026-10-01 鏈壒瀹屾垚锛歝urve_control鏈€浣砈W73.843376994495锛坆est34/stop52锛?42.404绉掞紝宄板€?726.49MiB锛夛紝瀹為檯涓夐€氶亾TOTAL80.377730438267锛?44665琛屾仮澶嶉娴嬩竴鑷达紱姣斿綋鍓峫ocal浣?.278375833663銆傞瀹氫箟por_posterior宸叉樉寮忓惎鍔≒ID3552锛屽綋鍓峞poch2/prior71.197115銆佸嘲鍊?726.49MiB锛涢潪鏈€缁堟垚缁┿€備袱椤逛弗鏍间覆琛岋紝鍚?46297鍙傛暟銆佸叾浠栫綉缁滃垵濮嬪寲/棰勭畻锛屼粎4缁磋緭鍏ユ潵婧愭敼鍙橈紝閰嶅宸€肩瓑寰呮湰鍊欓€夋渶缁堟仮澶嶃€傛帶鍒舵潈閲嶆殏鐣欑敤浜庨厤瀵癸紝涓嶅紑鍚洿澶氬€欓€夋垨鍏朵粬鎶橈紱鍘?闃舵鏈€濂?0.475161980049銆佸畬鏁碠OF81.891019297225鍜屾棫浜や粯淇濈暀銆傚浘璋便€佸疄闄呯粍鍚堟姤鍛婂拰PROJECTS宸叉洿鏂般€?

### por_posterior鏈€缁堝疄闄呯粍鍚?
SW=73.683727261052锛孴OTAL=80.321853031562锛屽鏃ф渶濂?0.182344392915锛泂trict808=False銆傛湰椤筨est/stop=20/38锛?72.643绉掞紝宄板€?726.5MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

### POR璇箟杈撳叆鏈€缁堟秷铻?
posterior-control SW=-0.159649733443锛泂ame846297瀹归噺涓庡叾浠栧弬鏁板垵濮嬪寲/鍘熸暟鎹?Q50/棰勭畻锛屼袱椤瑰潎瀹為檯鎭㈠鏈€浣炽€傚彧鏈夋湰4缁碢OR璇箟杈撳叆璁剧疆鐨勫崟鎶樿瘉鎹紝闈炲畬鏁碠OF鏀剁泭銆?
鍏抽棴鏈?缁碢OR璇箟缁熻璁剧疆锛涗笉鑷姩鎵弿鏇村鑰佸笀鎴栫粺璁★紝涓嶆硾鍖栦负鎵€鏈夎涔夌壒寰佹棤鏁堛€?

2026-10-01 v108缁撴潫锛歱osterior SW73.683727261052锛屽control73.843376994495涓?0.159649733443锛屽潎浣庝簬local锛屽叧闂?缁熻璁剧疆锛涜礋妫€鏌ョ偣/瀹屾暣鐘舵€佸強鐗瑰緛缂撳瓨娓呯悊47.57MiB骞剁暀娓呭崟銆備袱椤逛覆琛屽閽?180.004绉掞紝鍚惎鍔ㄩ棿闅?鍔犺浇/棰勭儹/璇勪环/IO锛岀紦瀛樻鍓?.691绉掑彟璁★紝涓嶄笌鍘嗗彶鏉ユ簮鎴愭湰娣峰悓銆倂109鎹负閫愮偣璇诲彇鍏ㄥ眬SW璁板繂锛屽悓瀹归噺geometry/content涓ら」锛屾柊澧?2672鍙傛暟闆惰緭鍑恒€佹畫宸?.25锛屽師pool/window/scoring淇濇寔锛屽惈婧?渚濊禆璁粌闃舵锛涘皻鏈缁冿紝涓嶈ˉ鍏朵粬鎶樸€?

2026-10-01 鏈壒宸插畬鎴恦108涓ら」鐪熷疄鎭㈠涓庡疄闄呯粍鍚堬細posterior-control SW=-0.159649733443锛屽叧闂?缁熻杈撳叆锛涜礋妯″瀷/瀹屾暣鐘舵€?鐗瑰緛缂撳瓨娓呯悊47.57MiB锛屾簮鐮?閰嶇疆/棰勬祴/鍝堝笇/鍒犻櫎娓呭崟淇濈暀銆備袱椤逛覆琛屼粠棣栨launch鑷虫渶缁坮eport澧欓挓1180.004绉掞紝鍗曢」642.404/472.643绉掞紱缂撳瓨鍑嗗9.691绉掑崟鍒楋紝鏃犲苟鍙戝姞閫熺粨璁恒€倂109宸叉樉寮忓惎鍔╣eometry PID28976锛屽凡鏍稿疄瀛樻椿锛宔poch8鏈€浣砈W72.970691691062锛屽嘲鍊?807.24MiB锛涘彧鏄揩鐓э紝content鏈惎鍔ㄣ€傛柊澧炴煡璇㈡敮璺?2672鍙傛暟锛屽叏858585锛岃緭鍑洪浂鍒濆鍖栥€伮?.25鐣岋紝鍘熺獥鍙?pool/璇勫垎涓嶅彉锛涗粠闅忔満瀹屾暣SW璁粌锛屽叏閰嶆柟鍚師鏉ユ簮4闃舵锛屾棤鍏朵粬鎶樻垨浜や粯鏇挎崲銆傞厤瀵硅褰曞櫒涓庡浘璋辨洿鏂板畬鎴愩€傜洰鏍囩浜?80.8/瀹屾暣OOF>82浠嶆湭瀹屾垚銆?

### geometry鏈€缁堝疄闄呯粍鍚?
SW=73.908896603003锛孴OTAL=80.400662301245锛屽鏃ф渶濂?0.103535123232锛泂trict808=False銆傛湰椤筨est/stop=49/67锛?73.170绉掞紝宄板€?807.2MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

2026-10-01 鏈壒瀹屾垚锛歡eometry SW73.908896603003锛宐est49/stop67锛?73.170绉掞紝宄板€?807.24MiB锛?44665鐐瑰疄闄呴噸杞戒竴鑷达紝瀹為檯鍥哄畾涓夐€氶亾TOTAL80.400662301245锛?渚濊禆闃舵鍚簮锛屼綆浜庡綋鍓峫ocal SW0.212856225155銆傞厤瀵筩ontent宸叉樉寮忓惎鍔≒ID23512骞舵牳瀹炲瓨娲伙紝epoch1/prior71.197115锛屽嘲鍊?795.52MiB锛涘彧鏄棭鏈熷揩鐓э紝灏氭棤閰嶅澧炵泭缁撹銆備袱鑰呭悓858585鍙傛暟/鍏朵粬鍒濆鍖?棰勭畻锛屽彧鏀筿uery杈撳叆銆備繚鐣欐帶鍒惰嚦鏈€缁堥厤瀵癸紱涓嶅啀澧炲姞鍊欓€夈€佷笉鍚姩鍏朵粬鎶樸€佷笉鏀逛氦浠樸€傚疄闄呯粍鍚堛€佸浘璋卞拰PROJECTS宸叉洿鏂帮紱褰撳墠4闃舵鏈€濂?0.475161980049銆佸畬鏁碠OF81.891019297225锛岀洰鏍囦粛鏈畬鎴愩€?

### content鏈€缁堝疄闄呯粍鍚?
SW=73.821136909607锛孴OTAL=80.369946408556锛屽鏃ф渶濂?0.134251015921锛泂trict808=False銆傛湰椤筨est/stop=34/52锛?80.807绉掞紝宄板€?795.5MiB銆備粎鍘熺浜斻€?渚濊禆璁粌闃舵锛屾潵婧愮畻鍏ワ紝绗?瀵嗗害鏈墽琛岋紱鏃犲叾浠栨姌銆?

### SW閫愮偣鍏ㄥ眬鏌ヨ鏈€缁堟秷铻?
content-geometry SW=-0.087759693396锛泂ame858585瀹归噺涓庡叾浠栧弬鏁板垵濮嬪寲/鍘熸暟鎹?Q50/棰勭畻锛屼袱椤瑰潎瀹為檯鎭㈠鏈€浣炽€傚彧鏈夋湰閫愮偣鏌ヨ杈撳叆璁剧疆鐨勫崟鎶樿瘉鎹紝闈炲畬鏁碠OF鏀剁泭銆?
鍏抽棴鏈煡璇㈡潵婧?娈嬪樊杈圭晫璁剧疆锛屼笉鑷姩鎵ご鏁版垨骞呭害锛屼笉娉涘寲涓烘墍鏈夎蹇嗚鍙栨棤鏁堛€?

2026-10-01 v109宸茬粨鏉燂細content SW73.821136909607锛実eometry73.908896603003锛屽尮閰嶅樊鍊?0.087759693396锛屽潎浣庝簬local锛屽叧闂湰鏌ヨ鏉ユ簮/娈嬪樊璁剧疆銆傝礋鏉冮噸/瀹屾暣鐘舵€佹竻鐞?6.91MiB骞剁暀娓呭崟锛涗袱椤逛覆琛屽閽?658.677绉掞紝鍗曢」873.170/680.807绉掞紝鍚姞杞?棰勭儹/IO/鍚姩闂撮殧銆倂110杞悜鍚?45913鍙傛暟鐨勫垎灞傚姩浣滃鍔变唬鐞嗭紝澶嶇敤v104/local鐪熷疄鎺у埗锛涘彧璁粌loss鏀瑰彉锛屽師楠岃瘉鍔ㄤ綔argmax涓嶆敼锛屽惈婧?闃舵锛屽皻鏃犳垚缁┿€?

2026-10-01 鏈壒瀹屾垚锛歷109鍐呭鏌ヨSW73.821136909607銆佷綅缃?3.908896603003锛屽樊鍊?0.087759693396锛屽潎浣庝簬local锛屽叧闂湰璁剧疆锛涜礋鏉冮噸/瀹屾暣鐘舵€佹竻鐞?6.91MiB锛岀暀婧愮爜/閰嶇疆/棰勬祴/瀹為檯缁勫悎/鍒犻櫎娓呭崟銆傛煡璇袱椤逛覆琛屽閽?658.677绉掞紝鍗曢」873.170/680.807绉掞紝鍚姞杞?棰勭儹/IO涓庡惎鍔ㄩ棿闅旓紝鏃犲苟鍙戝姞閫熺粨璁恒€倂110鏀硅缁冩晥鐢ㄤ唬鐞嗭紝鍘?45913鍙傛暟local缁撴瀯/鍒濆鍖?80-18/Q50/璇勫垎涓庨獙璇乤rgmax淇濇寔锛屽鐢╲104/local鍘嗗彶骞冲潶鎺у埗閬垮厤閲嶈銆俬ierarchical PID51128宸叉牳瀹炲瓨娲伙紝epoch8 SW鏈€浣?2.998922466155銆佸嘲鍊?721.17MiB锛涢潪鏈€缁堟敹鐩婏紝鍚簮4渚濊禆闃舵銆傝褰曞櫒銆佺嫭绔嬭褰曞拰鍥捐氨宸叉洿鏂帮紱绗簲>80.8/瀹屾暣OOF>82浠嶆湭瀹屾垚锛屼笉鍚姩鍏朵粬鎶樻垨淇敼鏃т氦浠樸€?

鏈€缁堬細SW=74.002091084435锛宧ierarchical-flat=-0.119661743722锛屽疄闄匱OTAL=80.433280369746锛宻trict808=False锛涘惈婧?璁粌闃舵锛屼粎鍘熺浜旓紝瀹屾暣OOF鏈獙璇併€俠est/stop=34/52锛?77.367绉掞紝宄板€?721.2MiB銆?

2026-10-01 v110缁撴潫锛歋W74.002091084435锛屾瘮flat -0.119661743722锛屽疄闄匱OTAL80.433280369746锛宐est34/stop52锛?77.367绉掋€佸嘲鍊?721.17MiB锛涘叧闂綋鍓嶆崯澶辫缃€傝礋鏉冮噸/last娓呯悊18.20MiB骞剁暀娓呭崟銆倂111鏀逛负PERM鍘熷潎鍊兼洿鏂拌寖鍥达紝鐑惎鍔ㄥ悓绱у噾鐖禼heckpoint锛実ate_route/mean_coadapt涓ら」锛屽潎lr3e-5锛屽€欓€夊彧澶?064娲昏穬鍧囧€煎彉閲忥紱楠ㄥ共/娣峰悎鏉冮噸/灏哄害鍥哄畾銆傛簮/POR/PERM鍒濊缁?PERM缁嗗寲/SW鍏?闃舵锛屽皻鏃犳垚缁╋紝鏃犲叾浠栨姌鎴栦氦浠樿鐩栥€?

gate_route鏈€缁堬細PERM=89.378394211144锛屽鐖?+0.000000000000锛涘疄闄匱OTAL=80.475161980049锛宻trict808=False銆傛墽琛?闃舵銆侀€夊畾鏈夋晥渚濊禆4锛屽惈婧愶紝浠呯浜旓紝鏈ˉ鍏朵粬鎶樸€俠est/stop=0/18锛?5.216绉掞紝宄板€?86.5MiB銆?

mean_coadapt鏈€缁堬細PERM=89.378394211144锛屽鐖?+0.000000000000锛涘疄闄匱OTAL=80.475161980049锛宻trict808=False銆傛墽琛?闃舵銆侀€夊畾鏈夋晥渚濊禆4锛屽惈婧愶紝浠呯浜旓紝鏈ˉ鍏朵粬鎶樸€俠est/stop=0/18锛?7.187绉掞紝宄板€?86.5MiB銆?

鏈€缁堟洿鏂拌寖鍥村鐓э細mean-gate_route PERM=+0.000000000000锛涘€欓€夊2064娲昏穬鍧囧€煎彉閲忥紝闈炲悓璁粌瀹归噺銆備袱椤逛粠鍚屼竴涓凡閫塒ERM鍑芥暟寮€濮嬶紝浠呭師绗簲璇佹嵁锛岄潪瀹屾暣OOF鏀剁泭銆?

2026-10-01 v111涓ら」宸茬粨鏉燂細gate_route鍜宮ean_coadapt鍧嘝ERM89.378394211144锛屽潎best0/stop18锛屽尮閰嶅鐩?锛岀湡瀹為噸杞?鍐荤粨鍙傛暟涓€鑷达紱瀹為檯涓夐€氶亾浠峊OTAL80.475161980049銆傛墽琛屽惈婧?闃舵锛岄€夊畾鏈夋晥渚濊禆浠?锛堢埗鍑芥暟锛夛紝涓嶈兘澹扮О缁嗗寲鏈夋敹鐩娿€備袱椤瑰崟椤?5.216/97.187绉掋€佸嘲鍊煎潎786.47MiB锛涙€诲閽?18.387绉掞紝launch鑷硆eport鏃堕棿閲嶅彔0.000绉掞紝瀹為檯妯″紡serial锛屼笉鑳介瀹ｇО鍙屽苟鍙戞垨鍔犻€熴€傞噸澶?璐熸潈閲嶅拰鐘舵€佹竻鐞?6.30MiB骞剁暀娓呭崟锛岀埗PERM/current best/浜や粯淇濈暀銆傚叧闂湰3e-5鏇存柊鑼冨洿锛屼笉鎵╁ぇ瀵嗗害灏哄害/鏉冮噸鎴栨壂LR銆傛棤GPU浠诲姟浠嶈繍琛岋紝鏈壒瀹屾垚锛岀洰鏍囧師绗簲>80.8涓庡師OOF>82鏈畬鎴愶紝鍚庣画鎹㈡満鍒讹紝涓嶈ˉ鍏朵粬鎶樸€?

2026-10-01 v112鎹㈠埌POR瀵嗗害鏃忥紝鍗曢」Cauchy涓巚101鍘嗗彶Student-t3鎺у埗锛涘悓绱у噾浣嶇疆璇诲嚭/鍘熷弬鏁板垵濮嬪寲锛屽浐瀹氬垵濮嬫瘡鍒嗛噺IQR鍖归厤锛孨LL/CDF鍚屾锛屽垵濮嬪嚱鏁颁笉鍚屼笖涓嶅尮閰嶆柟宸€傚彧鍘熸簮+鏂癙OR+PERM+SW鍥涗緷璧栭樁娈碉紝灏氭棤璁粌鏀剁泭銆傛棫Gaussian/t5宸叉湁璁粌涓嶉噸鎵紝v111鏇存柊鑼冨洿鍏抽棴锛屽叾浠栨姌/浜や粯涓嶅姩銆?

鏈€缁堬細POR=77.525094480331锛宑auchy-t3=+0.024726092686锛屽疄闄匱OTAL=80.482579807855锛宻trict808=False锛涘惈婧?璁粌闃舵锛屼粎鍘熺浜旓紝瀹屾暣OOF鏈獙璇併€俠est/stop=25/43锛?29.987绉掞紝宄板€?86.5MiB銆?

2026-10-01 v112瀹屾垚骞朵繚鐣欙細POR77.525094480331锛屽垵濮嬪寲77.274755692844锛屽涔犲鐩?0.250338787487锛涚浉瀵箆101瀹屾垚Student-t3鏁翠綋瀹舵棌澧炵泭+0.024726092686锛堜技鐒?CDF涓€璧锋敼锛屽垵濮嬪嚱鏁颁笉鍚岋級銆俠est25/stop43锛?29.987绉掞紝宄板€?86.47MiB锛?44665鐐筨est閲嶈浇鍜屽喕缁撳弬鏁颁竴鑷淬€傚浐瀹氫笁閫氶亾瀹為檯TOTAL80.482579807855锛岃緝鍓嶇揣鍑?0.007417827806锛屼粛浣庢棫18闃舵80.504197鍜?0.8闂ㄦ銆傚綋鍓嶆湁鏁?璁粌闃舵鍚簮锛屾ā鍨?鏉ユ簮/鍝堝笇鍜岀湡瀹炶瘎鍒嗗啓鍏URRENT_RECIPE.json锛涙湭琛ュ叾浠栨姌/瀹屾暣OOF锛屾棫浜や粯涓嶆敼銆傛牎姝ｅ浘璋卞鍒剁殑閿欒mode鏍囩涓篶auchy骞跺垹闄よ櫄鍋囨湭璁粌鑺傜偣锛岃缁冩簮鐮?hash涓嶅彉銆備笅鎵瑰彲娌垮凡璇佸疄瀵嗗害鏃忓眬閮ㄦ敹鐩婁綔鏈夐檺鏂板鐓э紱灏氭湭鍒涘缓鎴栧惎鍔ㄦ柊鍊欓€夛紝褰撳墠鏃犳椿璺冭缁冦€?

2026-10-01 v113娌縱112灞€閮ㄦ鏀剁泭锛氶瀹氫箟t3_cauchy/cauchy_t3涓や釜鍐呴儴涓撳鏃忚祴鍊硷紝鍚?4483鍙傛暟/source raw鍒濆鍖?棰勭畻锛屼技鐒朵笌CDF鍚屾锛涘垵濮嬪嚱鏁颁笉鍚屻€佸惈Cauchy鏃犳湁闄愬潎鍊?鏂瑰樊锛屼笉閫愯鎸戞棌鎴栧閮ㄦ姇绁ㄣ€傚師source/鏂癙OR/PERM/SW鍏?鏈夋晥闃舵锛屽綋鍓嶅皻鏃犳垚缁╋紝鏃犲叾浠栨姌/浜や粯瑕嗙洊銆?

cauchy_t3鏈€缁堬細POR=77.520670798696锛屽鍚岃川Cauchy=-0.004423681635锛涘疄闄匱OTAL=80.481252703364锛宻trict808=False銆傛墽琛?闃舵銆侀€夊畾鏈夋晥渚濊禆4锛屽惈婧愶紝浠呯浜旓紝鏈ˉ鍏朵粬鎶樸€俠est/stop=41/59锛?09.495绉掞紝宄板€?20.8MiB銆?

t3_cauchy鏈€缁堬細POR=77.540454326739锛屽鍚岃川Cauchy=+0.015359846408锛涘疄闄匱OTAL=80.487187761777锛宻trict808=False銆傛墽琛?闃舵銆侀€夊畾鏈夋晥渚濊禆4锛屽惈婧愶紝浠呯浜旓紝鏈ˉ鍏朵粬鎶樸€俠est/stop=69/80锛?23.388绉掞紝宄板€?20.8MiB銆?

鏈€缁堝垎閲忔棌璧嬪€煎鐓э細cauchy_t3-t3_cauchy POR=-0.019783528043锛涗袱椤瑰悓54483娲昏穬鍙傛暟涓庡師濮媟aw鍒濆鍖栵紝鍒濆棰勬祴鍑芥暟涓嶅悓锛涗粎鍥哄畾涓撳鏃忚祴鍊间笉鍚岋紝浠呭師绗簲璇佹嵁锛岄潪瀹屾暣OOF鏀剁泭銆?

2026-10-01 v113瀹屾垚锛歵3_cauchy POR77.540454326739锛宐est69/stop80锛屾瘮鍚岃川Cauchy+0.015359846408锛屾瘮鍚岃川t3+0.040085939094锛涘弽鍚?7.520670798696锛宐est41/stop59锛屾瘮Cauchy-0.004423681635浣嗗t3+0.020302411051锛屼袱best鍧囦綔涓哄眬閮ㄦ瀵圭収淇濈暀銆傝祴鍊煎弽鍚?姝ｅ悜=-0.019783528043锛涘垵濮嬪嚱鏁颁笉鍚岋紝涓嶅崟鐙綊鍥犺缁冩妧宸с€備袱椤圭湡瀹?44665鐐归噸杞?鍐荤粨鏍搁獙瀹屾垚锛屽疄闄呴噰鐢ㄦ鍚?闃舵TOTAL80.487187761777锛屾瘮鍓嶇揣鍑?0.004607953923锛屼粛浣庢棫18闃舵80.504197涓?0.8銆傚綋鍓嶉厤鏂?妯″瀷鍝堝笇鍐機URRENT_RECIPE.json锛屾棫閰嶆柟涓庝氦浠樹笉鍔紝瀹屾暣鍘烵OF81.891019鐩爣浠嶆湭瀹屾垚锛屼笉琛ュ叾浠栨姌銆傚疄娴嬪弻骞跺彂澧欓挓528.404绉掞紝launch鑷硆eport閲嶅彔414.475绉掞紱鍗曢」523.388/409.495绉掑惈浜夌敤銆佸嘲鍊煎悇920.85MiB锛屼笉澹扮О鍚屽伐浣滈噺涓茶鍔犻€熴€備粎鍐椾綑瀹屾垚last鐘舵€佹竻鐞?6.52MiB锛屾best/鍘焥ource/鎺у埗/璇佹嵁淇濈暀锛涚幇鏃犳椿璺冭缁冿紝鏃犺嚜鍔ㄩ槦鍒椼€?

# v114 fixed mixed-family POR width update scope
Focus/hypothesis: original t3 reference widths may need adaptation for the selected internal t3/Cauchy family assignment.
Controls: same restored v113 forward checkpoint and initial predictive function; own POR likelihood and synchronized CDF, original features/split/Q50/scoring unchanged. Existing active groups lr3e-5; location_scale additionally updates 2064 original width elements. This is an update-scope comparison, not equal active capacity. Width changes the numerical detached location-residual bound as well.
Explored/verified: implemented two explicitly launched arms, no final score yet. Atomic complete last states; max80/pat18/min10/bs64/64steps/cos80, normal/max2GPU.
Gain: not established. Parent POR77.540454326739, actual TOTAL80.487187761777. Source + initial POR + refinement + PERM + SW = 5 executed dependency training stages; if epoch0 is selected, effective selected stages remain4.
Not worth continuing: no negative evidence for this new bounded study yet. Closed cold-start/scaling/voting/decoder routes remain closed.
Done: scoped implementation and explicit launch. Pending: restored-best prediction integrity, actual three-channel assembly, scope delta and measured wall time. Full OOF>82 remains unmet; other folds require strict fifth>80.8 plus gain. Prior delivery unchanged.
Optional after completion: evidence-based next bounded study, no automatic queue.

V114 location_scale restored POR=77.540454326739; actual TOTAL=80.487187761777; parent gain=+0.000000000000; executed stages5, selected stages=4; fifth only, strict808=False; best/stop=0/18.

V114 location_only restored POR=77.548069494893; actual TOTAL=80.489472312223; parent gain=+0.007615168154; executed stages5, selected stages=5; fifth only, strict808=False; best/stop=13/31.

V114 final width-scope delta (scale-location) POR=-0.007615168154; same initial function, unequal active capacity; fifth only, no other folds.

Final: selected location_only, five total stages including source and initial POR; actual fifth TOTAL=80.489472312223, gain=+0.002284550446. Both144665-point best replays and frozen-parameter checks passed. Width candidate best0, no gain; close this width scope without generalizing all width learning. Remaining folds not started. Pair concurrent wall=206.466s, single201.478/132.363s with contention, peak920.85/920.87MiB; no equal-work serial benchmark. Sources/current and positive controls preserved; old delivery unchanged. FullOOF>82 still unmet.

# v115 compact PERM heavy-tail family comparison
Focus: test whether fixed Gaussian tails limit PERM compact readout; compare t3 and Cauchy with source Gaussian component IQR matched. Raw source parameters identical; initial predictive functions differ, so initial family change is not training gain.
Controls: retained v100 Gaussian PERM43431 active parameters, same source/init/data/Q50/optimizers/80-18/min10/bs64/64steps. Two fixed family arms; own PERM proper likelihood/CDF synchronized, original grid/action scoring unchanged. Original gates/routes/side train; all source density rows and backbone frozen. No external voting, test labels or row-wise selection.
Explored/verified: implementation prepared, no training score yet. Source-only initialization for new PERM; fixed POR=v114, SW=v104. Five total dependency stages including source and both POR phases. Two explicit GPU tasks maximum, atomic complete last states and explicit resume.
Gain: not established; Gaussian89.378394211144, prior actual compact TOTAL80.489472312223. Report own-family learning gain and matched-budget Gaussian difference separately. Cauchy has no finite mean/variance.
Not worth continuing: no evidence yet for this scoped family change. Prior failed warm PERM mean scopes and POR width scope are not reopened.
Pending: direct training, restored-best/population checks and actual three-target assembly; strict fifth>80.8 plus gain before remaining folds. Full originalOOF>82 remains unmet. Prior delivery preserved.
Optional: only evidence-based refinement with stage budget, no automatic queue.

V115 t3: restored PERM=88.396923292819; Gaussian delta=-0.981470918325; actual fifth TOTAL=80.145957490810;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=2/20, seconds=131.115, peakMiB=792.793.

V115 cauchy: restored PERM=88.391510248652; Gaussian delta=-0.986883962492; actual fifth TOTAL=80.144062925351;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=9/27, seconds=164.107, peakMiB=786.465.

Final decision: no new family adopted. PERM t3-Gaussian=-0.981470918325, Cauchy-Gaussian=-0.986883962492. Both final best checkpoints reproduce all144665 predictions and frozen parameters exactly. Initial family shifts caused ~1point loss not recovered by gate/routes adaptation. Close this fixed-IQR/frozen-density family scope; no general claim against all heavy-tail models. Current5-stage fifthTOTAL=80.489472312223, unchanged, strict808 false; original fullOOF>82 still unmet. Concurrent pair wall=169.044s; per-run/peak in pair_timing.json, no equal-work serial speedup evidence. Negative best/last checkpoints removed with manifest; source/current/positive controls and old delivery preserved.

# v116 SW fixed-capacity dilated depth filter
Focus: successful v104 learned65-tap local filter gains may depend on depth span. Compare dilation2/span129 and dilation4/span257 against retained dilation1/span65, with the same845913 total/6432 extra parameters and zero filter initialization.
Controls: same random SW initialization, raw206+relative input, original per-well conditioning/global/local backbone/labels/Q50/fifth split/score, own Gaussian NLL+utility1, SAM.1, DropKey.4, lr3e-4/cos80,80/18/min10/b64/64steps. Only local residual filter dilation/padding changes; no window/grid/voting changes. Zero filters give identical initial predictive function.
Explored/verified: two scoped implementations prepared, no scores. Serial explicit SW arms chosen because prior SW pair~28seconds/epoch each versus13-14solo did not show aggregate benefit. Max2GPU remains an upper limit; no automatic queue.
Gain: not established. Retained SW74.121752828158; current actual fifthTOTAL80.489472312223. Fixed PORv114/PERMv100 plus new random SW totals5 dependency stages including source and both POR phases. No old trained SW loaded.
Not worth continuing: prior spectral/query/hierarchical-utility/teacher changes have no net benefit and are not reopened. This receptive-span study has no result yet.
Pending: explicit dilation2 launch first, then decide second launch from evidence; restore best and actual3-target assembly, record same-capacity delta and real wall/peak. Strict fifth>80.8 plus gain before other folds. FullOOF>82 unmet. Keep prior delivery/positive controls.
Optional after completion: evidence-based retention/refinement only, within5-stage limit.

Live snapshot: dilation2 PID=71100, epoch=5, bestSW=72.175100151643; not final. InitialSW71.197115081472 matches original control initial function.845913 active parameters, measured~12s/epoch, peak=1722.197MiB, loading/warmup and validation counted by trainer. Dilation4 remains prepared/untrained, no queue. No fullCV or other folds started.

V116 dilation2: restored SW=74.036906311599; local dilation1 delta=-0.084846516558; actual fifth TOTAL=80.459776031428;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=617.022, peakMiB=1722.197.

Dilation2 final: restoredSW74.036906311599, against dilation1=-0.084846516558; actual3-target fifthTOTAL80.459776031428, no promotion. best32/stop50,617.022s including load/warmup/train/eval/replay/save, peak1722.197MiB. Checkpoint replay144665 exact and frozen parameters exact. Negative best/last removed with manifest (18.202MiB), compact result/source retained. Dilation4 explicitly launched after dilation2 terminal, PID28492; actual serial execution, not concurrent, no automatic queue. Current best5-stageTOTAL80.489472312223 unchanged; no other folds. Pending dilation4 final and full ablation decision.

# v117 compact PERM nonlinear joint-density readout
Focus/hypothesis: compact PERM previously adjusts only default gates and expert routes; fixed source density may constrain its gain. Add a bounded nonlinear191-32-24 residual per expert jointly adjusting mixture log-weights, locations and raw widths while keeping source density rows/backbone frozen.
Controls: original source initialization only, no v100 trained PERM loaded. Existing gate/side/routes training remains; new output zero initialized so initial full function equals v100 Gaussian control. New capacity means an architecture bundle, not equal-capacity ablation. All own PERM Gaussian likelihood/CDF and original nine-curve/206+63/input/data/Q50/fifth/scoring unchanged. Bounds fixed .25 logit/raw-width and .25 detached source component width for location.
Explored/verified: one predefined implementation prepared, no score yet.80/18/min10/bs64/64steps, existing LRgroups3e-5/3e-4, atomic complete state; max2GPU. Intended overlap with one active SW task, not two SW tasks; record observed wall/contention without speedup claims.
Gain: not established; historical PERM89.378394211144, current actual fifthTOTAL80.489472312223. Frozen PORv114 and SWv104 mean5 total training stages including source and2 POR phases; added joint head trained together with existing PERM components, no extra phase.
Not worth continuing: failed warm linear-mean and fixed heavy-tail PERM directions closed; this new nonlinear joint density is not yet tested.
Pending: direct train, restore best/144665 prediction integrity, actual3-target assembly, delta/peak/time; strict fifth>80.8 plus gain before other folds. FullOOF>82 remains unmet. Old delivery preserved.
Optional: follow positive structural evidence within stage budget; no automatic queue.

V117 joint_density: restored PERM=89.347354882560; Gaussian delta=-0.031039328584; actual fifth TOTAL=80.478608547219;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=2/20, seconds=185.930, peakMiB=786.465.

Final v117: restoredPERM=89.347354882560, against Gaussian gate-only=-0.031039328584; actual3-target fifthTOTAL=80.478608547219. best/stop=2/20, seconds=185.930, peak=786.465MiB;144665 exact replay/frozen params checks passed. Close this bounded joint-density architecture scope, not all density learning. Prior compact5-stageTOTAL80.489472312223 retained. Both old linear-mean and this nonlinear joint-density failed; do not continue small PERM density-row sweeps. Source/current/positive controls retained; negative best/last removed with manifest; SW dilation4 stays active, no training stopped for cleanup. No other folds/fullgoal claim.

V116 dilation4: restored SW=74.039683407705; local dilation1 delta=-0.082069420453; actual fifth TOTAL=80.460748015065;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=685.994, peakMiB=1722.197.

Final v116: dilation4SW=74.039683407705, original dilation1 delta=-0.082069420453, actual fifthTOTAL=80.460748015065, best32/stop50. Dilation4-minus-dilation2=+0.002777096106, a tiny secondary contrast; both worse than predeclared retained original dilation1. Close this fixed-capacity dilation mechanism, no more dilation/kernel-width sweeps. Both144665-point best replays/frozen checks passed. Current5-stageTOTAL80.489472312223 unchanged, strict808false; no other folds. Actual SW pair serial, measured pair wall=1368.167s including gap, singles617.022/685.994s, peak1722.197MiB each; second overlaps small PERM task, no standalone speedup claim. Both unpromising best/last outputs cleaned with separate manifests, prior local positive/source/current/delivery retained. FullOOF>82 remains unmet. Next direction must change mechanism rather than prolong these failed density/dilation scopes.

# v118 own POR likelihood/official soft-utility optimization
Focus: new positive mixed-family POR readout may optimize likelihood rather than original relative-error action reward. Two bounded loss arms alpha.25/.5, original utility soft-action temperature.05; retain proper mixed-family NLL and fixed t3/Cauchy CDF.
Controls: same actual v113 forward checkpoint/function,54483 trainable parameters, existing locations/new heads/routes lr3e-5, original density weights/scales/gate/backbone frozen. Reuse completed v114 location_only likelihood control with identical init/optimizer/budget/window/score; no redundant retrain. This replaces v114 refinement phase, not an additional phase: source+initial POR+new refinement+PERM+SW=5 total training stages.
Explored/verified: implemented, no training score yet. Original256-bin/800-action+default argmax unchanged; only training loss adds own POR truth utility under original tolerance.08/default.1 and log1p inverse. Training truths only; no foreign labels/voting/pseudo/row-wise oracle. DifferentiableCDF safe finite-edge arithmetic and constant tails; no scale updates. Original9curve/206+63/Q50/fifth split/population,80/18/min10/b64/64steps/cos80 unchanged. Atomic full last states, explicit resume; max2GPU.
Gain: not established; historical controlPOR77.548069494893, actualTOTAL80.489472312223. Soft utility is a proxy, not exact argmax-score equivalence. Earlier different all-backbone utility negatives do not prove this fixed-head mixed-family scope.
Not worth continuing: no evidence for this new scope yet; failed heavy-tail PERM/density residual/dilated SW scopes remain closed.
Pending: explicit two-arm launch, restore best and actual3-target score, compare alpha arms to control, measured per-run/pair timing and memory. Strict fifth>80.8+gain before other folds. FullOOF>82 remains unmet, delivered v97 untouched.
Optional: retain real positive controls only; no automatic queue or hidden6th phase.

V118 alpha050 restored POR=77.540454326739; actual TOTAL=80.487187761777; parent gain=+0.000000000000; executed stages5, selected stages=4; fifth only, strict808=False; best/stop=0/18.

V118 alpha025 restored POR=77.545558734992; actual TOTAL=80.488719084253; parent gain=+0.005104408253; executed stages5, selected stages=5; fifth only, strict808=False; best/stop=13/31.

V118 final utility loss delta (alpha050-alpha025) POR=-0.005104408253; same initial function, unequal active capacity; fifth only, no other folds.

Final v118: alpha025POR77.545558734992, own initial gain+0.005104408253 but likelihood control delta=-0.002510759901; alpha050POR77.540454326739 best0, control delta=-0.007615168154. Both144665-point best replays and frozen parameters exact. Neither improves fixed fair likelihood control, so utility mechanism not adopted; no further coefficient/temperature sweep under this scope. Actual3-target scores in arm assembled JSONs; retained5-stageTOTAL80.489472312223, strict808false, no other folds/fullgoal claim. Pair concurrent wall=435.668s, launch-to-report overlap=315.306s; singles/peaks in pair_timing.json, includes contention, no matched serial speedup evidence. Negative best/last removed with manifests; initial learning gain alone is not a positive trick ablation. Source/current and verified positive controls, original delivery preserved. Next attempt should change training/representation mechanism rather than continue these failed utility strengths.

# v119 POR feature adaptation scope
Focus: small output-only changes may be constrained by frozen source features. Compare updating expert input calibration only versus all expert non-head feature parameters, at1e-5, using the positive mixed t3/Cauchy POR function.
Controls: same v113 actual best checkpoint/function, existing54483 POR mean/side/point/routes parameters lr3e-5; historical v114 location-only control reused. Input-only old router/readout coefficient tensors frozen. Predictions of gate/weight/width can still change via modified hidden features, so do not claim those functions frozen. Expert eval mode retained with gradients, preventing a dropout policy change. Hook detachment/no_grad removed in this isolated module only. Update-scope ablation, not equal active capacity.
Explored/verified: implemented two explicit arms, no training result yet. Proper own-POR mixed-family likelihood and synchronizedCDF/old original action argmax unchanged. Original nine curves/206+63/split/Q50/population,80/18/min10/bs64/64steps. Existing groups cosine3e-5 to1e-5; new feature group starts ateta_min1e-5 and remains constant. No utility/prediction voting/foreign labels. Atomic complete state, explicit resume, max2GPU.
Gain: not established; likelihood controlPOR77.548069494893/actualTOTAL80.489472312223. Different earlier full-backbone anchored/t3 study lost0.029103 at larger scope; this mixed-family init/frozen-router/eval-mode and restricted-calibration contrast is separately tested, not presumed successful.
Not worth continuing: utility strengths, PERM tiny density heads and SW dilation are closed. No repeated projection/width expansion.
Pending: direct train, actual gradient groups/restored best/144665 population integrity, fixed3-channel assembly, fair control delta and measured time/peak. Source+initialPOR+new adaptation+PERM+SW=5 total stages, replacing v114 refinement, not appending6th. Strict fifth>80.8+gain before otherfolds; fullOOF>82 unmet, delivered package unchanged.
Optional: follow actual feature-scope positive evidence only; no automatic queue.

V119 calibration restored POR=77.540454326739; actual TOTAL=80.487187761777; parent gain=+0.000000000000; executed stages5, selected stages=4; fifth only, strict808=False; best/stop=0/18.

V119 features restored POR=77.540454326739; actual TOTAL=80.487187761777; parent gain=+0.000000000000; executed stages5, selected stages=4; fifth only, strict808=False; best/stop=0/18.

V119 final feature scope delta (features-calibration) POR=+0.000000000000; same initial function, unequal active capacity; fifth only, no other folds.

Final v119: both restoredPOR77.540454326739 best0/stop18, no gain vs own start, delta against same-init output-only likelihood control=-0.007615168154. Both144665-point replay/frozen-tensor integrity checks passed; feature/calibration gradient groups nonzero, so adaptation was actually attempted. Executed dependency phases5 but effective selected functions4 because epoch0 retained. Actual3-channel fifthTOTAL80.487187761777 for both; retain completed v1145-stageTOTAL80.489472312223. Close this low-rate/eval-mode warm feature scope, not all backbone adaptation; no LR/thaw-range sweep here. Pair concurrent wall=208.153s, overlap launch-to-report=164.693s, singles159.563/203.028s and peaks952.597/968.078MiB, active capacity differs; no equal-work serial speedup evidence. Negative duplicate best/last removed with manifest, parent/source/current positives and old delivery retained. No other folds; fullOOF>82 still unmet. Further work should move to a genuinely different representation/training strategy rather than extend closed small-head/density/utility/dilation/feature-unfreezing scopes.

# v120 SW train-only well sampling distribution
Focus: row-uniform anchor sampling gives long wells more gradient windows. Test well probability proportional to sqrt(training anchor count), then equal wells, with uniform anchor draw within chosen well. All original anchors remain eligible; no well selected/excluded using labels. Validation keeps official row-weighted scoring/population.
Controls: exactly v104 local random SW845913 parameters/initial function, original206+relative input/well conditioning/Q50/loss Gaussian+utility1/SAM.1/DropKey.4/lr3e-4/80-18/min10/b64/64steps and old decoder. Only training sampling changes. Counts from64 original training wells, original train-only Q/stats unchanged; no heldout-label sampling or prediction oracle. Historical row-uniform v104 control reused, not retrained.
Explored/verified: two implementations prepared, no current score. Earlier v14 small TCN/Q1/inner48-16 inverse-well loss weighting had negative0.069485 SW; no original fifth evidence and differs from current large local network/Q50/anchor sampling. This new bounded test does not presume balancing useful. Serial explicit SW launches retained due prior no aggregate dual-SW advantage, max2GPU; complete atomic sampler RNG/window/current/best/optimizer/schedule stopping state, explicit resume only.
Gain: not established; baselineSW74.121752828158/actualTOTAL80.489472312223. Window overlaps mean these probabilities change expected gradient distribution, not exact equal point reward weighting.
Not worth continuing: no new result yet; old inner weighting sweep, closed density/utility/dilation/unfreezing scopes not reopened. If sqrt fails clearly and earlier equal-well negative gives no reason for another run, prepared equal arm may remain untrained with explicit reason.
Pending: explicitly launch sqrt_well, restore best and actual3-channel validation, sampling manifest/time/peak; then evidence-based decision for second arm. Fixed PORv114/PERMv100 + new SW totals5 stages including source/2POR phases. Strict fifth>80.8+gain before other folds, original fullOOF>82 unmet. Old delivery preserved.
Optional: change direction based on measured gain, not automatic search/queue.

Live sqrt sampler: PID=24188, epoch=6, bestSW=72.442174672475, not final; initialSW71.197115081472 same as row-uniform v104 control,845913 parameters. Training64 wells/578891 anchors only; longest/shortest well mass ratio reduced from2.972596 to1.724122. Actual single SW process, measured~12s/epoch, peak1730.443MiB includes sampling tensors, data/warmup/eval counted by driver. Prepared equal_well remains untrained; no queue. Sampler RNG uses atomically saved windows generator, manifest preserved. Current prior5-stageTOTAL80.489472312223 unchanged, no remainingfolds.

V120 sqrt_well: restored SW=73.861179676692; row-uniform delta=-0.260573151465; actual fifth TOTAL=80.398271709210;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=21/39, seconds=484.116, peakMiB=1730.443.

Final v120 sqrt_well: restoredSW73.861179676692, against original row-uniform=-0.260573151465, actual3-target fifthTOTAL=80.398271709210; best21/stop39,484.116s including load/warmup/train/eval/save/replay, peak1730.443MiB.144665 exact replay and frozen tensors verified. Do not adopt sqrt redistribution. Equal_well intentionally NOT TRAINED/no score; current loss and earlier different inner-well weighting negative motivate changing direction, not evidence that this untrained arm failed. Close this mild SW count redistribution scope only. Retain current5-stageTOTAL80.489472312223, strict808false; no otherfolds/fullOOF completion. Negative best/last cleaned with manifest, source/current/positive controls/old delivery preserved. Next direction should preserve official training mass or change representation instead of increase this well-mass redistribution.

# v121 SW shared versus task-specific final residual readouts
Focus: default gating and conditional numeric density may compete in the same128-dimensional final readout. Add2 identical128-LN-64-GELU-128 bounded residual branches; shared arm sends their mean to both outputs, split sends branch0 to gate and branch1 to density. Shared backbone retained; no claim of fully independent models.
Controls: same original random v104 local SW body/input/prior, same33664 new parameters/equal879577 total trainable parameters, same branch initial tensors and zero residual outputs; original prior function equal. Both use the same separate F.linear gate/density implementation, so per-row computation differences do not define the contrast. Only branch-to-output connectivity changes. Parent-only controlv104 has less capacity; compare mechanism gain against matched shared arm separately from absolute score.
Explored/verified: implemented, no scores yet. Own SW Gaussian NLL+original soft utility1/SAM.1/DropKey.4, original row-uniform sampling/data/Q50/ninecurve206+relative/original fifth/decoder.80/18/min10/b64/64steps/lr3e-4/cos80. Source weights not loaded into SW, only training-side preprocessor/Q dependencies counted. Prior v11 smallTCN/discrete heads tested shared/split encoder scope; this is final residual connectivity on current large Gaussian/local network, not a replay of v11 or evidence of benefit.
Gain: not established; original SW74.121752828158 and compact actualTOTAL80.489472312223. Main fixed-capacity contrast split-minus-shared; preserve real local ablation gains even below best. Fixed PORv114/PERMv100 plus new SW totals5 phases including source/2POR phases.
Not worth continuing: closed loss/sampling/dilation/feature-unfreezing branches not reopened; no result for this new head scope yet.
Pending: explicit shared launch then split, restore best/144665 integrity and actual3-target assembly, capacity/ablation delta and per-run/total wall/peak. Serial SW chosen from prior contention measurements; max2GPU, full atomic last and explicit resume, no automatic queue/otherfolds. Strict fifth>80.8+gain before other folds; original fullOOF>82 unmet. Old delivery untouched.
Optional: retain proven connectivity trick only, no hidden extra training phase or external voting.

Live snapshot shared: PID=51456, epoch=6, bestSW=72.235014018835, not final. Actual879577 trainable parameters and initialSW71.197115081472 match planned source function; measured~13.2s/epoch, peak1898.744MiB, single SW process including warmup/load/eval per driver. Split is prepared/untrained, no automatic queue. Same-capacity connectivity comparison remains incomplete; cannot infer gain from current shared snapshot. Current prior5-stageTOTAL80.489472312223 unchanged, no other folds.

V121 shared: restored SW=73.873792847386; original local delta=-0.247959980772; actual fifth TOTAL=80.402686318953;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=34/52, seconds=709.672, peakMiB=1898.744.

Shared arm final: restoredSW73.873792847386, original local model delta=-0.247959980772, actual3-target fifthTOTAL=80.402686318953. best34/stop52,709.672s including load/warmup/train/eval/replay/save, peak1898.744MiB.144665 exact replay/frozen tensors passed. No shared candidate adoption. Keep shared checkpoint as required matched-control evidence until split completes; no automatic queue. Split explicitly launched after shared terminal. Actual serial SW arms, full ablation incomplete; do not call split beneficial/negative yet. Current prior5-stageTOTAL80.489472312223 unchanged, no otherfolds.

# v122 training utility matrix timing probe
Focus: large8192x257 by257x801 FP32 utility matrix multiplies may limit training speed. Compare IEEE versus TF32 for only this synthetic training-objective graph, without changing any active training source or final prediction/argmax precision.
Explored/verified: prepared short fixed-shape CUDA timing, actual utility matrix from retained training-side SW grid; synthetic probabilities/rewards, no labels/model updates. One benchmark plus one activeSW at most2GPU processes.8 warmup and32 measured forward/backward repetitions per precision, both own GPU stream synchronization/event timing and observed contention. No claim of real epoch/fullCV speedup from this isolated operation.
Gain: no timing result yet, no trained model or score; this is a runtime hypothesis, not accuracy improvement.
Not worth continuing: if isolated timing benefit small, no full-training precision experiment. If clear, a separately documented training-only approximation can be tested while validation/inference keeps original IEEE settings.
Pending: record elapsed/peak/precision/hardware/scope; updatePROJECTS/atlas. No training stage added by benchmark, no checkpoint, original best/fold splits/scoring/delivery remain unchanged. Original fullOOF>82 not achieved.

Final timing-only result: IEEE=2.651776ms versus TF32=2.308480ms per8192-point utility forward/backward, isolated ratio=1.148711;8 warmup/32 measured each while splitSW active. Whole benchmark runtime including source-grid load/warmup=0.397472s (Python startup outside this), peak210.849MiB. Estimated utility-only saving at2 evaluations*64steps is only~0.044seconds per epoch, compared with~13seconds actual epoch; this is an operation-count estimate, not measured epoch speedup. Do NOT launch full training precision experiment; keep original IEEE precision/settings in future trainers. No model/score or checkpoint produced, no claim of accuracy gain. Timing bottleneck hypothesis not worth further scope here; original goal remains unmet.

V121 split: restored SW=73.849287086711; original local delta=-0.272465741447; actual fifth TOTAL=80.394109302717;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=683.338, peakMiB=1899.771.

Final v121: sharedSW73.873792847386 versus splitSW73.849287086711; same879577 active parameters/init/loss, split-minus-shared=-0.024505760675. Matched connectivity contrast has no positive gain. Both also below original lower-capacity local SW74.121752828158; scope closes without generalizing all task-private heads. Both144665-point replay/frozen checks passed. Actual3-channel fifthTOTAL shared=80.402686318953, split=80.394109302717; retain prior5-stageTOTAL80.489472312223, strict808false, no otherfolds. Pair actual serial wall=1462.846s including gap=61.502s; singles709.672/683.338s, peak1898.744/1899.771MiB. Split briefly overlapped0.397s runtime-only TF32 probe, no full training precision change. Negative best/last removed with manifest; all source/current positives and delivery retained. Next work changes mechanism, no additional split-head width/bound sweep. FullOOF>82 still unmet.

# v123 same-radius parameter-scale weighted SAM direction
Focus: standard SAM global gradient direction may distribute perturbation poorly across differently scaled parameter tensors. Define d_p=(RMS(p)+0.01)*grad_p, then perturb delta_p=0.1*d_p/sqrt(sum(||d_p||^2)). This preserves total EuclideanL2 radius0.1, changes only direction; this is our bounded optimizer variant, NOT claimed standardASAM.
Controls: reuse completed v104 local SW/SAM.1 as same845913-parameter/random-init/model/input/loss/row-sampler/budget historical control. Original Gaussian ownSW NLL+utility1/DropKey.4/lr3e-4/80-18/min10/b64/64steps/cos80. Two passes same batch/RNG, clone exact parameter restoration before AdamW/clip2; no additional data, labels, output model, source-weight loading or decoder modification. Directions are local disposable tensors; full original optimizer/current/best/scheduler/RNG/window/stopping state atomically saved.
Explored/verified: implementation prepared, not trained/no score. One candidate only, no unbounded rho/LR/ASAM sweep. Historical v93/v94 SAM benefit belongs to different warm backbone; current local model itself remains original positive control, not retrained. Tensor-RMS direction is separate from those closed radius sweeps.
Gain: not established; historical SAM localSW74.121752828158, compact actualTOTAL80.489472312223. Compare fixed model/initial/budget mechanism delta; no claim of independent generalization from development-fold optimization. Extra optimizer kernels may cost time, record actual wall rather than assume faster.
Not worth continuing: source/param radius/precision/head-splitting negatives closed; no current result yet. If no gain, close this floor/direction scope without scanning new floors or radii.
Pending: explicit single GPU training, restored best/144665 integrity/actual3-channel assembly; preserve real positive control if delta positive. Fixed PORv114/PERMv100 + new SW totals5 phases incl source/2POR phases. Strict fifth>80.8+gain before otherfolds, original fullOOF>82 unmet. Old source/delivery preserved, no automatic queue.
Optional: follow demonstrated optimizer gain only within stage/time/memory budget.

Live v123 PID=89660, epoch=6, bestSW=72.970668780473, not final. Original SAM same-epoch historical best=72.268197681554, early best gap=+0.702471098919; not a completed mechanism gain and not OOF. Actual845913 trainable parameters, initialSW71.197115081472 same original prior. Measured~12.4s/epoch, peak1725.434MiB, oneGPU task, loading/warmup/eval counted by driver. Current prior5-stageTOTAL80.489472312223 retained, no remainingfolds. Config preserves same raw Euclidean SAM radius0.1 and clone/RNG two-pass path; only tensor-RMS-weighted perturbation direction changes.

V123 rms_direction: restored SW=73.897188251236; standard SAM delta=-0.224564576922; actual fifth TOTAL=80.410874710301;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=38/48, seconds=15089.669, peakMiB=1725.434.

Final v123: explicitly EARLY TERMINATED after authoritative checkpoint48 complete epochs, best38SW73.897188251236; original SAM control delta=-0.224564576922, actual3-target fifthTOTAL=80.410874710301.10 rounds without improvement before termination; initial trigger used stale33 snapshot, correction/original preserved in decision files. In-flight work discarded; atomic saved best restored in separate eval-only process, all144665 predictions/frozen tensors exactly match saved best. No trained source/config/checkpoint hash rewritten. Candidate not adopted; close this specificRMS+0.01 same-L2 perturbation direction without floor/radius/LR sweep. Current5-stageTOTAL80.489472312223 retained, no other folds/fullOOF completion.
Timing: actual launch-to-report wall=15102.457s; median epoch=12.862s but epoch18=14458.560s dominates wall. Kernel-Power507 recorded exit from Modern Standby during this long epoch. Report observed wall honestly; exact entry/cause not fully proved, no4-hour active-compute claim. Added training_awake_guard.py: NEW trainer can use with TrainingAwake(): main(), temporary SYSTEM_REQUIRED while training, releases on exit, no display or permanent power-plan edit. Request acquire/release verified once; existing trained source unchanged. Future manual stop/suspend remains user-controlled.
Negative best/last removed with manifest after replay; source/current/positive controls/delivery retained. Next training must adopt scoped keep-awake request to reduce accidental idle stalls; do not reopen negative optimizer direction.

# v124 SW finite-threshold CDF calibration loss
Focus: augment retained local Gaussian model/NLL/utility1 with quality-weighted squared error of predicted cumulative probability against own SW truth at255 finite original density edges. This is finite-threshold Brier calibration, NOT claimed exact continuousCRPS or exact official-score loss. Default99.9 atom enters CDF at its transformed location; use true SW label transformed under original train-only stats.
Controls: same v104 local random SW845913 parameters/init/priors/features/Q50/row sampler, proper Gaussian NLL+original utility1/SAM.1/DropKey.4/lr3e-4/80-18/min10/b64/64steps, original GaussianCDF/bin grid/action argmax/fifth score/population unchanged. Only added CDF coefficient1 or4 differs. Historical no-CDF control reused; no old trainedSW weights, extra model/data/labels/voting/oracle. Probability masses already computed for utility are reused via cumsum; no new decoder grid or expensive additional CDF evaluation.
Explored/verified: implemented two bounded arms, no scores. Prior POR action-utility changes, SW hierarchical action proxy and fixed family negatives are separate from this proper threshold-probability target. Temporary SYSTEM_REQUIRED context wired into NEW trainer, release on exit; previous frozen sources untouched. Added explicit stage early-stop request at epoch boundary(min10), preserving full atomic state and then normal best restoration/replay; no forced termination needed for this trainer.
Gain: not established; original SW74.121752828158 and actual currentTOTAL80.489472312223. Report same-capacity loss gain vs no-CDF and coefficient contrast separately. Finite-threshold calibration does not guarantee higher official action score.
Not worth continuing: no present evidence yet. Closed head connectivity/SAM direction/sampling/dilation/POR utility/unfreezing scopes remain closed. Stop weak trials with documented epoch-boundary requests if evidence supports, no automatic sweep.
Pending: explicitly cdf1 first, restore best/144665 integrity and actual3-channel assembly, real time/peak including any gaps. Decide cdf4 launch from evidence; no queue. Fixed PORv114/PERMv100 + newSW totals5 phases incl source/2POR phases. Strict fifth>80.8+gain before otherfolds, original fullOOF>82 unmet. Old delivery preserved.
Optional: follow a true measured calibration gain within stage/time/memory budget, not alter argmax scoring post hoc.

Live snapshot: cdf1 PID=69456, epoch=25, best SW=73.866179307971 at epoch24; running, gain not established. cdf4 remains untrained and unqueued. No remaining folds or delivery changes.

V124 cdf1: restored SW=74.051716256449; original no-CDF loss delta=-0.070036571709; actual fifth TOTAL=80.464959512125;5 stages including source and2 POR phases; strict808=False; no other folds. best/stop=32/50, seconds=733.464, peakMiB=1722.197.

Final v124: cdf1 best32/stop50 SW74.051716256449, delta vs local control -0.070036571709; actual fifth TOTAL80.464959512125 vs current80.489472312223 (-0.024512800098). All144665 rows best replay exact and frozen parameters exact. Serial training 733.464s, launch-to-report wall 738.282s, median epoch 13.952s, peak 1722.197MiB. cdf4 untrained/skipped; no measured cdf4 result, no coefficient sweep. Negative best/last deleted with SHA manifest (18.205MiB). Current5-stage recipe and submitted artifacts retained, no other folds started. No active v124 training. Goal full original OOF>82 remains unmet.

Prepared v125 fixed/learned POR tail-family probabilities from positive v113; identical0.95/0.05 initial mixtures, own properNLL/CDF; no scores yet,5 dependency phases including source. Max2 small POR jobs; no other folds or delivery edits.

## Latest user focus: one target until gain exceeds0.25
Concentrate subsequent bounded exploration on POR; keep PERM(v100) and SW(v104) fixed. Compact control POR77.54806949489308; also preserve historical original-fifth POR77.551936801885. Require POR>77.801936801885 (strict gain>0.25 against stronger historical reference), not just improved TOTAL or partial snapshot. Mechanism-positive ablations can guide continued POR work below this milestone. No switching to PERM/SW before this milestone absent new user direction. Original fullOOF>82 and strict original fifth TOTAL>80.8+gain promotion gate unchanged; five recipe training phases incl source. No automatic remaining folds. Maintain PROJECTS/exploration/atlas, max2GPU and frozen history/deliveries.

Prepared v126 POR gate/weight output-scope arms from positive v113; fixed other channels, no new scores or launch yet. Historical POR>77.8019368 milestone,5 recipe phases.

V125 fixed: restored POR=77.551813371420; delta vs v114=+0.003743876527, vs historical=-0.000123430465; actual fifth TOTAL=80.490595475181;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=13/31, seconds=237.983, peakMiB=920.849.

V125 learned: restored POR=77.545093940609; delta vs v114=-0.002975554284, vs historical=-0.006842861276; actual fifth TOTAL=80.488579645938;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=13/31, seconds=240.303, peakMiB=920.850.

V125 final: fixed POR77.551813371420, TOTAL80.490595475181, gain vs compactv114 +0.003743876527, versus historical -0.000123430465; retained fixed positive checkpoint. Learned POR77.545093940609, matched learned-minus-fixed -0.006719430811: learning16 tail logits has no gain here, do not sweep itsLR. Both best13/stop31, all144665 replay exact. Pair wall280.473s, overlap206.979s, singles{'fixed': 237.98260540000047, 'learned': 240.30304889997933}; no measured serial equal-work speedup. Cleanup 31.972MiB recorded. Historical POR>77.8019368 milestone and TOTAL>80.8 not met; continue exclusivelyPOR.

V126 explicitly launched two concurrent POR jobs: gatePID56892 and weightPID67496; exact process command lines verified. Runtime snapshot {"gate": {"status": "running", "pid": 56892, "stage": "gate", "epoch": 8, "best": 77.54045432673898, "best_epoch": 0, "bad": 8, "peak_MiB": 920.85546875}, "weight": {"status": "running", "pid": 67496, "stage": "weight", "epoch": 3, "best": 77.54045432673898, "best_epoch": 0, "bad": 3, "peak_MiB": 920.8740234375}}; training snapshots only, no final gain. Other targets fixed; POR strict historical+0.25 milestone remains pending.

Latest two-route authorization: route1 concentrates POR output optimization until historical original-fifth gain>0.25; route2 independently replaces neural backbone, starting random privatePOR ConvGRU v127. Old backbone need not be retained. Max2 jobs across routes, no automatic queue, original fifthTOTAL>80.8+gain before remainingfolds; recipe<=5 phases includingsource, original OOF>82 unmet. Other-target fixed predictions retained for fair POR assembly. This overrides any prior ban on new random/backbone attempts within this scoped route.

V126 gate: restored POR=77.540454326739; delta vs v114=-0.007615168154, vs historical=-0.011482475146; actual fifth TOTAL=80.487187761777;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=0/18, seconds=134.192, peakMiB=920.855.

V126 weight: restored POR=77.540454326739; delta vs v114=-0.007615168154, vs historical=-0.011482475146; actual fifth TOTAL=80.487187761777;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=0/18, seconds=132.467, peakMiB=920.874.

V126 finished both gate/weight best0/stop18 POR77.540454326739 vs v114 -0.007615168154, no new gain. All144665 replay/frozen exact, actualTOTAL80.487187761777. Closed this linear update scope, no LR/factor sweep; delete negative best/last 37.469MiB with SHA manifest. Pair timing {"concurrent": true, "pair_wall_seconds": 165.311885, "overlap_seconds": 110.503468, "single_reported_seconds": {"gate": 134.1915901999746, "weight": 132.46718979999423}, "peak_MiB": {"gate": 920.85546875, "weight": 920.8740234375}, "serial_equal_work_not_measured": true}; not a measured serial speedup. Resume two-route design: first new expressive joint POR head; second independently new random ConvGRU backbone.

Prepared POR route v128 joint local 16-component location readout on frozenv113, no score; paired with independent new v127 backbone route(max2). Goal historical POR+0.25 and original fullOOF>82 remain unmet.

Explicit two-route launch verified by actual PID command: 30752 mode convgru; live snapshot {"status": "running", "pid": 30752, "stage": "convgru", "epoch": 22, "best": 74.05493959779567, "best_epoch": 21, "bad": 1, "peak_MiB": 965.03173828125}. Training only, no verified final gain; combined max2GPU. Other channel predictions fixed, deliveries unchanged; historical POR>77.8019368 and original fullOOF>82 unmet.

Explicit two-route launch verified by actual PID command: 24272 mode local_joint; live snapshot {"status": "running", "pid": 24272, "stage": "local_joint", "epoch": 12, "best": 77.56050447331172, "best_epoch": 9, "bad": 3, "peak_MiB": 922.1650390625}. Training only, no verified final gain; combined max2GPU. Other channel predictions fixed, deliveries unchanged; historical POR>77.8019368 and original fullOOF>82 unmet.

V127 convgru: restored POR=74.054939597796; delta vs v114=-3.493129897097, vs historical=-3.496997204089; actual fifth TOTAL=79.441533343094;4 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=21/39, seconds=390.055, peakMiB=965.211.

V128 local_joint: restored POR=77.560504473312; delta vs v114=+0.012434978419, vs historical=+0.008567671427; actual fifth TOTAL=80.493202805749;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=9/27, seconds=211.513, peakMiB=922.165.

Two-route batch final: v128POR77.560504473312 best9/stop27, gain historical+0.008567671427, actualTOTAL80.493202805749,5 phases inclsource. Current recipe updated, positivebest/last retained; strictchannel+0.25 and TOTAL80.8 not met. V127randomConvGRU POR74.054939597796 best21/stop39, negative -3.496997204089, finishednegativebest/last cleaned11.691MiB with SHA manifest. All144665 replay exact, pair timing{"pair_wall_seconds": 394.662387, "overlap_seconds": 216.375362, "no_measured_serial_equal_work_speedup": true}, no serial speedup claim. NextPORpoint ablation tests depth-context contribution; nextbackbone changes architecture and supervised multitarget representation, no oldnetworkweights or extra labels.

Prepared v129 point ablation of positive v128, identical initialization/stored capacity but effectivekernelDOF differs; no score yet.

Prepared backbonev130 random joint hierarchicalattention, architecture/representation changed from negativeprivateConvGRU. Real3channel TOTAL selection,2standalonephases countingoriginalsourceQ; futureprivatePORrecipe<=5. No score or launch yet.

Explicitly launched point_joint PID86428, exact livecommand verified; snapshot {"status": "running", "pid": 86428, "stage": "point_joint", "epoch": 23, "best": 77.55790267084286, "best_epoch": 16, "bad": 7, "peak_MiB": 922.1650390625}. Two sharedslots, no final gain yet, no otherfolds or delivery changes.

Explicitly launched joint PID24264, exact livecommand verified; snapshot {"status": "running", "pid": 24264, "stage": "joint", "epoch": 6, "best": 77.84129728092265, "best_epoch": 6, "bad": 0, "peak_MiB": 1049.71923828125}. Two sharedslots, no final gain yet, no otherfolds or delivery changes.

Prepared v131 conditionalPOR location+logweight jointhead from frozenv113; v128positive historicalcontrol, new1040outputparams, no score/launch yet.

V129 point_joint: restored POR=77.561376174833; delta vs v114=+0.013306679940, vs historical=+0.009439372948; actual fifth TOTAL=80.493464316205;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=33/51, seconds=342.627, peakMiB=922.165.

V130 joint source verified: actual original-fifth scores{'POR': 73.73051180222829, 'PERM': 88.65145390526791, 'SW': 73.73768594153564, 'TOTAL': 78.95535248704972},2 stages incl originalsourceQ; strict808=False; no remainingfolds. best/stop=18/36, seconds=360.855, peakMiB=1049.719;144665x3 replay exact.

V129 final pointPOR77.561376174833 best33/stop51, gain historical+0.009439372948, actualTOTAL80.493464316205;5 phases. Local-minus-point=-0.000871701521, no confirmed localcontext benefit; effectivekernelDOF differs. Retain both positivecontrols and pointbest/last, currentrecipepoint. POR+0.25 and TOTAL80.8 unmet.

Prepared design selected before any launch/config: use pointreadout after finalv129point77.561376 beatslocal77.560504 by0.000871701521; newconditional logweighthead compared to v129point. Earlierlocal-preparation text superseded without changing trainedsource. Frozenv113sameinit, no trainedpointheadwarmstart; still5phases inclsource.

V130 closed: newsharedbackbone best18/stop36 actualoriginalfifthTOTAL78.955352487050, POR73.730511802228/PERM88.651453905268/SW73.737685941536. All144665x3 bestreplay/frozen exact, alltargets below retained controls. Finishednegativebest/lastdeleted 23.374MiB with SHAmanifest; source/config/history/preds retained. Nextstudent explores train-only internal-feature alignment against same architecture/control; noteacherpredictions or pseudolabels, noextra data.

Prepared v132 newbackbone train-onlyhiddenfeaturealignment againstcompletedv130control; frozenownsource teacher/no predictions/pseudolabels/inference, randomstudent,2phasesinclsource. No score yet.

V132 failed before anyoptimizerstep: fixedteacher256window cannot read512. Preserve failedsource/config, cleanupinitialstate; new v132r1 trial keepsstudent512, teachercentral256/alignedcommon128core, aux24832actualparams. No measuredgain/negative exists for failedv132; r1 explicitlaunch next, max2 withPORv131.

Explicitlaunch verified by actualcommand PID91284; snapshot {"status": "complete", "pid": 91284, "stage": "location_weight", "target": "POR", "target_score": 77.72901411274366, "initial_score": 77.54045432673898, "against_initial": 0.18855978600467438, "against_v97_target": 0.17707731085864964, "best_epoch": 27, "stopped_epoch": 45, "validation_rows": 144665, "predictions_reproduced_exactly": true, "frozen_parameters_exact": true, "gradient_groups": {"new_joint_location_weight_POR_head": 0.16152802109718323}, "seconds": 268.4852590000082, "peak_MiB": 922.1806640625, "complete_five_folds": false, "early_terminated": false}. Trainingonly, no finalgain;max2 acrossroutes. POR historical+0.25 and originalfullOOF>82 unmet, no otherfolds/delivery changes.

Explicitlaunch verified by actualcommand PID24968; snapshot {"status": "running", "pid": 24968, "stage": "aligned", "epoch": 7, "best": 77.89858094587699, "best_epoch": 6, "bad": 1, "peak_MiB": 1121.14794921875}. Trainingonly, no finalgain;max2 acrossroutes. POR historical+0.25 and originalfullOOF>82 unmet, no otherfolds/delivery changes.

V131 location_weight: restored POR=77.729014112744; delta vs v114=+0.180944617851, vs historical=+0.177077310859; actual fifth TOTAL=80.543755697579;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=27/45, seconds=268.485, peakMiB=922.181.

V131 positive final best27/stop45 POR77.729014112744, historical gain+0.177077310859, point-only incremental+0.167637937911; actualTOTAL80.543755697579, five recipephases. All144665 bestreplay/frozen exact. Positivebest/last retained; CURRENT_RECIPE updated, historical+0.25 and strictTOTAL80.8 stillfalse. Extra1040params distinguish capacity; no identicalcapacity claim. Realtraining268.485s/peak922.181MiB. New boundedconditionalgate study follows this measuredpositive fromsamefrozenv113, not trainedv131warmstart.

Prepared v133 conditionalloc/weight/gatePORhead (+65params) on samefrozenv113; comparepositivev131,5recipephases, no score yet.

Explicit actual process verified PID77064, mode location_weight_gate, runtime snapshot {"status": "running", "pid": 77064, "stage": "location_weight_gate", "epoch": 9, "best": 77.54045432673898, "best_epoch": 0, "bad": 9, "peak_MiB": 921.857421875}. Trainingonly, no finalgain;2sharedslots. CurrentverifiedPOR77.729014112744/TOTAL80.543755697579,5phases; strictPOR+0.25 andTOTAL80.8/fullOOF82 unmet. No otherfolds or delivery changes.

Explicit actual process verified PID24968, mode aligned, runtime snapshot {"status": "running", "pid": 24968, "stage": "aligned", "epoch": 22, "best": 78.90513703645937, "best_epoch": 18, "bad": 4, "peak_MiB": 1121.14794921875}. Trainingonly, no finalgain;2sharedslots. CurrentverifiedPOR77.729014112744/TOTAL80.543755697579,5phases; strictPOR+0.25 andTOTAL80.8/fullOOF82 unmet. No otherfolds or delivery changes.

Prepared v134 width128 extension of positivev131 conditionaldensityhead (82720 vs33184trainableparams), sameinitialfrozenv113function. No launch/score yet; no trainedv131weightsloaded,5recipephases.

V133 location_weight_gate: restored POR=77.540454326739; delta vs v114=-0.007615168154, vs historical=-0.011482475146; actual fifth TOTAL=80.487187761777;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=0/18, seconds=144.001, peakMiB=921.857.

V133 closed defaultgate extension: POR77.540454326739 best0/stop18, incrementalvspositivev131-0.188559786005; actualTOTAL80.487187761777,5phases. All144665replay/frozen exact; no newgain. Negativebest/last cleaned18.919MiB with SHAmanifest, keep v131positive/current. Timing{"training_seconds": 144.00128050000058, "launch_to_report_wall_seconds": 149.278195, "peak_MiB": 921.857421875, "overlap_other_backbone": true, "warmup_data_in_wall": true, "serial_equal_work_not_measured": true}. NextPORv134 widens only successfullocation/logweight nonlinearhead; no moregatebound/LR sweep.

V132r1 joint source verified: actual original-fifth scores{'POR': 74.69914236435433, 'PERM': 89.11417805271635, 'SW': 73.25866038340264, 'TOTAL': 79.24023616194795},2 stages incl originalsourceQ; strict808=False; no remainingfolds. best/stop=37/55, seconds=718.312, peakMiB=1121.148;144665x3 replay exact.

V132r1 finalpositive alignment: originalfifthTOTAL79.240236161948 best37/stop55; gainvsnoaux {"POR": 0.9686305621260374, "PERM": 0.4627241474484407, "SW": -0.47902555813300296, "TOTAL": 0.2848836748982251}. Predictionarchitecture identical, extra24832trainingauxparams; not equalactivecapacity. All144665x3replay/frozenteacherexact. Keepbest/last aspositivecontrol despitebelowSOTA/gate, no otherfolds. Newv135 replacesinputstemwithcurve tokens whilekeepingalignment, untrained.

V134 wide128: restored POR=77.869090371913; delta vs v114=+0.321020877020, vs historical=+0.317153570028; actual fifth TOTAL=80.585778575329;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=34/52, seconds=363.200, peakMiB=922.284.

V134 VERIFIED POR milestone:77.869090371913 best34/stop52, historicalgain+0.317153570028 (>0.25) andwidth64gain+0.140076259170. Actualthree-channelTOTAL80.585778575329, five totaldependencyphases; strictTOTAL80.8 stillfalse. All144665 replay/frozen exact, keeppositivebest/last andcurrentrecipe. This meets userone-channel+0.25 milestone, NOT fullgoal; firstroute now independently targets PERM readout, fixedPORv134/SWv104. OriginalfullOOF>82 unmet, no otherfolds/delivery edits.

PORhistorical+0.25milestone verified:v13477.869090 (+0.317154). Firstroute now independentPERMconditionaldensityv136 on frozenoriginalsource; fixedpositivePOR/SW, finalrecipe5phases. Secondroute curve-tokennewbackbonev135 retainingpositivealignment. Max2jobs, strictTOTAL80.8/OOF82 goalsstillunmet; no otherfolds/deliveryedits.

Live training snapshot 2026-10-01T13:22:11: v135 backbone; {"status": "running", "pid": 61172, "stage": "curve_tokens", "epoch": 33, "best": 79.14776547995906, "best_epoch": 30, "bad": 3, "peak_MiB": 3199.2373046875}; process command line verified. This supersedes pending-launch text. Partial fifth-fold score only; no final gain claim, no remaining folds.

Live training snapshot 2026-10-01T13:22:11: v136 PERM; {"status": "running", "pid": 57644, "stage": "perm_conditional", "epoch": 39, "best": 89.38793609578403, "best_epoch": 38, "bad": 1, "peak_MiB": 786.46533203125}; process command line verified. This supersedes pending-launch text. Partial fifth-fold score only; no final gain claim, no remaining folds.

V137 prepared/untrained: new random joint convolutional pyramid64/32/16 replaces temporal attention, positive train-only alignment retained. No launch while two current jobs live; no score/gain yet, two phases including source.

V136 final PERM=89.387936095784, compactcontrolgain=+0.009541884640; actualTOTAL=80.589118234954, fivephases inclsource;strict808=False, no otherfolds. best/stop=38/56, seconds=531.053, peakMiB=786.465; all144665replay/frozen exact.

V136 final positive retained: PERM89.387936095784, compact gain+0.009541884640; actual TOTAL80.5891182349535; restored-best replay/frozen integrity verified. Five phases including source; no other folds. Next v138 width256 replaces PERM phase from same original source, no v136 warm-start.

V138 explicitly launched, actual PID62004 command verified; snapshot {"status": "running", "pid": 62004, "stage": "wide256", "epoch": 7, "best": 89.3447601747529, "best_epoch": 0, "bad": 7, "peak_MiB": 786.46533203125}. Training partial, no gain yet. This supersedes prepared-only text. Positive v136 control retained; no warm-start, final recipe still5 phases. Parallel with v135; max2GPU.

V135 joint source verified: actual original-fifth scores{'POR': 74.28300589439002, 'PERM': 89.14599947824355, 'SW': 73.31932541216233, 'TOTAL': 79.14776547995906},2 stages incl originalsourceQ; strict808=False; no remainingfolds. best/stop=30/48, seconds=1161.755, peakMiB=3199.237;144665x3 replay exact.

V138 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.574006662593, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=151.874, peakMiB=786.465; all144665replay/frozen exact.

Final v135 vs aligned control: {"POR": -0.4161364699643144, "PERM": 0.031821425527198244, "SW": 0.06066502875968638, "TOTAL": -0.09247068198888542}. Overall TOTAL loss and higher compute close this joint architecture; small PERM/SW positive comparison evidence retained in best checkpoint, no blanket rejection of curve encoding. No width/depth/LR extension.
Timing {"training_seconds": 1161.7552319999959, "launch_to_report_wall_seconds": 1166.910198, "peak_MiB": 3199.2373046875, "overlap_other_route": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}; removedMiB=31.545778274536133. No remaining folds, originalOOF>82 not achieved.

Final v138 PERM89.344760174753 best0/stop18, versus width128 v136 -0.043175921031 and compactcontrol -0.033634036391. No gain; close width expansion, retain v136 positive model. Negative best/last removed with SHA manifest after actual assembly/replay.
Timing {"training_seconds": 151.87421049998375, "launch_to_report_wall_seconds": 157.391062, "peak_MiB": 786.46533203125, "overlap_other_route": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}; removedMiB=22.19795799255371. No remaining folds, originalOOF>82 not achieved.

V137 explicitly launched after v135/v138 terminal handles absent; live PID26384 command verified. Initial training snapshot {"status": "running", "pid": 26384, "stage": "pyramid", "epoch": 2, "best": 77.10771502998041, "best_epoch": 2, "bad": 0, "peak_MiB": 1120.16455078125}. Supersedes prepared-only state. Architecture entirely convolutional temporal context, 1107499 trainable including auxiliary projections, no old student loaded; first epoch11.161s, no OOM. Initial/partial learning not mechanism gain. Original fifth TOTAL>80.8+gain and fullOOF>82 unmet; no other folds. Channel route retained v136 positive after v138 negative, current actual recipeTOTAL80.589118234954.

V139 prepared: SW Student3 proper likelihood/utility/CDF on same random local architecture, no old SW weights; replace SW phase, five total. Fixed positivePORv134/PERMv136; no gain yet.

Live process and command verified, snapshot {"status": "running", "pid": 69512, "stage": "student3", "epoch": 1, "best": 71.19711508147167, "best_epoch": 0, "bad": 1, "peak_MiB": 1794.25732421875}. Both v137/v139 run in normal GPU, maximum2, no OOM. Supersedes prepared-only text. Partial scores, no final gain claim; current verified compact recipe remainsTOTAL80.589118234954/five phases. No remaining folds or delivery changes.

Live process and command verified, snapshot {"status": "running", "pid": 26384, "stage": "pyramid", "epoch": 15, "best": 78.58002190630785, "best_epoch": 14, "bad": 1, "peak_MiB": 1120.16455078125}. Both v137/v139 run in normal GPU, maximum2, no OOM. Supersedes prepared-only text. Partial scores, no final gain claim; current verified compact recipe remainsTOTAL80.589118234954/five phases. No remaining folds or delivery changes.

V140 prepared/untrained, no GPU launch: POR width256 from same frozen v113 parent, test measured width64->128 positive route, no trained v134 head warm-start. Replaces PORhead phase, fixedPERM136/SW104, final five phases. Wait channel-line v139 terminal before explicit launch.

Concurrency timing snapshot {"snapshot_only": true, "backbone_solo_median_epoch_seconds": 10.667198800001643, "backbone_overlap_median_epoch_seconds": 22.298820449999766, "SW_overlap_median_epoch_seconds": 23.39720274999854, "peak_backbone_MiB": 1120.16455078125, "peak_SW_MiB": 1794.25732421875, "equal_work_total_wall_not_yet_measured": true, "warmup_first_epoch_excluded_from_solo": true}. GPUsharing slows each job; do not claim twofold speedup or infer total equivalent-work wall from one epoch. Both current jobs remain live, no interruption for benchmarking.

Verified wait 2026-10-01T13:38:12: actual PID/command confirmed live; {"status": "running", "pid": 26384, "stage": "pyramid", "epoch": 31, "best": 79.15430096458167, "best_epoch": 27, "bad": 4, "peak_MiB": 1120.16455078125}. No restart or candidate launch, partial fifth snapshot only. Still below final reference; no confirmed gain/no otherfolds. Latest verified compact recipeTOTAL80.589118234954.

Verified wait 2026-10-01T13:38:12: actual PID/command confirmed live; {"status": "running", "pid": 69512, "stage": "student3", "epoch": 17, "best": 73.51140270155435, "best_epoch": 15, "bad": 2, "peak_MiB": 1794.25732421875}. No restart or candidate launch, partial fifth snapshot only. Still below final reference; no confirmed gain/no otherfolds. Latest verified compact recipeTOTAL80.589118234954.

V141 prepared/untrained, no launch: positive aligned random backbone with three private nonlinear predictive branches instead of linear75; allstudentrandom, original teacher training-hidden only. Wait v137 terminal, max2GPU,2source-counted phases, no gain/folds yet.

Latest user direction: prioritize analysis and further expansion of proven POR head gains; transfer to all output heads only after validated POR evidence. v139 SW early-stop requested at epoch24 boundary (below control0.546008), no forcedkill. v140 width256 from same v113 is ready, launch only after SW handle terminal. Current v137 backbone may finish unchanged; v141 remains prepared, no automatic launch under this priority. Original strictfifthTOTAL80.8+gain/fullOOF82/max2/five total phases remain.

V139 student3: restored SW=73.617842845787; Student3 minus Gaussian SW delta=-0.503909982371; actual fifth TOTAL=80.412749741124;5 stages including source and2 POR phases; fixed PERM v136; strict808=False; no other folds. best/stop=25/26, seconds=612.391, peakMiB=1794.257.

V139 early stopped at26/best25: SW73.617842845787, retainedGaussian delta-0.5039099823705584; actualTOTAL80.41274974112382. Restore/replay/frozen exact; weak partial-bounded attempt, not proof of convergence or universal Student3 failure. Negative best/last cleanedMiB=18.204355239868164. {"training_seconds": 612.3913287000032, "launch_to_report_wall_seconds": 617.014697, "peak_MiB": 1794.25732421875, "overlap_other_backbone": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. POR priority replaces channel job, no resumption/folds.

Requested gain analysis completed: [GAIN_ANALYSIS.md](GAIN_ANALYSIS.md), exact frozen fifth144665 predictions, no fit or selection. Point vs local no context gain; logweight+location at64 +0.167638, capacity128 +0.140076, each positive10/16 wells; whole vs v113 +0.328636, nondefault +1.394254/default -0.239516. Source defaultgate and widths frozen, gain conditional density/main continuous cases. Capacity/training differences limit causal claims. V140 explicit POR launch PID88516 after SW terminal, same frozen v113 andzerohead, no v134trainedweights; restorebest and compare against77.869090. Transfer later to each head requires own measured ablation, no automatic sixth phase.

V135 joint source verified: actual original-fifth scores{'POR': 74.83751384945637, 'PERM': 88.79109150916375, 'SW': 73.21761366153558, 'TOTAL': 79.15430096458167},2 stages incl originalsourceQ; strict808=False; no remainingfolds. best/stop=27/45, seconds=820.027, peakMiB=1120.165;144665x3 replay exact.

V142 prepared single same-capacity POR locationbound2 vs positive v134 bound1, width128/82720params, same random/raw initial function; no old trained head. Requested analysis motivates conditional density range test, fullPOR selection. Five phases incl source; transfer later only if measured beneficial.

Final v137 original-fifth TOTAL79.154300964582 best27/stop45, control deltas{"POR": 0.13837148510204145, "PERM": -0.3230865435526056, "SW": -0.04104672186706182, "TOTAL": -0.08593519736628252}. Overall below alignedcontrol, close this joint configuration; preservebest for scoped positivePOR comparison, redundantlast cleanedMiB=32.72123336791992. {"training_seconds": 820.0271517000219, "launch_to_report_wall_seconds": 824.212579, "peak_MiB": 1120.16455078125, "overlap_SW_and_POR": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. Exact144665x3/frozen replay, no otherfolds. Latest user prioritizes POR mechanism amplification, v141 stays prepared/unlaunched.

Requested POR gain analysis saved in results/deep_learning/por_joint_density_width_v140_20261001/GAIN_ANALYSIS_ZH.md. Both POR studies actually live, snapshot {"status": "running", "pid": 88516, "stage": "wide256", "epoch": 44, "best": 77.9121028885816, "best_epoch": 30, "bad": 14, "peak_MiB": 924.5458984375}. Partial results only; first amplify measured POR mechanism then assess transfer to PERM/SW. Five phases, no extra folds/delivery changes.

Requested POR gain analysis saved in results/deep_learning/por_joint_density_width_v140_20261001/GAIN_ANALYSIS_ZH.md. Both POR studies actually live, snapshot {"status": "running", "pid": 4352, "stage": "locbound2", "epoch": 19, "best": 77.59738834861389, "best_epoch": 17, "bad": 2, "peak_MiB": 922.2841796875}. Partial results only; first amplify measured POR mechanism then assess transfer to PERM/SW. Five phases, no extra folds/delivery changes.

V140 wide256: restored POR=77.912102888582; delta vs v114=+0.364033393689, vs historical=+0.360166086697; actual fifth TOTAL=80.602021989954;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=350.769, peakMiB=924.546.

V140 FINAL confirmed POR77.912102888582, width256 minus128 +0.043012516668; historical+0.360166086697. ActualTOTAL80.602021989954, five source-counted phases. best30/stop48,350.769seconds/924.546MiB;144665 exactreplay/frozen. Keep positivebest/last and new CURRENT_RECIPE; strict80.8 false, no extra folds. Requested amplification verified; next transfer SW conditional readout from original source, no v104 trained weights.

V143 SWconditionalhead transfer prepared after positive PORcapacity final; from originalsource only, no SW104trainednetwork, ownGaussianNLL,128head. Final recipe5phases, actualfixedPOR140/PERM136assembly; no gain yet.

por_joint_density_width_v144 prepared underlatest explicituserrequest; Width512 vs verified256 POR77.912102888582; same headmechanism/loss/bounds, new512stem/GLU1024/32outputs, 724000active parameters. No old trained256head loaded, all newhidden random andzerooutputs. Capacity change, not sameinitial random matrices/trajectory. Replace PORphase, originalfive-phaserecipe, no score yet.

por_residual_distribution_v145 prepared underlatest explicituserrequest; Residual distribution correction on256head: output48 instead of32, correct16locations+16logweights+16scale raw parameters. Added4112active params vs control, not samecapacity. New scale raw delta0.5*tanh(u/0.5), positive component referencewidth still softplus(raw)+0.025; this is a bounded softplus-parameter residual, not exact multiplicative log-SD. Originalparent rawpluszero retains exactinitialfunction. Proper mixedStudent3/Cauchy NLL andCDF both use corrected width; Cauchy wholemixture still has nofinite mean/variance. Locationdelta bound uses frozen original referencewidth, gate/familyassignments/oldpredictor frozen. No extra mixture components/prediction votes. Replace PORphase, originalfive-phaserecipe, no score yet.

V142 locbound2: restored POR=77.831487046825; delta vs v114=+0.283417551932, vs historical=+0.279550244940; actual fifth TOTAL=80.577837237427;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=34/52, seconds=379.708, peakMiB=922.284.

V143 final SW=73.622384767129, sourcegain=+0.000000000000, retainedGaussian delta=-0.499368061029; actualTOTAL=80.427243168594, five phases incl source;strict808=False, no otherfolds. best/stop=0/18, seconds=115.324, peakMiB=786.465; all144665replay/frozen exact.

Closed locbound2: Locationbound2 loses to samecapacity bound1 by0.037603, close this bound extension. targetscore=77.8314870468251, best/stop=34/52, actualTOTAL=80.57783723742709. Exact144665/frozen replay; negative best/last cleanedMiB=19.861730575561523. Timing {"training_seconds": 379.7081502999936, "launch_to_report_wall_seconds": 385.394452, "peak_MiB": 922.2841796875, "overlap_other_POR": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. No broad family/head failureclaim, no extra folds.

Closed sw_conditional: SW head transfer best0, no gain versus frozen source; retain original SW104. targetscore=73.62238476712884, best/stop=0/18, actualTOTAL=80.42724316859398. Exact144665/frozen replay; negative best/last cleanedMiB=19.66194725036621. Timing {"training_seconds": 115.3244775000203, "launch_to_report_wall_seconds": 120.422902, "peak_MiB": 786.46533203125, "overlap_other_POR": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. No broad family/head failureclaim, no extra folds.

Explicit userrequested job launch verified by live PID/command; snapshot {"status": "running", "pid": 66720, "stage": "wide512", "epoch": 22, "best": 77.85456615085442, "best_epoch": 17, "bad": 5, "peak_MiB": 930.58349609375}. Supersedes prepared/untrained status. Current control POR25677.912102888582/actualTOTAL80.602021989954, new partial scores not finalgain. Max2GPU,80-18, five total phases; no old trained256head loaded, no folds/deliveryoverwrite.

Explicit userrequested job launch verified by live PID/command; snapshot {"status": "running", "pid": 82756, "stage": "residual256", "epoch": 4, "best": 77.54045432673898, "best_epoch": 0, "bad": 4, "peak_MiB": 923.87060546875}. Supersedes prepared/untrained status. Current control POR25677.912102888582/actualTOTAL80.602021989954, new partial scores not finalgain. Max2GPU,80-18, five total phases; no old trained256head loaded, no folds/deliveryoverwrite.

V144 wide512: restored POR=77.854566150854; delta vs v114=+0.306496655961, vs historical=+0.302629348969; actual fifth TOTAL=80.584760968636;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=17/35, seconds=257.146, peakMiB=930.583.

Latest explicituser: stop512, test192.512 was already naturallycomplete35/best17 POR77.854566, no resume. v146192 prepared same frozenv113/loss/LR/budget; no256/512trainedhead loaded, replacephase/five total. Residualv145 continuesunchanged, no more512 launches.

V144512closed after natural35/best17 POR77.854566150854, delta256=-0.057536737727176046,actualTOTAL=80.58476096863589. Userexplicitstop, no resumption. Negativebest/last cleanedMiB=32.09226036071777, exactSHAmanifest/144665bestreplay/frozen evidence retained. {"training_seconds": 257.1462739999988, "launch_to_report_wall_seconds": 261.558137, "peak_MiB": 930.58349609375, "overlap_SW_and_residual_POR": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. New192studyv146 replaceschanneljob, max2withresidual, no folds/delivery changes.

V146 width192 explicitly launched, live PID79160 commandverified; snapshot {"status": "running", "pid": 79160, "stage": "wide192", "epoch": 3, "best": 77.54045432673898, "best_epoch": 0, "bad": 3, "peak_MiB": 922.392578125}. Supersedespreparedstatus, no finalgainyet. 512finished35 andcleanednegativeweights, notresumed. Current256verifiedPOR77.912103/TOTAL80.602022 retained. Max2withresidualv145, originalfivephases/noextra folds/delivery changes.

Latest explicituser requests384PORhead. v147 prepared samefrozenv113/loss/budget,444704activeparams, no256/512headwarmstart. Residualv145 alreadycomplete35/POR77.676996 below256; new384usesfreeGPUslot beside192. Five total phases, no more512/no otherfolds.

V145 residual256: restored POR=77.676996071434; delta vs v114=+0.128926576541, vs historical=+0.125059269549; actual fifth TOTAL=80.531489944810;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=312.207, peakMiB=923.871.

V145 residualdistribution finalPOR77.676996071434 vs256delta=-0.23510681714769532, actualTOTAL=80.53148994480973,best17/stop35. This exact boundedrawscale extension negative, no blanketrejection ofallresidualdensity methods. Exact144665/frozenreplay, negativeweights cleanedMiB=22.765783309936523. {"training_seconds": 312.2068451999803, "launch_to_report_wall_seconds": 317.02129, "peak_MiB": 923.87060546875, "overlap_POR512_and192": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. Source/config/results retained, no folds. User384studyv147 takesreleasedslot.

V147384explicitlaunch verified actuallivePID31088/command, snapshot {"status": "running", "pid": 31088, "stage": "wide384", "epoch": 6, "best": 77.54045432673898, "best_epoch": 0, "bad": 6, "peak_MiB": 927.78857421875}. Supersedespreparedonlystatus. Control256POR77.912103/TOTAL80.602022 remainsverifiedbest; partial384score no finalgain. Originalfive phases/max2/80-18 preserved, source frozen, no folds/deliveryoverwrite.

V146 wide192: restored POR=77.786804168476; delta vs v114=+0.238734673583, vs historical=+0.234867366591; actual fifth TOTAL=80.564432373922;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=34/52, seconds=405.644, peakMiB=922.393.

V146192final POR77.786804168476 vs256delta=-0.12529872010566123,actualTOTAL=80.56443237392234,best34/stop52. Exact144665/frozenreplay, no capacitygain; negativeweights cleanedMiB=21.119054794311523. {"training_seconds": 405.6442259999749, "launch_to_report_wall_seconds": 415.31758, "peak_MiB": 922.392578125, "overlap_residual_and_POR384": true, "loading_warmup_in_wall": true, "serial_equal_work_not_measured": true}. Source/config/result evidence retained; no more192/default automatic sweep.384remainsactive, current256bestrecipe unchanged, no folds/deliveryoverwrite.

V147 wide384: restored POR=77.791188185864; delta vs v114=+0.243118690971, vs historical=+0.239251383979; actual fifth TOTAL=80.565747579139;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=22/40, seconds=290.900, peakMiB=927.789.

V148 prepared/untrained: properresiduallogdensity correction trainingNLL andCDFsame normalization; fixed original256intervals/actions, no old256headweights loaded. Extend userrequestedresidualmechanism after negativewidth/scale variants; five total phases, no score yet.

Final capacity study: width384 POR77.791188185864, delta256 -0.120914702717; best22/stop40, 290.900 training seconds, peak927.789MiB. Exact144665 replay and actual assembly TOTAL80.565747579139 complete. Negative best/last removed with SHA manifest. Width64/128/192/256/384/512 all complete;256 remains best77.912103. No width extension or remaining folds.

Verified launch PID74320: proper residual density readout running, snapshot epoch26, partial POR best77.769460078679; not final gain vs256. Initial full-fifth POR77.540454326739 identical to frozen source. Focus distribution readout per latest user direction; no further width sweeps/private backbone launch. Five total source-counted phases; no remaining folds.

V148 tilt: restored POR=77.889012260011; delta vs v114=+0.340942765118, vs historical=+0.337075458126; actual fifth TOTAL=80.595094801383;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=362.568, peakMiB=1312.311.

Final v148: POR77.889012260011, delta vs predefined256 control -0.023090628570; actual TOTAL80.595094801383. Best30/stop48, exact144665/frozen replay. 362.568 training seconds, peak1312.311MiB, single GPU; loading/warmup included in launch-to-report wall (timing.json). Proper interval tilt did not exceed best readout; close this configuration, no further tilt-bound/LR sweep. Negative checkpoints removed with SHA manifest. Best recipe remains v140 TOTAL80.602021989954/POR77.912102888582. Next hypothesis is structured component-shared distribution readout, not yet implemented/launched; READOUT_DIRECTION_ZH.md records evidence. No otherfolds, five phases, fullOOF>82 incomplete.

V149 distribution readout: [study](results/deep_learning/por_component_readout_v149_20261001/README.md). Two predefined matched shared/communicate component heads, same frozenv113 source; five total phases. Prepared/untrained, no score; focusPOR readout, otherbackbone remains held, no otherfolds.

Verified explicit dual launch: shared PID45244 and communicate PID22848, both command lines and actual training epochs confirmed live. Both initial POR77.540454326739 and290610 trainableparameters. Snapshot shared epoch6, communicate epoch4, no improvement yet; peak956.931MiB each, noOOM. Shared epochs ~6.4s and communication ~8.9s under overlap; no completed equal-work wall comparison yet. Final gain/actual assembly pending; do not promote.

V150 prepared/untrained: [smooth conditionalCDF readout](results/deep_learning/por_smooth_cdf_readout_v150_20261001/README.md),8Beta basis weights, properPDF derivative andCDF synchronized. No source/trainedhead mutation, fivephases; held until a realGPUslot opens. Currentv149twojobs active, no gain established.

V149 shared: restored POR=77.672952461232; delta vs v114=+0.124882966339, vs historical=+0.121015659347; actual fifth TOTAL=80.530276861749;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=320.591, peakMiB=956.931.

V149 communicate: restored POR=77.645461857202; delta vs v114=+0.097392362309, vs historical=+0.093525055317; actual fifth TOTAL=80.522029680540;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=322.662, peakMiB=956.931.

V149 closed: shared POR77.672952461232/TOTAL80.530276861749; communicate POR77.645461857202/TOTAL80.522029680540. Matched communication delta-0.027490604030; no gain. Both best17/stop35, exact144665 replay/frozen integrity, actual three-target assembly. Negative weights removed with SHAmanifest. Timing/concurrent total wall in timing.json; no serial equal-work benchmark. No further component-attention sweep; v150smoothCDF explicitly launched PID68944 after both terminal, one GPU. Best remains v140POR77.912103/TOTAL80.602022; strict808/fullOOF82 unmet, no otherfolds.

Explicit launch PID68944 verified command line and actual epoch2 training after v149 both complete. Initial original-fifth POR77.540454326739 identical to frozen source;233000activeparameters; peak924.940MiB, oneGPU, noOOM or stderr. Initial progress is not gain; final best restoration/assembly pending. No remainingfolds or new automatic candidates.

Latest user: if new ideas lack gain, augment mechanisms already proven useful. V151 [jointRPS study](results/deep_learning/por_ranked_probability_v151_20261001/README.md) keeps effective256location/logweight structure and capacity, adds true default/continuous CDF supervision only, fromsameparent/fivephases. No negative module stacking or extra sixthphase. Prepared/untrained; v150 continues unchanged, max2.

V150 bernstein8: restored POR=77.769405183474; delta vs v114=+0.221335688581, vs historical=+0.217468381589; actual fifth TOTAL=80.559212678422;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=17/35, seconds=248.118, peakMiB=924.940.

V150 final POR77.769405183474/TOTAL80.559212678422; delta vs effective256 -0.142697705108; best17/stop35. Exact144665/frozen replay, actual three-target assembly. Training seconds248.118, peakMiB924.940; launch/report wall and overlap in timing.json. Negative weights removed with SHAmanifest. Close thisCDF shape config; latestuser directs augmentation of proven gains, v151same230944parameter256readout plus jointRPS launched, source-counted5phases. No otherfolds; bestv140 unchanged.

Verified launch PID29608, actual training observed, snapshot epoch8, partialbest77.540454326739, peakMiB924.547. Initial POR77.540454326739,230944parameters, exact effective256architecture/initialization retained; only addsjointRPS. No newgain established. V150 terminal now, onlyone training process remains.

V152 [effective256 plus matchedexpertinteraction](results/deep_learning/por_expert_interaction_v152_20261001/README.md), additive/product lowrank32, commoninitialweights retained/newbranchzero,247712parameters. Prepared/untrained, no score; no negative module stacked, fivephases/max2, v151independent continues.

Verified additive launch PID61136: commandline and actual epoch5 confirmed live. Initial POR77.540454326739,247712parameters; peak955.997MiB, noOOM/stderr. Product still prepared/untrained, no autoqueue. V151 PID29608 remains live alongside; max2 maintained. Partial results not gain.

V151 live snapshot epoch25, partialbestPOR77.820742086193, bestepoch17; not final gain vs25677.912103. Additivev152 now overlaps, no more than2jobs. No otherfolds or currentrecipe change.

Verified live snapshot epoch35, bestPOR77.926380760724 at30, partial delta vs samecapacity v140+0.014277872142. This is a training snapshot, not final replay/assembly or fullOOF gain. Preserve all active states; final restoration pending. Productv152 still not launched while twoGPUjobs occupied.

V151 joint_rps: restored POR=77.926380760724; delta vs v114=+0.378311265831, vs historical=+0.374443958839; actual fifth TOTAL=80.606305351597;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=624.994, peakMiB=924.547.

V151 final positive adopted in NEW CURRENT_RECIPE.json: restored POR77.926380760724, delta samecapacity v140+0.014277872142; actual TOTAL80.606305351597 (+0.004283361643). Default delta+0.008478438272, numeric+0.025155211367, positive10/16wells; descriptive frozen analysis only, no selection byrow/well. Best30/stop48, exact144665/frozen replay; training624.994seconds, peak924.547MiB, overlap/wall in timing.json. Preserve positivebest/last and previouscontrol/delivery. Five phases, strict808 false/nootherfolds, completeOOF82notachieved. Next bounded RPScoefficient expansion may be prepared only fromsamefrozenparent replacingheadphase, no negative module stacking.

V152 additive: restored POR=77.857372986694; delta vs v114=+0.309303491801, vs historical=+0.305436184809; actual fifth TOTAL=80.585603019388;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=469.133, peakMiB=955.997.

Product explicitly launched PID62976 after v151 terminal, initial POR77.540454326739/247712parameters; live actual epoch11, best77.540454326739, noOOM/stderr. Additive naturallycompleted48/best30, restored POR77.857372986694/TOTAL80.585603019388, delta vs direct256 -0.054729901887. Exact144665/frozen replay. Keep additive control weights until paired product comparison finalized; no restart. OneGPUjob now live; product final/paired gain pending. V151positive recipe80.606305 retained, no otherfolds.

V153 [expand positive jointRPS coefficient1to2](results/deep_learning/por_ranked_probability_v153_20261001/README.md), same256readout/230944parameters, one bounded candidate fromsamev113, no extra stage/no negative modules. Prepared/untrained, positivev15177.926381 preserved; v152product active, max2.

Verified explicit launch PID43724, actual epoch4 and commandline live; initial POR77.540454326739/230944parameters, same asRPS1 control. Current partialbest77.540454326739, peak924.547MiB, noOOM/stderr. Only coefficient2 changed; original positivev151 checkpoint andrecipe untouched. Two GPUjobs with productv152; final gain/actualassembly pending, no otherfolds.

V152 product: restored POR=77.848494843975; delta vs v114=+0.300425349082, vs historical=+0.296558042090; actual fifth TOTAL=80.582939576572;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=412.136, peakMiB=955.997.

V152 final paired negative: additivePOR77.857372986694/productPOR77.848494843975; product minus additive-0.008878142719. Both below effective256/RPSpositive; exact144665/frozen replay and actual assemblies complete. Bothbest30/stop48. Four negativeweights deleted with SHAmanifest after pairfinal; all compact evidence/predictions kept. Timing and combined staggered wall in timing.json. Close rank32 add/product configuration, no negative stack into positive recipe; v153coeff2 continues alone. OriginalfifthTOTALbest80.606305, strict808/fullOOF82 unmet/nootherfolds.

Verified live PID43724 snapshot epoch24, partialPORbest77.817462477236; no finalgain vs RPS1 control77.926381. V152paired negative finalized/cleaned, now oneGPUjob only. No more candidates launched this batch; current positive5phase recipe80.606305 retained, no otherfolds.

V154 [PERM effective128 plus evidence-positive jointRPS](results/deep_learning/perm_ranked_probability_v154_20261001/README.md), same82720parameters/initialization asv136, Gaussian/log10 ownPERM labels. Prepared/untrained, no score; fivephases/fixedPOR151/SW104, max2 besidePOR153. No negative stack or otherfolds.

Verified explicit launch PID79880 and actual epoch2; source initial PERM89.344760174753,82720parameters, matched effective128head initialization. Partialbest89.344760174753, peak786.465MiB, noOOM/stderr. Two GPUjobs withPOR153; finalgain/replay/actual fixedPOR151+SW104 assembly pending. No otherfolds or more candidates.

V153 joint_rps: restored POR=77.901655016833; delta vs v114=+0.353585521940, vs historical=+0.349718214948; actual fifth TOTAL=80.598887628429;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/48, seconds=536.930, peakMiB=924.547.

V155 [effectiveSWlocal+jointRPS1](results/deep_learning/sw_ranked_probability_v155_20261001/README.md), same845913parameters/randominitial/utility1/SAM0.1/Gaussian; prepared/untrained, no score. Fivephases/fixedPOR151PERM136, no oldSWheadwarmstart; POR153 now terminal so explicitlaunch allowed besidePERM154(max2).

V153 final negative: coefficient2POR77.901655016833/TOTAL80.598887628429; deltaRPS1 -0.024725743891/NLL -0.010447871749. Best30/stop48, exact144665/frozen replay/actual assembly; training536.930s/peak924.547MiB, wall/overlap timing.json. Negativeweights deleted with SHAmanifest; keep positiveRPS1/newrecipe80.606305 and oldcontrols. Close coefficient2, no new coefficient grid. SW155explicitlaunched afterterminal alongsidePERM154(max2); no otherfolds/fullOOF82unmet.

Explicitlaunch PID53596 verified actual epoch2 and commandline. InitialSW71.197115081472/845913parameters same randomlocal104 start, partialbest71.197115081472, peak1728.449MiB, noOOM. First epoch32.045s with PERM154 overlap, not standalone equal-work speed evidence. Existing CUDAwarn_only attention/interpolation warnings observed; no bitwise training reproducibility claim. Initiallearning is not gain vs SW10474.121753; final replay/assembly pending. Onlytwo jobs, no otherfolds.

V154 final PERM=89.405564856919, compactcontrolgain=+0.027170645775; actualTOTAL=80.612475417994, fivephases inclsource;strict808=False, no otherfolds. best/stop=31/49, seconds=561.870, peakMiB=786.465; all144665replay/frozen exact.

V154 final positive adopted in NEW CURRENT_RECIPE.json: PERM89.405564856919, delta samecapacityv136+0.017628761135; actual TOTAL80.612475417994 (+0.006170066397). Best31/stop49, exact144665/frozen replay, training561.870s/peak786.465MiB, wall/overlap timing.json. Defaultdelta-0.001962329657/numeric+0.055721972822, positive7/16wells: modest singlefold gain, not universal/default improvement. Preserve positivebest/last and oldcontrols. Source-counted5phases, strict808false/nootherfolds, completeOOF82unmet. ActiveSW155 original plannedPERM136 assembly remains unchanged; any latestPOR151/PERM154/SW155 assembly must be separately evaluated afterSWfinal.

SWverifiedlive PID53596 snapshot epoch13, partialbest73.396265624043; initiallearning not gainvsSW10474.121753. PERM154positive finished, now oneGPUprocess; original fixedPERM136 assembly plan unchanged, no source/config mutation. Current combined positivePOR151/PERM154/SW10480.612475 recipe recorded separately.

V156 [PERMpositive jointRPS coefficient1to2](results/deep_learning/perm_ranked_probability_v156_20261001/README.md), same82720parameters/initialization, onlylosscoefficient changes; prepared/untrained. PositivePERM15489.405565 kept. SW155active, explicitnewlaunchmax2/fivephases, no otherfolds.

V155 restored SW=73.670351522101, delta v104=-0.451401306057; actual TOTAL=80.448314894477, five phases/source included, strict808=False; no otherfolds. best/stop=20/21; exact144665/frozen replay.

Latest explicit user two lines: continuePOR on positivefoundation; other heads may use different mechanisms, MAIN PERM. SW155prioritystop atsavedepoch21/best20, exactsavedbestreplay73.670352; early/incomplete, not convergence conclusion, no resume. V157 [PORpositiveRPS plus smallgate](results/deep_learning/por_bounded_gate_rps_v157_20261001/README.md), prepared/untrained,231201parameters/+257, fivephases, max2 besidePERM156.

V155 priority-stop final (INCOMPLETE): saved completeepoch21/best20 restoredSW73.670351522101, below retainedSW104by0.451401306057. User latest focusesPORaugmentation/MAINPERM, not full-budget convergence negative. Exact144665/frozen replay/actual original-planned assembly verified. Best/last cleanedwith SHAmanifest, no auto-resume. Timing records stop/replay/overlap; frozen copiedv104 trainingentry lacked TrainingAwake call, finalizer used it; no retrospective source mutation. POR157 explicitnewlaunch and PERM156 now2jobs; latest positive5phase recipe80.612475 untouched.

V156 CUDAillegalmemory runtimefailure after savedepoch8, no final gain/convergence claim; source/config/error/last preserved, processmissing confirmed, no automaticrestart. V158 [PERMLaplace independentreadout](results/deep_learning/perm_laplace_rps_v158_20261001/README.md), same82720rawparams but differentinitialfunction; prepared/untrained. Latestuser MAINPERM plusPORpositiveaugmentation; explicitlaunchmax2 besidePOR157, source-counted5phases/nootherfolds.


## POR澧炶ˉ涓嶱ERM涓绘敾锛氬惎鍔ㄦ牳楠?

POR157灏忛棬鎺?RPS鍜孭ERM158 Laplace+RPS鍧囨甯歌缁冿紝鏈€澶氫袱GPU浠诲姟锛涙殏鏈秴杩囧悇鑷鎺у埗銆侾ERM156涓篊UDA闈炴硶鍐呭瓨璁块棶杩愯澶辫触锛屼繚鐣欑幇鍦猴紝涓嶈嚜鍔ㄦ仮澶嶏紱SW155鍥犱紭鍏堢骇鍋滄锛屼笉浣滀负鏀舵暃璐熺粨鏋溿€傚綋鍓嶅凡楠岃瘉绗簲鎶樼粍鍚?0.612475417994锛屼粛鏈揪涓ユ牸80.8锛涘畬鏁碠OF>82鏈畬鎴愩€備笅涓€姝ュ畬鎴愮湡瀹炶缁冨強鏈€缁堟仮澶?鍚堝苟锛屼繚鎸佷簲闃舵锛屼笉琛ュ叾浠栨姌銆傛紨杩涘浘璋眂urate/build宸叉垚鍔熸洿鏂般€?


v159 [PERM灏哄害璇诲嚭](results/deep_learning/perm_scale_readout_v159_20261001/README.md)锛氭湁鐣宺aw-scale澧炶ˉGaussian/RPS姝ｈ矾绾匡紝澧炲姞2064鍙傛暟锛屼簲闃舵锛屽悓鍒濆鍑芥暟锛涘噯澶囨湭璁粌锛岀瓑寰呯┖浣嶏紝鏃犺嚜鍔ㄩ槦鍒椼€?

V158 final PERM=89.332375289863, compactcontrolgain=-0.046018921280; actualTOTAL=80.586859069525, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=251.808, peakMiB=786.465; all144665replay/frozen exact.


## 鏈€缁堢粨璁轰笌鍏抽棴

Laplace PERM89.332375289863锛孏aussian/RPS瀵圭収宸€尖垝0.073189567056锛涚0杞渶浣筹紝绗?8杞棭鍋滐紝鏃犲涔犳敼鍠勩€傚疄闄呬笁閫氶亾TOTAL80.58685906952451锛涚鐩樻渶浣抽噸杞藉拰鍐荤粨瀹屾暣鎬с€?44665琛屽疄闄呯粍鍚堝凡楠岃瘉銆傚叧闂湰閰嶇疆锛屼笉娉涘寲鎵€鏈塋aplace鏈哄埗鏃犳晥銆傝礋鏉冮噸宸叉寜SHA/璺緞娓呭崟鍒犻櫎锛屾潵婧?閰嶇疆/棰勬祴/鏃ュ織/缁撴灉淇濈暀銆傝缁?51.808绉掞紝涓嶱OR157骞跺彂锛屽崟椤规椂闂翠笉鑳戒綔涓茶閫熷害缁撹銆?


## 宸叉牳楠屽疄闄呭惎鍔?

PID44704锛宯ormal GPU锛屼复鏃禨YSTEM_REQUIRED宸插惎鐢紱涓嶱OR157鍙屽苟鍙戙€傚垵濮婸ERM89.344760174753涓嶨aussian鍘熷嚱鏁颁竴鑷达紝84784鍙缁冨弬鏁帮紱绗?杞揩鐓у皻鏃犳敹鐩婏紝宄板€?86.465MiB锛屾棤stderr閿欒銆傚緟鏈€缁堟仮澶?瀹為檯鍚堝苟楠岃瘉锛涙紨杩涘浘璋卞凡鏇存柊銆?


v160 [POR瀹瑰樊鏁堢敤澧炶ˉ](results/deep_learning/por_rps_utility_v160_20261001/README.md)锛氬悓230944瀹归噺/RPS1锛屽鍔犺缁冭嚜韬爣绛炬晥鐢ㄤ唬鐞?.1锛涘師瑙ｇ爜涓嶆敼锛屼簲闃舵銆傚噯澶囨湭璁粌銆佹棤鍒嗘暟锛岀瓑寰匬OR157鑷劧缁撴潫銆?

V157 joint_rps: restored POR=77.763278807566; delta vs v114=+0.215209312673, vs historical=+0.211342005681; actual fifth TOTAL=80.563544832047;5 phases incl source; channel gain>0.25=False; strict808=False; no otherfolds. best/stop=30/48, seconds=599.604, peakMiB=924.223.

V159 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/18, seconds=178.450, peakMiB=786.465; all144665replay/frozen exact.


## 鏈€缁堟敹鐩婂拰鍏抽棴

涓庢RPS鎺у埗宸€?0.163101953158锛屾渶浣?鍋滄30/48锛涘疄闄匱OTAL80.563544832047锛?44665琛屾仮澶嶅拰鍐荤粨瀹屾暣鎬у凡楠岃瘉銆傛棤鏂板鏀剁泭锛屼笉缁х画鎵弿鏈満鍒讹紱璐熸潈閲嶆寜SHA娓呭崟鍒犻櫎锛岃瘉鎹繚鐣欍€傝缁?99.604绉掞紝涓庡叾浠栦换鍔″苟琛岋紱涓嶄綔鐙珛閫熷害缁撹銆?


## 鏈€缁堟敹鐩婂拰鍏抽棴

涓庢RPS鎺у埗宸€?0.060804682166锛屾渶浣?鍋滄0/18锛涘疄闄匱OTAL80.591193779236锛?44665琛屾仮澶嶅拰鍐荤粨瀹屾暣鎬у凡楠岃瘉銆傛棤鏂板鏀剁泭锛屼笉缁х画鎵弿鏈満鍒讹紱璐熸潈閲嶆寜SHA娓呭崟鍒犻櫎锛岃瘉鎹繚鐣欍€傝缁?78.450绉掞紝涓庡叾浠栦换鍔″苟琛岋紱涓嶄綔鐙珛閫熷害缁撹銆?


## 鐢ㄦ埛鏈€鏂版棭鍋滐細寰皟patience14

寰皟杩炵画14杞棤鏀瑰杽鏃╁仠锛宮ax80/min10淇濇寔锛岃鐩栨棫patience18銆傚凡瀹屾垚v157/v159涓嶈拷鏀瑰巻鍙诧紱v159鍦ㄦ敹鍒版柊鎸囦护澶勭悊鏈熼棿鑷劧瀹屾垚best0/stop18銆傚凡鍚姩v160淇濈暀鍐荤粨80/18婧愮爜/閰嶇疆锛屼笓椤瑰彧璇荤洃鎺ID62172鍦╞ad14鍐欏凡鏈夎疆闂村仠姝㈣姹傦紝淇濆瓨鏈€浼樺苟鎶ュ憡瀹為檯杞锛屼笉鏇存敼璁粌婧愮爜/涓嶉噸鍚?涓嶈嚜鍔ㄩ槦鍒楋紱鍚庣画鏂版簮鐮佺洿鎺?4銆?


## 棣栬疆鐪熷疄楠岃瘉

v160鍒濆POR77.540454326739锛屼笌鍚岀埗鍒濆涓€鑷达紝230944鍙缁冨弬鏁般€傚凡瀹屾垚绗?杞紝鏆傛棤鏀剁泭锛屽嘲鍊?24.549MiB锛泂tderr鏃犻敊璇€侾ID62172璁粌涓嶱ID68380闈濭PU鏃╁仠鐩戞帶鍧囨牳楠岋紝鐩戞帶浠呭啓bad14杞棿鍋滄璇锋眰锛屾棤鎭㈠/闃熷垪銆傚浘璋眂urate/build鎴愬姛鏇存柊銆?


v161 [PERM榛樿闂ㄦ帶鍗忓悓](results/deep_learning/perm_gate_coadapt_v161_20261001/README.md)锛氬熀浜嶳PS榛樿/闈為粯璁よ瘖鏂紝淇濈暀鏈夋晥128瀵嗗害璇诲嚭锛屽紑鏀惧師gate琛屼綆LR锛屾柊澧?58娲昏穬鍙傛暟/鏋舵瀯涓嶆墿瀹广€傚噯澶囨湭璁粌锛?0/14棰勭畻銆佷簲闃舵銆?


## 棣栬疆鍚姩鏍搁獙

PID50992锛宯ormal GPU/涓存椂SYSTEM_REQUIRED锛屼笌POR160鍙屽苟鍙戙€傚垵濮婸ERM89.344760174753涓庡師鍑芥暟涓€鑷达紱棣栬疆89.307024687938锛屾殏鏈鐩婏紱82978鍙缁冨弬鏁般€乸eak786.465MiB銆傚凡瀹屾垚棣栬疆鎰忓懗鐫€涓ょ粍姝ｆ搴?鍐荤粨绛惧悕妫€鏌ラ€氳繃锛岄厤缃?0/14纭锛宻tderr鏃犻敊璇€傚緟鏈€缁堟渶浣抽噸杞?瀹為檯涓夊ご鍚堝苟锛屼笉灏嗗揩鐓х畻鏈€缁堟敹鐩娿€傚浘璋卞凡鏇存柊銆?


## 棣栬疆鍚姩鏍搁獙

PID50992锛宯ormal GPU/涓存椂SYSTEM_REQUIRED锛屼笌POR160鍙屽苟鍙戙€傚垵濮婸ERM89.344760174753涓庡師鍑芥暟涓€鑷达紱棣栬疆89.307024687938锛屾殏鏈鐩婏紱82978鍙缁冨弬鏁般€乸eak786.465MiB銆傚凡瀹屾垚棣栬疆鎰忓懗鐫€涓ょ粍姝ｆ搴?鍐荤粨绛惧悕妫€鏌ラ€氳繃锛岄厤缃?0/14纭锛宻tderr鏃犻敊璇€傚緟鏈€缁堟渶浣抽噸杞?瀹為檯涓夊ご鍚堝苟锛屼笉灏嗗揩鐓х畻鏈€缁堟敹鐩娿€傚浘璋卞凡鏇存柊銆?


v162 [PERM灞€閮ㄨ〃绀洪€傞厤](results/deep_learning/perm_local_representation_v162_20261001/README.md)锛氫袱涓撳鍚勫皬鍨?鐐规湁鐣岄€傞厤鍣?鏈夋晥128/RPS锛屽師楠ㄥ共鍐荤粨锛涙柊澧?7216鍙傛暟銆?0/14銆佷簲闃舵銆傚噯澶囨湭璁粌鏃犲垎鏁帮紝绛夊緟鐜版湁涓ら」缁撴潫锛屼笉寮€绗笁GPU浠诲姟/鑷姩闃熷垪銆?


POR160涓嶱ERM161缁х画鍙屽苟鍙戯紝鍧囨殏鏃犺秴姝ｆ帶鍒舵敹鐩婏紱宸插噯澶嘝ERM162琛ㄧず灞傞€傞厤銆佹湭鍚姩锛宎tlas宸叉垚鍔熸洿鏂般€傚畬鏁碠OF>82浠嶆湭瀹屾垚锛屽綋鍓嶅凡楠岃瘉缁勫悎80.612475銆?

V161 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=203.253, peakMiB=786.465; all144665replay/frozen exact.


## 鏈€缁堢粨璁?

PERM89.344760174753锛岀浉瀵笹aussian/RPS姝ｆ帶鍒垛垝0.060804682166锛屽垵濮嬫渶浣筹紝鏃犲涔犳敼鍠勶紱best0/stop14锛?4杞棭鍋滆鍒欏凡瀹為檯鐢熸晥銆傚疄闄呬笁澶碩OTAL80.591193779236锛?44665鐐规渶浣虫仮澶?鍐荤粨涓€鑷淬€備袱涓搴︾粍鍧囨锛屼絾鏃犳垚缁╂敹鐩婏紝鍏抽棴鏈棬鎺ф洿鏂伴厤缃€傝礋鏉冮噸SHA娓呭崟鍒犻櫎锛屾潵婧?閰嶇疆/鏃ュ織/棰勬祴/缁撴灉淇濈暀銆傝缁?03.253绉掞紝鍚姩鑷虫姤鍛婂閽?10.845绉掞紝涓嶱OR160骞惰銆佸寘鍚暟鎹?鍒濆/鏈€缁堣瘎浼帮紝涓嶅綋浣滃畬鏁翠簲闃舵鑰楁椂銆?


## 瀹為檯鍚姩涓庨杞獙璇?

PID59780宸叉牳楠屽瓨娲伙紝normal GPU銆佷复鏃禨YSTEM_REQUIRED銆佷笌POR160鍙屽苟鍙戯紱鍒濆PERM89.344760174753涓庡師鍑芥暟涓€鑷达紝99936娲昏穬鍙傛暟銆傚凡瀹屾垚绗?杞紝鏈€浣充粛鍒濆锛岀2杞?9.313387736773锛宲eak786.465MiB锛宻tderr鏃犻敊璇€傝缁冪殑涓ょ粍姝ｆ搴﹀拰鍐荤粨绛惧悕妫€鏌ュ凡閫氳繃锛?0/14閰嶇疆纭銆傚皻鏈瘉瀹炴敹鐩婏紝寰呮渶浣崇鐩樻仮澶?瀹為檯涓夐€氶亾缁勫悎锛涘浘璋卞凡鏇存柊銆?


## 瀹為檯鍚姩涓庨杞獙璇?

PID59780宸叉牳楠屽瓨娲伙紝normal GPU銆佷复鏃禨YSTEM_REQUIRED銆佷笌POR160鍙屽苟鍙戯紱鍒濆PERM89.344760174753涓庡師鍑芥暟涓€鑷达紝99936娲昏穬鍙傛暟銆傚凡瀹屾垚绗?杞紝鏈€浣充粛鍒濆锛岀2杞?9.313387736773锛宲eak786.465MiB锛宻tderr鏃犻敊璇€傝缁冪殑涓ょ粍姝ｆ搴﹀拰鍐荤粨绛惧悕妫€鏌ュ凡閫氳繃锛?0/14閰嶇疆纭銆傚皻鏈瘉瀹炴敹鐩婏紝寰呮渶浣崇鐩樻仮澶?瀹為檯涓夐€氶亾缁勫悎锛涘浘璋卞凡鏇存柊銆?


v163 [PERM浣嶭R/RPS](results/deep_learning/perm_low_lr_rps_v163_20261001/README.md)锛氬巻鍙?54棣栨澧炵泭16杞紝褰撳墠pat14浼氭埅鏂紱闄嶄綆澶碙R鑷?e-4锛屽叾浠栨湁鏁堟満鍒朵笉鍙樸€傛簴澶囨湭璁粌銆佷簲闃舵銆侀绠?0/14銆?

V162 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=212.983, peakMiB=786.465; all144665replay/frozen exact.


## 鏈€缁堢粨璁轰笌鍏抽棴

80/14棰勭畻涓婸ERM89.344760174753锛屾棤鍒濆鏀瑰杽锛涚浉瀵瑰巻鍙叉湁鏁?54鈭?.060804682166锛宐est0/stop14銆傚疄闄匱OTAL80.591193779236锛?44665琛屾渶浣抽噸杞戒笌鍐荤粨绮剧‘锛屼袱姊害缁勪负姝ｃ€傚綋鍓嶉厤缃?棰勭畻鏃犳敹鐩婏紝鍘嗗彶154闇€16杞鏀瑰杽锛屼笉鑳芥硾鍖栧叏閮ㄩ€傞厤鍣ㄦ棤鏁堛€傚叧闂湰閰嶇疆锛岃礋鏉冮噸SHA娓呭崟娓呯悊锛屽叾浠栬瘉鎹繚鐣欍€傝缁?12.983绉掞紝鍚姩鑷虫姤鍛婂閽?19.845绉掞紝鏁版嵁/鍒濆/鏈€缁堣瘎浼拌鍏ワ紝涓嶱OR160骞惰锛涗笉鏄畬鏁翠簲闃舵鑰楁椂銆?


## 瀹為檯鍚姩鏍搁獙

PID27280瀛樻椿锛宯ormal GPU/涓存椂SYSTEM_REQUIRED锛屼笌POR160鍙屽苟鍙戙€傚垵濮?9.344760174753锛屼笌鍘熷嚱鏁颁竴鑷达紱82720鍙傛暟锛岄杞?9.327379522056锛屾殏鏃犳敹鐩娿€傚疄闄呴厤缃甿ax80/pat14/lr1e-4纭锛涢杞搴?鍐荤粨妫€鏌ラ€氳繃锛宲eak786.465MiB锛屾棤stderr閿欒銆傜瓑寰呭疄闄呮渶浣虫仮澶?缁勫悎/鏈€缁堝鐩婄粨璁猴紝鍥捐氨宸叉洿鏂般€?


## 瀹為檯鍚姩鏍搁獙

PID27280瀛樻椿锛宯ormal GPU/涓存椂SYSTEM_REQUIRED锛屼笌POR160鍙屽苟鍙戙€傚垵濮?9.344760174753锛屼笌鍘熷嚱鏁颁竴鑷达紱82720鍙傛暟锛岄杞?9.327379522056锛屾殏鏃犳敹鐩娿€傚疄闄呴厤缃甿ax80/pat14/lr1e-4纭锛涢杞搴?鍐荤粨妫€鏌ラ€氳繃锛宲eak786.465MiB锛屾棤stderr閿欒銆傜瓑寰呭疄闄呮渶浣虫仮澶?缁勫悎/鏈€缁堝鐩婄粨璁猴紝鍥捐氨宸叉洿鏂般€?


v164 [POR璇诲嚭娈嬪樊娣卞害](results/deep_learning/por_deep_readout_v164_20261001/README.md)锛歭inear/nonlinear296864鍚屽閲忓鐓э紝鍘熸湁鏁?56/RPS1锛屾畫宸唴閮ㄥ琛ワ紝鍘熺埗棰勬祴鍐荤粨銆傚噯澶囨湭璁粌锛屼袱椤规棤鍒嗘暟锛?0/14銆佷簲闃舵锛涗笉鍚姩绗笁GPU浠诲姟銆佷笉鑷姩闃熷垪銆?

V160 joint_rps: restored POR=77.873725032741; delta vs v114=+0.325655537848, vs historical=+0.321788230856; actual fifth TOTAL=80.596678699599;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/45, seconds=861.459, peakMiB=924.549.


## 鏈€缁堢粨璁?

POR77.873725032741锛屾RPS1宸€尖垝0.052655727983锛涘疄闄呬笁澶碩OTAL80.596678699599銆俠est30/stop45锛涚44杞産ad14瑙﹀彂涓撻」鍋滄锛屼笅涓€杞凡寮€濮嬶紝瀹為檯bad15鍋滐紱涓嶅啋绉版伆濂?4杞€?44665鏈€浣虫仮澶?鍐荤粨绮剧‘宸查獙璇併€傚叧闂晥鐢?.1閰嶇疆锛屼笉鍚﹀畾鎵€鏈夋晥鐢紱璐熸潈閲峉HA娓呭崟鍒犻櫎锛屽叾浠栬瘉鎹繚鐣欍€傝缁?61.459绉掞紝鍚姩鍒版姤鍛婂閽?69.027绉掞紝鍚暟鎹?鍒濆/鏈€缁堣瘎浼帮紝涓嶱ERM澶氶」涓叉帴骞惰銆佹渶澶?GPU锛涗笉鏄叏浜旈樁娈佃€楁椂銆?

V163 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=184.407, peakMiB=786.465; all144665replay/frozen exact.


## 鏈€缁堢粨璁?

PERM89.344760174753锛屽垵濮嬫渶浣?鍋滄14锛屾棤瀛︿範鏀瑰杽锛涙RPS鎺у埗宸€尖垝0.060804682166锛屽疄闄匱OTAL80.591193779236銆?44665鏈€浣虫仮澶?鍐荤粨绮剧‘锛涗綆LR1e-4鍦?0/14浠嶆棤鏀剁泭锛屼笉缁х画LR缃戞牸銆備粎闄愭棰勭畻/閰嶇疆锛屼笉鍚﹀畾鎵€鏈夊涔犵巼浼樺寲銆傝礋鏉冮噸SHA娓呭崟娓呯悊锛岃瘉鎹繚鐣欍€傝缁?84.407绉掞紝鍚姩鍒版姤鍛婂閽?91.893绉掞紝鏁版嵁/鍒濆/鏈€缁堣瘎浼拌鍏ワ紝涓嶱OR160骞惰鍚庣煭鏆傜嫭鍗狅紱涓嶆槸浜旈樁娈垫€昏€楁椂銆?


## 瀹為檯鍙屽苟鍙戝惎鍔?

linear PID40840銆乶onlinear PID67592鍧囨牳楠屽瓨娲汇€乶ormalGPU/涓存椂SYSTEM_REQUIRED銆傚垵濮嬩袱椤筆OR77.540454326739锛屼笌鍚岀埗鍑芥暟涓€鑷达紱296864鍙傛暟銆乵ax80/pat14鍧囩‘璁わ紝stderr鏃犻敊璇€傚揩鐓э細{"linear": {"pid": 40840, "epoch": 6, "best": 77.54045432673898, "best_epoch": 0, "bad": 6}, "nonlinear": {"pid": 67592, "epoch": 2, "best": 77.54045432673898, "best_epoch": 0, "bad": 2}}銆傛棤鏈€缁堟敹鐩婄粨璁猴紝寰呬袱椤规仮澶?瀹為檯缁勫悎/鍏钩宸€硷紱鍥捐氨宸叉洿鏂般€?


## 瀹為檯鍙屽苟鍙戝惎鍔?

linear PID40840銆乶onlinear PID67592鍧囨牳楠屽瓨娲汇€乶ormalGPU/涓存椂SYSTEM_REQUIRED銆傚垵濮嬩袱椤筆OR77.540454326739锛屼笌鍚岀埗鍑芥暟涓€鑷达紱296864鍙傛暟銆乵ax80/pat14鍧囩‘璁わ紝stderr鏃犻敊璇€傚揩鐓э細{"linear": {"pid": 40840, "epoch": 6, "best": 77.54045432673898, "best_epoch": 0, "bad": 6}, "nonlinear": {"pid": 67592, "epoch": 2, "best": 77.54045432673898, "best_epoch": 0, "bad": 2}}銆傛棤鏈€缁堟敹鐩婄粨璁猴紝寰呬袱椤规仮澶?瀹為檯缁勫悎/鍏钩宸€硷紱鍥捐氨宸叉洿鏂般€?


v165 [PERM浼樺寲鍣ㄥ尮閰嶅鐓(results/deep_learning/perm_sam_rps_v165_20261001/README.md)锛氬師鏈夋晥128/RPS1锛岄瀹氫箟AdamW/SAM0.05銆?2720鍚屽閲?鍚屽垵濮嬪寲/80-14锛屽噯澶囨湭璁粌銆佹棤鍒嗘暟锛屽緟164閲婃斁妲戒綅锛屼笉鑷姩鎺掗槦銆?


## 鐪熷疄涓湡蹇収

{"linear": {"status": "running", "epoch": 16, "best": 77.58681772050492, "best_epoch": 12, "bad": 4}, "nonlinear": {"status": "running", "epoch": 11, "best": 77.54045432673898, "best_epoch": 0, "bad": 11}}銆備袱椤硅繘绋嬪疄闄呮牳楠屽瓨娲伙紱绾挎€х増鏈凡鏀瑰杽鍒濆浣嗕粛鏈秴姝ｆ帶鍒讹紝闈炵嚎鎬ф殏鏃犲鐩娿€傚尮閰嶆秷铻嶅皻鏈粨鏉燂紝涓嶅０绉版渶缁堟満鍒舵敹鐩娿€傚凡鍑嗗PERM165 AdamW/SAM0.05鍚岄绠椾袱椤癸紝鏈惎鍔ㄣ€佹棤绗笁GPU浠诲姟銆傚浘璋卞凡鏇存柊銆?


## 鐪熷疄涓湡蹇収

{"linear": {"status": "running", "epoch": 16, "best": 77.58681772050492, "best_epoch": 12, "bad": 4}, "nonlinear": {"status": "running", "epoch": 11, "best": 77.54045432673898, "best_epoch": 0, "bad": 11}}銆備袱椤硅繘绋嬪疄闄呮牳楠屽瓨娲伙紱绾挎€х増鏈凡鏀瑰杽鍒濆浣嗕粛鏈秴姝ｆ帶鍒讹紝闈炵嚎鎬ф殏鏃犲鐩娿€傚尮閰嶆秷铻嶅皻鏈粨鏉燂紝涓嶅０绉版渶缁堟満鍒舵敹鐩娿€傚凡鍑嗗PERM165 AdamW/SAM0.05鍚岄绠椾袱椤癸紝鏈惎鍔ㄣ€佹棤绗笁GPU浠诲姟銆傚浘璋卞凡鏇存柊銆?


## 鍚岃疆鍓嶇紑璇婃柇锛堥潪鏈€缁堟秷铻嶏級

涓ら」鍏卞悓瀹屾垚19杞細绾挎€у墠缂€鏈€浣?7.808192160603锛岄潪绾挎€?7.833969698770锛岄潪绾挎€р垝绾挎€?0.025777538167锛涘潎鏈揪鍒版鎺у埗77.926380760724銆備袱杩涚▼宸叉牳楠宭ive锛屾棤stderr閿欒銆傚彧浣滆缁冧腑蹇収锛屾湭瀹屾垚涓嶈兘澹扮О鏈€缁堟満鍒跺鐩娿€?


## 鍚岃疆鍓嶇紑璇婃柇锛堥潪鏈€缁堟秷铻嶏級

涓ら」鍏卞悓瀹屾垚19杞細绾挎€у墠缂€鏈€浣?7.808192160603锛岄潪绾挎€?7.833969698770锛岄潪绾挎€р垝绾挎€?0.025777538167锛涘潎鏈揪鍒版鎺у埗77.926380760724銆備袱杩涚▼宸叉牳楠宭ive锛屾棤stderr閿欒銆傚彧浣滆缁冧腑蹇収锛屾湭瀹屾垚涓嶈兘澹扮О鏈€缁堟満鍒跺鐩娿€?


宸插畬鎴怭ERM154鍥哄畾棰勬祴閿欒鏉ユ簮璇婃柇锛岃瘉鎹results/deep_learning/perm_ranked_probability_v154_20261001/ERROR_SOURCES.md锛涚害45.2%澶卞垎鏉ヨ嚜榛樿/杩炵画鍔ㄤ綔閿欓厤锛岃繛缁綆浼颁负涓汇€備粎鎻忚堪鎬у垎鏋愶紝涓嶆敼棰勬祴銆佷笉鍔犻獙璇佹爣绛炬淳鐢熺壒寰侊紝涓嶅綋浣滄満鍒舵敹鐩娿€?


宸插畬鎴怭ERM154鍥哄畾棰勬祴閿欒鏉ユ簮璇婃柇锛岃瘉鎹results/deep_learning/perm_ranked_probability_v154_20261001/ERROR_SOURCES.md锛涚害45.2%澶卞垎鏉ヨ嚜榛樿/杩炵画鍔ㄤ綔閿欓厤锛岃繛缁綆浼颁负涓汇€備粎鎻忚堪鎬у垎鏋愶紝涓嶆敼棰勬祴銆佷笉鍔犻獙璇佹爣绛炬淳鐢熺壒寰侊紝涓嶅綋浣滄満鍒舵敹鐩娿€?


v166 [PERM绉佹湁楠ㄥ共/RPS](results/deep_learning/perm_private_backbone_rps_v166_20261001/README.md)锛氫富鏀籔ERM锛屽師鍙屼笓瀹禸ody1e-5/鏈夋晥澶?e-4锛岀湡瀹瀊ody姊害锛岃緭鍑鸿/router鍙傛暟鍥哄畾锛汼OURCE鍒濆鍖栵紝浜旈樁娈碉紝80-14銆傚噯澶囨湭璁粌鏃犲垎鏁帮紱寰?64/165閲婃斁妲戒綅鏄惧紡鍚姩銆佷笉鑷姩鎺掗槦銆?

V164 linear: restored POR=77.891844743355; delta vs v114=+0.343775248462, vs historical=+0.339907941470; actual fifth TOTAL=80.602114612783;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/44, seconds=670.027, peakMiB=925.553.


## 鏄惧紡鍚姩鎺у埗椤?

POR164 linear宸插畬鎴愶紝鏆備繚鐣欐潈閲嶈嚦閰嶅姣旇緝缁撴潫锛涢潪绾挎€т粛杩愯銆傚凡鐢ㄧ┖妲芥樉寮忓惎鍔?65 AdamW鎺у埗锛孲AM灏氭湭鍚姩锛屾棤绗笁GPU浠诲姟鎴栬嚜鍔ㄩ槦鍒椼€傜鏈夐骞?66宸插噯澶囦絾灏氭棤璁粌/鍒嗘暟锛岀瓑寰呯幇鏈夐瀹氫箟瀵圭収銆?


## 鏄惧紡鍚姩鎺у埗椤?

POR164 linear宸插畬鎴愶紝鏆備繚鐣欐潈閲嶈嚦閰嶅姣旇緝缁撴潫锛涢潪绾挎€т粛杩愯銆傚凡鐢ㄧ┖妲芥樉寮忓惎鍔?65 AdamW鎺у埗锛孲AM灏氭湭鍚姩锛屾棤绗笁GPU浠诲姟鎴栬嚜鍔ㄩ槦鍒椼€傜鏈夐骞?66宸插噯澶囦絾灏氭棤璁粌/鍒嗘暟锛岀瓑寰呯幇鏈夐瀹氫箟瀵圭収銆?

V164 nonlinear: restored POR=77.885830334051; delta vs v114=+0.337760839158, vs historical=+0.333893532166; actual fifth TOTAL=80.600310289992;5 phases incl source; channel gain>0.25=True; strict808=False; no otherfolds. best/stop=30/44, seconds=697.551, peakMiB=925.399.


## 鏈€缁堥厤瀵圭粨璁?

绾挎€OR77.891844743355锛岄潪绾挎€?7.885830334051锛涘悓瀹归噺/鍒濆鍖?80-14闈炵嚎鎬р垝绾挎€?0.006014409304锛屾棤闈炵嚎鎬ф敹鐩婏紝涓よ€呬篃浣庝簬姝PS鎺у埗77.926380760724銆備袱椤规渶浣?0/鍋滄44锛屽叏144665鎭㈠鍜屽疄闄呬笁澶村悎骞躲€佸喕缁撶簿纭€傛棭鏈熷墠缂€姝ｅ樊鍊间笉鏄渶缁堝鐩婏紝宸茶鏈€缁堢粨鏋滄帹缈汇€傚叧闂65920鍙傛暟/鐣?.1娈嬪樊缁撴瀯锛屼笉娉涘寲鍏ㄩ儴娣卞ご鏃犳晥銆傚洓涓礋鏉冮噸鎸塖HA娓呭崟鍒犻櫎锛岃瘉鎹繚鐣欍€備袱椤规€讳綋鍚姩鍒版渶鍚庢姤鍛婂閽?39.384绉掞紝鍗曢」璁粌/鏄惧瓨瑙乼iming.json锛涢敊寮€鍚姩鍙屽苟鍙戯紝鍚暟鎹?棰勭儹/璇勪及锛屼笉褰撲綔瀹屾暣浜旈樁娈佃缁冩椂闀裤€?

V165 adamw: actual PERM=89.344760174753, TOTAL=80.591193779236; delta historicalRPS=-0.060804682166; best/stop=0/14, seconds=131.987, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.


v167 [PERM闅愯棌琛ㄧず绾︽潫](results/deep_learning/perm_aligned_backbone_rps_v167_20261001/README.md)锛?66鍚屽閲?鍒濆鍖?80-14锛屽彧鍔?.1璁粌渚у師SOURCE闅愯棌MSE锛宼eacher涓嶉娴嬭捀棣?涓嶆帹鐞嗭紝鏃犳柊璁粌闃舵锛屼簲闃舵銆傚噯澶囨湭璁粌锛屽緟绌烘Ы鏄惧紡鍚姩銆傞樁娈靛帇缂╁凡鎸夌敤鎴锋挙閿€锛屼笉鏀瑰彉褰撳墠鏈夋晥閰嶆柟銆?

V166 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=249.026, peakMiB=1050.815; all144665replay/frozen exact.

V165 sam: actual PERM=89.352248368881, TOTAL=80.593814647181; delta historicalRPS=-0.053316488038; best/stop=10/24, seconds=397.458, peakMiB=786.465; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.

V165 completed matched80/14 pair: SAM-AdamW PERM=+0.007488194128. Equal updates, unequal compute; original fifth only, not fullOOF.


v168 [PERM SAM闂ㄦ帶](results/deep_learning/perm_sam_gate_rps_v168_20261001/README.md)锛氬熀浜?65鏈€缁堝眬閮⊿AM+0.007488锛屽鍔犲師gate鏇存柊鑼冨洿锛?2978娲昏穬鍙傛暟銆佸悓SOURCE鍒濆/80-14锛屼簲闃舵锛涘噯澶囨湭璁粌锛屾棤宸茶161/165澶村姞杞姐€?


Completed perm_conditional: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.591193779236; negative best/last deleted with SHA manifest after exact replay. Training=249.026s, launch/report wall=255.981s, peak=1050.815MiB; includes data/initial/final eval, other jobs overlap, not fullfivephase wall. Comparisons retain immutable configs/results/predictions.


Completed adamw: best/stop=0/14, PERM=89.344760174753, no initial gain, actualTOTAL=80.591193779236; negative best/last deleted with SHA manifest after exact replay. Training=131.987s, launch/report wall=140.226s, peak=786.465MiB; includes data/initial/final eval, other jobs overlap, not fullfivephase wall. Comparisons retain immutable configs/results/predictions.


## 鏈€缁圫AM鏈哄埗鏀剁泭

鍚?0/14銆?2720鍙傛暟銆佸垵濮嬪寲銆佹暟鎹?璇勫垎锛孲AM鈭扐damW PERM=+0.007488194128锛汼AM89.352248368881锛屼粛浣庝簬鍘嗗彶鏈夋晥154鐨?9.405564856919銆傚皬骞呭崟鎶樺崟seed瀹炴祴鏈哄埗鏀剁泭锛屼繚鐣橲AM鏈€浣冲拰last锛涗笉鍗囩骇涓篠OTA/瀹屾暣OOF鏀剁泭锛岀户缁竴椤归棬鎺ц寖鍥撮獙璇併€係AM姣忔洿鏂板弻鍓嶅悗鍚戯紝鎺у埗涓€娆★紝绛夋洿鏂颁笉绛夎绠楋紱瀹屾垚涓ら」鎬诲閽?18.049绉掋€佸崟椤硅€楁椂/鏄惧瓨瑙乼iming.json銆傞敊寮€鍚姩涓斿叾浠栨帰绱㈢┛鎻掞紝涓嶄綔绾袱浠诲姟骞跺彂鍔犻€熺粨璁恒€?


## 鐢ㄦ埛鎾ら攢闃舵鍘嬬缉

鐢ㄦ埛鍥炲鈥滅畻浜嗭紝灏辫繖鏍峰惂鈥濓紝淇濈暀褰撳墠闃舵鏁帮紝涓嶅啀鎺ㄨ繘鍘嬬缉銆傚綋鍓嶆湁鏁圥OR閾捐矾3闃舵銆丳ERM2闃舵锛涘叡浜師鏉ユ簮鍙涓€娆★紝鍏ㄦ柟娉曞叡5闃舵銆備繚鐣欏凡楠岃瘉閰嶆柟鍜屾潈閲嶏紝缁х画鍘熸湁浼樺寲/80-14/绗簲鎶樹弗鏍?0.8绛涢€夈€傛湭鍚姩闃舵鍘嬬缉璁粌銆?


## 鐢ㄦ埛鎾ら攢闃舵鍘嬬缉

鐢ㄦ埛鍥炲鈥滅畻浜嗭紝灏辫繖鏍峰惂鈥濓紝淇濈暀褰撳墠闃舵鏁帮紝涓嶅啀鎺ㄨ繘鍘嬬缉銆傚綋鍓嶆湁鏁圥OR閾捐矾3闃舵銆丳ERM2闃舵锛涘叡浜師鏉ユ簮鍙涓€娆★紝鍏ㄦ柟娉曞叡5闃舵銆備繚鐣欏凡楠岃瘉閰嶆柟鍜屾潈閲嶏紝缁х画鍘熸湁浼樺寲/80-14/绗簲鎶樹弗鏍?0.8绛涢€夈€傛湭鍚姩闃舵鍘嬬缉璁粌銆?

V167 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=235.218, peakMiB=1072.628; all144665replay/frozen exact.

V168 final PERM=89.350850358669, compactcontrolgain=-0.027543852475; actualTOTAL=80.593325343606, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=346.955, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁堢粨璁猴細v167 PERM=89.344760174753锛岀浉瀵箆166 +0.000000000000锛涘疄闄呬笁澶碩OTAL=80.591193779236銆俠est/stop=0/14锛?0/14锛?44665鐐圭鐩樻仮澶嶉娴嬪強鍐荤粨鍙傛暟鏍搁獙閫氳繃銆傛湭瓒呰繃鏈夋晥v154锛屼笉鏇挎崲閰嶆柟銆佷笉琛ユ姌銆備笉缁х画鏈疆0.1闅愯棌閿氬畾绯绘暟鎵弿锛涚粨璁轰粎闄愯璁剧疆/14杞棭鍋溿€?

鏈€缁堢粨璁猴細v168 PERM=89.350850358669锛岀浉瀵箆165 -0.001398010212锛涘疄闄呬笁澶碩OTAL=80.593325343606銆俠est/stop=10/24锛?0/14锛?44665鐐圭鐩樻仮澶嶉娴嬪強鍐荤粨鍙傛暟鏍搁獙閫氳繃銆傛湭瓒呰繃鏈夋晥v154锛屼笉鏇挎崲閰嶆柟銆佷笉琛ユ姌銆傞棬鎺у琛ユ湭鎵╁ぇSAM鏀剁泭锛涗繚鐣檝165鍗曠嫭SAM灏忓箙姝ｆ敹鐩婏紝鍏抽棴鏈疆闂ㄦ帶澧炶ˉ锛屼笉鎵弿闂ㄦ帶瀛︿範鐜囥€?

鍑嗗v169锛歋AM/RPS鍔燩ERM log10璇樊涓夎鏁堢敤锛岀郴鏁?.1锛涙棤鎴愮哗锛屼簲闃舵涓嶅彉锛?0/14锛屾渶澶?GPU銆傝褰時esults/deep_learning/perm_sam_utility_v169_20261001/README.md銆?

鍑嗗v170锛氭柊269缁磋緭鍏ャ€?4缁村灏哄害TCN鍒嗗竷淇缂栫爜鍣紱鍘烻OURCE淇濈暀鍐荤粨鍏堥獙锛涙棤鎴愮哗锛屼簲闃舵涓嶅彉锛?0/14锛屾渶澶?GPU銆傝褰時esults/deep_learning/perm_raw_tcn_v170_20261001/README.md銆?

v169鍏ュ彛缂栫爜澶辫触锛泇171鏂扮増鏈慨澶嶏紝鏈哄埗/棰勭畻涓嶅彉锛屾棤鎴愮哗銆傝褰昜README](results/deep_learning/perm_sam_utility_v171_20261001/README.md)銆?

v170鍏ュ彛缂栫爜澶辫触锛泇172鏂扮増鏈慨澶嶏紝鏈哄埗/棰勭畻涓嶅彉锛屾棤鎴愮哗銆傝褰昜README](results/deep_learning/perm_raw_tcn_v172_20261001/README.md)銆?

宸叉樉寮忓惎鍔≒ID39440锛宯ormal GPU锛涙潵婧愬嚱鏁板垵濮婸ERM89.344760174753锛屽彲璁粌鍙傛暟82720锛?0/14銆傚綋鍓嶅揩鐓'status': 'running', 'pid': 39440, 'stage': 'perm_conditional', 'epoch': 1, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 1, 'peak_MiB': 786.46533203125}锛涘皻鏈渶缁堟仮澶嶏紝鏀剁泭灏氭湭璇佸疄銆?
璁粌鏁堢敤绯绘暟0.1璁板綍浜巌nitialization.train_only_expected_logratio_utility_weight锛沜onfig.utility_weight鏄户鎵跨殑鏃W alpha鍗犱綅0锛屾湰瀹為獙瀹為檯PERM鎹熷け涓篘LL+BCE+RPS1+0.1鏁堢敤銆係AM鍗婂緞0.05锛?娆″墠鍚庡悜銆?

宸叉樉寮忓惎鍔≒ID73544锛宯ormal GPU锛涙潵婧愬嚱鏁板垵濮婸ERM89.344760174753锛屽彲璁粌鍙傛暟54176锛?0/14銆傚綋鍓嶅揩鐓'status': 'running', 'pid': 73544, 'stage': 'perm_conditional', 'epoch': 3, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 3, 'peak_MiB': 786.46533203125}锛涘皻鏈渶缁堟仮澶嶏紝鏀剁泭灏氭湭璇佸疄銆?
TCN杈撳叆269=鍘?06+璁粌渚?3骞呭€肩壒寰侊紱涓嶈鍙栧師闅愯棌鐗瑰緛浣滄柊缂栫爜鍣ㄨ緭鍏ャ€係OURCE鍐荤粨鍒嗗竷鍏堥獙淇濈暀锛涗笉鏄暣妯″瀷鍐峰惎鍔ㄦ浛鎹紱active54176涓?2720瀵圭収涓嶇瓑瀹归噺銆?

V172 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=160.280, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁堬細PERM=89.344760174753锛宐est/stop=0/14锛屾棤鍒濆鏀剁泭锛涘疄闄匱OTAL=80.591193779236銆?44665閲嶈浇/鍐荤粨鏍￠獙閫氳繃锛岃缁?60.280s锛屽嘲鍊?86.465MiB锛涜礋鏉冮噸娓呯悊鐣橲HA娓呭崟銆傚仠姝㈡湰杞粎淇浣嶇疆/鏉冮噸鐨勫師濮嬬壒寰乀CN锛涗笉娉涘寲鍒版墍鏈塗CN锛屼笉鎵搴︺€?

鍑嗗v173绉佹湁PERM闅忔満TCN+娉ㄦ剰鍔涘畬鏁撮骞诧紝鏈缁冩棤鎴愮哗锛屾浛鎹ERM闃舵銆佹€?闃舵銆傝褰昜README](results/deep_learning/perm_private_attention_v173_20261001/README.md)銆?

璁粌蹇収锛堥潪鏈€缁堬級锛歷171 {'status': 'running', 'pid': 39440, 'stage': 'perm_conditional', 'epoch': 14, 'best': 89.36208510714584, 'best_epoch': 10, 'bad': 4, 'peak_MiB': 786.46533203125}锛沘ctive82720銆傚皻鏃犳渶缁堢鐩樻仮澶嶅拰涓夊ご缁勫悎锛屼笉鏀瑰彉褰撳墠閰嶆柟銆?

璁粌蹇収锛堥潪鏈€缁堬級锛歷173 {'status': 'running', 'pid': 64464, 'stage': 'perm_conditional', 'epoch': 4, 'best': 87.79233514602183, 'best_epoch': 3, 'bad': 1, 'peak_MiB': 788.1005859375}锛沘ctive685873銆傚皻鏃犳渶缁堢鐩樻仮澶嶅拰涓夊ご缁勫悎锛屼笉鏀瑰彉褰撳墠閰嶆柟銆?
CUDA鎻掑€煎強memory-efficient attention鍙嶅悜鏈墂arn_only闈炵‘瀹氭€ц鍛婏紝鏈姤杩愯閿欒锛涗笉澹扮О璁粌杞ㄨ抗浣嶇骇鍙鐜帮紝鏈€缁堟仮澶嶉娴嬩竴鑷存€т粛闇€楠岃瘉銆傚垵濮嬪寲68.383772436967锛屽墠2杞?6.192730/87.420202锛涜繖鏄嫭绔嬮殢鏈烘ā鍨嬪涔犺繘灞曪紝闈炶秴杩囩儹鍚姩瀵圭収鐨勬敹鐩娿€?

V171 final PERM=89.362085107146, compactcontrolgain=-0.016309103998; actualTOTAL=80.597257505573, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=588.786, peakMiB=786.465; all144665replay/frozen exact.

V173 final PERM=88.847573085146, compactcontrolgain=-0.530821125998; actualTOTAL=80.417178297873, fivephases inclsource;strict808=False, no otherfolds. best/stop=20/34, seconds=286.037, peakMiB=788.101; all144665replay/frozen exact.

鏈€缁堝眬閮ㄦ鏀剁泭锛歅ERM89.362085107146锛屽悓瀹归噺/鍒濆鍖?80-14/SAM鍗婂緞/姝ユ暟/Q/璇勫垎瀵圭収v165宸€?0.009836738265锛屼絾姣攙154 -0.043479749773銆傚疄闄匱OTAL80.597257505573锛宐est/stop=10/24锛?44665鎭㈠涓€鑷翠笌鍐荤粨鏍搁獙閫氳繃銆備繚鐣檅est/last姝ｇ粨鏋溿€傞澶栨晥鐢–DF璁＄畻浣跨瓑姝ユ暟涓嶇瓑璁＄畻閲忥紱涓嶅０绉板畬鏁碠OF鏀剁泭锛屼笉鏇挎崲鏈夋晥閰嶆柟銆?

鏈€缁堢嫭绔嬮骞诧細PERM88.847573085146锛岀浉瀵箆154 -0.557991771773锛屽疄闄匱OTAL80.417178297873锛沚est/stop=20/34锛宎ctive685873锛?44665纾佺洏鎭㈠涓€鑷达紝璁粌286.037s锛宲eak788.101MiB銆傝櫧鐒堕殢鏈哄垵濮嬪埌瀛︿範鍚庢彁鍗囷紝浣嗘湭瓒呰繃鏉ユ簮/鏈夋晥澶淬€佹病鏈夊悓瀹归噺娑堣瀺姝ｈ瘉鎹紱涓嶄綔涓烘湁鏁堝鐩娿€傛竻鐞嗚礋鏉冮噸鐣橲HA锛涙湰杞喎鍚姩娉ㄦ剰鍔涢骞插仠姝㈠欢浼革紝涓嶆硾鍖栨墍鏈夐骞层€?

鍑嗗v174鏉′欢CDF鐩戠潱锛屽湪v171宸茶瘉瀹炲眬閮ㄦ敹鐩婂熀纭€涓婁繚鎸丼AM/鏁堢敤锛涗笉鍔犺浇宸茶缁?71澶达紝浜旈樁娈碉紝灏氭湭璁粌銆俒璁板綍](results/deep_learning/perm_conditional_cdf_v174_20261001/README.md)銆?

v174鏄惧紡鍚姩PID75048锛屽師SOURCE鍒濆PERM89.344760174753銆乤ctive82720锛涘綋鍓嶈缁冨揩鐓'status': 'running', 'pid': 75048, 'stage': 'perm_conditional', 'epoch': 2, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 2, 'peak_MiB': 786.46533203125}锛屽皻鏈瘉瀹炴敹鐩娿€傛湭鍚姩鍏朵粬浠诲姟鎴栬ˉ鎶樸€?

鍑嗗v175鐙珛PERM鍥哄畾閲囨牱娑堣瀺锛?.5琛?0.5浜曞潎鍖€锛屽叾浠栬缃悓173锛涙湭璁粌鏃犲垎鏁帮紝涓嶅姞杞?73鏉冮噸锛屽師鎬?闃舵銆俒璁板綍](results/deep_learning/perm_well_sampling_v175_20261001/README.md)銆?

宸插惎鍔≒ID16724銆乶ormal GPU锛屽垵濮婸ERM68.383772436967涓?73涓€鑷达紝685873鍙傛暟銆?4璁粌浜曢敋鐐规渶灏?379銆佹渶澶?3017锛岃閲囨牱浜曟洕鍏?.756446%~2.248610%锛屾柊娣峰悎閲囨牱1.159473%~1.905555%銆傚揩鐓'status': 'running', 'pid': 16724, 'stage': 'perm_conditional', 'epoch': 4, 'best': 88.0536673951041, 'best_epoch': 4, 'bad': 0, 'peak_MiB': 798.00732421875}锛岄潪鏈€缁堟垚缁┿€?
CUDA鎻掑€?attention鍙嶅悜warn_only闈炵‘瀹氭€ц鍛婏紝鍚?73锛屾湭鍙戠敓杩愯閿欒锛涗笉澹扮О璁粌閫愭浣嶇骇涓€鑷淬€?

鍑嗗v176Gaussian瑙ｆ瀽璁粌鏁堢敤锛屼粎鏇挎崲鍒嗙鏁堢敤璁＄畻锛屾湭璁粌鏃犲垎鏁帮紱174/175浠嶈繍琛岋紝涓嶅惎绗笁GPU銆傝褰昜README](results/deep_learning/perm_analytic_utility_v176_20261001/README.md)銆?

V175 final PERM=88.893539547789, compactcontrolgain=-0.484854663355; actualTOTAL=80.433266559798, fivephases inclsource;strict808=False, no otherfolds. best/stop=21/35, seconds=294.565, peakMiB=798.007; all144665replay/frozen exact.

鏈€缁堥厤瀵规鏀剁泭锛歅ERM88.893539547789锛岀浉瀵?73 +0.045966462643锛涘悓瀹归噺/鍒濆鍖?鎹熷け/80-14/Q/瑙ｇ爜妫€鏌ラ€氳繃銆傚垵濮嬪潎68.383772436967锛岄殢鏈虹獥鍙ｅ簭鍒楀洜閲囨牱鏈哄埗涓嶅悓锛屽弽鍚戝瓨鍦╳arn_only闈炵‘瀹氭€ц鍛婏紝涓嶅０绉伴€愭浣嶇骇涓€鑷存垨璺ㄦ姌绋冲畾鏀剁泭銆傚疄闄匱OTAL80.433266559798锛宐est/stop21/35锛?44665纾佺洏鎭㈠涓€鑷村拰鍐荤粨鏍￠獙閫氳繃銆備綆浜巚154-0.512025309130锛屼繚鐣欐娑堣瀺best/last锛屽綋鍓嶆湁鏁堥厤鏂逛笉鏇挎崲銆佸叾浠栨姌鏈惎鍔ㄣ€?

鍑嗗v177淇濈暀浜曞潎琛★紝鍙姞瀹屾暣绉佹湁PERM SAM0.05锛涚浉瀵?75鍚屽弬鏁?鍒濆鍖?loss/閲囨牱銆佺瓑姝ユ暟涓嶇瓑绠楀姏锛屾湭璁粌鏃犳垚缁┿€俒璁板綍](results/deep_learning/perm_well_sam_v177_20261001/README.md)銆?

V174 final PERM=89.364985921273, compactcontrolgain=-0.013408289871; actualTOTAL=80.598272790518, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=528.507, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁堟潯浠禖DF姝ｆ秷铻嶏細PERM89.364985921273锛岀浉瀵?71 +0.002900814127锛涘悓瀹归噺/鏉ユ簮鍒濆鍖?80-14/SAM/鏁堢敤/Q/瑙ｇ爜涓€鑷达紝鍙湁CDF鐩戠潱鑼冨洿鏀瑰彉銆傚疄闄匱OTAL80.598272790518锛宐est/stop=10/24锛?44665鏈€浣虫仮澶嶉娴?鍐荤粨涓€鑷达紱淇濈暀姝ｇ粨鏋渂est/last銆備粛浣庝簬鏈夋晥154锛屼笉鏇挎崲鐜版湁閰嶆柟銆佷笉鍚姩鍏朵綑鎶樸€?

v176宸叉樉寮忓惎鍔≒ID66468锛宯ormal GPU/涓存椂SYSTEM_REQUIRED锛宎ctive82720銆傚綋鍓嶈缁冨揩鐓'status': 'running', 'pid': 66468, 'stage': 'perm_conditional', 'epoch': 4, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 4, 'peak_MiB': 786.46533203125}锛岄潪鏈€缁堟敹鐩婏紱鏈€澶?GPU銆佸畬鏁存柟娉?闃舵銆佸叾浣欏洓鎶樻湭鍚姩銆?

v177宸叉樉寮忓惎鍔≒ID14800锛宯ormal GPU/涓存椂SYSTEM_REQUIRED锛宎ctive685873銆傚綋鍓嶈缁冨揩鐓'status': 'running', 'pid': 14800, 'stage': 'perm_conditional', 'epoch': 12, 'best': 88.34151401360728, 'best_epoch': 6, 'bad': 6, 'peak_MiB': 800.61279296875}锛岄潪鏈€缁堟敹鐩婏紱鏈€澶?GPU銆佸畬鏁存柟娉?闃舵銆佸叾浣欏洓鎶樻湭鍚姩銆?
鍘熺鏈夊垵濮嬪寲鍑芥暟涓?75涓€鑷达紝瀹屾暣缃戠粶SAM姊害鑼冨洿涓庢棫鐙珛澶翠笉鍚岋紱CUDA鎻掑€?attention鍙嶅悜warn_only闈炵‘瀹氭€ц鍛婂瀵圭収锛屾湭鍑虹幇杩愯閿欒锛屾渶缁堝疄闄呮仮澶嶅緟鍋氥€?

鍑嗗178闈炵嚎鎬ч粯璁ogit鑱斿悎璇诲嚭锛屽熀浜?74鏈哄埗浣嗕笉鐢ㄥ叾璁粌鏉冮噸銆乤ctive+129銆佹€?闃舵锛涘皻鏈缁冿紝鏃犵涓塆PU浠诲姟銆俒璁板綍](results/deep_learning/perm_nonlinear_gate_v178_20261001/README.md)銆?

鍑嗗179璁粌渚OURCE闅愯棌杩佺Щ锛宻tudent棰勬祴685873涓嶅彉銆乤ux+33024锛屾棤teacher棰勬祴鎹熷け鎴栨帹鐞嗭紝鏃?77鐑惎鍔紱鏈缁冦€佷笉鍚涓塆PU銆俒璁板綍](results/deep_learning/perm_hidden_transfer_v179_20261001/README.md)銆?

V177 final PERM=88.970890837572, compactcontrolgain=-0.407503373572; actualTOTAL=80.460339511223, fivephases inclsource;strict808=False, no otherfolds. best/stop=24/38, seconds=551.783, peakMiB=800.613; all144665replay/frozen exact.

鏈€缁堝畬鏁撮骞睸AM灞€閮ㄦ鏀剁泭锛歅ERM88.970890837572锛岀浉瀵?75 +0.077351289783锛涘悓妯″瀷/鍒濆鍖?閲囨牱/loss/Q/棰勭畻/浼樺寲姝ユ暟鏍搁獙閫氳繃锛汼AM2閬峷sAdamW1閬嶏紝涓嶇瓑绠楀姏銆傚疄闄匱OTAL80.460339511223锛宐est/stop24/38锛?44665纾佺洏鎭㈠/鍐荤粨涓€鑷达紱淇濈暀姝ｇ粨鏋渂est/last銆備粛浣庝簬鏈夋晥154锛屼笉鏇挎崲閰嶆柟/琛ユ姌銆?

V176 final PERM=89.359879687511, compactcontrolgain=-0.018514523632; actualTOTAL=80.596485608701, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=609.569, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁堣В鏋愭晥鐢ㄨ礋鏇挎崲锛歅ERM89.359879687511锛岀浉瀵?74 -0.005106233761锛沘ctualTOTAL80.596485608701锛宐est/stop10/24锛?44665纾佺洏鎭㈠棰勬祴鍙婂喕缁撲竴鑷达紝鍚屽閲?鍒濆鍖?棰勭畻/Q/瑙ｇ爜妫€鏌ラ€氳繃銆傝礋鏉冮噸宸叉竻鐞嗙暀SHA锛涗繚鐣?74鍒嗙鏁堢敤涓嶆浛鎹㈤厤鏂癸紝涓嶆壂鎻忕Н鍒嗙簿搴︺€備粎杩欐璁粌鏁堢敤鏇挎崲璐熺粨鏋滐紝涓嶆硾鍖朑aussian瑙ｆ瀽绉垎鏃犳晥锛涘苟鍙戞敼鍙樿绠楄祫婧愶紝涓嶈兘鍑崟椤?09.569s澹扮О绉垎瀹炵幇鏇存參銆?

v178鏄惧紡鍚姩PID7844锛宯ormal GPU/涓存椂SYSTEM_REQUIRED銆傝缁冨揩鐓'status': 'running', 'pid': 7844, 'stage': 'perm_conditional', 'epoch': 4, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 4, 'peak_MiB': 786.46533203125}锛宎ctive82849锛岄潪鏈€缁堟敹鐩婏紱鐩墠鏈€澶?浠诲姟銆佸叏鏂规硶浠?闃舵锛屾湭琛ユ姌銆?

v179鏄惧紡鍚姩PID5808锛宯ormal GPU/涓存椂SYSTEM_REQUIRED銆傝缁冨揩鐓'status': 'running', 'pid': 5808, 'stage': 'perm_conditional', 'epoch': 12, 'best': 88.64084730011055, 'best_epoch': 12, 'bad': 0, 'peak_MiB': 823.6640625}锛宎ctive718897锛岄潪鏈€缁堟敹鐩婏紱鐩墠鏈€澶?浠诲姟銆佸叏鏂规硶浠?闃舵锛屾湭琛ユ姌銆?
鍒濆student PERM68.383772436967涓?77涓€鑷达紱鏁欏笀浠卼rain forward璁＄畻闅愯棌鐩爣锛宔val涓嶈繍琛屾暀甯堛€傞娴?85873/aux33024銆乤ctive718897锛涘師SOURCE鍐荤粨銆佹棤鏂伴璁粌/鏁欏笀棰勬祴鎹熷け/鏁欏笀鎺ㄧ悊銆侰UDA鎻掑€煎拰attention鍙嶅悜warn_only闈炵‘瀹氭€ц鍛婂悓瀵圭収锛屾湭鎶ヨ繍琛岄敊璇紝涓嶅０绉颁綅绾ц缁冨鐜般€?

鍑嗗180锛氬悓174璇诲嚭锛岃縼绉?0%琛?50%浜曢噰鏍凤紝鍚屽閲忋€佹棤鐑惎鍔紝灏氭湭璁粌锛?78/179鍦ㄨ繍琛岋紝涓嶅惎绗笁浠诲姟銆俒璁板綍](results/deep_learning/perm_head_well_sampling_v180_20261001/README.md)銆?

V178 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=373.289, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁堥潪绾挎€х姸鎬佹敮璺棤鏀剁泭锛歅ERM89.344760174753锛岀浉瀵?74 -0.020225746520锛沚est/stop0/14锛宎ctualTOTAL80.591193779236锛宎ctive+129闈炵瓑瀹归噺锛?44665鎭㈠/鍐荤粨楠岃瘉閫氳繃銆傚悓Source鍒濆鍑芥暟/棰勭畻/Q/瑙ｇ爜锛屽叧闂湰杞叡浜玹runk銆伮?gate鏀矾锛屼笉鎵晫闄?瀹藉害锛屼笉娉涘寲鎵€鏈夐潪绾挎€ate锛涜礋鏉冮噸娓呯悊鐣橲HA銆?

鍑嗗181杈撳叆鏁翠簳FiLM鏉′欢鍖栵紝鍙敤well bank杈撳叆锛屼笉鍔犺浇鏃W/177鏉冮噸銆佹棤teacher锛屾湭璁粌锛屾棤绗笁浠诲姟銆俒璁板綍](results/deep_learning/perm_well_conditioned_v181_20261001/README.md)銆?

V179 final PERM=88.881649640932, compactcontrolgain=-0.496744570212; actualTOTAL=80.429105092399, fivephases inclsource;strict808=False, no otherfolds. best/stop=21/35, seconds=767.048, peakMiB=823.664; all144665replay/frozen exact.

鏈€缁堥殣钘忚縼绉昏礋缁撴灉锛歅ERM88.881649640932锛岀浉瀵?77 -0.089241196640锛沚est/stop21/35锛宎ctualTOTAL80.429105092399銆傞娴嬬粨鏋?85873涓嶅彉锛宎ux璁粌+33024銆乼eacher浠呰缁冦€佸叏灞€SAM姊害鑼冩暟鏀瑰彉锛岄潪瀹屽叏鍚宎ctive瀹归噺娑堣瀺銆?44665鏈€浣虫仮澶?鍐荤粨楠岃瘉閫氳繃锛岃礋鏉冮噸娓呯悊鐣橲HA锛屽師SOURCE鍙?77姝ｇ粨鏋滀繚鐣欍€傚叧闂湰娆?.1涓や笓瀹秇idden鐩戠潱锛屼笉鎵郴鏁?灞傦紝涓嶆硾鍖栨墍鏈夎〃寰佽縼绉汇€?

鍑嗗182Student-t5 PERM锛宭og鍒嗛噺鐭╁尮閰嶃€佸垵濮婥DF涓嶅悓锛孭DF/CDF/utility涓€鑷达紝鏈缁冩棤鍒嗘暟锛屼笉鍔犺浇174澶达紝鎬?闃舵銆俒璁板綍](results/deep_learning/perm_student5_v182_20261001/README.md)銆?

V180 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=333.678, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁堜簳鍧囪　杩佺Щ鍒拌鍑烘棤鐩婏細PERM89.344760174753锛岀浉瀵?74 -0.020225746520锛沚est/stop0/14锛宎ctualTOTAL80.591193779236锛屽悓瀹归噺/鍒濆鍖?loss/浼樺寲/Q/瑙ｇ爜鏍搁獙閫氳繃锛?44665鎭㈠涓€鑷淬€傛竻鐞嗚礋鏉冮噸鐣橲HA锛涗繚鐣?75楠ㄥ共鍘熼噰鏍锋鏀剁泭锛屼絾涓嶈法鏋舵瀯璁ゅ畾閲囨牱娉涘寲锛涗笉鎵瘮渚嬨€佽鍑虹户缁敤174鍘熻閲囨牱銆?

v181宸叉樉寮忓惎鍔≒ID63344銆乶ormal GPU銆佷复鏃禨YSTEM_REQUIRED銆傚綋鍓嶈缁冨揩鐓'status': 'running', 'pid': 63344, 'stage': 'perm_conditional', 'epoch': 33, 'best': 88.85628937090793, 'best_epoch': 27, 'bad': 6, 'peak_MiB': 826.8720703125}锛宎ctive839761锛岄潪鏈€缁堟敹鐩婏紱max2浠诲姟/瀹屾暣鏂规硶5闃舵/鏈ˉ鎶樸€?
浜昩ank鍙惈鍘?06杈撳叆锛?28娣卞害閲囨牱銆佹棤label/浜旾D锛涢澶栨潯浠剁紪鐮佸櫒瀹归噺鏀瑰彉SAM鍏ㄥ眬姊害鑼冩暟锛屼笉鏄瓑瀹归噺娑堣瀺銆傚垵濮嬪嚱鏁颁繚鎸侊紝鎻掑€?attention鍙嶅悜warn_only璀﹀憡濡?77锛屼笉澹扮О浣嶇骇璁粌澶嶇幇銆?

v182宸叉樉寮忓惎鍔≒ID85420銆乶ormal GPU銆佷复鏃禨YSTEM_REQUIRED銆傚綋鍓嶈缁冨揩鐓'status': 'running', 'pid': 85420, 'stage': 'perm_conditional', 'epoch': 5, 'best': 89.33653883690049, 'best_epoch': 0, 'bad': 5, 'peak_MiB': 786.46533203125}锛宎ctive82720锛岄潪鏈€缁堟敹鐩婏紱max2浠诲姟/瀹屾暣鏂规硶5闃舵/鏈ˉ鎶樸€?
鍒濆PERM89.33653883690049锛孏aussian鍘烻OURCE89.3447601747529锛汣DF鍙樺寲瀵艰嚧鍒濆鍑芥暟涓嶅悓锛屾姤鍛婃渶缁坰core鍜屽鍚勮嚜鍒濆鐨勫彉鍖栵紝涓嶈兘鎶婂垵濮婸DF宸紓褰撲綔璁粌澧炵泭銆?

鍑嗗183淇濈暀浜曞潎琛?SAM鍙皢楠ㄥ共96鈫?28锛屾棤teacher/鏉′欢鍖栵紝涓嶅姞杞?77鏉冮噸锛屾湭璁粌鏃犲垎鏁般€俒璁板綍](results/deep_learning/perm_private_width128_v183_20261001/README.md)銆?

V181 final PERM=88.856289370908, compactcontrolgain=-0.522104840236; actualTOTAL=80.420228997890, fivephases inclsource;strict808=False, no otherfolds. best/stop=27/41, seconds=639.703, peakMiB=826.872; all144665replay/frozen exact.

鏈€缁堟暣浜曟潯浠跺寲鏃犵泭锛歅ERM88.856289370908锛岀浉瀵?77 -0.114601466664锛沚est/stop27/41锛宎ctualTOTAL80.420228997890銆傚垵濮嬪嚱鏁?棰勭畻/loss/SAM/閲囨牱/Q/瑙ｇ爜涓€鑷达紝棰濆鏉′欢缂栫爜鍣ㄥ閲忔敼鍙楽AM姊害鑼冨洿銆佷笉鏄瓑瀹归噺娑堣瀺锛?44665鎭㈠/鍐荤粨楠岃瘉閫氳繃銆傛竻鐞嗚礋鏉冮噸鐣橲HA锛涘叧闂湰128鐐?.5FiLM璁剧疆锛屼笉鎵偣鏁?鐣岄檺锛屼笉娉涘寲鍏ㄩ儴鏁翠簳涓婁笅鏂囥€傚師177淇濈暀锛屼笉琛ユ姌銆?

鍑嗗184Source绉佹湁PERM楠ㄥ共鍗忓悓锛屼繚鐣?74姝AM/CDF/utility锛宧ead3e-4/body1e-5锛岃緭鍑?router鍙傛暟鍐荤粨锛屼笉鍔犺浇166/174澶达紝鏈缁冩棤鍒嗘暟锛?闃舵銆俒璁板綍](results/deep_learning/perm_source_body_sam_v184_20261001/README.md)銆?

V182 final PERM=89.336538836900, compactcontrolgain=-0.041855374243; actualTOTAL=80.588316310987, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=327.517, peakMiB=786.465; all144665replay/frozen exact.

鏈€缁坱5鍒嗗竷鏃犵泭锛歅ERM89.336538836900锛岀浉瀵?74 -0.028447084372锛涜嚜宸卞垵濮?9.336538836900銆佸涔犲鐩?銆乥est/stop0/14锛宎ctualTOTAL80.588316310987銆傚弬鏁?浼樺寲/棰勭畻/Q涓€鑷翠絾鍒濆CDF涓嶅悓銆乴og鍒嗛噺鐭╁尮閰嶏紱PDF/CDF/utility/decoder鍚屾棌锛?44665鎭㈠涓€鑷?鍐荤粨閫氳繃銆傝礋鏉冮噸娓呯悊鐣橲HA锛涗繚鐣橤aussian锛屼笉鎵玠f鎴栨贩鍚堟棌锛岀粨璁轰粎闄愭湰df5璁剧疆銆?

鍏ュ彛183澶辫触锛氶娆¤瘎浼皐ide_forward NameError F鏈鍏ワ紱灏氭湭璁粌銆佹棤鍒嗘暟锛屼笉鏄ā鍨嬭礋缁撴灉锛屽師婧愮爜/config/log鍐荤粨銆?85鏂扮増鏈彧琛瀵煎叆锛屽悓缁撴瀯/鍒濆/棰勭畻锛屼笉缁窇澶辫触杈撳嚭銆?
[185璁板綍](results/deep_learning/perm_private_width128_v185_20261001/README.md)銆?

## 2026-10-01 鐢ㄦ埛纭淇濈暀褰撳墠闃舵鏁?
鐢ㄦ埛绛斿鈥滅畻浜嗭紝灏辫繖鏍峰惂鈥濓細淇濈暀 POR 3 闃舵銆丳ERM 2 闃舵锛涘叡浜?SOURCE 浠呰涓€娆★紝鍏ㄦ柟娉曞悎璁?5 闃舵锛屼笉鍐嶇户缁帇缂┿€傚綋鍓嶅凡鏍搁獙绗簲鎶樻渶浣?TOTAL=80.61247541799393锛岄厤鏂逛负 v154 CURRENT_RECIPE.json锛涢潪瀹屾暣浜旀姌 OOF銆傜幇鏈?v184/v185 璁粌缁х画锛屼笉鏂板浠诲姟銆傛湰娆″疄鏃跺揩鐓э細v184 绗?杞紝鏈€浣?PERM=89.3447601747529锛堢0杞級锛泇185 淇鐗堝凡姝ｅ父璁粌鑷崇10杞紝鏈€浣?PERM=88.45696328292092銆傚潎涓鸿缁冧腑蹇収锛屽皻鏈舰鎴愭柊澧炴渶缁堟敹鐩娿€?


2026-10-01 batch update: atlas refreshed through v185; v181/v182 final negative replay/actual assembly retained, negative weights cleaned; v183 implementation failure has no model score, source frozen; v184/v185 actual live handles verified, no new tasks or remaining folds. Current verified five-stage fifth TOTAL80.612475417994 remains below strict80.8.

- [v186 PERM???????0.3](results/deep_learning/perm_utility_strength_v186_20261001/README.md)??SOURCE????82720????174??CDF/SAM????0.1?0.3?????????????warmstart??5???

V185 final PERM=88.988533638321, compactcontrolgain=-0.389860572823; actualTOTAL=80.466514491485, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=467.565, peakMiB=833.329; all144665replay/frozen exact.


Final width128 PERM=88.988533638321, versus96 control +0.017642800749, best/stop16/30. Actual144665-point assemblyTOTAL80.466514491485; exact best replay/frozen verified. Positive best/last retained, no SOTA or fullOOF claim. Capacity and random hidden initialization differ, initial predictive function and other controls match; not equal compute. Close failed183 entry separately, no score. Single training467.564975s, peak833.329MiB, overlapped184; not full five-phase cost. Gain supports this capacity direction but no automatic width queue.


v186 explicitly launched PID70168, normal GPU guard verified other184 only; temporary SYSTEM_REQUIREDTrue. Initial PERM89.344760174753, active82720, same174 source initialization. Actual utility coefficient0.3, original placeholder SW utility_weight0 remains unused. Training snapshot, no final gain; no remaining folds.

- [v187 ??PERM????192](results/deep_learning/perm_private_width192_v187_20261001/README.md)???185????????SAM/???????128?192???128?????????????????????????

V184 final PERM=89.350304704679, compactcontrolgain=-0.028089506464; actualTOTAL=80.593134364710, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=969.181, peakMiB=1055.937; all144665replay/frozen exact.


Final actual PERM89.350304704679, best/stop10/24; opening body versus matched positive174 -0.014681216593, no body-scope gain. Same-active-capacity166 secondary training-combination gain +0.005544529926, changed SAM/RPS/utility together, not isolated contribution attribution; retain best/last as secondary positive control. ActualTOTAL80.593134364710,144665 exact replay/frozen verified; current154 unchanged. Close this body-update route, no LR/body grid. Single training969.180803s, peak1055.937MiB, staggered185 then186 overlap, not equal-compute or full5-stage time.


v187 explicitly launchedPID58908, normalGPU guard only186otherjob, temporarySYSTEM_REQUIREDTrue. InitialPERM68.38377243696718 same185 predictive priors, active2314801, no downstream/SOURCE weights. Current training snapshot only. CUDA interpolation/attention backward warn_only nondeterminism as177/185, no claim of bit-identical training. No remainingfolds.

- [?????????](results/deep_learning/cross_domain_research_20261001/README.md)????????????????????/??/??????????????????????????

- [v188 PERM???????](results/deep_learning/perm_kernel_score_v188_20261001/README.md)???????????174??/???/??????0.5??????????????????????????5???

V186 final PERM=89.351642639109, compactcontrolgain=-0.026751572035; actualTOTAL=80.593602641760, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=653.065, peakMiB=786.465; all144665replay/frozen exact.


Final utility0.3 PERM89.351642639109 best/stop10/24, versus same-capacity0.1 control174 -0.013343282164; actualTOTAL80.593602641760. Initial prediction/data/budget/Q/SAM/lr/capacity matched;144665 exact best replay/frozen verified. Own initial +0.006882 is not gain over0.1. Close this strength increase, retain0.1 positive control; no weight grid. Negative best/last cleanup authorized with SHA manifest; keep source/config/results. Training653.064628s, peak786.465MiB, staggered184 then187 overlap; not full5phase time.

??????????results/deep_learning/cross_domain_research_20261001/LITERATURE_IDEAS.md?3????/13?????GPU?188???????187??????????


v188 explicitly launchedPID15724, normalGPU guard verified only187otherjob; temporary SYSTEM_REQUIREDTrue. InitialPERM89.344760174753 matches174, active82720 unchanged, fixed kernel buffers do not add trainable parameters. Live state {'status': 'running', 'pid': 15724, 'stage': 'perm_conditional', 'epoch': 4, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 4, 'peak_MiB': 786.46533203125}, training snapshot not final gain.

- [v189????MixStyle](results/deep_learning/perm_mixstyle_v189_20261001/README.md)???????185???/????????????????SAM??????????/???????5???


2026-10-01 batch progress: v189 training-only MixStyle prepared with identical185 capacity/random initialization, no GPU launch; v187 still actually live, late best at epoch39 PERM88.952629531642, versus185 -0.035904106679 (training snapshot, not final). v188 live epoch14 best89.34998331538223, no final kernel gain. Continue existing two jobs; no remaining folds, strict80.8 unmet. Atlas updated with prepared189 and current snapshots.

- [v190??Haar/????29](results/deep_learning/perm_wavelet_detail_v190_20261001/README.md)????????185?????detail DW??Haar?long29????3072??/??????????????????/??????

V187 final PERM=88.952629531642, compactcontrolgain=-0.425764679502; actualTOTAL=80.453948054147, fivephases inclsource;strict808=False, no otherfolds. best/stop=39/53, seconds=1023.044, peakMiB=917.312; all144665replay/frozen exact.

V188 final PERM=89.349983315382, compactcontrolgain=-0.028410895761; actualTOTAL=80.593021878456, fivephases inclsource;strict808=False, no otherfolds. best/stop=10/24, seconds=710.778, peakMiB=786.465; all144665replay/frozen exact.


V187 final PERM=88.952629531642, versus v185 -0.035904106679; actualTOTAL=80.453948054147; best/stop=39/53. All144665 exact restored predictions/frozen verified. Own initial improvement is not mechanism gain; negative best/last cleanup with SHA manifest, source/config/compact result retained, controls intact. Close this192 expansion, retain128, no further width grid. Training=1023.043530s, peak=917.312MiB, actualstaggered2GPU, data/initial/final eval included, no separate warmup/assembly excluded, not complete5phase cost. No remainingfolds.


V188 final kernel PERM=89.349983315382, versus same-capacity174 -0.015002605890; actualTOTAL=80.593021878456; best/stop10/24. All144665 best reload/frozen verified. Own initial+0.005223 not gain over174. Close this finite-support Bartlett kernel score0.1, no bandwidth/coefficient grid; not all kernel scores rejected. Negative best/last cleanup with SHA, positive174 intact. Training=710.777643s, peak=786.465MiB, actualstaggered2GPU, data/initial/final eval included, no separate warmup/assembly excluded, not complete5phase cost. No remainingfolds.


v189 perm_conditional explicitly launchedPID59824, normal GPU/temporary SYSTEM_REQUIREDTrue/guard max2 verified. InitialPERM68.38377243696718 matches185 priors; active1112113; state{'status': 'running', 'pid': 59824, 'stage': 'perm_conditional', 'epoch': 9, 'best': 86.56838007670231, 'best_epoch': 8, 'bad': 1, 'peak_MiB': 833.4072265625} (training snapshot only). CUDA interpolation/attention backward warn_only nondeterminism as baseline; no bitwise trajectory claim. MixStyle adds0parameters, training partner IDs not predictions, SAM replays RNG.


v190 wavelet explicitly launchedPID25524, normal GPU/temporary SYSTEM_REQUIREDTrue/guard max2 verified. InitialPERM68.38377243696718 matches185 priors; active1115185; state{'status': 'running', 'pid': 25524, 'stage': 'wavelet', 'epoch': 5, 'best': 88.17673457945067, 'best_epoch': 5, 'bad': 0, 'peak_MiB': 845.0908203125} (training snapshot only). CUDA interpolation/attention backward warn_only nondeterminism as baseline; no bitwise trajectory claim. Wavelet adds3072 parameters, long29 matched arm remains untrained/unlaunched, no fair pair gain yet.

- [v191 SOURCE??IQF32](results/deep_learning/perm_quantile_readout_v191_20261001/README.md)??82720??Gaussian???????+??????CDF??????SAM????????????/noGPUlaunch??5????shell?????????????????????????????

V190 wavelet final PERM=88.935724895823, compactcontrolgain=-0.442669315320; actualTOTAL=80.448031431610, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=625.707, peakMiB=845.091; all144665replay/frozen exact.

V189 final PERM=88.465433532158, compactcontrolgain=-0.912960678986; actualTOTAL=80.283429454328, fivephases inclsource;strict808=False, no otherfolds. best/stop=45/45, seconds=815.547, peakMiB=833.407; all144665replay/frozen exact.


Final MixStyle early-screen PERM88.465433532158, versus185 -0.523100106163, actualTOTAL80.283429454328; best/stop45/45. Early request observed42/best36/bad6; before boundary processing candidate improved at45, reported exactly rather than hidden. Same capacity/init/data/loss/SAM/sampling/max80/pat14, but evidence-based termination45 differs from natural control stopping; not proof of whole80 behavior or universal style failure.144665 exact reload/frozen verified. Closep.5/alpha.1 first-local cross-well stats only, no mixing-strength grid; negative checkpoints cleaned with SHA,185 positive retained. Training815.546929s/peak833.407MiB, data/start/final prediction included, staggered actualmax2GPU, not full5phase cost.


Wavelet arm final PERM88.935724895823 best16/stop30, actualTOTAL80.448031431610, exact restored144665/frozen verified; versus original185 -0.052808742498 (capacity differs). Long29 matched control is now active, pair incomplete, no frequency attribution yet. Preserve wavelet best/last until fair pair evidence; no completed pair gain. Single625.707167s/peak845.091MiB, overlaps189, not full pair or fivephase cost.


v190 long29 explicitly launchedPID91652, normalGPU/guard max2/temporary SYSTEM_REQUIREDTrue verified, active1115185, initialPERM68.383772436967; current{'status': 'running', 'pid': 91652, 'stage': 'long29', 'epoch': 14, 'best': 88.69633347496752, 'best_epoch': 12, 'bad': 2, 'peak_MiB': 833.38134765625} is training snapshot, not final mechanism gain. Wavelet matched arm already finalized88.935724895823; pair still incomplete, same active1115185 and same output priors; fair pair comparison pending.


v191 iqf explicitly launchedPID67908, normalGPU/guard max2/temporary SYSTEM_REQUIREDTrue verified, active82720, initialPERM89.345246688523; current{'status': 'running', 'pid': 67908, 'stage': 'iqf', 'epoch': 5, 'best': 89.34524668852337, 'best_epoch': 0, 'bad': 5, 'peak_MiB': 786.46533203125} is training snapshot, not final mechanism gain. SOURCE original89.344760174753 initial representation delta+0.000486513770, not learned improvement. Inherited Gaussian diagnostic keys describe old SOURCE/old parameterization; actual new output isIQF32, no new component logweight correction. Target-space exponential tails have finite z moments, physical PERM moments not asserted. Same-step cache reset before each optimizer step and eval.

Correction to progress narration: actual MixStyle final delta versus185 is -0.523100106163, not earlier mental arithmetic -0.523457. Official control_delta.json/assembly remain authoritative.

V190 long29 final PERM=88.953509395268, compactcontrolgain=-0.424884815876; actualTOTAL=80.454256006416, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=547.440, peakMiB=833.381; all144665replay/frozen exact.

V191 iqf final PERM=89.345246688523, compactcontrolgain=-0.033147522620; actualTOTAL=80.591364059055, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=495.978, peakMiB=786.465; all144665replay/frozen exact.

- [v192 matched multi-order add/multiply](results/deep_learning/perm_multiorder_interaction_v192_20261001/README.md): prepared; random private128 with positive SAM/well-sampling, no trained185 weights, pair gain unverified, five phases.

- [v192 matched multi-order add/multiply](results/deep_learning/perm_multiorder_interaction_v192_20261001/README.md): prepared; random private128 with positive SAM/well-sampling, no trained185 weights, pair gain unverified, five phases.

FINAL paired result: wavelet PERM88.935724895823, long29 88.953509395268; frequency decomposition minus exactly matched long kernel -0.017784499445. Correct wavelet-versus185 arithmetic -0.052808742498, supersedes earlier manually transcribed -0.052808742498. Both final16/30,144665 exact reload/frozen, same1115185 params/initial function/data/loss/sampling/budget; no wavelet gain. Close this two-level first-detail setting only, no level/grid, neither arm improves185. Actual long29 TOTAL80.454256006416; original154 retained. Training long29 547.440191s/peak833.381348MiB; not full5phase cost.

FINAL paired result: wavelet PERM88.935724895823, long29 88.953509395268; frequency decomposition minus exactly matched long kernel -0.017784499445. Correct wavelet-versus185 arithmetic -0.052808742498, supersedes earlier manually transcribed -0.052808742498. Both final16/30,144665 exact reload/frozen, same1115185 params/initial function/data/loss/sampling/budget; no wavelet gain. Close this two-level first-detail setting only, no level/grid, neither arm improves185. Actual long29 TOTAL80.454256006416; original154 retained. Training long29 547.440191s/peak833.381348MiB; not full5phase cost.

FINAL IQF32 PERM89.345246688523, best0/stop14, learned gain0; minus Gaussian174 -0.019739232749. Initial representation delta +0.000486513770 is not learning. Actual TOTAL80.591364059055,144665 exact reload/frozen. Close this 32-node piecewise-linear/exponential-tail SOURCE-conditioned representation with existing NLL/RPS/utility/SAM only; not all quantile functions. Training495.978274s/peak786.465332MiB, no full5phase time.

FINAL IQF32 PERM89.345246688523, best0/stop14, learned gain0; minus Gaussian174 -0.019739232749. Initial representation delta +0.000486513770 is not learning. Actual TOTAL80.591364059055,144665 exact reload/frozen. Close this 32-node piecewise-linear/exponential-tail SOURCE-conditioned representation with existing NLL/RPS/utility/SAM only; not all quantile functions. Training495.978274s/peak786.465332MiB, no full5phase time.

Sequence expansion complete: [SOFTS/Sonnet/CATS/xLSTM source-backed ideas](results/deep_learning/cross_domain_research_20261001/SEQUENCE_IDEAS.md). Proposed migrations and cost estimates unverified. v192 add/multiply explicitly launched; no automatic research queue.

v192 add/multiply both live and GPU guard max2 verified; normal/temporarySYSTEM_REQUIREDTrue. Identical1162673 active params (+50560 versus185), original same predictive initial68.38377243696718; common core and zero-residual initialization, not bitwise CUDA trajectory. Same seed/data/loss/SAM0.05/sampling/80-14, one replacement PERM phase/whole recipe5. Launch snapshots only, no mechanism gain yet.

v192 add/multiply both live and GPU guard max2 verified; normal/temporarySYSTEM_REQUIREDTrue. Identical1162673 active params (+50560 versus185), original same predictive initial68.38377243696718; common core and zero-residual initialization, not bitwise CUDA trajectory. Same seed/data/loss/SAM0.05/sampling/80-14, one replacement PERM phase/whole recipe5. Launch snapshots only, no mechanism gain yet.

Finalized negatives v190 both arms/v191 IQF: six best/last checkpoints deleted with SHA manifests after exact replay and actual assembly. Preserve positive185/174/current154 and all active192 outputs. v190 paired total launch-to-report wall1330.530043s includes inter-arm gap and overlap other studies, not dedicated benchmark/full5phase time.

Finalized negatives v190 both arms/v191 IQF: six best/last checkpoints deleted with SHA manifests after exact replay and actual assembly. Preserve positive185/174/current154 and all active192 outputs. v190 paired total launch-to-report wall1330.530043s includes inter-arm gap and overlap other studies, not dedicated benchmark/full5phase time.

- [v193 SOURCE-referenced adaptive mass](results/deep_learning/perm_adaptive_mass_v193_20261001/README.md): prepared/untrained, proper bounded density transport; no GPU launch while192both live, five stages.

- [v193 SOURCE-referenced adaptive mass](results/deep_learning/perm_adaptive_mass_v193_20261001/README.md): prepared/untrained, proper bounded density transport; no GPU launch while192both live, five stages.

Verified live v192 add/multiply process cmdlines, snapshots {"add": {"status": "running", "pid": 85080, "stage": "add", "epoch": 17, "best": 88.88500972902482, "best_epoch": 16, "bad": 1, "peak_MiB": 867.146484375}, "multiply": {"status": "running", "pid": 13760, "stage": "multiply", "epoch": 12, "best": 88.58452172937969, "best_epoch": 8, "bad": 4, "peak_MiB": 874.61181640625}}; results interim only. v193 adaptive probability mass prepared and registered with own/source density plus exactJacobian, same82720 capacity and original decoder, not launched while max2 slots occupied. Whole5phase best80.612475 unchanged, originalOOF>82 still unmet. No automatic remainingfolds or queue.

Verified live v192 add/multiply process cmdlines, snapshots {"add": {"status": "running", "pid": 85080, "stage": "add", "epoch": 17, "best": 88.88500972902482, "best_epoch": 16, "bad": 1, "peak_MiB": 867.146484375}, "multiply": {"status": "running", "pid": 13760, "stage": "multiply", "epoch": 12, "best": 88.58452172937969, "best_epoch": 8, "bad": 4, "peak_MiB": 874.61181640625}}; results interim only. v193 adaptive probability mass prepared and registered with own/source density plus exactJacobian, same82720 capacity and original decoder, not launched while max2 slots occupied. Whole5phase best80.612475 unchanged, originalOOF>82 still unmet. No automatic remainingfolds or queue.

- [v194 local nine-curve core](results/deep_learning/perm_curve_core_v194_20261001/README.md): prepared/untrained, matched mean/weighted STAR-style aggregation; retain206/fullresolution, noGPUlaunch or queue, five stages.

- [v194 local nine-curve core](results/deep_learning/perm_curve_core_v194_20261001/README.md): prepared/untrained, matched mean/weighted STAR-style aggregation; retain206/fullresolution, noGPUlaunch or queue, five stages.

V192 add final PERM=88.885009729025, compactcontrolgain=-0.493384482119; actualTOTAL=80.430281123231, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=554.660, peakMiB=867.146; all144665replay/frozen exact.

v193 adaptive explicitly launchedPID77412, normalGPU guard found only192multiply, temporarySYSTEM_REQUIREDTrue. Active82720. InitialPERM89.344788677861, minusSOURCE+0.000028503108; algebraic uniform identity does not yield exactly identical measured decoding, no initial difference counted as learning. Possible finite-precision/readout implementation variation is inference, not proven cause. Same-SAM-step cache cleared eachstep/eval. Source/config hashes now frozen; no automatic retries. Snapshot {'status': 'running', 'pid': 77412, 'stage': 'adaptive', 'epoch': 1, 'best': 89.34478867786068, 'best_epoch': 0, 'bad': 1, 'peak_MiB': 786.46533203125}, no final mechanism gain.

v193 adaptive explicitly launchedPID77412, normalGPU guard found only192multiply, temporarySYSTEM_REQUIREDTrue. Active82720. InitialPERM89.344788677861, minusSOURCE+0.000028503108; algebraic uniform identity does not yield exactly identical measured decoding, no initial difference counted as learning. Possible finite-precision/readout implementation variation is inference, not proven cause. Same-SAM-step cache cleared eachstep/eval. Source/config hashes now frozen; no automatic retries. Snapshot {'status': 'running', 'pid': 77412, 'stage': 'adaptive', 'epoch': 1, 'best': 89.34478867786068, 'best_epoch': 0, 'bad': 1, 'peak_MiB': 786.46533203125}, no final mechanism gain.

v192 add final PERM88.885009729025, actual TOTAL80.430281123231, best16/stop30, training554.660232s/peak867.146484MiB,144665 exact replay/frozen. Multiply still live; preserve both pair checkpoints until fair final comparison. No pair gain/other folds;185 and current154 unchanged.

v192 add final PERM88.885009729025, actual TOTAL80.430281123231, best16/stop30, training554.660232s/peak867.146484MiB,144665 exact replay/frozen. Multiply still live; preserve both pair checkpoints until fair final comparison. No pair gain/other folds;185 and current154 unchanged.

## Latest user: retain proven mechanisms, small-module or additive changes
Keep verified best architecture/recipe/weights. Subsequent trials retain positive mechanisms and change only one small module or add a residual/auxiliary term; broad readout/backbone replacement is held. Current192multiply/193adaptive may complete already-authorized fixed comparisons unchanged, max2. Prepared194 random-private curve-core pair is HELD, no launch under this direction. New additive experiments must count all training phases honestly; do not load trained154 and call another stage free. Source-only initialization with same effective architecture plus one change can replace the existing head phase and remain5 stages. Original splits/Q/scoring/80-14/promotion strict80.8/OOF>82 preserved.

## Latest user: retain proven mechanisms, small-module or additive changes
Keep verified best architecture/recipe/weights. Subsequent trials retain positive mechanisms and change only one small module or add a residual/auxiliary term; broad readout/backbone replacement is held. Current192multiply/193adaptive may complete already-authorized fixed comparisons unchanged, max2. Prepared194 random-private curve-core pair is HELD, no launch under this direction. New additive experiments must count all training phases honestly; do not load trained154 and call another stage free. Source-only initialization with same effective architecture plus one change can replace the existing head phase and remain5 stages. Original splits/Q/scoring/80-14/promotion strict80.8/OOF>82 preserved.

- [v195 one additive CRPS term](results/deep_learning/perm_additive_crps_v195_20261001/README.md): retains proven154 architecture/NLL/jointRPS, no addedparams, SOURCEinit/five stages; prepared/untrained.194held per latestuser.

- [v195 one additive CRPS term](results/deep_learning/perm_additive_crps_v195_20261001/README.md): retains proven154 architecture/NLL/jointRPS, no addedparams, SOURCEinit/five stages; prepared/untrained.194held per latestuser.

V192 multiply final PERM=88.892166252243, compactcontrolgain=-0.486227958901; actualTOTAL=80.432785906357, fivephases inclsource;strict808=False, no otherfolds. best/stop=16/30, seconds=574.888, peakMiB=874.612; all144665replay/frozen exact.

FINAL v192 paired: multiply PERM88.892166252243 minus add88.885009729025=+0.007156523218, actual multiplyTOTAL80.432785906357. Same1162673 params/initialhidden/priors/loss/Q/sampling/SAM/LR/80-14; both best16/stop30 and144665 exact reload/frozen actual3head assembly. Local mechanism-positive singlefold/singleseed; not significance claim, both below185 and154. Keep multiply best/last as localpositive, hold furtherprivate replacements under latest user small/additive direction; keep original154 best. Add negative checkpoints eligible cleanup after fullpair proof, source/config/preds/delta preserved. Pair totalwall649.418650s actualstaggered2GPU/overlap193, trainingadd554.660232/multiply574.887780s, not equalcompute/full5phase.

v195 explicit launchPID87044, normalGPU guard only193otherjob, temporarySYSTEM_REQUIREDTrue. Same82720 params/originalGaussian output/NLL/BCE/jointRPS; only analyticCRPS0.1 added, no new module or backbone replacement. InitialPERM89.3447601747529 exactly sameSOURCE/control165adamw score; learned gain not shown yet. Snapshot {'status': 'running', 'pid': 87044, 'stage': 'perm_conditional', 'epoch': 3, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 3, 'peak_MiB': 786.46533203125}, original154 positive weights/delivery unchanged; five phases and no otherfolds.

v195 explicit launchPID87044, normalGPU guard only193otherjob, temporarySYSTEM_REQUIREDTrue. Same82720 params/originalGaussian output/NLL/BCE/jointRPS; only analyticCRPS0.1 added, no new module or backbone replacement. InitialPERM89.3447601747529 exactly sameSOURCE/control165adamw score; learned gain not shown yet. Snapshot {'status': 'running', 'pid': 87044, 'stage': 'perm_conditional', 'epoch': 3, 'best': 89.3447601747529, 'best_epoch': 0, 'bad': 3, 'peak_MiB': 786.46533203125}, original154 positive weights/delivery unchanged; five phases and no otherfolds.
v192 add negative best/last deleted after full paired replay/assembly/control_delta with SHA manifest; multiply localpositive best/last retained, all original154/185 controls and active193/195 unchanged.

- [v196 small additive curve core](results/deep_learning/perm_local_core_patch_v196_20261001/README.md): retains154 entire effective readout/loss and frozenSOURCE; one small stem residual, mean/weighted matchedpair; prepared/untrained, noGPUlaunch yet.194held.

- [v196 small additive curve core](results/deep_learning/perm_local_core_patch_v196_20261001/README.md): retains154 entire effective readout/loss and frozenSOURCE; one small stem residual, mean/weighted matchedpair; prepared/untrained, noGPUlaunch yet.194held.

Latest user explicitly raises total training stage cap to8, superseding5. Prioritize freezing learned positive functions with small additive repairs. New197 warm154+one patch honestly6stages, ownPERM loss/fixedscore/max2/80-14. Existing sources/deliveries unchanged. Prepared196 source-retrained variant HELD in favor of197 preserving learned154 weights; no automatic queue/remainingfolds.

Latest user explicitly raises total training stage cap to8, superseding5. Prioritize freezing learned positive functions with small additive repairs. New197 warm154+one patch honestly6stages, ownPERM loss/fixedscore/max2/80-14. Existing sources/deliveries unchanged. Prepared196 source-retrained variant HELD in favor of197 preserving learned154 weights; no automatic queue/remainingfolds.

V195 final PERM=89.344760174753, compactcontrolgain=-0.033634036391; actualTOTAL=80.591193779236, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=211.378, peakMiB=786.465; all144665replay/frozen exact.

V193 adaptive final PERM=89.344788677861, compactcontrolgain=-0.033605533283; actualTOTAL=80.591203755324, fivephases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=342.805, peakMiB=786.465; all144665replay/frozen exact.

## Latest user: each output head may add at most one stage
Use verified five-stage154 combined recipe as baseline. POR current SOURCE+113+151 may add one; PERM current SOURCE+154 may add one; SW104 may add one. SOURCE shared once; whole method at most8. Warm154+197 is the only added PERM stage, whole6. Mean/weighted candidates are parallel alternatives from154, never sequential197mean->197weighted or197->another trainedrepair. Future PERM attempts may be siblings from original154; adopted chain only one addition. POR/SW each retain one possible addition. Do not conceal retraining/refitting as no stage, no autoqueue/otherfolds; strict80.8+gain/fullOOF82/max2/80-14 unchanged.

## Latest user: each output head may add at most one stage
Use verified five-stage154 combined recipe as baseline. POR current SOURCE+113+151 may add one; PERM current SOURCE+154 may add one; SW104 may add one. SOURCE shared once; whole method at most8. Warm154+197 is the only added PERM stage, whole6. Mean/weighted candidates are parallel alternatives from154, never sequential197mean->197weighted or197->another trainedrepair. Future PERM attempts may be siblings from original154; adopted chain only one addition. POR/SW each retain one possible addition. Do not conceal retraining/refitting as no stage, no autoqueue/otherfolds; strict80.8+gain/fullOOF82/max2/80-14 unchanged.

FINAL v193 PERM89.344788677861, best0/stop14, learned gain0; control delta-0.020197243412, actual TOTAL80.591203755324,144665 exact replay/frozen realassembly. Close this bounded adaptive transport only, not allAdaCat. Training342.804890s/peak786.465332MiB, actualstaggered193/195 overlap, data/initial/final eval included, standaloneassembly excluded/no separatewarmup; not full5phase/equalcompute cost. Negative best/last eligiblecleanup with SHA; source/config/evidence preserved.

FINAL v195 PERM89.344760174753 best0/stop14, learnedgain0, matched165AdamW delta0; actualTOTAL80.591193779236,144665 exact replay/frozen. Added analyticCRPS0.1 no gain at80-14; original jointRPS benefit154 retained, no coefficient grid. Training211.377959s/peak786.465332MiB, actualstaggered193/195 overlap, data/initial/final eval included, standaloneassembly excluded/no separatewarmup; not full5phase/equalcompute cost. Negative best/last eligiblecleanup with SHA; source/config/evidence preserved.

v197 mean/weighted verified live; both start exact learned154 PERM89.405564856919, only20056 newpatch params. Whole6stages, per-target addition counts POR0/PERM1/SW0 and max8. Same base learned weights frozen and extra initial tensors, max2 normalGPU/temporarySYSTEM_REQUIREDTrue. Pair not complete/no learnedgain yet; only one candidate may be adopted, no sequentialpatch warmstart.193/195 negative four best/last deleted after full144665 replay/assembly with SHA manifests, positive154/192multiply/currentdelivery/active197 preserved.

v197 mean/weighted verified live; both start exact learned154 PERM89.405564856919, only20056 newpatch params. Whole6stages, per-target addition counts POR0/PERM1/SW0 and max8. Same base learned weights frozen and extra initial tensors, max2 normalGPU/temporarySYSTEM_REQUIREDTrue. Pair not complete/no learnedgain yet; only one candidate may be adopted, no sequentialpatch warmstart.193/195 negative four best/last deleted after full144665 replay/assembly with SHA manifests, positive154/192multiply/currentdelivery/active197 preserved.

- [v198 frozen best + scale-only patch](results/deep_learning/perm_frozen_scale_patch_v198_20261001/README.md): retains154 allweights, only4656newparams/rawscale+-0.1; alternative addedPERMstage from154, not197stack, whole6. Prepared/untrained.

- [v198 frozen best + scale-only patch](results/deep_learning/perm_frozen_scale_patch_v198_20261001/README.md): retains154 allweights, only4656newparams/rawscale+-0.1; alternative addedPERMstage from154, not197stack, whole6. Prepared/untrained.

V197 mean final PERM=89.405564856919, compactcontrolgain=+0.027170645775; actualTOTAL=80.612475417994, sixphases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=137.915, peakMiB=786.465; all144665replay/frozen exact.

- [v199 frozen SW positive + small density repair](results/deep_learning/sw_frozen_density_patch_v199_20261001/README.md): only4656newparams, original104 fixed/NLL+utility/SAM retained, one SWaddition/whole6; prepared/untrained.

- [v199 frozen SW positive + small density repair](results/deep_learning/sw_frozen_density_patch_v199_20261001/README.md): only4656newparams, original104 fixed/NLL+utility/SAM retained, one SWaddition/whole6; prepared/untrained.

V197 weighted final PERM=89.405564856919, compactcontrolgain=+0.027170645775; actualTOTAL=80.612475417994, sixphases inclsource;strict808=False, no otherfolds. best/stop=0/14, seconds=153.217, peakMiB=786.465; all144665replay/frozen exact.

V198 scale final PERM=89.405831181353, compactcontrolgain=+0.027436970209; actualTOTAL=80.612568631546, sixphases inclsource;strict808=False, no otherfolds. best/stop=1/15, seconds=126.469, peakMiB=786.465; all144665replay/frozen exact.

- [v200 frozen POR151 + scale-only patch](results/deep_learning/por_frozen_scale_patch_v200_20261001/README.md): only8752newparams, learned151/113/SOURCE fixed, one PORaddition/whole6; combinedPERM1987/withSW1998, eachheadone. Prepared/untrained.

- [v200 frozen POR151 + scale-only patch](results/deep_learning/por_frozen_scale_patch_v200_20261001/README.md): only8752newparams, learned151/113/SOURCE fixed, one PORaddition/whole6; combinedPERM1987/withSW1998, eachheadone. Prepared/untrained.

Verified current research best six-stage TOTAL80.612568631546, versus five-stage154+0.000093213552, PERM+0.000266324434. Tiny singlefold/singleseed gain, not significance or promotion; best1/stop15/126.468553s/786.465MiB, all144665 actualassembly/reload/frozenverified. New198CURRENT_RECIPE preserves old154/deliveries; keep positive198best/last. Source-counted6, addedhead POR0/PERM1/SW0, max8. FuturePERM candidates siblings from154 only, not198warmstack.

FINAL197 mean/weighted eachPERM89.405564856919 best0/stop14; learnedgain0 and pairdelta0, actualTOTAL80.612475417994 forboth, all144665 reload/frozen/assembly exact. Small mean/weighted localcurve patch no gainedrepair, preserve base154; close patch16/series16/core8 setting, no grid/universalSOFTSclaim. Both negativeweights eligiblecleanup withSHA; keep source/config/pred/evidence. Trainingmean137.915155/weighted153.217394s; totalwall222.931725s actualstaggered2, not full6/equalcompute.

Latest user prioritizes simplePORstageablation and no unnecessaryextra stages. HoldPORscale200 (entry failed wrongparentfolder beforeGPU admission/firstbatch, no score, preservefailure source/log/status, no automaticrepair/retry). Default practicalrecipe remains five-stage15480.612475417994; optional198sixstage80.612568631546 retained as micropositive, not necessarydefaultphase. New201source/prefix/directskipmiddle ablation, noaddedPORphase; othernewPORaugmentationheld.

Latest user prioritizes simplePORstageablation and no unnecessaryextra stages. HoldPORscale200 (entry failed wrongparentfolder beforeGPU admission/firstbatch, no score, preservefailure source/log/status, no automaticrepair/retry). Default practicalrecipe remains five-stage15480.612475417994; optional198sixstage80.612568631546 retained as micropositive, not necessarydefaultphase. New201source/prefix/directskipmiddle ablation, noaddedPORphase; othernewPORaugmentationheld.

- [v201 simplePOR stage ablation](results/deep_learning/por_stage_ablation_v201_20261001/README.md): Sourceonly/prefix2/full3/directskipmiddle2, fixedotherheads; oldpredsreused, one???train only, noextraPORstages.

- [v201 simplePOR stage ablation](results/deep_learning/por_stage_ablation_v201_20261001/README.md): Sourceonly/prefix2/full3/directskipmiddle2, fixedotherheads; oldpredsreused, one???train only, noextraPORstages.

V199 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds158.374369/peakMiB957.755859. No otherfolds.

Read-only prefix ablation, fixed PERM154/SW104: [{"name": "source_only", "POR_training_stages": 1, "total_training_stages": 3, "scores": {"POR": 77.27786784460355, "PERM": 89.40556485691894, "SW": 74.12175282815758, "TOTAL": 80.41792154315785}, "delta_vs_full_POR": -0.6485129161203105, "delta_vs_full_TOTAL": -0.19455387483608888}, {"name": "source_113", "POR_training_stages": 2, "total_training_stages": 4, "scores": {"POR": 77.54045432673898, "PERM": 89.40556485691894, "SW": 74.12175282815758, "TOTAL": 80.49669748779847}, "delta_vs_full_POR": -0.3859264339848778, "delta_vs_full_TOTAL": -0.11577793019546334}, {"name": "source_113_151", "POR_training_stages": 3, "total_training_stages": 5, "scores": {"POR": 77.92638076072386, "PERM": 89.40556485691894, "SW": 74.12175282815758, "TOTAL": 80.61247541799393}, "delta_vs_full_POR": 0.0, "delta_vs_full_TOTAL": 0.0}]
Original SOURCE uses original T3 family; later stages use mixed T3/Cauchy. These are actual prefix recipes, not a pure family-matched mechanism ablation. Direct skip-middle training is separate.

2026-10-01 user reiterates max2 concurrent GPU jobs and do not stop other training for ablation. Prior PERM/SW jobs naturally complete. V201 direct skip-middle PID29320 running, no forced stops. Read-only SOURCE evaluation completed alongside it with guard reporting exactly one other GPU job; now one training live. Default remains five phases; optional198 tiny gain does not justify automatic extra phase. V199 has no gain. Prefix actual TOTAL 1/2/3 POR stages=80.417921543158/80.496697487798/80.612475417994. No remainingfolds or automatic launches.


V201 final POR77.415345964852, gain-0.511034795872, actualTOTAL80.459164979232; fourphases inclsource/two POR stages, strict808=False; all144665replay/frozenexact, best/stop1/15, trainingseconds155.154987/peakMiB924.547363. No otherfolds.

V201 direct skip-middle completed best1/stop15 POR77.415345964852; lower than11377.540454 and15177.926381. All prefix recipes verified, keep three POR phases. V202 lower LR3e-5 scale-only4656patch explicitly launched PID53300; same154 parent, no198 weights; one optional PERM stage replaces198, not stacks. No GPU job stopped; peak concurrent two, one now live. 197/199/201 negative checkpoints removed with SHA manifests after full144665 replay/assembly. Goal fullOOF>82 still unmet; no remainingfolds.


V203 explicit POR scale-only8752 repair launch PID50216, fresh corrected151 joint_rps path, LR3e-5; failure200 remains immutable/no GPU result. V202 PERM4656 scale LR3e-5 PID53300 running independently; max2 GPU jobs, no original function replaced, no sequentialrepair stacking. Each candidate alone six phases, combining only if proven at mostseven. V201 closed negative and default remains original five phases.


V202 scale final PERM=89.406833212247, matched198gain=+0.001002030895; actualTOTAL=80.612919342359, sixphases inclsource;strict808=False, no otherfolds. best/stop=2/16, seconds=168.443, peakMiB=786.465; all144665replay/frozen exact.

V204 failed after GPU admission during model construction due inherited capture method name collision, before epochs/initial score; no efficacy result. Preserve source/traceback. V205 fresh source only renames capture_context, same planned module/loss/data, explicit launch, not queue/retry of frozen file.


V203 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds180.127673/peakMiB921.752930. No otherfolds.

V202 restored PERM89.406833212247, +0.001268355328 vs154 and +0.001002030895 vs198; actualTOTAL80.612919342359, optional6phase recipe retained, not default or fullOOF. V205 context patch verified initial89.405564856919/6992parameters; admitted alongside203, at most2GPU. PyTorch warns adaptive pooling backward nondeterministic under existing warn_only setting: inference/replay checks remain required, no exact training-trajectory reproducibility claim. V204 construction failed no epochs/score, source retained. No remainingfolds.


V203 completed POR77.926380760724 best0/14, no learned gain, restored/frozen/replay/actualassembly done. Negative best/last removed with SHA manifest; no more scale-only scans on this POR setting. Keep151 effective stages and source. V205 remains running, early PERM89.408616231012 is not final.


V205 attention final PERM=89.408616231012, matched198gain=+0.002785049659; actualTOTAL=80.613543398926, sixphases inclsource;strict808=False, no otherfolds. best/stop=1/15, seconds=151.016, peakMiB=786.465; all144665replay/frozen exact.

V207 prepared then explicitly launched as one same-architecture lowerLR comparison; control205 restored PERM89.408616231012 +0.003051374093 vs154. Both independently warm154, not205weights; six overall phases if adopted, never sequential repair. Only LR3e-5 changes, original adaptive pool warning remains disclosed, no exact trajectory reproducibility claim. Not yet proved gain; no wider grid automatic.


V205 actual assembly completed TOTAL80.613543398926/PERM89.408616231012, +0.003051374093 vs154. Retain positive best/last and optional6phase recipe; default5 because gain tiny, strict808false. V206 local feature adapter running PID81288; V207 same205 architecture lowerLR3e-5 running PID84032, exactly twoGPU admitted. No source changes to old trained files; oneextra perhead only, candidates not stacked. Goal fullOOF>82 unmet.


V206 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds206.650378/peakMiB922.250977. No otherfolds.

V207 attention final PERM=89.408649040241, matched205gain=+0.000032809230; actualTOTAL=80.613554882157, sixphases inclsource;strict808=False, no otherfolds. best/stop=1/15, seconds=185.537, peakMiB=786.465; all144665replay/frozen exact.

V208 startup failed after GPU/data admission before epochs due manually mistyped parent SHA constant; actual104 diskSHA matches original199/plan07e0...f58670. Failure/source preserved, no efficacy score. V209 fresh source corrects only SHAconstant; independent explicitlaunch, no active trainedsource mutation.


V206 closed no gain POR77.926380760724 best0/14,206.650378s/922.251MiB; exactreplay/frozen/actualassembly done, negativebest/last removed withSHAmanifest. V207 restored PERM89.408649040241, matched205+0.000032809230 tiny; optional6 recipe retained, default5 unchanged and no morecontextLRsweep. V208 mistypedSHA failedbeforeinitial/epochs, preserved. V209 correctedmasssource explicitlylaunchPID37104, initSW74.121752689769 differs104 by-0.000000138389 fromfloor/normalization; initialreported separately, 12576newparams, no initialgainclaim. V210 localfeatureadapter explicitPID44400 independentlywarm104; alternatives209/210 neverstack, oneextraSWphase. Max2GPU, no originaleffectiveweight mutation, no remainingfolds/fullOOF>82 unachieved.


SW209/210 confirmed live, no forcedstop. Prepared211 fairutility alpha0vs1 ablation and212 finitebound2vs0.5 expansion, same12576capacity and104initialfunction, independentnew directories. NoGPU launch/automaticqueue for211/212 while2jobsactive. Only at mostone adoptedSW repair, sourcecounted6overall alone; default5, fullOOF>82unmet.


V210 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds360.206421/peakMiB1018.911621. No otherfolds.

V210 finished SW74.121752828158 best0/14 no learnedgain,360.206421s/1018.912MiB; all144665 replay/frozen/actualassembly, negativebest/lastdeleted withSHAmanifest. No further samefeatureadapter sweep. V211 matching utilityalpha0ablation explicitlylaunch PID88568,209live unchanged; max2GPU, oneSWstage alternatives not stacking. V212 bound2 prepared/no automaticqueue. Default5 maintained, fullOOF>82 not achieved.


V209 final SW74.142184487495, gain+0.020431659337, actualTOTAL80.619626498762; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds448.372218/peakMiB908.493652. No otherfolds.


V209 complete best9/stop23 SW74.142184487495; learnedvsinitial+0.020431797726, vsoriginal104+0.020431659337;448.372218s/908.494MiB, exact144665 frozen/replay/actualassembly. V211 independent utilityalpha0 PID88568 running. V212 finitebound2 matchedcomparison explicitly launched PID21932, original104 warmstart/no209weights; now twoGPU, neither trial stacksstages. Keep209 positive models, default5 remains; no remainingfolds/fullOOF>82 unmet.


V209 actual optional6 TOTAL80.619626498762; actual optional7 (PERM207+SW209) TOTAL80.620705962925, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.

Prepared214 backbone-side rank8 update tofirstglobalSW attentionQKV,3072newparams andall104oldweights/readout retained; noGPUlaunch/queue while211 and212 live. No rank/layer grid planned. CurrentSWhead trials remainrunning undermax2; fullOOF>82 unmet and remainingfolds notstarted.


V211 final SW74.147622622627, gain+0.025869794469, actualTOTAL80.621529846058; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds551.402990/peakMiB908.493652. No otherfolds.

V212 final SW74.144913000682, gain+0.023160172525, actualTOTAL80.620581478378; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds542.029938/peakMiB908.493652. No otherfolds.

V211 restored SW74.147622622627 +0.025869794469 vs104, +0.005438135132 vs same209 utility1; best10/stop24,551.40299s/908.494MiB. V212 bound2 SW74.144913000682, +0.002728513187 vs209; best9/stop23,542.029938s/908.494MiB. Both actual144665 frozen/restored/assembly done, positives kept, no broaderutility/boundgrid. V214 first-attention rank8 backbone repair explicitPID60644; V213 posterior view explicitPID86544, repeatmatchedcontrol prepared/noqueue; at most2GPU and oneextraSWstage alternatives. Default5 retained, no remainingfolds/fullgoal>82 unmet. V214 interpolation/attention backward warnings mean no exact trajectory reproducibility claim; inference/replay checks remain required.


V211 actual optional6 TOTAL80.621529846058; actual optional7 (PERM207+SW209) TOTAL80.622609310221, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.

Prepared215 complementary input-hidden view from originalsharedSOURCE (no outputpredictions used), all104/SOURCEweightsfrozen; repeat/source modes21280sameparams. OriginalSOURCEalreadycountedonce, no additionalsource trainingphase. NoGPUlaunch/queue while213posterior and214rank8live; default5 unchanged, fullgoal>82unmet.


V214 final SW74.124181854091, gain+0.002429025933, actualTOTAL80.613325577071; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/15, trainingseconds341.289236/peakMiB1255.336914. No otherfolds.

V213 posterior final SW74.134546961565, actualTOTAL80.616953364687; learned gain+0.012794271796, vs211-0.013075661062. Six optionalphases, default5; no fair viewgainclaim until repeat/posterior paircomplete.

V214 completed SW74.124181854091 +0.002429025933 vs104 best1/15,341.289236s/1255.337MiB; actual144665assembly/frozen/restore done; localpositive retained but no layer/rankgrid. V213 posterior completed SW74.134546961565 +0.012794133407 vs104, -0.013075661062 vs211, best5/19,371.503420s/1069.142MiB; parameter-matchedrepeat pending. Explicit213repeat PID84596 and215source PID54776 nowlaunched, twoGPUmax, noextraSOURCEtraining/no predictionvotes. AllcandidatesoneSWrepair alternative, default5 remains/fullOOF>82 unmet.


V215 failed during initial evaluator before initialscore/epochs: SOURCE cosine patch requires original256window, full1024 unsupported. Failure/source preserved, not negativeefficacy. V216 fresh source uses only central256 original SOURCE window around sameanchor; fills256hidden side into current1024tensor but loss/evaluation select central128 only. SW original1024context/bins/actions unchanged. Sourcecomputed inbotharms. No additionalsource training/queue.


V216 Source feature encoder uses original256point window and actualscored central128; initialSW function must equal104 except existing1e-12floor/renormalization. Sourcebranchinferencetime included, no Source1024feature claim. V213repeat/216source live undermax2, no automatictrial queue. Positive214 tiny and213posterior retainedpendingattribution; no layer/rankgrid or widerprob-viewgrid. Current best verifiedoptional7 TOTAL80.622609310221 below80.8, default5/fullgoal>82 unchanged.


V213 repeat final SW74.137964012421, actualTOTAL80.618149332486; learned gain+0.016211322652, vs211-0.009658610206. Six optionalphases, default5; no fair viewgainclaim until repeat/posterior paircomplete.

V213 matchedposterior-minusrepeat SW-0.00341705085586, posterior74.134546961565 versusrepeat74.137964012421; explicit posteriorfeatures didnothelp thispair, bothbelow21174.147622622627. Close this view/no broadergrid. Repeat retainedas213control and reusedfunctionally for216, no redundant216repeat GPUrun; Source inferencetime not matched, see216 CONTROL_REUSE.md. Current216source continues asonlylivejob (cap2), default5/fullgoal>82 unchanged.


V216 source final SW74.136821087878, actualTOTAL80.617749308896; learned gain+0.015068398109, vs211-0.010801534749. Six optionalphases, default5; no fair viewgainclaim until repeat/source paircomplete.

216 completed SW74.136821087878, below reused213repeat by0.001142924543 and211 by0.010801534749; source-view closed. Best optional7 fold5 TOTAL80.622609310221/default5 80.612475417994. New217/218 independent SW alternatives/max2 GPU; no remaining folds.

CPU diagnostic104->211: default100786 rows SW98.715632->98.637659 (-0.077974); continuous43879 rows17.631889->17.896277 (+0.264388), weighted contributions -0.054323/+0.080193. Diagnostic only, no fitted weights/oracle.219 tolerance-target loss prepared only;217/218 remain sole two GPU jobs.

V217 final SW74.123007594015, gain+0.001254765857, actualTOTAL80.612914586044; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/12, trainingseconds256.543644/peakMiB966.999023. No otherfolds.

V218 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds283.698458/peakMiB1255.786621. No otherfolds.

218 final SW74.121752828158, best0/14,no gain; actual assembly recorded.219 active;220 small distance-bias alternative prepared. Max2 GPU,no remainingfolds.217 gate variant below211, no continuation.

218 negative best/last checkpoints deleted after verified restored-state replay/actual assembly; SHA and bytes in CHECKPOINT_DELETION_MANIFEST.json. Positive controls and active219/220 preserved.219 mass target smoothing and220 four attention distance biases are the only active GPU jobs. Best optional7TOTAL80.622609310221 remains below80.8; no remaining folds.

220 failed first backward with CUDA efficient-attention LSE strideH alignment, initial74.121752828158/no trained score. Preserve source/config/error; new221 explicit float32 attention implementation.219 active; at most2 GPU tasks.

222 prepared only: positive211 mass-readout width32->128,49536 parameters,104 parent/oneSW addition.219 and221 sole active GPU tasks; no queue/otherfolds.219 seven-phase combination recorder prepared, actual score pending completion.

V219 final SW74.148179431612, gain+0.026426603454, actualTOTAL80.621724729203; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop7/21, trainingseconds304.111063/peakMiB908.494629. No otherfolds.

V211 actual optional6 TOTAL80.621724729203; actual optional7 (PERM207+SW219) TOTAL80.622804193366, at mostone extra perhead. Both fixed144665assembly, no rowwise model choice. Default5 kept because gains small; below80.8, no remainingfolds/fullOOF claim.

V221 final SW74.122157243259, gain+0.000404415101, actualTOTAL80.612616963279; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop3/13, trainingseconds225.275674/peakMiB1211.433594. No otherfolds.

219 final SW74.148179431612, matched211 +0.000556808984. Actual optional7 TOTAL80.622804193366 (+0.000194883145 vs previous7), best7/stop21,304.111063s/908.494629MiB. Restored RAM-state replay exact/frozen exact; no diskreload claim. Positive kept, gain too small for smoothing grid/default extra phase.221 finished13 best3 SW74.122157243259 (vs104+0.000404415101); four-bias scope closed without broader conclusion.222 mass width128 and223 original65tap local filter are current two explicit GPU alternatives; no stacking/otherfolds.

224 prepared/untrained: transfer positive SW mass-readout idea onto frozenPOR151;16672parameters,own categoricalNLL+BCE+jointRPS1,no existingstage discarded,onePORextra,total6 or8 with207/219.222 requested early boundary finalization below32widthcontrol;223 remainslive, max2GPU.

V222 final SW74.141314487722, gain+0.019561659564, actualTOTAL80.619321998841; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop7/13, trainingseconds245.395531/peakMiB933.126465. No otherfolds.

V223 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds271.268866/peakMiB1269.003906. No otherfolds.

222 final SW74.141314487722, gainvs104+0.019561659564 but width128 below32 by0.006308134905; best7/stop13,245.395531s/933.126465MiB, close width expansion.223 finalSW74.121752828158 best0/14 no gain,271.268866s/1269.003906MiB; negative best/last deleted withSHA manifest after actual assembly, rest evidence retained. Current live224POR mass (initial77.926320949376 vs15177.926380760724, float renormalization -0.000059811348 not learned gain) and225SW lastFF residual. Max2GPU/default5/best optional7TOTAL80.622804193366, no otherfolds.

V224 final POR77.926320949376, gain-0.000059811348, actualTOTAL80.612457474589; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds206.436611/peakMiB986.931152. No otherfolds.

224 finalPOR77.926320949376, best0/14, zero learned gain, original151 delta-0.000059811348 from floor/floatnormalization; actualassembly/replay/frozen exact.206.436611s/986.931152MiB. Negative best/last deleted withSHA manifest, original151 preserved.226 nativeNLL/BCE+RPS lowLR context-only fine-tune active, initial77.926380760724 exact parent score.225 requested earlyfinalization below211/219;227 input-conditioned residual prepared/noqueue. Best optional7TOTAL80.622804193366/default5 unchanged,nootherfolds.

V225 final SW74.128789410226, gain+0.007036582068, actualTOTAL80.614938221718; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/14, trainingseconds234.072998/peakMiB1199.011230. No otherfolds.

225 finalSW74.128789410226, gain104+0.007036582068 but below retained219 by0.019390021386; best1/stop14,234.072998s/1199.011230MiB. Restored-state replay/frozen exact/actualassembly; positive kept, no FF width/layer expansion.226 live native context fine-tune,227 live input-conditioned residual (one smallmodule);228 prepared true-label RPS-only matchedloss ablation after early226 decline. Max2GPU/no stacks/no otherfolds.

V226 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/12, trainingseconds162.520420/peakMiB924.547363. No otherfolds.

226 finalPOR77.926380760724,best0/stop12,zero gain,162.520420s/924.547363MiB; restored-state replay/frozen exact/actual assembly. Negative best/last deleted withSHA manifest; parent151 and all positive models preserved.228 matched RPS-only native-head refinement explicitly launched from151,no226weights.227 conditionpatch active; actual22224params corrects arithmetic22160 plan annotation without changing source. Max2GPU.

V227 final SW74.121752828158, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds242.930387/peakMiB1612.750000. No otherfolds.

V228 final POR77.926380760724, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds183.171263/peakMiB924.547363. No otherfolds.

V229 amplitude final SW74.121752828158, actualTOTAL80.612475417994; learned gain+0.000000000000, vs211-0.025869794469. Six optionalphases, default5; no fair viewgainclaim until repeat/amplitude paircomplete.

227 finalSW74.121752828158,best0/14,no gain,242.930387s/1612.75MiB;228 finalPOR77.926380760724,best0/14,no gain,183.171263s/924.547363MiB.229 amplitude finalSW74.121752828158,best0/14,no gain,165.671455s/1397.282227MiB; repeat untrained/held as negativeamplitude does not warrant control cost, no featuregain claim. All actual144665 assemblies/replays/frozen exact; negative best/last deleted withSHA manifests, positive104/151/154 preserved.230PERM conditionalmass and231SW additiveBiGRU16 are current twoexplicitGPUjobs, default5/bestoptional7TOTAL80.622804193366 unchanged. No otherfolds.

V230 final PERM89.413121051402, gain+0.007556194483, actualTOTAL80.615120086063; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds178.528288/peakMiB791.250000. No otherfolds.

230 finalPERM89.413121051402, gain154+0.007556194483/gain207+0.004472011161, best10/stop24,178.528288s/791.25MiB. Initial floor+float delta+0.000008777740, learnedgain+0.007547416743 separated. Actual optional7(POR151+PERM230+SW219)TOTAL80.624369397272, gain previous7+0.001565203906, all144665 fixedchannel assembly; default5 unchanged/below80.8/no otherfolds.231 failed firstbackward(no epochs) because GRU trainflag was absent;232freshfixexplicitGRUtraining.233 width64 sibling from154 prepared; no230 repair stacking.

231 failedfirstbackward because intended GRU trainingmode switch was missing,no epoch efficacy score;232freshsource explicitlysets recurrent.train(mode) and has completed realepochs.233PERM width64 and232SW memory are sole GPU jobs. Current bestactualoptional7TOTAL80.624369397272,fullOOF82 stillunmet,no otherfolds.

V232 final SW74.133475000603, gain+0.011722172445, actualTOTAL80.616578178350; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop2/16, trainingseconds252.972477/peakMiB1205.791992. No otherfolds.

V233 final PERM89.409673958189, gain+0.004109101270, actualTOTAL80.613913603438; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds213.481596/peakMiB791.437500. No otherfolds.

232 finalSW74.133475000603,best2/16,252.972477s/1205.791992MiB, gain104+0.011722172445 but below219 by0.014704431009.233 finalPERM89.409673958189,best6/20,213.481596s/791.4375MiB, width64 below32 by0.003447093213; close widthgrid,positive controls retained. Both actual144665 assemblies/restoredreplay/frozen exact.234PERM tolerance-loss and235SW joint memory+mass(one phase,26880newparams) explicitly launched/max2GPU. Bestoptional7TOTAL80.624369397272 unchanged,default5 kept/no otherfolds.

Gain diagnosis,not trainingfit: PERM default66.037% scores97.278705->97.281652,continuous33.963%74.096912->74.113431; aggregate+0.007556. SW default69.669%98.715632->98.644703,continuous30.331%17.631889->17.881934; weighted gains-0.049416/+0.075842,net+0.026427. ContinuousSW zero-official-score66.389%,pred>=99.9 30.281%; latter is action range,not directgateclassification. Frozen-small-repair scope limits correction; representation/generalization hypothesis remains unproven, no noise/data-ceiling claim. Diagnostics saved CHANNEL_GAIN_STRATA.json and CONTINUOUS_ERROR_DIAGNOSTIC.json.

V234 final PERM89.405573634659, gain+0.000008777740, actualTOTAL80.612478490203; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds150.714973/peakMiB791.251953. No otherfolds.

V235 joint final SW74.129980102971, actualTOTAL80.615354964179; learned gain+0.008227413202, vs211-0.017642519656. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

234 PERM tolerance final learned gain0/best0/stop14;235 SW joint final74.129980102971, +0.008227 learned but below219. Next prepared236 joint gate/mass and237 local PERM mass, siblings from original104/154, one addition each, max2, no otherfolds. Best optional7 TOTAL80.624369397272/default5 unchanged; fullOOF82 unmet.

User confirms preserve gains and strengthen continuous representation/readout cooperation.236/237 failed missing PLAN beforemodel/no efficacy result. Fresh238 SW gate+mass and239 PERM local mass launch; no changes to default5/best optional7/parent weights. Max2/80-14/oneaddition each, strict808 before otherfolds.

Training-only stratum diagnostic: {"POR": {"training_anchors": 578891, "continuous_fraction": 0.3207702338695526, "quality_weighted_continuous_fraction": 0.31910228729248047}, "PERM": {"training_anchors": 578891, "continuous_fraction": 0.31064051389694214, "quality_weighted_continuous_fraction": 0.30894896388053894}, "SW": {"training_anchors": 578891, "continuous_fraction": 0.27884870767593384, "quality_weighted_continuous_fraction": 0.27711787819862366}}; potential next matched loss test separates continuous likelihood normalization; hypothesis unproved, no current trainer changes.

240/241 prepared: matched continuous-likelihood normalization vs238SWjoint/230PERMpointwise; training-only denominator/Qunchanged/no validation fitting. No launch while238239 active, max2/80-14/oneaddition each. Default5 and bestoptional7 unchanged.

V238 final SW74.125862431785, gain+0.004109603628, actualTOTAL80.613913779264; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/12, trainingseconds231.232769/peakMiB903.000000. No otherfolds.

V239 final PERM89.412863604436, gain+0.007298747517, actualTOTAL80.615029979625; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds246.963993/peakMiB791.269531. No otherfolds.

238 final SW74.125862431785,best1/stop12/earlyterminated; learned+0.004109742017 but below219 by0.022316999826.239 final PERM89.412863604436,best6/stop20; learned+0.007289969777 but below pointwise230 by0.000257446966. Positive sibling checkpoints retained; no evidence local9 improves pointwise control. Next240/241 loss-normalized matched siblings, no repair-weight chaining, max2/oneaddition each. Default5/bestoptional7 unchanged; strict808 and fullOOF82 unmet.

242 prepared backbone/readout coadapt: original104 finalFF existingparams at3e-5 plus zero masshead1e-4, hardbinNLL/BCE/SAM; one SW phase, no235trainedweights. Hold until slotsfree;240241continue unchanged, no qualifying808/full82yet.

V240 final SW74.126307795273, gain+0.004554967116, actualTOTAL80.614069656485; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop1/12, trainingseconds240.383194/peakMiB903.000000. No otherfolds.

SW240 final74.126307795273,best1/stop12/earlyterminated; versus238+0.000445363488, versus219-0.021871636338. RestoredRAM/frozencheck/actual144665assembly complete. Keep positive sibling, close this normalization scope rather than broad negative claim.242 finalFF+mass joint launched as one SWaddition from original104, no trainedrepair weights;241 remains active with snapshot89.413463686933 (unconfirmed final). Default5/bestoptional7 unchanged.

V241 final PERM89.413463686933, gain+0.007898830014, actualTOTAL80.615240008499; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds292.098263/peakMiB791.250000. No otherfolds.

241 final matched PERM loss normalization89.413463686933 vs23089.413121051402,+0.000342635531; best10/stop24. Actualoptional7TOTAL80.624489319708, gain+0.000119922436, same144665fifthpopulation. Tiny positive retained; default5 unchanged. SW242 currently training87424params(74848existingFF+12576mass),peak976.59MiB; no proven gain. Historical interpolation CUDAbackward warn-only nondeterminism observed whenFF enabled; no deterministic bitwise rerun claim. No otherfolds/fullOOF proof.

243SW regret-margin ranking retains219 softtarget/mass;244PERM conditional-only RPS retains241 normalizedNLL. Same104/154 originalparent, no controlrepairweights, oneheadaddition/sixalone/max2/80-14.243 may use free slot alongside242;244held until actual slotfree. Gain unproved; best optional7TOTAL80.624489319708/default5 preserved.

V242 joint final SW74.121752689769, actualTOTAL80.612475369558; learned gain+0.000000000000, vs211-0.025869932858. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

242 final no learned gain,best0/stop13/earlyterminated,SW74.121752689769,214.8334s/976.59MiB; actualassembly complete. Delete redundant best/last checkpoints with manifest; close this finalFF lowLR+binNLL coadapt, not allbackbone optimization.243 SWdecisionrank and244 PERMconditionalCDF now live max2; oneaddedstageeach. Bestoptional7TOTAL80.624489319708/default5 intact; gate808/fullOOF82 unmet.

V243 final SW74.129922413897, gain+0.008169585739, actualTOTAL80.615334773003; sixphases inclsource/one addedSWstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds293.088040/peakMiB908.494629. No otherfolds.

V244 final PERM89.413440514835, gain+0.007875657916, actualTOTAL80.615231898265; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds264.101818/peakMiB791.250000. No otherfolds.

243finalSW74.129922413897 below219;244finalPERM89.413440514835 below241 by0.000023172098, positives retained.245 matched raw206+relative versus repeated hidden207 local branch prepared; same21684params/seed/loss/80-14, original104only,no trained219weights; max2/oneSWaddition/sixalone. Bestoptional7TOTAL80.624489319708 intact, no qualification/fullOOF.

246 prepared additive-vs-logmass pair with identical.001 uniform-mixture initial/12609params; original104frozen/219loss/no trainedrepairs. Initial deterministic score changes must separate fromlearning. Hold until245finished, max2/80-14/oneSWaddition. Bestoptional7TOTAL80.624489319708 preserved; no808qualification/full82proof.

V245 raw final SW74.138695788526, actualTOTAL80.618405454123; learned gain+0.016943098757, vs211-0.008926834101. Six optionalphases, default5; matched raw-vs-repeat features, one-stage sibling.

V245 repeat final SW74.139868705360, actualTOTAL80.618815975015; learned gain+0.018116015591, vs211-0.007753917267. Six optionalphases, default5; matched raw-vs-repeat features, one-stage sibling.

245 matched pair stopped at15both: raw74.138695788526 vsrepeat74.139868705360, delta-0.001172916834; inputs did not help within this bounded run, no convergence/broadfeature-negative claim. Both positivevs104 weights retained. Individualtrainer429.52/429.73s; concurrent pairwall processcreation-to-lastreport438.498s, includesstartup/data/model/initialeval, excludespreparation/CPUassembly; no serial-speedup claim. Next246newdistribution matched pair, original104/no245weights/oneSWaddition/max2. Bestoptional7 unchanged80.624489319708, no otherfolds.

246 bothlive/initialeval complete: SW74.121701580675 equal acrossarms, vs104-0.000051247483 deterministic mixture/floating change, not traininggain. Both12609params/sameinitial. Newdistribution pair uses max2; owntrainedgain unproved. Originalbest models preserved.

246 stilllive: additive5bestinitial, parametersmoving/weakfirstgradient, controlsmallgain.247 prepared matched add.25 own-label newcomponentCE to bypass smallresponsibility suppression; no246weights, same12609params/initial, held untilslotfree, oneSWphase/max2/80-14. Bestoptional7 unchanged80.624489319708; no qualifyingfold/fullOOF.

246 both15/best7/bad8: request paired boundary earlytermination, additivebelowcontrol; no convergence claim.247 owncomponentaux ready;248PERM learneduniformbackground prepared from154/no241weights, +33params disclosed, retains241 normalizedNLL/RPS. Launch onlyactualfree slots/max2/oneaddition each. Bestverified7 unchanged pending246restore/assembly.

V246 additive final SW74.121881069311, actualTOTAL80.612520302398; learned gain+0.000179488636, vs211-0.025741553316. Six optionalphases, default5; paired additive-vs-multiplicative distribution mechanism, identical initial mixture.

V246 multiplicative final SW74.149145254062, actualTOTAL80.622062767061; learned gain+0.027443673388, vs211+0.001522631435. Six optionalphases, default5; paired additive-vs-multiplicative distribution mechanism, identical initial mixture.

246 paired bounded final18both: additive74.121881069311 vsmultiplicative74.149145254062, gap-0.027264184751; newdistribution alone not useful here, no convergence/globalmechanism claim. ControlSW vs219+0.000965822451. Actualoptional7TOTAL80.624827357565, gain+0.000338037858. Both source/controlweights kept (additive247dependency); default5 unchanged. Concurrent pairwall toreport525.936s vsindividual519.04/518.34s; no serialspeedup claim. Next247aux+248PERMtransfer, original104/154only, oneaddition each,max2. No808/full82proof.

Bounded research agent delivered3primarypapers; inspiration docs/deep_learning/research/continuous_readout_suggestions_20261001.md.249 prepared8piecewise-affine monotone horizontal-value vsvertical-probability warp,4392params each/identityinit/frozen104/oneSWaddition, no smallmixture. Not RQS fullreproduction, no untrainedgain.247248continue/max2; newpairheld untilactualslotsfree.

V247 aux final SW74.121888655505, actualTOTAL80.612522957566; learned gain+0.000187074831, vs211-0.025733967122. Six optionalphases, default5; paired additive-vs-multiplicative distribution mechanism, identical initial mixture.

V248 final PERM89.413465595388, gain+0.007900738469, actualTOTAL80.615240676458; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds371.503725/peakMiB789.903809. No otherfolds.

247 finalSW74.121888655505,best5/stop19,vs246additive+0.000007586194 only; firstgradientindicator0.0047076 vs0.0001221 grew but no substantive scorebenefit. Close this tiny-mixture/auxiliary scope, no coefficientgrid.248 finalPERM89.413465595388,best10/24,vs241+0.000001908455 only, initial89.404398334646 differsparent89.405565; actualmicro7TOTAL80.624828025525, gain+0.000000667959, no material improvement/default5 unchanged. Positive/controlweights retained.2494392param pair launched from104; frozenparent/defaultgate, no repairedweights, max2/oneSWaddition. Goal808/full82unmet.

V249 vertical final SW74.127693242658, actualTOTAL80.614554563069; learned gain+0.005940552889, vs211-0.019929379969. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

V249 horizontal final SW74.124751573162, actualTOTAL80.613524978745; learned gain+0.002998883393, vs211-0.022871049465. Six optionalphases, default5; joint one-stage compound trial,not pure mechanism ablation.

V249 final: horizontal SW74.124751573162 (best2/stop16), vertical74.127693242658 (best1/stop15); horizontal-minus-vertical -0.002941669497. Both naturally patience14 stopped; neither beats retained SW24674.149145254062. Initial74.121752689769 separately reported; RAM-best restoration/frozen parameters/actual144665-row composition verified, not independent disk reload. Close this eight-segment affine transport configuration, no warp width/coefficient grid; no conclusion about all monotone transport. Positive minor gains kept. Concurrent pair wall 307.569s includes startup/data/model/initial evaluation, individual301.496/265.416s, excludes code preparation/CPU assembly; no serial speedup claim. Default5 and optional7 TOTAL80.624828025525 unchanged; no other folds or active GPU training remain. Next work: consider jointly adapting a small continuous-value representation and readout in one added phase, preserving retained controls; not yet implemented or launched.

250 prepared matched readout/coupled terminal continuous representation residual,20961params/oneSWaddition/original104 only. Preserve optional7 TOTAL80.624828025525; max2/80-14/no otherfolds.

250 actual launch: readout PID65604/coupled PID15932, same initial74.121701580675 and20961params. Epoch2 snapshots readout74.126919573085/coupled74.088084576167; early/unproved, both live, peak773.251/788.132MiB. No OOM/errors, no otherfolds.

251 prepared continuous body3e-5+mass1e-4 versus frozen-body head control, SAM0both, original104 atom fixed. OneSWaddition/no repaired weights; held while250max2live.

V250 readout final SW74.140580389878, actualTOTAL80.619065064596; learned gain+0.018878809204, vs211-0.007042232749. Six optionalphases, default5; same-capacity one-stage representation/readout coupling comparison.

V250 coupled final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; same-capacity one-stage representation/readout coupling comparison.

250 bounded final readout74.140580389878 best7/stop16; coupled74.121701580675 best0/stop13/no learnedgain. Unequal achieved budgets disclosed; no convergence or strict fair fullbudget mechanism gain claim. Both below retained24674.149145254062. RAM restoration/frozen/full144665actual composition done. Keep positive readout; remove redundant coupled checkpoints with manifest. Concurrent pair wall 251.182s vsindividual241.700/244.675s includes startup/model/data/initialeval, excludes prep/CPUassembly; no serial speedup claim.251 head/body launched only after250PIDs terminal, max2/oneSWaddition, no otherfolds.

251 actual launch headPID25000/bodyPID42372; both initial74.121701580675. Head12609params epoch1 best74.133401970608, body858522params epoch1score73.946483489217/no gain yet. Live/max2; peaks667.695/1464.630MiB, noOOM. Body backward emits interpolation/attention nondeterminism warn_only; no bitwise rerun reproducibility claim. Original atom remains frozen and full source checkpoint unchanged; early snapshot not final evidence.

252 prepared frozen104 mass+ownCDF RPS1 joint/conditional matched12609; retains246mechanism/219CE/SAM.1, no repairedweights. Held until251slotsfree; max2/oneSWaddition/no otherfolds.

V251 head final SW74.133401970608, actualTOTAL80.616552617852; learned gain+0.011700389934, vs211-0.014220652019. Six optionalphases, default5; one-stage continuous-backbone versus frozen-body readout comparison.

V251 body final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; one-stage continuous-backbone versus frozen-body readout comparison.

251 final head74.133401970608 best1/stop15 naturalpatience14; body74.121701580675 best0/stop12 earlytermination/no learnedgain. Body-minus-head -0.011700389934, unequalbudgets and backbone CUDAwarn_only nondeterminism disclosed; no convergence/globalbackbone rejection. RAMrestore/frozen atom/actual144665composition done. Positive head kept; negative body checkpoints removed withSHA manifest. Concurrent pairwall 227.838s; individual193.433/220.512s, includesstartup/model/data/initialeval, excludesprep/CPUassembly, no measuredserialspeedup. Close current3e-5continuousbody+mass/AdamW/SAM0 setting, no LRgrid.252joint/conditionalCDFsupervision new sibling original104 pair launched onlyafter251PIDs ended; max2/oneSWaddition/nootherfolds. Bestoptional7 unchanged80.624828025525.

252 actual launch jointPID27552/conditionalPID24456, initial74.121701580675same/12609params. Original104 frozen, ownCE plusRPS1, no trained246 weights. Both live/max2; initialevaluation only, gain unproved.

246 updated own validation diagnosis: continuous score 17.884647 vs104 17.631889, gain 0.252758; default score change -0.070724. Diagnostic only; not training weights/row selection or a gate classifier accuracy measure. Frozen gate does not prevent changes in final default action.

253 prepared16770param separated gate/mass global versus balanced SAM, equal theoretical totalrho.1, original104only/oneSWaddition. Hypothesis not proven; held while252max2live.

252 live verified PID27552/24456 epoch7: joint74.148580477702, conditional74.149228207812, conditionalvsretained246+0.000082953750 only. Snapshot not final/materialgain. Both1101MiB/max2; no otherfolds.253prepared held until slotsfree.

V252 joint final SW74.148580477702, actualTOTAL80.621865095335; learned gain+0.026878897028, vs211+0.000957855075. Six optionalphases, default5; CDF supervision joint-vs-conditional on retained mass mechanism, same initial mixture.

V252 conditional final SW74.149228207812, actualTOTAL80.622091800873; learned gain+0.027526627138, vs211+0.001605585185. Six optionalphases, default5; CDF supervision joint-vs-conditional on retained mass mechanism, same initial mixture.

252 bounded pair final14both/best7: joint74.148580477702, conditional74.149228207812; conditionalvs246+0.000082953750 only. Actual newoptional7 TOTAL80.624857059337, previousoptional7delta+0.000029033812, not material; default5 unchanged. All144665actual composition/frozen/RAMbest replay; no diskreload/fullOOF/promotion claim. Both positive checkpoints kept, no coefficientgrid. Pairwall380.194s, individual373.941/373.596s/1101MiB, startup/data/model/initialeval included, prep/CPUassemblyexcluded, no serialspeedup claim.253global/balanced factorSAM launched onlyafter252PIDs terminal; max2/oneSWaddition.

253 real launch globalPID45612/balancedPID77268, matched initial74.121701580675/16770params. Epoch1global74.128904879153/balanced74.128795529416, early/unproved; both groups training/no errors/OOM, peak1119.827MiB. Epoch1about24.04s withconcurrentpair; bypassing unusedutility has no demonstrated speedup versus previousdifferentstudy, no attribution claim. Original104frozen, no otherfolds.

254 prepared PERM raw206+relative versus repeat128hidden side207/local9, matched21717params, retains248 normalizedNLL/RPS/background mechanism but starts154only. OnePERMaddition/max2/80-14; held while253live/nootherfolds.

253 live verified epoch5: global74.131134840323/balanced74.132890964633, earlygap+0.001756124309/bothbelowretained25274.149228. First owntraining-only gradient norms mass0.005389/state0.008164; globalrho allocation0.05509/0.08346 versusbalanced0.07071each. Evidence of allocation change only, not proven gradient interference/gain across trajectory. DiagnosticJSON saved, no validationgradients/trainingweight fitting.254PERM matched raw/readout prepared/held while2GPUlive.

V253 global final SW74.131134840323, actualTOTAL80.615759122252; learned gain+0.009433259649, vs211-0.016487782304. Six optionalphases, default5; matched global versus balanced group SAM, one joint SW phase.

V253 balanced final SW74.132890964633, actualTOTAL80.616373765760; learned gain+0.011189383958, vs211-0.014731657994. Six optionalphases, default5; matched global versus balanced group SAM, one joint SW phase.

253 bounded matched12both/best5 finalglobal74.131134840323 versusbalanced74.132890964633, relative+0.001756124309, both belowretained25274.149228207812. Owntraining gradient partition changed, no material gain or convergence claim. RAMbest/frozen/full144665 actualcomposition done. Positive/control checkpoints kept; noSAMrho/groupweight grid. Concurrent pairwall318.879s vsindividual311.496/311.376s; startup/data/model/initialeval included,prep/CPUassemblyexcluded,nomeasuredserialspeedup.254PERM raw/repeat launched onlyafter253PIDs terminal,max2/onePERMaddition. Bestoptional7TOTAL80.624857059337/default5unchanged/nootherfolds.

254 actual launch rawPID2980/repeatPID50424, sameinitialPERM89.404398334646/21717params, max2. Original154frozen/no repaired248weights; onejointPERMstage. Initialeval completed/no errors; no gain yet.

255 prepared PERM coordinate-conditioned conditionalmass scorer matched5313params bilinear/nonlinear, same248 positive loss/background mechanisms/from154only. OnePERMaddition/max2/80-14; held until254terminal/nootherfolds.

V254 final PERM89.410293214370, gain+0.004728357451, actualTOTAL80.614130343102; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop4/17, trainingseconds187.801940/peakMiB790.049316. No otherfolds.

V254 final PERM89.409306142997, gain+0.003741286078, actualTOTAL80.613784868121; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/17, trainingseconds188.345957/peakMiB789.947266. No otherfolds.

254 bounded matched17both finalraw89.410293214370 best4 versusrepeat89.409306142997 best5, relative+0.000987071374 tiny, both belowretained24889.413465595388. Source/frozen/RAMbest/full144665 actualcomposition done viafresh recorder_r1, old mass-only recorder ignoredstage/no score; training source unchanged. Positive/controlweights retained; no rawdepth/width scan. Pairwall196.773s, individuals187.802/188.346s, peaks790.049/789.947MiB, startup/model/data/initialevalincluded/codeprepCPUassemblyexcluded/no serialspeedup claim.255coordinate querypair launched onlyafter254PIDs terminal, max2/onePERMaddition. Bestoptional7TOTAL80.624857059337unchanged/default5preserved/nootherfolds.

255 actual launch bilinearPID39824/nonlinearPID69904, bothinitial89.404398334646/5313params. Epoch2bilinearbestinitial/nonlinear89.405634949173, early/unproved; both gradienttrain/noerrors/OOM, peaks790.252MiB. OnePERMaddition/max2, no otherfolds. Fresh record_perm_coordinate_readout_v255_r1.py handlesstage atfinal.

256 prepared CE versusfixedq.2 continuous generalized likelihood on identical248mass/background/normalizedNLL+RPS mechanism from154only. No label/Q changes; objective not properloglikelihood claim. OnePERMaddition/max2/80-14, heldwhile255live.

V255 final PERM89.405114632526, gain-0.000450224393, actualTOTAL80.612317839456; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/12, trainingseconds251.374102/peakMiB790.252441. No otherfolds.

V255 final PERM89.405634949173, gain+0.000070092254, actualTOTAL80.612499950283; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop2/12, trainingseconds250.074362/peakMiB790.252441. No otherfolds.

255 bounded12both: bilinear89.405114632526 best3 vsnonlinear89.405634949173 best2; nonlinearrelative+0.000520316647 only, absolute belowretained24889.413465595388. Smalllearned gains/controlweights retained, no coordinate width/frequency scan. RAMbest/frozen/full144665actualcomposition viarecorder_r1 complete; no independentdiskreload or convergence claim. Pairwall257.709s vsindividual251.374/250.074s/790.252MiB; includesstartup/data/model/initialeval, excludescodeprep/CPUassembly, no measuredserialspeedup.256 CE/q02 likelihood pair launched onlyafter255PIDs terminal,max2/onePERMaddition/nootherfolds. Bestoptional7TOTAL80.624857059337/default5unchanged.

256 actual launch CE PID62520/q02 PID62420, sameinitial89.404398334646/12609params. Epoch1CE89.406764227678/q0289.406750098120, no earlyrelative gain, peak789.904MiB/noerrors/OOM. Lossmagnitudes differ byobjective/not accuracy evidence. Bothrealgradient updates/max2/original154 frozen/onePERMaddition. No otherfolds.

257 prepared smallPERM raw207->32/two crosscurve-depthblocks dilations1/4 jointmasshead; point/local matched29185storedparams, effectivecontextfreedom disclosed. Carries254boundedrelative rawgain, starts154only/no trainedrepairs/onePERMaddition. CE248loss retained, no unproved256loss compounded; max2/80-14/heldwhile256live.

V256 final PERM89.413465595388, gain+0.007900738469, actualTOTAL80.615240676458; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/17, trainingseconds180.099758/peakMiB789.903809. No otherfolds.

V256 final PERM89.413048870990, gain+0.007484014071, actualTOTAL80.615094822919; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/17, trainingseconds180.489797/peakMiB789.903809. No otherfolds.

256 bounded17both CE89.413465595388 best10 vsq0289.413048870990 best6, delta-0.000416724398. No loss-mechanism gain; no qgrid. RestoredRAM/frozen/full144665actualassembly; CE checkpoint state/predictions exact existing248, verified CPU tensor comparison. Remove redundantCE best/last and unpromisingq02 checkpoints withmanifest after this evidence; preserve248positive weights. Pairwall186.983s, individuals180.100/180.490s/789.904MiB, startup/data/model/initialevalincluded/prepCPUassemblyexcluded,no measuredserialspeedup.257newsmallfeaturebackbonepoint/local launched onlyafter256PIDs terminal/max2/onePERMaddition/nootherfolds. Bestoptional7TOTAL80.624857059337unchanged.

257 actuallaunch pointPID64632/localPID60836, bothsameinitial/29185storedparams. Epoch3best point89.406580519480 versuslocal89.406499698385 (best2both); early/unproved. Bothgradienttrain/noerrors/OOM, peaks790.941MiB,10.2-10.6s recentepoch/concurrentpair (not serialbenchmark). Original154frozen/onePERMaddition; no otherfolds.

258 prepared input-only PERM wholewell128x206mean/std412->32/readout26849storedparams, reference train-onlyconstant versus currentwellinputs. No labelstats/IDs/forbiddencurves, same248loss/from154only/onePERMaddition/max2/80-14. Held while257live/nootherfolds.

V257 final PERM89.409216200327, gain+0.003651343409, actualTOTAL80.613753388187; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/17, trainingseconds194.156798/peakMiB790.940918. No otherfolds.

V257 final PERM89.409182616679, gain+0.003617759760, actualTOTAL80.613741633910; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/17, trainingseconds194.175286/peakMiB790.940918. No otherfolds.

257 bounded17both point89.409216200327 best5/local89.409182616679 best6, local-minus-point -0.000033583648, bothbelowretained24889.413465595388. No temporal mechanism benefit here; no branch depth/dilation/width grid, no fullconvergence/generaltemporalnegative claim. RAMbest/frozen/all144665actualcomposition done; preserve source/config/score/SHA and remove redundant checkpoints forboth unpromisingbranches withmanifest. Currentbest248/252 unchanged. Pairwall201.105s versusindividual194.157/194.175s/790.941MiB, startup/data/model/initialevalincluded/prepCPUassemblyexcluded, no measuredserialspeedup.258reference/wellinputcontext launched onlyafter257PIDs terminal,max2/onePERMaddition/nootherfolds. Bestoptional7TOTAL80.624857059337unchanged.

258 actual launch referencePID66400/wellPID41832, bothsameinitial/26849params. Epoch2reference89.408134584882/well89.409475154060, earlyrelative+0.001340569178 butbelowretained248; snapshot not proved finalgain. Bothgradienttrain/noerrors/OOM, peaks790.477/790.296MiB. Existinginput-only128x206bank/trainreference, no labelstats/wellIDs or heldout fitting. Original154frozen/onePERMaddition/max2/nootherfolds.

248 retainedPERM diagnosis full144665: continuous score74.115001 vsoriginal74.096912, defaultscore97.281366 vsoriginal97.278705; diagnostic only/no trainingweights or rowselect. Continuous zero-scorefraction0.065171, finaldefaultactionfraction0.118151 (not gateclassifier accuracy).

259 prepared SW jointbody+mass with continuous-label rank representation auxiliary0vs0.1, same858522params/original104only/frozenatom. OneSWaddition/max2/80-14; no projector/teacher/pseudolabels/validationfitting. Primarysource authorloss/paper checked; mechanismtransfer not fullpaperreproduction. Research docs/deep_learning/research/representation_pivot_suggestions_20261001.md.

V258 final PERM89.411293594790, gain+0.005728737871, actualTOTAL80.614480476249; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds186.966599/peakMiB790.476562. No otherfolds.

V258 final PERM89.411012124353, gain+0.005447267434, actualTOTAL80.614381961596; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds187.184188/peakMiB790.295898. No otherfolds.

V258 closed: reference PERM 89.411293594790, well 89.411012124353, delta -0.000281470438; natural patience14 stop19 both. Actual144665 composition/RAMbest replay/frozen checks complete, no independent diskreload. Pair wall 193.287s, includes startup/data/initialeval/finalrestore, excludes codeprep. Wholewell mean/std no gain; retain248/default5/optional7, no otherfolds.
V259 launched CE/RNC joint continuous SW backbone/readout, original104 atom fixed, one addedSWphase, max2GPU/80-14. Gain unproved; study results/deep_learning/sw_rank_representation_v259_20261001.

V259 both arms failed before first optimizer step: torch argsort stable overload requires dim as keyword. No trained gain; immutable source/log/error kept. V260 new source fixes only argsort(dim=-1,stable=True), resets both arms from original104, identical hypothesis/config.

V260 live verified: PIDs74868/77992, both completed epoch1 with finite loss and RNC70 sampledpoints; ~1496.63MiB peak each. Epoch1 CE73.943206247637/RNC73.929819934377, both below identicalinitial74.121701580675; no gain yet, continue80/14. V258 negative checkpoints removed with SHA manifest after actualcomposition; sources/config/results/predictions retained. Atlas batch built1014experiments/191deadends/213unresolved; unresolved ancestry remains explicit.

V261 prepared, not launched: bounded zero-initialized8352representation+12609mass jointly, original104 frozen, rankaux0vs.1. OneSWaddition; wait260terminal/max2.

V259 ce final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; same-architecture continuous-representation auxiliary0 versus0.1 one-stage study.

V259 rnc final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; same-architecture continuous-representation auxiliary0 versus0.1 one-stage study.

V260 terminal early12 both/best0 initializationSW74.121701580675/actualTOTAL80.612457481375. RNC-minus-CE restoredbest0. All144665 composition/RAMreplay/frozenatom verified; no independentdiskreload. Individual223.858171/226.733864s, peak1496.630371MiB, concurrent wall232.331s inclstartup/data/initialeval/finalrestore/excludesprep. Close fullcontinuousbody update for this loss, not allRNC/backbones. V261 launched bounded adapter+mass matched CE/RNC, original104 entirelyfrozen/oneSWaddition/max2/80-14; gains unproved.

V261 live verified PIDs86312/18912 at epoch3: CE74.131990259295/RNC74.134371736808, snapshot delta+.002381477513, NOT final gain and below retainedSW25274.149228207812.20961trainableparams/804.753MiB peak perarm. Continue matched80-14, no remainingfolds. V259 failed andV260 early-negative checkpoints deleted with SHA manifests; source/config/error/results retained/currentpositives untouched. Atlas batch built1017experiments/192deadends/214unresolved; no all-parent-verification claim.

V262 prepared stronger private continuous backbone TCN/BiGRU from raw207, original104 frozen, independent component CE prevents smallmixture coefficient suppressing learning. OneSWaddition; untrained/wait261terminal/max2.

V259 ce final SW74.140580389878, actualTOTAL80.619065064596; learned gain+0.018878809204, vs211-0.007042232749. Six optionalphases, default5; same-architecture continuous-representation auxiliary0 versus0.1 one-stage study.

V259 rnc final SW74.134371736808, actualTOTAL80.616892036022; learned gain+0.012670156134, vs211-0.013250885819. Six optionalphases, default5; same-architecture continuous-representation auxiliary0 versus0.1 one-stage study.

V261 FINAL (recorder copied label V259 is historical typo; this evidence is V261): CE74.140580389878/RNC74.134371736808, delta-0.006208653070, bothearly14/best7and3; actualTOTAL80.619065064596/80.616892036022, no808. Individual209.635357/209.040677s, peak804.753MiB, pairwall217.325s. CEpredictionequal250=True, tensorstateidentity notchecked. Close bounded rank.1; no convergence/generalRNC claim. V262 launched two new continuousspecialist raw207 backbones TCN98497/BiGRU96513 parameters, matched initialSW74.122050668856/nonidentityalpha.01. OneSWphase; owncomponentCE.5+normalizedmixtureCE jointly trained, no teacher/no oldrepair weights. Gainunproved, preserve currentbest/default5/delivery, nootherfolds.

V262 live verified PIDs69224/44720. TCN epoch3/bestinitial74.122050668856, BiGRUepoch4/best2 74.122942685760. No meaningful gain yet. Trainable98497/96513, peak938.692/1295.686MiB respectively, finite owncomponentCE and losses. Continue80/14 max2; no qualification/otherfolds. Prior and original atom frozen. V261 checkpoints removed after fullcomposition; CEpred equal retained250 control; deletionmanifest kept. Atlas1020experiments/193deadends/215unresolved, no completeancestry claim.

V262 train-only16anchor CPUdiagnostic: TCNepoch9 continuousalpha mean.00022797, GRUepoch12 .00043499; newcomponent CE down but finalmodel nearlyignores branch. V263 prepared directprior-relative bounded3 mass/no learnedmixturegate, same newencoders joint oneSWstage, fresh104/no262weights; held until262terminal/max2.

V262 bigru final SW74.122942685760, actualTOTAL80.612891868155; learned gain+0.000892016904, vs211-0.024679936867. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V262 tcn final SW74.122529455623, actualTOTAL80.612747237607; learned gain+0.000478786767, vs211-0.025093167004. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V262 FINAL TCN74.122529455623/best12/early16, GRU74.122942685760/best2/natural16; learnedinitial+.000478786767/+.000892016904, no meaningfulgain vsretained252. ActualTOTAL80.612747237607/80.612891868155/full144665finite/RAMbest/frozenexact; no independentdiskreload. Individual213.611151/177.972639s, peak938.692/1295.686MiB; pairwall222.291s includingstartup/data/initialeval/finalrestore, excludesprep. Architectureparamcounts differ98497/96513. Sampledtrainingcontinuous alpha nearzero onlydiagnostic, no validationfit. V263 launched directpriorrelative distribution/newencoders/no learnedmixgate/fresh104/oneSWphase/max2/80-14. Gains unproved, originalcurrentbest/default5/delivery untouched.

V263 liveverified PIDs9332/45272. TCNepoch2 SW74.083129319825; GRUepoch4 SW74.071211629977; both bestinitial74.121701580675, no learnedgain yet. Finite losses/gradients, peaks937.782/1295.659MiB. Continue80-14 max2. V262 negative checkpoints removed after full144665composition; CHECKPOINT_DELETION recordsSHA/paths; original104/retained252/delivery untouched. Atlas1023experiments/194deadends/216unresolved, no completeancestry claim.

V264 prepared PERM stateless trainingwell risk equalmean vs entropicsoftworst, matched12609 effective248architecture/fresh154/no repairedweights; held263GPUslots, gainunproved.

V263 tcn final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V263 bigru final SW74.121701580675, actualTOTAL80.612457481375; learned gain+0.000000000000, vs211-0.025921041953. Six optionalphases, default5; new private continuous specialist TCN versus BiGRU, unmatched-capacity architecture comparison.

V263 FINAL bothSW74.121701580675/best0/natural14stop/no learnedgain, actualTOTAL80.612457481375, full144665composition/RAMbest/frozenverified/no diskreload. TCN188.605411s/937.782MiB, GRU151.468424s/1295.659MiB, concurrentwall194.192s inclstartup/data/initialeval/finalrestore/excludesprep. Close thisnewrawspecialist/readout route, no width/bound scan. V264 launched matched12609PERM wellmean/softworstrisk, fresh154 only/no repairedweights/onePERMaddition/max2/80-14. Gainsunproved. Research nextPORlateblock V-REx documented only, not trained; docs/deep_learning/research/representation_large_gain_pivot_20261001.md.

V264 liveverified PIDs47772/62424: wellmeanepoch4/best89.410577810895, robuste5/best89.412670737129, both below retained24889.413465595388. Differentepoch snapshot not final pairedgain; continued80-14/max2. Same12609params/789.904MiB peak, finitegroupstatistics, trainwellaggregation only. V263 negativeweights removed aftercomposition/SHAmanifest, original104/current252 intact. Atlas1026experiments/195deadends/217unresolved, incompleteancestry explicit.

V265 prepared copiedPOR terminaldec0/combine +retained151readout jointupdate, nativepredictionloss plustrainwellnormalizedvariance0vs.1/ramp5. Frozen113prior preserved, onePORaddition/eight with248/252. Held264max2, no trainedgain.

V264 final PERM89.415397483681, gain+0.009832626762, actualTOTAL80.615916837361; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds222.600644/peakMiB789.903809. No otherfolds.

V264 final PERM89.414811070835, gain+0.009246213917, actualTOTAL80.615711592865; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds219.909478/peakMiB789.903809. No otherfolds.

V265 both failed beforefirst optimizer step: cdf.diff(-1) interpreted as negativeorder, requires dim=-1. Initialfunction equals151POR77.926380760724 both, trainable265632/newcopies34688. No trained result. NewimmutableV266 fixesdiff(dim=-1) only, fresh151/copyinit/resetoptimizers/samehypothesis. Failure source/log/checkpoint evidence tracked, no mechanismconclusion.

V264 FINAL wellmeanPERM89.415397483681/best8/natural22, robust89.414811070835/best10/natural24, robustminuscontrol-0.000586412845, wellmeanvs248+0.001931888293. Actual144665/RAM/frozenexact/no diskreload. Individual222.600644/219.909478s/789.904MiB, pairwall228.643s. Retainbothpositive/control models. Actual264wellmean+252+151 sevenphase TOTAL80.625533220240, delta+0.000676160903 MICROnotmaterial/no808/nootherfolds/default5unchanged. OldPLAN obsolete26849 newparameters corrected by PLAN_ERRATA actual12609 without mutation. V265 firststepfailedonly, V266 freshdiffAPIfix launched onePORphase/copylateblock+151head/control-vrex; gainunproved.

V266 liveverified PIDs64340/13884 epoch3. ExactinitialPOR77.926380760724 both; currentcontrol77.539310605749/vrex77.531868980095, beststillinitial/no learnedgain yet. Finite nativeNLL (negative is density logscore, not failure), V-RExfinite/28traininggroups,265632updates/34688newcopiedparams, peak925.151MiB. Dropoutofallcopiedlateblocks eval/gradients enabled, old113frozen; max2/80-14. V265failed initialweights deleted withSHAmanifest; original151/264positivepreserved. Currentoptional7microTOTAL80.625533220240, notmaterial/no808/nootherfolds. Atlas1032experiments/197deadends/219unresolved, unresolvedancestryexplicit.

V267 prepared PERM context+mass jointupdate, carry264equalwell loss mechanism withoutrepairweights, conservative trainingonlyMixStyle p.25/strength.1 control. Held266twojobs/max2, no gainclaim.

V266 control finalPOR77.926380760724, actualTOTAL80.612475417994; gainvs151+0.000000000000. Sixphases alone, onePORaddition/no otherfolds; restoredRAM replay, not independent diskreload.

V266 vrex finalPOR77.926380760724, actualTOTAL80.612475417994; gainvs151+0.000000000000. Sixphases alone, onePORaddition/no otherfolds; restoredRAM replay, not independent diskreload.

V266 FINAL control/vrex POR77.926380760724 both/best0/natural14stop, learnedgain0; actual144665assembly/RAMbest/frozenexact/no diskreload. SoloTOTAL80.612475417994 is olddefault5 function replay, not newgain. Individual246.521914/248.109755s/925.151MiB, concurrentwall253.429s inclstartup/data/initialeval/finalrestore/excludesprep.265632trainable/34688newcopiedlateparams; close thiscoadapt/variance design, no beta/encoderwidthscan. V267 launched ownPERM154context+mass coadapt andsame264wellmean mechanism, p.25/strength.1 training-onlystyle versus matchedcontrol, onePERMaddition/max2/80-14. Gainunproved, currentoptional7 80.625533220240/default5/delivery unchanged/nootherfolds.

V267 liveverified PIDs84480/36084 bothepoch3: control89.370909727960/mixstyle89.368278460947, bestinitial89.404398334646/no learnedgain yet, below retained26489.415397483681. Same95329trainable(82720oldcontext+12609newmass), peak786.465MiB, bothgradgroupsnonzero/finite losses, traininggroupstats31/27, no extraGPU. Continue80-14/max2. Originalsourcefrozen, evalstyleidentity. V266negative best/last deleted afterfull144665assembly/SHAmanifest, original151/current264preserved. Atlas1035experiments/198deadends/220unresolved, incompleteancestryexplicit. Currentoptional7TOTAL80.625533220240/default5/delivery unchanged; no808/nootherfolds.

V268 prepared frozen154/newlearnedprototype memory15552 vsMLP15548 +12609mass, zero feature residual/onePERMphase, retain264equalwell mechanism withoutweights; no sample/labelretrieval. Held267max2, gainunproved.

V267 final PERM89.404398334646, gain-0.001166522273, actualTOTAL80.612067135198; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds241.300178/peakMiB786.465332. No otherfolds.

V267 final PERM89.404398334646, gain-0.001166522273, actualTOTAL80.612067135198; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds237.493554/peakMiB786.465332. No otherfolds.

V267 FINAL control/mixstyle PERM89.404398334646 both/best0/natural14stop/no learnedgain, below26489.415397483681. Full144665assembly/RAMbest/frozenSOURCE checks/no diskreload. Individual241.300178/237.493554s/786.465MiB, pairwall246.882s inclstartup/data/initialeval/finalrestore/excludesprep. Closecontext+mass coadapt and conservativeStyle here, no layer/strength grid. V268 launched frozen154 newrepresentationMLP28157/memory28161 incl12609mass, bound.25/zero feature residual/sameinitial89.404398334646, onePERMphase/equalwell learning/max2/80-14. Memory is purelatent parameters/no sample-labelretrieval. Gainsunproved; current7TOTAL80.625533220240/default5/delivery unchanged/nootherfolds. Research docs/deep_learning/research/perm_learned_memory_readout_20261001.md notesnear-capacity and delayedupstreamgradients.

V268 liveverified PIDs88004/68720 bothepoch6: MLPbest89.413337894012/memorybest89.413778421887, bothe5 best, below retained26489.415397483681. Snapshotmemory advantage+.000440527875 not finalgain; continue80-14. Actual28157/28161trainable/790.642and790.361MiB peak, sameinitial89.404398334646, finitegroupstats/losses. Source154fullyfrozen, memory no labels/samplestore. V267negativeweights deleted afterfull144665composition/SHAmanifest, positives264/154 retained. Atlas1038experiments/199deadends/221unresolved, incompleteancestryexplicit. Current7TOTAL80.625533220240/default5/delivery unchanged/no808/nootherfolds.

V269 prepared SW calibratedclass-balanced gateBCE vsordinary, same16770/bound2/conditionalCDF1/twofactorSAM, frozen104/no repairweights. Trainingprioronly/offsettraining not inference, oneSWaddition; held268slots/unproved.

V268 final PERM89.415840060079, gain+0.010275203160, actualTOTAL80.616071739100; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds227.018359/peakMiB790.641602. No otherfolds.

V268 final PERM89.415277247011, gain+0.009712390092, actualTOTAL80.615874754526; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds236.025189/peakMiB790.360840. No otherfolds.

V268 FINAL MLP PERM89.415840060079; memory-minus-MLP-0.000562813068. Actual seven-stage TOTAL80.625688121979, previous7delta+0.000154901739, MICROnotmaterial. Full144665 finite/official composition/RAM replay/frozen proof, no independent diskreload. Both checkpoints retained as positive/control; default5 and delivery unchanged/no808/nootherfolds. Representation gradient small, no memory-width sweep. V269 two SW jobs running. User requests larger-potential direction: next PERM raw-feature multiscale representation directly coupled to distribution readout, original154 retained/one joint additional phase; prepare while max2 occupied.

## 2026-10-01 larger-potential PERM direction (prepared, not trained)
User asks larger gains. Next controlled idea: original-input explicit scales[1,4,16] versus repeated[1,1,1], each scale directly supervised conditional distribution with nonzero readout; fixed within-stage alpha0->.1 schedule, original154 frozen, no268 repair weights or learned mixture gate. This differs from negative257 zero mass residual; small268 representation gradients are observed but not proven cause of final low gain. Research/plan: [PERM direct multiscale distribution](docs/deep_learning/research/perm_direct_multiscale_distribution_20261001.md). Not full TimeMixer reproduction, no gain promised, no GPU launched/automatic queue. Current269twoSWjobs remain active(max2); next branch jointly trains in one PERM additional phase, replacing optional repair if beneficial. Latest actually composed7-stage TOTAL80.625688121979, +0.000154901739 micro; default5/deliveries unchanged, original fullOOF>82 unmet.

V270 prepared actual PERM terminal encoding dec0/combine copies plus retained154 nativehead jointupdate vs same nativehead-only. No repairweights, original prior/gate fixed/nativeNLL/RPS1/decoder, onePERMphase/max2/80-14. Held269slots, not trained/unproved.

V269 control final SW74.133230187542, actualTOTAL80.616492493778; learned gain+0.011528606867, vs211-0.014392435085. Six optionalphases, default5; samecap16770/statebound2/equalradiusSAM/conditionalRPS1, calibratedclass-balanced BCE versus ordinaryBCE.

V269 balanced final SW74.143370538826, actualTOTAL80.620041616728; learned gain+0.021668958151, vs211-0.004252083802. Six optionalphases, default5; samecap16770/statebound2/equalradiusSAM/conditionalRPS1, calibratedclass-balanced BCE versus ordinaryBCE.

V269 FINAL ordinarySW74.133230187542/calibratedbalanced74.143370538826, pairedgain+0.010140351284, retained252delta-0.005857668987; both best5/naturalstop19, actual144665/RAMreplay/frozenproof/no independentdiskreload. Keep useful positive mechanism/control checkpoints; do not replace252. Individual449.070910/445.477191s/1119.828MiB, concurrentwall455.858s inclstartup/data/initialeval/finalrestore/excludesprep. TrainingQ-defaultprior.7228821516/logitshift-.9588033485 onlytrain, not inference. No classweight/bound sweep. V270 launched PERMhead-only82720 vs copiedlateencoder joint117408trainable, exactinitial15489.405564856919 verified during training. Max2/80-14/onePERMphase/nootherfolds. Currentbestactualoptional7TOTAL80.625688121979 unchanged/default5/deliveries preserved/fullOOF82unmet.

V270 live process commands verified bothGPUjobs, head_only/late_joint at epoch4 bestinitial89.405564856919/best0, no gain yet, eachpeak786.465MiB. Ownnative154head82720 versus head+actualcopiedlateblocks117408parameters; originalprior/gate frozen, no repairedheads loaded. Continue80-14/max2, onePERMphase/nootherfolds. Latestactualoptional7TOTAL80.625688121979 unchanged, strict808/fullOOF82 unachieved.

V271 prepared raw206+relative directcategoricalbranch, repeat1/1/1 vs maskedpool1/4/16, own directCE auxiliary independentofalpha/nonzeroreadout, fixedalpharamp0->.1 withinonePERMstage. Original154frozen/no repairweights; hold270twojobs, gainunproved/nootherfolds.

V270 final PERM89.405564856919, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds145.626368/peakMiB786.465332. No otherfolds.

V270 final PERM89.405564856919, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds147.725743/peakMiB786.465332. No otherfolds.

V270 FINAL bothPERM89.405564856919/best0/natural14/learnedgain0; actualTOTAL80.612475417994 olddefault5 replay, not improvement. Joint lateencodinggradient.05622446537/head.46505635977 nonzero, but nativehead/lateencoderadaptation didnotbeatinitial. Actual144665/frozenproof/RAMbest/no diskreload. Individual145.626368/147.725743s/786.465MiB, concurrentwall156.894s inclstartup/data/initialeval/finalrestore/excludesprep. Close thisscope at1e-5/3e-5, noLRgrid/broadbackbonefailureclaim. V271directcomponent rawrepresentation+readout launched repeat/multiscale same57568params, initial89.405566626894 slightly+.000001769975 fromdiscretization, notlearnedgain; onePERMstage/max2/80-14. Currentretainedoptional7TOTAL80.625688121979 unchanged/nootherfolds.

Userupdatedgoal: future candidates must updatebackbone+readout, no head-only launches. Current271freshrawfeatureencoders+categoricalheads alreadyjointtrain; prepared272fullcopiedSOURCEexpertencoders1e-5/directreadout3e-4, original154prior retained/frozen, effective271directsupervision mechanism but no271weights. OnePERMadditionalphase/max2/80-14, held271/unproved. Finished270negativecheckpoints removed withSHAmanifest afterfullassembly, positives/active/deliveries intact.

V271 final PERM89.439391359490, gain+0.033826502571, actualTOTAL80.624314693894; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds237.511100/peakMiB788.226562. No otherfolds.

V271 final PERM89.438759265115, gain+0.033194408196, actualTOTAL80.624093460863; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds239.721348/peakMiB787.983398. No otherfolds.

V273preparedchannelroute fresh96widegatedTCNdepth9/d1,4,16 +directPERMdistributionjointtrain, carry271positivelearningmechanism withoutweights; original154fixedprior/onePERMphase/max2/80-14. Differentcapacitynotmatchedcausalablation. Paired272fullwarmfeaturebackbone route; gainsunproved/nootherfolds.

V271 FINAL repeatPERM89.439391359490/multiscale89.438759265115 bothbest8/natural22; multiscale-minus-repeat-0.000632094375, repeatvs268+0.023551299411. Original154 retained, directrepresentation+readout learning gain; extra poolscale notproven. Actual144665/RAM/frozenproof/no independentdiskreload. Sevenphase actualTOTAL80.633931076773, previous7delta+0.008242954794, stillbelow808/nootherfolds/fullOOF82unmet; default5/delivery unchanged. Bothpositive/controlweights retained. Individual237.511100/239.721348s/788.227and787.983MiB, concurrentwall247.236s inclstartup/data/initialeval/finalrestore/excludesprep. V272fullcopiedexpertencoder+directreadout live; v273fresh96temporalbackbone+directreadout launched besideit, max2/80-14/onePERMphase/nomutatedhistoricsource/norepairedweightschain.

V272 liveprocesscommand verified PID70928/epoch11/best89.4207751145966; trainable1346678, bothfeaturebackbone anddirectreadout updated. Max2GPU/80-14/onePERMphase; oldSOURCE/154priorfrozen/no271trainedweights/nootherfolds. This supersedes prepared/heldtext.

V273 liveprocesscommand verified PID88956/epoch3/best89.42320118267263; trainable132064, bothfeaturebackbone anddirectreadout updated. Max2GPU/80-14/onePERMphase; oldSOURCE/154priorfrozen/no271trainedweights/nootherfolds. This supersedes prepared/heldtext.

V274prepared useful271rawencoder/directdensity jointtraining same57568/init/loss, fixedcomponent-share.1->.25/5epochramp only; comparefrozenexisting271control/no unchanged rerun/no271trainedweights, onePERMphase/max2/80-14. Held272/273slots/unproved/nootherfolds.

V273 final PERM89.434129958417, gain+0.028565101498, actualTOTAL80.622473203518; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds232.049750/peakMiB789.281250. No otherfolds.

V275prepared SWjoint newrawrepeatencoder+directcategoricalhead57568, fixedprobabilitymix vspriorrelativetilt sameencoder/nonzeroauxhard-softCE1/ramp5, original104prior/atomfixed/no252weights/oneSWphase/max2/80-14. Earlier262/263 hadauxCE.5, nofalsefirstsupervisionclaim. Held272/274/unproved/nootherfolds.

V272 final PERM89.424282481650, gain+0.018717624731, actualTOTAL80.619026586650; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop17/31, trainingseconds496.498249/peakMiB971.438965. No otherfolds.

V274 final PERM89.482758477027, gain+0.077193620108, actualTOTAL80.639493185032; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds212.270299/peakMiB788.226562. No otherfolds.

V272 FINAL PERM89.424282481650, actualTOTAL80.619026586650, vsretained271-0.015108877840; best/stop17/31, actual144665/RAM/frozenverified/no independentdiskreload. Positivelearningfromprior but backbonevariant doesnotbeat cheaperretainedrawencoder; no fair isolatedcapacity claim, no adoption/no newwidth-orLRgrid. Individual496.498s/peak971.439MiB, launchwall502.894s inclstartup/data/initialeval/finalrestore/excludesprep, overlapexplicit/no measuredserialspeedup. Source/config/score/predictions kept; unpromisingredundantweights removablewithSHAmanifest, original154/271/active274/275 retained.

V273 FINAL PERM89.434129958417, actualTOTAL80.622473203518, vsretained271-0.005261401073; best/stop8/22, actual144665/RAM/frozenverified/no independentdiskreload. Positivelearningfromprior but backbonevariant doesnotbeat cheaperretainedrawencoder; no fair isolatedcapacity claim, no adoption/no newwidth-orLRgrid. Individual232.050s/peak789.281MiB, launchwall238.655s inclstartup/data/initialeval/finalrestore/excludesprep, overlapexplicit/no measuredserialspeedup. Source/config/score/predictions kept; unpromisingredundantweights removablewithSHAmanifest, original154/271/active274/275 retained.

V274 FINAL alpha25PERM89.482758477027/best8/natural22, vsalpha.1 samebody/loss/init/control+0.043367117537; directrawencoder+readout jointlylearned, no inferencealpha scan/no oldrepairweights. Actual144665/RAM/frozenproof/no independentdiskreload; actualsevenphaseTOTAL80.649109567911/previous7delta+0.015178491138, strict808false/nootherfolds/fullOOF82unmet/default5/deliveryunchanged. Retainpositiveweights/control271. Individual212.270299s/788.227MiB/launchwall222.251s inclstartup/data/initialeval/finalrestore/excludesprep, overlapdifferenttasks/no equalworkserialclaim. V275 SWnewrawencoder/directhead mix/tilt bothGPUactive, allnewbackbones+readoutsjointtrain/max2/80-14/oneSWphase.

V275 both actualGPUprocesses verified active; SWrawfeaturebackbone+directdistribution jointlytrain,57568 each, probabilitymix versuspriorrelativetilt. InitialSW74.121752689769 common vs104tinyfloor/discretizationdifference notgain. Staggeredlaunch/partialoverlap with274 recorded, max2/80-14/oneSWphase, nootherfolds. V272/273unpromisingweights deleted afteractualassembly/SHAmanifest, original154/positive271/current274/active275 untouched. CurrentactualsevenTOTAL80.649109567911 below808/fullOOF82unmet; default5/deliveryunchanged.

V276prepared training-inputquantilePLE9rawcurves/max8segments vs scalarrepeat sameactive-dimension/NNparams, mergecloseknots/no zero-padchannels/label-Q fitselection. Newrawfeatureencoder+readout jointtrain/fixedshare.25/directCE, original154prior/no274weights/onePERMphase/max2/80-14. ResearchNUMERIC_PLE_RESEARCH registered; gainunproved/nootherfolds.

V275 mix final SW74.137236359285, actualTOTAL80.617894653889; learned gain+0.015483669516, vs211-0.010386263342. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V275 tilt final SW74.121752689769, actualTOTAL80.612475369558; learned gain+0.000000000000, vs211-0.025869932858. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V275 FINAL mixSW74.137236359285/best34/natural48; tilt74.121752689769/best0/natural14; pairedmixgain+0.015483669516, vsretained252-0.011991848527. Actual144665/RAMbest/frozenproof/no diskreload; retain usefulmix/control forfusionmechanism, notreplace252/noextraSWrepairchain. Individual454.294959/214.124284s/1242.015and1241.765MiB; staggeredpartlyconcurrentwall461.826s inclstartup/data/initialeval/finalrestore/excludesprep, notequal-startspeedcomparison. Diagnostic-only default/continuousstrata recorded, nottrainingweights ororacle. Currentactual7TOTAL80.649109567911/default5/deliveryunchanged/below808/fullOOF82unmet. V276numericPLE/scalarcompare body+readout bothjointtrain live/max2/80-14/onePERMphase.

V276 final PERM89.472074255939, gain+0.066509399020, actualTOTAL80.635753707651; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds239.697639/peakMiB787.984375. No otherfolds.

V276 final PERM89.482745940151, gain+0.077181083232, actualTOTAL80.639488797125; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds243.965727/peakMiB787.984375. No otherfolds.

V276 pair finalized: PLE89.482745940151 vs scalar89.472074255939, fair delta+0.010671684212; versus retained274 -0.000012536876, no adoption. Both best8/stop22, new encoder/readout jointly trained. Retain useful positive and matched control. No remaining folds.

V277 prepared: copied104 late/full backbone plus directly supervised categorical component, no repair-chain, joint encoder/readout; unproved.

V277 launched and actual processes verified: latePID28564/fullPID47148; identical initialSW74.121752689769, trainable144816/861424 respectively. Joint backbone/readout, no teacher or prior repair weights. CUDA backward warnings mean no bitwise training reproducibility claim. Terminal results and assembly pending.

V277full failed initial training CUDA OOM (no trained result). Late remains active; newV278 full checkpointed source from original104, same function/init/loss/budget prepared.

Live verified full: epoch2, best74.1233306382582, peakMiB1505.5302734375; not final gain, assembly pending.

Live verified late: epoch10, best74.1231236298602, peakMiB1343.3193359375; not final gain, assembly pending.

V279 prepared, heldGPU slots: new377inputneighborhood encoder+continuousvaluecoordinate32basis, ordered/shuffled causalpair. Nottrained/unproved.

V277 late final SW74.123123629860, actualTOTAL80.612955198590; learned gain+0.001370940091, vs211-0.024498992767. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V278 early-stop requested: At epoch12 best at2 remains74.123330638258, below SW25274.149228; validation falls while directCE decreases. Same late scope already final below retained. Bounded early stop at boundary, not convergence claim.

V279 ordered failed startup: infinite terminal-edge midpoint createsNaN coordinate; initial0 invalid/no validscore. Shuffled neverlaunched. New280 finiteoriginalphysicalcenter correction, notrainedweights resume.

V278 full final SW74.123330638258, actualTOTAL80.613027651529; learned gain+0.001577948489, vs211-0.024291984369. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

Closed paired scope: lateSW74.123123629860 natural16, fullSW74.123330638258 bounded13; full-late+0.000207008398, below252-0.025897569554, no adoption. Actualassembled144665/RAMreplay/frozenproof, no independentdiskreload. Budgets differ, not fullconvergence causalproof; close this fixedLR/alpha/readout scope only. Unpromising checkpoints may be cleaned after assembly.

Finished unpromising277/278 and failed279 checkpoints cleaned: 6files/85251138bytes, SHAmanifest in v278. Source/config/log/report/predictions and original104/positive252/active280 retained.

V280 both live verified; originalinitialSW74.121752689769 restored finitecoordinate fix,31248trainable. At epoch2 orderedbest74.125062885965/shuffled74.122079887753, peak1141.03125MiB each. Early snapshots not finalgain. Branch/valueencoder nonzero secondstep gradients recorded. No otherfolds.

Prepared281PERM jointnewencoder/densityalpha.5 provenroute continuation; prepared282SWjointstate/density newencoder ordinary/balanced pair. Bothhold280GPU slots. No newgain/noextra repair stages.

V280 ordered final SW74.130726840713, actualTOTAL80.615616322388; learned gain+0.008974150944, vs211-0.016895781914. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V280 shuffled final SW74.128705295243, actualTOTAL80.614908781474; learned gain+0.006952605474, vs211-0.018917327384. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V280 final orderedSW74.130726840713/best5/stop19 vs shuffled74.128705295243/best4/stop18; fairdelta+0.002021545471, below252-0.018501367099. Actualall144665 assembly/RAM/frozenverified, no diskreload. Notadopted/nootherfolds. Perstrata diagnostics saved, notoracleselection. Timingwall255.320978s/individual249.536854,243.394834s/peak1141.03125MiB; codeprep/CPUassembly excluded. Close thisconfig, not wholeenergyfamily.

V280 stratifieddiagnostic: orderedcontinuous17.475277159184 vsoriginal10417.631889060480 andshuffled17.480330711490. Default98.796697 vsoriginal98.715632 creates overallgain. No continuousbenefit; noadoption. Finishedunpromisingpaircheckpoints deletedSHAmanifest, source/config/report/predictions/diagnostic retained, active281/282 andpositive104/252 preserved.

Live verified alpha50: epoch7, best89.498908539006; not finalgain. Atomicbest selects currentmixalpha too. Nootherfolds.

Live verified ordinary: epoch8, best74.12175268976877; not finalgain. Atomicbest selects currentmixalpha too. Nootherfolds.

V282ordinary boundedstop requested after15+ epochs; best74.122510 belowretained252/275. Compare balanced with same actualcompleted epoch budget once ordinaryterminal; no convergenceclaim or unequalbudget finalfairgain.

V282 ordinary final SW74.122510119287, actualTOTAL80.612740469889; learned gain+0.000757429518, vs211-0.025112503340. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V281 final PERM89.498908539006, gain+0.093343682087, actualTOTAL80.645145706724; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds230.226585/peakMiB788.226562. No otherfolds.

V281actualoptional7 TOTAL80.654762089603, PERM89.498908539006, gainvs274TOTAL+0.005652521693. Selectedbest3 actualalpha0.300000012, .5schedulereachedlaterbutnotselected. Currentbestretained, newrecipe/predSHA, original5/old274recipe/delivery preserved. Strict808false/fullOOF82unmet/nootherfolds; RAMproof+actualcomposition, no independentdiskreload.

Verifiedlive alpha30: {"status": "running", "pid": 72020, "stage": "alpha30", "epoch": 2, "best": 89.41754394419156, "best_epoch": 2, "bad": 0, "peak_MiB": 788.2265625}; notfinalgain. 282balanced comparison cap18 definedseparately, originalmaximum80/schedule80 unchanged.

Verifiedlive balanced: {"status": "running", "pid": 48848, "stage": "balanced", "epoch": 2, "best": 74.12175268976877, "best_epoch": 0, "bad": 2, "peak_MiB": 1241.642578125}; notfinalgain. 282balanced comparison cap18 definedseparately, originalmaximum80/schedule80 unchanged.

PreparedPOR284 rawfeature encoder+directcategorical distribution withtruejointRPS; frozen151 prior, onePORaddedphase, held283/282GPUslots. No newgain claimed.

PreparedSW285 jointencoder/state/density share.25, carriedfairclassbalance mechanism but no282weights/extraSWphase. Comparefixed18prefix andabsolute252 separately.

V282 balanced final SW74.143119674512, actualTOTAL80.619953814218; learned gain+0.021366984743, vs211-0.004502948115. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V283 final PERM89.493314539170, gain+0.087749682251, actualTOTAL80.643187806782; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds307.513264/peakMiB788.226562. No otherfolds.

POR284 STOPPED invalidSWscorecolumn, initial0.000678 notPORscore. New286 correctsEvaluator toself.c andknownordinaryinitialscorecheck, original151restart/no284weights. SW285continues.

V282 final ordinarySW74.122510119287/best14/bounded18 vsbalanced74.143119674512/best17/bounded24. Matched18prefix samebest17 state/pred exactlymatches finalmodel, fairdelta+0.020609555225; actualoverrun24 not concealed. Below252-0.006108533300, noadoption/nootherfolds. Positivebalance/control/matchedproof kept; actual144665assembly/RAM/frozenproof, no independentdiskreload. Strata: {"SW104": {"default": 98.71563225845625, "continuous": 17.631889060480088, "overall": 74.12175282815869, "continuous_zero_fraction": 0.6661728845233483}, "SW252": {"default": 98.64493054166374, "continuous": 17.884868139910104, "overall": 74.14922820781278, "continuous_zero_fraction": 0.6639166799608013}, "ordinary": {"default": 98.65618782716787, "continuous": 17.770924133590587, "overall": 74.12251011928777, "continuous_zero_fraction": 0.6643952688074022}, "balanced": {"default": 98.7661680040883, "continuous": 17.586257646102915, "overall": 74.14311967451279, "continuous_zero_fraction": 0.6669021627657877}}.

V283 finalPERM89.493314539170/best8/stop22, vs281-0.005593999836, vs274+0.010556062143. Notadopted, usefulfixed30 positivecontrol kept, close localsharegrid. Bestcombo28180.654762 preserved/nootherfolds.

286 failed before initialevaluation: inheritedEvaluator constructor missingmode forwarding. New287 correctsforwarding, same target-correct evaluator/initialknownscore guard andoriginal151freshstart; no286weights. No286 validscore/training result.

Actualprocessverified raw: {"status": "running", "pid": 90284, "stage": "raw", "epoch": 4, "best": 77.94491086237232, "best_epoch": 3, "bad": 1, "peak_MiB": 981.986328125}. POR287normalinitialscoreexact15177.926380760724 afterevaluator target/constructor corrections; actualtrainingcounts only287 valid, not284/286. No finalgain/otherfolds claim.

Actualprocessverified balanced: {"status": "running", "pid": 24904, "stage": "balanced", "epoch": 38, "best": 74.15237674585131, "best_epoch": 32, "bad": 6, "peak_MiB": 1241.642578125}. POR287normalinitialscoreexact15177.926380760724 afterevaluator target/constructor corrections; actualtrainingcounts only287 valid, not284/286. No finalgain/otherfolds claim.

Invalid284 lastcheckpoint cleaned16,250,023bytes withSHAmanifest; compactinvalid-evalhistory/config/source retained. Corrected287 startsoriginal151, noinvalidweights dependence.

Lastlive snapshot: epoch8 best77.95085361757371, not finalgain; checkpointrestore/actualcombinedscore pending.

Lastlive snapshot: epoch43 best74.15237674585131, not finalgain; checkpointrestore/actualcombinedscore pending.

PreparedPERM288 width96 direct-supervised rawencoder184000params, controlledagainstpositive283width48. No283/281weights/extraPERMphase, held287/285slots.

V285 balanced final SW74.152376745851, actualTOTAL80.623193789187; learned gain+0.030624056083, vs211+0.004754123224. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V285finalSW74.152376745851/best32/stop46, versus252+0.003148538039. Actualoptional7TOTAL80.655864077917/delta+0.001101988314; all144665actual/RAMrestore/frozen proof, notdiskreload. Matched18historybest74.149603837559/delta282+0.006484163047, no retainedprefixcheckpoint forreplay; do notattributefull46gainsolelytoalpha .25 vsbounded18control. Strata{"SW104": {"default": 98.71563225845625, "continuous": 17.631889060480088, "overall": 74.12175282815869, "continuous_zero_fraction": 0.6661728845233483}, "SW252": {"default": 98.64493054166374, "continuous": 17.884868139910104, "overall": 74.14922820781278, "continuous_zero_fraction": 0.6639166799608013}, "SW285": {"default": 98.72446679129324, "continuous": 17.712561177589066, "overall": 74.15237674585221, "continuous_zero_fraction": 0.6657626655119762}}. Default5/oldrecipes/deliveries preserved, strict808false/nootherfolds/fullOOF82unmet.

PreparedSW289 statebound2to4, sameeffectivejointencoder/heads/share.25 andnatural80-14 budget aspositive285; no285weights/extraSWphase. HeldPOR287/PERM288slots.

PreparedPOR290 share25 samepositive287mechanism, no287weights/extraPORphase, newencoder/readoutjoint.

V287 final POR77.952002347840, gain+0.025621587116, actualTOTAL80.620161894129; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop9/23, trainingseconds346.776188/peakMiB981.986328. No otherfolds.

V288 final PERM89.482470023653, gain+0.076905166734, actualTOTAL80.639392226351; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop8/22, trainingseconds201.223310/peakMiB797.685059. No otherfolds.

V291 prepared new PERM sparse curve reacquisition encoder/readout point/window pair; no training/score yet. [study](results/deep_learning/perm_sparse_reacquisition_v291_20261002/README.md). One added PERM phase, original154 only, max2GPU/80-14, no other folds.

V290 final POR78.023526555255, gain+0.097145794531, actualTOTAL80.641619156353; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds447.645340/peakMiB981.986328. No otherfolds.

V290final POR78.023526555255/best11/stop25, gain151+.097145794531/gain287+.071524207415; actualoptional8 TOTAL80.685007816277, default5/old7/deliveries preserved. Strict808false/nootherfolds/fullOOF82unmet. PERM288negative width96 closed and best/last cleanedSHAmanifest, controls kept. V291point explicitly launched, windowheld, companionSW289 max2GPU.

V289 balanced final SW74.153726428999, actualTOTAL80.623666178289; learned gain+0.031973739231, vs211+0.006103806372. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V289 finalSW74.153726428999/best32/stop46, same80-14 budget vs285delta+.001349683148. Actualoptional8TOTAL80.685480205378, eachheadone addedphase. Strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW285": {"default": 98.72446679129287, "continuous": 17.71256117758909, "overall": 74.15237674585131}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}}. Smallgain/notmaterial breakthrough, no bound6/8grid. Original5/oldrecipes/deliveries unchanged; no otherfolds, strict808false/fullOOF82unmet.

V292 prepared PORshare50 newencoder/readout fromoriginal151, motivated verified290gain+.071524207415 vs287. OnePORaddedphase, no290weights, heldGPUslot. [study](results/deep_learning/por_direct_share_v292_20261002/README.md).

V291 final PERM89.515554793257, gain+0.109989936338, actualTOTAL80.650971895712; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds231.108907/peakMiB788.697754. No otherfolds.

V291point finalPERM89.515554793257/best5/stop19, vs281+.016646254251. Actualoptional8TOTAL80.691306394366, all144665/frozen/RAMbestreplay verified, notdiskreload. Windowpairedfinalpending, no causalpairgain claim yet. New36466encoder/readout joint,8phases/eachhead+1, strict808false/nootherfolds/fullOOF82unmet. POR292 explicit launched original151/share50, no290weights.

V291 final PERM89.510323066916, gain+0.104758209997, actualTOTAL80.649140791493; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds216.946259/peakMiB788.697754. No otherfolds.

V291paired final point89.515554793257 vswindow89.510323066916, fairdelta+0.005231726341; bothbest5/stop19/same36466params/initial89.405566626894, jointbody/readout grads and training masks changed. Adoptpoint, +.016646254251 vs281. Actual8TOTAL80.691306394366, strict808false/nootherfolds. Pair isolates selection granularity only, not individual sparsemax effects. Bothpositive/control weights kept. Timing{"first_process_started_utc": "2026-10-01T16:42:03.718643+00:00", "point_started_epoch": 1790872923.7186427, "window_started_epoch": 1790872989.5238235, "concurrency": 2, "maximum_GPU_jobs": 2, "terminal": true, "batch_wall_start": "first OS process creation; includes data prep and warmup", "code_preparation_excluded": true, "batch_wall_seconds": 290.45420360565186, "individual_seconds": {"point": 231.1089074999909, "window": 216.94625869998708}, "peak_MiB": {"point": 788.69775390625, "window": 788.69775390625}, "completed_utc": "2026-10-01T16:46:54.172846+00:00", "wall_end": "last training report write mtime; CPU composition excluded", "startup_overlap_partial": true}. Atlasbatch updated.

V293 prepared samepositive291sparse architecture/newbody+readout, conditional local17 versus matched291point; original154 only/no291weights/one addedPERMphase. [study](results/deep_learning/perm_sparse_local_v293_20261002/README.md). No newgain claimed.

Actualprocess verifiedPID74224, snapshot{"status": "running", "pid": 74224, "stage": "raw", "epoch": 15, "best": 78.0087907758016, "best_epoch": 3, "bad": 12, "peak_MiB": 981.986328125}, unfinished/no finalgain. Jointnewencoder/readout, ownloss/max2GPU/80-14, oneaddedtargetphase, nootherfolds.

Actualprocess verifiedPID34580, snapshot{"status": "running", "pid": 34580, "stage": "local", "epoch": 1, "best": 89.4055666268939, "best_epoch": 0, "bad": 1, "peak_MiB": 788.69775390625}, unfinished/no finalgain. Jointnewencoder/readout, ownloss/max2GPU/80-14, oneaddedtargetphase, nootherfolds.

V292 final POR78.008790775802, gain+0.082410015078, actualTOTAL80.637198422517; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds195.790577/peakMiB981.986328. No otherfolds.

V294 prepared SWjoint sparse curve selection encoder/state/density point/window pair, original104 only/no289/291weights, oneaddedSWphase. [study](results/deep_learning/sw_sparse_reacquisition_v294_20261002/README.md). No newgain yet.

V292 finalPOR78.008790775802/best3/stop17, versus290-.014735779454, selectedalpha0.30000001192092896 not .5. Rejectshare50 escalation, retain25; no.75/1grid. Actualassembly retained, redundantbest/lastdeletedSHAmanifest, positives151/287/290 untouched. Nootherfolds/fullOOF82unmet.

SW294 point explicitstartedPID83792 andinitialscoreexact10474.121752689769; snapshot{"status": "running", "pid": 83792, "stage": "point", "epoch": 5, "best": 74.13335156100214, "best_epoch": 3, "bad": 2, "peak_MiB": 1147.96630859375}. Windowheld293 process pending finalrestore; max2GPU. No finalgain/otherfolds.

V293 final PERM89.514219912256, gain+0.108655055337, actualTOTAL80.650504687362; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds180.873825/peakMiB788.697754. No otherfolds.

V293local17 finalPERM89.514219912256/best5/stop19 vs291point-.001334881001. Same36466params/init/loss and prior154; only selectorlogits pooling changed. Rejectlocal17/closewindowgrid, point positivekept. Redundantbest/last cleanedSHAmanifest afteractualassembly; nootherfolds/fullOOF82unmet.

V294two actualprocesses verifiedPID83792/21088, snapshots{"point": {"status": "running", "pid": 83792, "stage": "point", "epoch": 10, "best": 74.15181714457486, "best_epoch": 6, "bad": 4, "peak_MiB": 1147.96630859375}, "window": {"status": "running", "pid": 21088, "stage": "window", "epoch": 2, "best": 74.12175268976877, "best_epoch": 0, "bad": 2, "peak_MiB": 1145.0009765625}}, initial104score exact, no finalgain yet. Max2GPU/80-14, newencoder/readouts jointly train, oneSWaddedphase.

V295 prepared PORjoint sparsecurve encoder/readout pair fromoriginal151, no290/291weights, onePORaddedphase, held294slots. [study](results/deep_learning/por_sparse_reacquisition_v295_20261002/README.md). No training/score yet.

V294 point final SW74.151817144575, actualTOTAL80.622997928740; learned gain+0.030064454806, vs211+0.004194521948. Six optionalphases, default5; new36515sparse_rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V294point finalSW74.151817144575/best6/stop20, vs289-.001909284424; strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}, "SW294point": {"default": 98.86811932865155, "continuous": 17.380759647269663, "overall": 74.15181714457486}}. Actual144665/RAMbest/frozen verification, no diskreload. Windowongoing, pairedfinalpending; noadoption/cleanup beforecomparison. POR295point explicitlylaunchedPID78300 withwindowheld, companionSW294windowPID21088/max2GPU.

V296 prepared history-supported SWrankone firstlinear4member sharedencoder/distribution; independent vs pooledcontinuousCE pair, original104 only/no294weights, oneSWaddedphase. [study](results/deep_learning/sw_rankone_continuous_v296_20261002/README.md). No newgain/score yet, heldGPUslots.

V294 window final SW74.145463813500, actualTOTAL80.620774262864; learned gain+0.023711123731, vs211-0.002158809127. Six optionalphases, default5; new36515sparse_rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V294paired final point74.151817144575/best6/stop20 vswindow74.145463813500/best14/stop28, same36515params/initial74.121752689769/80-14 budget, delta+0.006353331075. Bothbelowretained289; strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}, "SW294point": {"default": 98.86811932865155, "continuous": 17.380759647269663, "overall": 74.15181714457486}, "SW294window": {"default": 98.77535448509161, "continuous": 17.572885559050093, "overall": 74.14546381349975}}. Pointcontinuous worsens, noadoption/noSWcontinuous breakthrough. Keepfairgranularitypositive/control checkpoints and predictions; close standalone sparseSW route. Actual144665assembly/RAMbest/frozen checks, notdiskreload/fullOOF. Batchtiming{"point_process_start_utc": "2026-10-01T16:51:45.825622+00:00", "point_start_epoch": 1790873505.825622, "maximum_concurrent_GPU": 2, "terminal": true, "includes_prep_initial_train_final": true, "excludes_codeprep_CPUcomposition": true, "window_start_epoch": 1790873577.4945474, "concurrency": 2, "not_isolated_speed_comparison": true, "batch_wall_seconds": 462.9522874355316, "individual_seconds": {"point": 256.6073596000206, "window": 383.6069831000059}, "peak_MiB": {"point": 1147.96630859375, "window": 1145.0009765625}, "partial_overlap": true, "wall_end": "last trainingreport write; no codeprep/CPUcomposition"}.

V296independent actualprocessPID68436 launched alongsidePOR295pointPID78300; snapshot{"status": "running", "pid": 68436, "stage": "independent", "epoch": 1, "best": 74.12175268976877, "best_epoch": 0, "bad": 1, "peak_MiB": 2424.8310546875}, pooledheldslot. New37535sharedencoder/rankone/readout/state jointlytrained, original104 only/no294weights; oneSWaddedphase/max2GPU. No gain claimed.

Research completed, no training by research agent: [historically positive TabM transfer](results/deep_learning/cross_domain_research_20261001/TABM_CONTINUOUS_ENSEMBLE_RESEARCH_20261002.md), [ConR false-neighbor backup, research only](results/deep_learning/cross_domain_research_20261001/CONR_FALSE_NEIGHBOR_RESEARCH_20261002.md). Historicalfold/fullOOF not directly subtracted from currentfold.

V295 final POR78.006867884719, gain+0.080487123995, actualTOTAL80.636621555193; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds364.607663/peakMiB981.824707. No otherfolds.

V295point finalPOR78.006867884719/best11/stop25, vs290-.016658670536, noadoption/windowcomparisonpending. WindowPID38784 andSW296independentPID68436 actualprocessverified, states{"por_sparse_reacquisition_v295_20261002": {"status": "running", "pid": 38784, "stage": "window", "epoch": 1, "best": 77.94138902452742, "best_epoch": 1, "bad": 0, "peak_MiB": 981.82470703125}, "sw_rankone_continuous_v296_20261002": {"status": "running", "pid": 68436, "stage": "independent", "epoch": 9, "best": 74.15261198144826, "best_epoch": 6, "bad": 3, "peak_MiB": 2424.8310546875}}. Allnewencoder/readouts jointlytrained, max2GPU/80-14/oneaddedphaseperhead/max8. Currentactualrecipe80.691306394366 unchanged; nootherfolds/fullOOF82unmet.

V297 prepared NEWrawbranch input-only wellFiLM pair(actual vs64trainreference), prior104actualbank unchangedbotharms, jointnewencoder/density/state/condition, oneSWaddedphase; nooldrepairweights. [study](results/deep_learning/sw_raw_well_condition_v297_20261002/README.md). NoGPUscore yet, held295/296slots.

V295 final POR78.006433806844, gain+0.080053046121, actualTOTAL80.636491331830; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds538.699527/peakMiB981.824707. No otherfolds.

V296 independent final SW74.153530676552, actualTOTAL80.623597664932; learned gain+0.031777986783, vs211+0.005908053925. Six optionalphases, default5; new37535rankone_sharedencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V295final paired point78.006867884719 vswindow78.006433806844, delta+0.000434077875; bothbest11/stop25/same36466params/initialfunction. Bothbelow290 by.01666/.01709; tiny localgranularity gain is not practicalbreakthrough. Rejecttransfer/nofurtherselectorgrid, redundantbest/last deletedSHAmanifest afteractualassembly. Timing{"point_start_epoch": 1790873806.9309926, "maximum_concurrent_GPU": 2, "terminal": true, "includes_prep_initial_train_final": true, "excludes_codeprep_CPUcomposition": true, "window_start_epoch": 1790874232.04695, "point_individual_seconds": 364.6076627000002, "point_peak_MiB": 981.82470703125, "partial_overlap": true, "batch_wall_seconds": 970.6491668224335, "individual_seconds": {"point": 364.6076627000002, "window": 538.6995268999599}, "peak_MiB": {"point": 981.82470703125, "window": 981.82470703125}, "wall_end": "last trainingreport write, excludes laterCPUassembly/codeprep"}. Strict808false/fullOOF82unmet.

V296independent finalSW74.153530676552/best17/stop31/peak2424.831MiB, versus289-.000195752447. Strata{"SW252": {"default": 98.64493054166336, "continuous": 17.88486813991017, "overall": 74.14922820781207}, "SW289": {"default": 98.72311025198368, "continuous": 17.72012680313723, "overall": 74.1537264289993}, "SW296independent": {"default": 98.83016481677804, "continuous": 17.473587002886575, "overall": 74.15353067655211}}. Full144665/RAMbest/frozenverified, notdiskreload. PooledstartedPID21508, pairedgainpending; noadoption/currentcombo80.691306 unchanged. Newrankgains group actualgradient.00184643, allnewbody/readout/state jointlytrained.

ActualprocessPID21508 pooled launched, snapshot{"status": "running", "pid": 21508, "stage": "pooled", "epoch": 2, "best": 74.12175268976877, "best_epoch": 0, "bad": 2, "peak_MiB": 2424.8310546875}, unproved. Original104 only, oneSWaddedphase/newencoder+readout/state joint/max2GPU/80-14; nootherfolds.

ActualprocessPID73284 well launched, snapshot{"status": "running", "pid": 73284, "stage": "well", "epoch": 3, "best": 74.12210756298634, "best_epoch": 3, "bad": 0, "peak_MiB": 1280.80224609375}, unproved. Original104 only, oneSWaddedphase/newencoder+readout/state joint/max2GPU/80-14; nootherfolds.

Research: [SW newrawbranch wellconditioning, notfirstwholewellmethod](results/deep_learning/cross_domain_research_20261001/SW_RAW_WELL_CONDITIONING_RESEARCH_20261002.md). New297paired smallFiLM prepared/actualwell armstarted; prior104 keepsactualbankbotharms, reference64traininputs only.

V297 well final SW74.160030223020, actualTOTAL80.625872506196; learned gain+0.038277533251, vs211+0.012407600393. Six optionalphases, default5; new65265raw_wellencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

## Verified well-arm completion; reference control active
Well completed at best17/stop31, SW74.16003022301975; own-best RAM replay/frozen weights and actual144665 CPU composition checked, not independent disk reload. Optional8 POR290/PERM291/SW297 well TOTAL80.69351272227345, +0.00220632790714 versus prior optional8. No >80.8 qualification, no remainingfolds/fullOOF; default5 and earlier deliveries unchanged. Saved separate WELL_EIGHT_STAGE_RECIPE.json and predictions; retain well positive checkpoint while reference arm completes.
Default SW98.82679621708131 versus28998.72311025198368, but continuous SW17.502752767347182 versus28917.72012680313723 and25217.88486813991017. Thus this is a default-state gain, NOT improved continuous readout. Whole-well causality remains unproved until same-capacity reference comparison. Individual563.871727s/peak1280.802MiB includes data prep/initial/train/finalrestore; pair wall pending. Reference PID75804 launched after well exit, overlaps296pooled; at most2GPU. No descriptor/bound grid based on this small absolute result.

V296 pooled final SW74.153750372820, actualTOTAL80.623674558626; learned gain+0.031997683051, vs211+0.006127750193. Six optionalphases, default5; new37535rankone_sharedencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Paired completion
Independent SW74.153530676552, pooled74.153750372820; independent-minus-pooled-0.000219696268. Pooled-minus289+0.000023943821. Continuous scores independent17.473587002887, pooled17.469171752586, below25217.88486813991017. No continuous breakthrough or adoption; close this independent-v-pooled auxiliaryCE setting, not historical TabM family. Original104 retained; allnew encoder/rankgains/readouts actuallytrained. Checkpoints retained temporarily as completed matched controls until current bank comparison closes; no automatic extra stages/folds. Paired wall1533.245s, individual independent687.077s/pooled762.144s, partial overlap/gap prevents speed ranking. Actual144665 assembly/own-best RAM replay, no independent disk reload.
V298 scalar launched; conditional held until GPU slot. Preserve291 weights, use architecture/randominit only and original154 frozen prior. Same sparse encoder+continuous readout train jointly; bound actualshare[0,.6], learn one scalar or feature-dependent49param share, activecapacitydifference48 disclosed. No extra sequentialphase; max8/strict80.8/fullOOF82 pending.

V297 reference final SW74.154114272860, actualTOTAL80.623801923640; learned gain+0.032361583091, vs211+0.006491650233. Six optionalphases, default5; new65265raw_wellencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final matched contrast
Well SW74.160030223020, reference74.154114272860; matched well-minus-reference+0.005915950160. Continuous well17.502752767347, reference17.463982441198, matched+0.038770326149; BOTH must be compared to retained28917.72012680313723/25217.88486813991017 before declaring continuous gain. Optional8 actualTOTAL80.693512722273, strict808 false; separate positive candidate and matched control retained, no additional stages/folds. Individual well563.872s/reference366.630s, pairwall969.381s incl stagger and overlap against296, not speed ranking. Bank scope evidence applies only to newraw FiLM, not globally to all lithology/fluid modeling. Further descriptor/bound sweeps not justified by this small gain; next priority substantive continuous representation.
V299 flat launched; hierarchical held for freeGPUslot. Fresh SW encoder/state/group-conditioned coarse+fine readout jointly3e-4, train-only8groups of original256bins, flat versus conditionalnormalization matchedparams/init; rawbody pointwise limitation disclosed. Original104prior/no289 or297 trainedweights. OneSWaddedphase/max8/80-14/strict808/fullOOF82pending.

V298 final PERM89.466090565545, gain+0.060525708626, actualTOTAL80.633659416013; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds214.688933/peakMiB788.700195. No otherfolds.

V298 final PERM89.409738092787, gain+0.004173235869, actualTOTAL80.613936050548; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds194.641268/peakMiB788.701660. No otherfolds.


## Final paired negative
Scalar PERM89.466090565545, conditional89.409738092787; conditional-minus-scalar-0.056352472757, scalar-minus291-0.049464227712. Continuous scalar74.037494684582/conditional74.091162808666 versus29174.001066339761. Actual selected scalarshare0.184601694345, nominal .3 is not actual alpha. Newbody/readout/allocation grads nonzero, original154 frozen/RAMbest replay/144665actualassembly. No adoption: close this early adaptive allocation setting, retain original291 and earlier candidates. Pairwall279.053s, individual scalar214.689s/conditional194.641s; partial overlap, no speedclaim. Redundant negative weights eligible for manifest cleanup after this record; compact predictions/source/config evidence retained.
Clarification of v298: conditional continuous74.091162808666 vs scalar74.037494684582 (paired+.053668124085) and retained29174.001066339761 (+.090096468906), while default97.287981640665 versus29197.494555208792 loses. Frozen gate does NOT freeze official default action since density changes utility competition. No adoption, but retain conditional positive-mechanism/control weights; future state+density coupling is a distinct fresh joint phase, not adaptive bound grid or stacked repair.
Checkpoint cleanup pending: automatic approval rejected native fixed-path deletion twice with blocked_by_policy, no files removed. Manifest records planned targets/deleted=false; conditional298 and pooled296 positives plus allactive/current models retained. Training continues.
V300 joint_gate launched; fixed_gate held for slot. Fresh joint sparseencoder/readout/conditionalshare/state from original154 only; no298weights, state residualbound2 and original ownBCE/NLL/RPS/decoder. Goalrecoverdefaulttradeoff while preserving continuousgain, max8whole/onePERMaddedstage/strict808/OOF82pending.

V299 flat final SW74.153482146373, actualTOTAL80.623580679369; learned gain+0.031729456604, vs211+0.005859523746. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.

V299 hierarchical final SW74.155778635294, actualTOTAL80.624384450492; learned gain+0.034025945525, vs211+0.008156012667. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final matched result
Flat SW74.153482146373, hierarchical74.155778635294; paired overall+0.002296488921, continuous flat17.382210710024/hierarchical17.404116921379, pairedcontinuous+0.021906211355. Hierarchical-minus297-0.004251587726, not adopted. Compare absolutecontinuous25217.88486813991017; no continuous breakthrough from a local relative benefit alone. Same actual8boundaries/fit161423rows/58777params and sourceinitialfunction, allnewencoder/readout/state grads verified/144665composition/RAMbest replay, not independentdiskreload. Individual flat374.755s/hierarchical371.262s, pairwall420.451s, partial overlap with otherjobs, no speedranking. Do not sweep groupcount/quantile/headwidth on thissmall overall effect. No additionalfolds, preserve current optional8/default5/oldfullOOFdelivery.
V299 finalized: hierarchy SW74.155778635294 versusflat74.153482146373 (+.00229649), continuous17.404116921379 versus17.382210710024 (+.02190621); below current297 overall and252 continuous, no adoption/groupcount grids. Positive/localcontrol evidence preserved, cleanup remains policyblocked. V300 bothjoint/fixedGate running; sameinitial154 andfixedGate earlytrajectory reproduces298conditional, finalpending.
V301 density_owned launched while300jointGate continues; no more than2GPU. Change only state gradient into freshrawbody, statehead stilllearns; density encoder/readout jointlytrain, same57617capacity/init/loss as289, no289trainedweights/newsequentialphase. Explicit historicalcontrol scope and continuum references, strict808/OOF82 unmet.

V300 final PERM89.409738092787, gain+0.004173235869, actualTOTAL80.613936050548; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop3/17, trainingseconds210.240520/peakMiB787.851074. No otherfolds.

V300 final PERM89.419650223819, gain+0.014085366900, actualTOTAL80.617405296409; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds438.768973/peakMiB787.854004. No otherfolds.


## Final paired outcome
Fixed PERM89.409738092787, joint89.419650223819, pairedwhole+0.009912131032/pairedcontinuous+0.412532501309, joint-minus291-0.095904569438. Strata recorded separately; no adoption or statebound/LR grid. Fixed control exactlyreplays298conditional144665preds=True; ownbestRAM/frozen/actualassembly, no independentdiskreload. Active49parameterdifference intendedupdate-scope. Individual joint438.769s/fixed210.241s, pairwall447.835s partialoverlap/no speedranking. Localpositivecontrols retained; checkpointcleanup for unpromising redundantstates remains policyblocked, no deletion claim. Currentoptional8/default5 intact, strict808/OOF82 unmet.
V302 grouped/full explicitlylaunched as twoGPUtasks after300/301terminal. Fresh76592POR encoder/directreadout/numericinteraction jointlytrain from original151 prioronly, no oldcurveattention/290weights. Groups3tokenskeep QK active; identityonlycontrol rejected. Historicalnegative numericattention disclosed, this is connectivity/placement transfer; max8/onePORphase/strict808/fullOOF82pending.

V301 density_owned final SW74.148904173757, actualTOTAL80.621978388954; learned gain+0.027151483988, vs211+0.001281551130. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final gradient-routing outcome
SW74.148904173757/best17/stop31, versus289-0.004822255242; continuous17.494506745789, versus289-0.225620057348/252-0.390361394121. Not adopted; no detachratio/BCE/LR grid. Actualencoder/readout/state nonzerograd/frozen/RAMbest144665assembly; no independentdiskreload. Individual294.684s/peak1241.643MiB, overlaps300/no speedranking. Decoded atom actionfraction is a posterior decision diagnostic, not gateprobability/classifier accuracy; no roworacle or stratum-specific model selection. Currentrecipes unchanged; finishedcleanup remains blocked by automaticpolicy, no deletionperformed.
Next bounded diagnosis (prepared, NOT evaluated): v300 joint-best29 fixedcheckpoint, zero only newstate delta at inference and retain its jointlytrained encoder/density/share; compare to existingfull89.419650/continuous74.503695 and291whole89.515555. Onevariant/no weightsweep/no roworacle/no extra phase, same originalpopulation/scoring. This separates inference-state delta from the representation benefit of state-supervised jointtraining; no claimthat pruned variant was checkpointselected or independently trained. Evaluate only after an actualGPUslot frees, no trainingstop. If no gain, close inference-state pruning, then consider ONE new same-phase decisionutility migration (oldv87/v171 negatives disclosed), not arbitrary calibrationgrid.


## Fixed-checkpoint inference state pruning
Actual PERM89.406477687966, versusfull300-0.013172535853/retained291-0.109077105291; default97.280204083215/continuous74.096685623378. Actualoptional8 TOTAL80.655335735422, strict808=False; onlyzero newstate at fixedjointbest29, allotherweights unchanged, source/diskcheckpoint hashes preserved. ONE predefined ablation, no stateweight sweep/roworacle/newphase, not optimized for prunedscore. Do not equatepruninganalysis with independent remainingfold verification. GPUeval3.011s incldata/init/diskload/eval/CPUcomposition/record, peak786.465MiB; not trainingtiming.

V302 final POR78.023865741485, gain+0.097484980761, actualTOTAL80.641720912222; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop20/34, trainingseconds733.248072/peakMiB1813.379395. No otherfolds.

V302 final POR78.030552794145, gain+0.104172033421, actualTOTAL80.643727028020; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds811.177571/peakMiB1813.378418. No otherfolds.

Final v302: grouped POR78.023865741485, full POR78.030552794145; matched +0.006687052660, full versus290 +0.007026238890. Actual eight-stage TOTAL80.695620593940, versus297optional8 +0.002107871667; one added stage per head, no chained290weights. Single seed/warn-only attention nondeterminism: small gain, not robust or fullOOF proof. Retain full/grouped positive controls; do not scan connectivity/width on this small result. Strict808 false, no remaining folds. Pair wall817.606s; individual grouped733.248s/full811.178s.

Fixed best29 state pruning completed: PERM89.406477687966 versus unpruned89.419650223819 (-.013172535853), continuous74.096685623378 versus74.503695309976 (-.407009686598). Non-state weights unchanged, one variant/no retraining/no per-row selection. Close inference pruning; original291 remains selected. GPU evaluation3.010677s/peak786.465MiB includes data/init/load/eval/assembly, not training timing.

Explicit v304 launched PID81008: new raw SW backbone and readout jointly train from original104 only, one added SW stage. Keep aligned state gradients; project only conflicts with density. No extra parameters, no changed inference, max2GPU/80-14. Gain unproved; pending whole-best/strata/conflict/cost results. Research: ../cross_domain_research_20261001/SW_ASYMMETRIC_PCGRAD_RESEARCH_20261002.md.

Explicit v305 POR PLE launched PID60056; original151 only/fresh302 architecture, 76816activeparams (+224). Train-only quantile token embedding and allnew rawencoder/readout jointly optimized in onePORphase. No302weights; 80-14/max2/max8 unchanged. Gain unproved.

V304 density_priority final SW74.153254610940, actualTOTAL80.623501041968; learned gain+0.031501921172, vs211+0.005631988313. Six optionalphases, default5; new57568rawencoder/directdensity jointlytrain, fixedprobabilitymix versusprior-relative tilt/auxCE1/SAM0/conditionalRPS1.


## Final gradient-routing outcome
SW74.153254610940/best32/stop46, versus289-0.000471818059; continuous17.713332876240, versus289-0.006793926897/252-0.171535263670. Not adopted; no priority/threshold/BCE grid. Actualencoder/readout/state nonzerograd/frozen/RAMbest144665assembly; no independentdiskreload. Individual401.474s/peak1373.719MiB, single GPU, historical contention differs/no speedranking. Decoded atom actionfraction is a posterior decision diagnostic, not gateprobability/classifier accuracy; no roworacle or stratum-specific model selection. Currentrecipes unchanged; finishedcleanup remains blocked by automaticpolicy, no deletionperformed.

Research completed: ANCHORED_LOWRANK_RAW_RESEARCH.md, originalDCN paper/officialcode and old79.448021 configuration boundary. Retain only one smallerjointcross/additive proposal for future evidence-based decision, no GPU queue. v304 final wholeSW-.000471818059/continuous-.006793926897 versus289 despite mean38.519% conflicts; closed threshold/priority grids. v305 sole activeGPU task; retained8stage80.695620593940 unchanged.

V305 final POR78.021651568327, gain+0.095270807603, actualTOTAL80.641056660275; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds232.126179/peakMiB1817.208984. No otherfolds.

Final305 POR78.021651568327, versus302-0.008901225818, versus290-0.001874986928; best/stop6/20. Adopted=False; +224capacity/nonconcurrent control/single-seed nondeterministicattention means no pure mechanism/robustness claim. Individual232.126s/peak1817.209MiB; trainquantilefit included. Whole-best144665RAM/frozenexact, not independentdiskreload, no fullOOF/remainingfolds. If negative close knot/endpoint/LR grid.

Explicitv306 pair launched: cross PID61712, additive PID31260. Both freshbody38272params/newreadout joint, original151 only, max2GPU. Initialsourcefunction77.926380760724 and actualactivecount confirmed, no extra stages or oldweights. Pending terminalpair/strata/realwall.

Prepared v307 PERM numeric-curve grouped/full pair, not launched. Original154 only/fresh291 sparsebody +19024freshattention=55490params, onejointPERMphase. History check found no equivalent291-stem migration; v194 patchSTAR is distinct and HOLD/untrained. BothPOR306GPUtasks stillactive, max2 respected; no automatic queue. Pending actualslot/explicitlaunch/finalpair and808qualification.

V306 final POR78.025027984200, gain+0.098647223476, actualTOTAL80.642069585037; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds421.501598/peakMiB982.318359. No otherfolds.

V306 final POR78.020232389922, gain+0.093851629198, actualTOTAL80.640630906753; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds421.089864/peakMiB982.427734. No otherfolds.

Final306 cross78.025027984200/additive78.020232389922; paired+0.004795594278; absoluteadoption=False, selectedcross. Same38272params/initialfunction, effective interaction family differs. No oldrepairweights, onePORaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall443.469s, individuals{'cross': 421.50159800000256, 'additive': 421.0898638999788}. If cross<=additive close rank/layer/scalegrid; localpositivebutabsolutebelowchampion not adopted. Bestcheckpoint blocknorms recorded to distinguish zero-init function from trained interaction.

Explicit v307 launched groupedPID78172/fullPID18960; original154 initial89.405565 verified by live INITIAL,55490activeparams, allnew sparsebody/readout/interactionjointonephase. Maximum2GPU actualprocesses confirmed. Pending restoredwholePERM-best/pairedscope/strata/actual8assembly. No otherfolds.

Prepared v308, NOT launched: preserve allfresh302full modules; add1728params two rank8 sharedLNupdates, cross/additive same78320params/init. Allnewbody/attention/cross/readout inONEjointPORphase fromoriginal151only, no302/306trainedweights. Current307 occupies2GPU slots. Pending explicitlaunch afterslots, score/strata/pair/time/actualcomposition; no autoqueue, no gainyet.

V307 final PERM89.474368549659, gain+0.068803692740, actualTOTAL80.636556710453; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds382.965970/peakMiB1813.562988. No otherfolds.

V307 final PERM89.488856959473, gain+0.083292102555, actualTOTAL80.641627653888; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds381.287035/peakMiB1813.562012. No otherfolds.

Final307 full89.488856959473/grouped89.474368549659; paired+0.014488409814; absoluteadoption=False, selectedfull. Same55490params/init/initialfunction, only tokenconnectivity differs; selectors point in both. No oldrepairweights, onePERMaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall388.586s, individuals{'full': 381.28703549999045, 'grouped': 382.96597009996185}. If absolute belowretained291 close thisplacement/head/widthgrid; localgain recorded separately. Bestcheckpoint bridgenorms recorded to distinguish zero-init function from trained interaction.

Explicit308 joint pair launched: cross PID44656, additive PID76968. INITIAL77.926380760724 and78320activeparams both confirmed from live logs; original151 only, parent302architecture retained/no302 or306 weights, allnewbody/attention/cross/readoutonephase. Max2GPU, current8stage80.695620593940 unchanged. Pending finalpair/strata/time/actualassembly.

Prepared309, NOT launched: fixed.05 realDEFAULT-label auxiliary CE on finalnewq; samefresh307full55490params/init, original154only/allnewencoder-readoutjointonephase. Corrected explanation: originaljointRPS alreadysupervisesDEFAULT, so onlydirectCEadded/semanticbias notlikelihoodfix. Historyresearch pending; current308occupies2GPU. No automaticqueue/coefscan/otherheadmigration.

Research completed: [DEFAULT anchor CE primary/history](../cross_domain_research_20261001/PERM_DEFAULT_ANCHOR_CE_RESEARCH_20261002.md). Original labels are realtruth log10 and not uniformlyreplaced. Grid uses nonDEFAULT trainingmin/max centers, internalmidpoints, outeredges?inf; DEFAULT actualz/canonicaldefault_z expectedbin0, no separate.01numericcenter. DirectCE only pushes tailmass, not sub-bin DEFAULT variation. Initializationrecords bins/counts/firstfiniteedge/canonicalbin and outsideedges(typically0due infiniteedges); no gridextension or labelrewrite. ExistingjointRPS DEFAULT supervision and semanticbias retained. No equivalentcurrentraw directDEFAULTCE found in searchedhistory; not globalnovelty.

V308 final POR78.021345596608, gain+0.094964835884, actualTOTAL80.640964868759; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop11/25, trainingseconds643.216918/peakMiB1814.011230. No otherfolds.

Explicit309 launched PID87620; original154 only,55490params/samefresh307fullinit andforward, allnewbody/readout/attentionjointonephase. Fixed.05DEFAULT auxiliaryCE, actualtrainbins[0]/counts[399064]; no gridextension/decoderchange. Overlaps308additive, max2GPU. Gainunproved; pending wholebest/CONT-defaulttradeoff/actual8/time.

308cross final: POR78.021345596608, best11/stop25, versus retained302 -.009207197537; extra1728params group actuallylearns (firstgroupnorm .00158365), all144665RAM/frozenexact. Individual643.217s/peak1814.011MiB underdualGPU. 308additive stilllive andrecentbest78.028308460582 at29; pairfinalpending, do not call entirecombination negative yet. 309actualdefaultbin0training launched aftercrossPIDmissing, no thirdGPU.

V308 final POR78.028308460582, gain+0.101927699859, actualTOTAL80.643053727952; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds976.572693/peakMiB1813.394043. No otherfolds.

Final308 cross78.021345596608/additive78.028308460582; paired-0.006962863975; absoluteadoption=False, selectedadditive. Same78320params/init/fullcurve, onlyanchorproductdiffers. No oldrepairweights, onePORaddedphase, no independentdiskreload/fullOOF/otherfolds. Pairwall983.066s, individuals{'cross': 643.2169180000201, 'additive': 976.5726931000245}. Ifbothbelow302 close combinationgrid; scopedgainnotabsoluteadoption. Bestcheckpoint blocknorms recorded to distinguish zero-init function from trained interaction.

V309 final PERM89.506715700712, gain+0.101150843794, actualTOTAL80.647878213322; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds331.898410/peakMiB1813.562012. No otherfolds.

Final309 PERM89.506715700712; versus307full{'PERM': 0.01785874123898168, 'default_PERM': 0.06280395301871522, 'continuous_PERM': -0.06953340475654102}, versusretained291{'PERM': -0.008839092544718596, 'default_PERM': -0.013538330957288736, 'continuous_PERM': 0.0002981773214827399}; adopted=False. WholePERM-best/stop6/20,144665RAM/frozenexact, no independentdiskreload or fullOOF. DEFAULT-only true-label regularizer notproperconditionalNLL correction; checkCONTtradeoff, no coeff/edge/decodergrid. Individual331.898s/peak1813.562MiB; historicalcontrol notconcurrent/timefairnessnotclaimed. OnePERMaddedphase, currentdelivery intact, nootherfolds.

Direction decision afterFINAL: localoverall+.017858741239/default+.062803953019 butCONT-.069533404757 consumes307continuousgain. Absolute89.506715700712<29189.515554793257; noadoption. Do not automaticallyrunSWdefaultCE transfer despite lowpriorityresearch suggestion. Preserve scopedpositive evidence/control, no coefficient/grid/decoder scans. Nextprioritize continuous depthrepresentation with allnewbackbone/readoutjointonephase; researchpending.

V310 prepared: SW297 mechanism preserved fresh, gated depth/channel residual pair73953params, original104only, joint encoder/readout one added SWphase. Gains unproved; next explicit max2 launch. Current8 TOTAL80.695620593940 unchanged.

Explicit pair launched: depth PID85880, channel PID76952; max2 normal GPU; gains pending real results.

Running snapshot (not final gain): {"depth": {"status": "running", "pid": 85880, "stage": "depth", "epoch": 8, "best": 74.1349342495239, "best_epoch": 5, "bad": 3, "peak_MiB": 1390.0986328125}, "channel": {"status": "running", "pid": 76952, "stage": "channel", "epoch": 7, "best": 74.14086578545242, "best_epoch": 4, "bad": 3, "peak_MiB": 1714.6171875}}; both below retainedSW74.16003022301975; current optional8TOTAL80.69562059394032 unchanged. Training continues under own whole-SW best and patience14.

V311 prepared/untrained: fresh297well jointbody plus physical ordered/shuffled conditional chain, same65265params/uniform init/original104only. No added loss or extra phase, no automatic launch while310twoGPU active. Gain unproved; existing recipe retained.

V312 prepared/untrained: preservedfresh291point jointbody/readout plus104 anchored group response parameters, rational/polynomial36570pair, original154only. No newloss/defaultCE/extra phase/automaticqueue. Gain unproved; current8 unchanged.

V310 depth: SW74.160199923047, actual TOTAL80.625931901205; fresh73953 encoder/readout jointly train, original104 prior only; one added SW phase, no fullOOF or remaining folds.

V311 ordered explicitly launched PID90368 while310channel final epochs occupy other slot; shuffled not launched yet, no automatic queue. First real gradients/init/results pending.

V310 channel: SW74.155625583198, actual TOTAL80.624330882258; fresh73953 encoder/readout jointly train, original104 prior only; one added SW phase, no fullOOF or remaining folds.

Final310 whole scores {'depth': 74.16019992304726, 'channel': 74.15562558319847}; depth-channel +0.004574339849; CONT paired -0.013838135467; strata {'depth': {'default': 98.84885176097337, 'continuous': 17.452652710549867}, 'channel': {'default': 98.83626121843403, 'continuous': 17.466490846016587}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 649.342s; no fullOOF or remaining folds.

310 closed: depth-channel whole+0.004574339849/CONT-0.013838135467; depth versus297 whole+.000169700028/CONT-.050100056797. Actual8 diagnosticTOTAL80.695679988950, delta+0.000059395010, strict808false, not adopted for continuous focus. Retained8TOTAL80.69562059394032 unchanged; evidence/checkpoints positive/control preserved, no generic kernel sweep. Pairwall649.342s, depth632.700s/channel644.247s; two-process timing not serial ranking.

Explicit311 pair now running: ordered PID90368/shuffled PID82616, each65265jointparams, original104 initialSW74.121752689769 exact. Source/activeconfigs frozen; gains pending final ownwhole-best.

V311 running snapshot, not final gain: {"ordered": {"status": "running", "pid": 90368, "stage": "ordered", "epoch": 9, "best": 74.14793432961145, "best_epoch": 6, "bad": 3, "peak_MiB": 1484.1552734375}, "shuffled": {"status": "running", "pid": 82616, "stage": "shuffled", "epoch": 4, "best": 74.13525951818056, "best_epoch": 4, "bad": 0, "peak_MiB": 1484.1552734375}}; real source initialization74.121752689769/65265params, actual joint training passed gradient/frozen checks. Retained8unchanged; remainingfolds not started.

V311 ordered: SW74.147934329611, actualTOTAL80.621638943503; fresh65265 joint encoder/conditional-chain readout, original104only; oneSWphase, nofullOOF/remainingfolds.

V312 rational explicitly launched PID80048, original154 prior initialPERM89.405566626894/36570params; polynomial not launched yet. Actual joint encoder/readout/shape gradients monitored by training; no standalone preflight. No third GPU.

V311 shuffled: SW74.149063522486, actualTOTAL80.622034161009; fresh65265 joint encoder/conditional-chain readout, original104only; oneSWphase, nofullOOF/remainingfolds.

Final311 whole scores {'ordered': 74.14793432961145, 'shuffled': 74.14906352248619}; ordered-shuffled -0.001129192875; CONT paired -0.086347350318; strata {'ordered': {'default': 98.87157026578004, 'continuous': 17.360031882823368}, 'shuffled': {'default': 98.8355981992487, 'continuous': 17.44637923314177}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 689.697s; no fullOOF or remaining folds.

V312 final PERM89.515804404879, gain+0.110239547960, actualTOTAL80.651059259780; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds230.112487/peakMiB881.833496. No otherfolds.

313 prepared: fresh297well jointencoder/readout/state with existingARCHIE_PROXY q20 background coordinate; ownwell vs train64reference,65313params/208stem/48zeroaddedweights/original104only. Data-fitting/splits/Q unchanged; gains unproved, no automatic queue.

V312 final PERM89.515098900854, gain+0.109534043935, actualTOTAL80.650812333371; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds144.507527/peakMiB847.802246. No otherfolds.

Final312 whole scores {'rational': 89.51580440487884, 'polynomial': 89.51509890085404}; rational-polynomial +0.000705504025; CONT paired +0.002335749759; strata {'rational': {'default': 97.49399611826219, 'continuous': 74.00288840401272}, 'polynomial': {'default': 97.49412903900604, 'continuous': 74.0005526542539}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 532.047s; no fullOOF or remaining folds.

V312 final: rational89.515804404879 vs polynomial89.515098900854; pairedwhole+.000705504025/CONT+.002335749759; versus291whole+.000249611622/CONT+.001822064252. Actual newoptional8TOTAL80.695707958008, delta+0.000087364068, strict808false. Retained as micropositive only, default5/deliveries unchanged; no extra phase/newgrid/otherfolds or robust gain claim. Current recipe here; old302recipe frozen. Rational230.112s/poly144.508s, totalwall532.047s incl sequentiallaunchgap/different GPUoverlap, not isolatedspeedranking.

V311 closed: ordered74.147934329611/shuffled74.149063522486, pairedwhole-.001129192875/CONT-.086347350318; both below297 and252. Noadoption or chain/bias/lossgrid. FullRAM/frozen144665replay, actual6composition, best/stop6/20 and17/31, wall689.697s includingstagger/variableoverlap; negative scope not all ordinalmethods.

Explicit313 pair launched: wellPID86808/referencePID59544, original104initialSW74.121752689769/65313params, training64 q20reference no labels. Current optional8 updated by312micropositive to80.695707958008, not a313gain;313ownresults pending.

314 prepared: POR fresh302full preserved plus104shared response shape parameters,76696rational/polynomial jointpair, original151only/no302orPERMweights. Transfer312tiny localpositive code; no claim PORgain, extra loss/phase or automatic queue.

V313 well: SW74.160209600351, actualTOTAL80.625935288262; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V313 reference: SW74.160973221838, actualTOTAL80.626202555782; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final313 whole scores {'well': 74.16020960035114, 'reference': 74.16097322183785}; well-reference -0.000763621487; CONT paired +0.002906752418; strata {'well': {'default': 98.82888103513555, 'continuous': 17.49855552377331}, 'reference': {'default': 98.8312426199958, 'continuous': 17.49564877135523}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 514.752s; no fullOOF or remaining folds.

313 final: well74.160209600351/reference74.160973221838; pairedwhole-.000763621487/CONT+.002906752418. BothCONTbelow297/252; fixedq20setting closed, retainedSW297 unchanged, no percentile/Rw/physicalgrid. Wholebest17/stop31 both; individual400.854/399.969s, pairwall514.752s with stagger/partialoverlap; not isolatedspeedranking.

314 explicitly running PID86736/rational andPID53836/polynomial,76696params, own151sourceinitial verified, nothirdGPU. Snapshotnotfinal {"rational": {"status": "running", "pid": 86736, "stage": "rational", "epoch": 11, "best": 78.01658032834617, "best_epoch": 6, "bad": 5, "peak_MiB": 1814.08740234375}, "polynomial": {"status": "running", "pid": 53836, "stage": "polynomial", "epoch": 11, "best": 78.01613681353473, "best_epoch": 6, "bad": 5, "peak_MiB": 1813.98974609375}}; legacyparent only_scope_differs flag describes302history, currentcurvefullheldfixed/shape_modevaries; seeMETADATA_CLARIFICATION. Sources/configshashes frozen. Retainedoptional8TOTAL80.695707958008 unchanged.

V314 live snapshot: rational epoch 23, best POR 78.02234630977468 at 20; polynomial epoch 23, best POR 78.02234526264449 at 20. Both still running; no final gain claim. CPU paired finalizer prepared and syntax checked. Retained optional8 TOTAL80.69570795800789 unchanged; no other folds or third GPU job.

V315 prepared, untrained: [SW bounded allocation](results/deep_learning/sw_adaptive_share_v315_20261002/README.md); fresh encoder/state/density jointly train, scalar1 vs conditional49 active allocation params disclosed. No GPU launch while314pair live.

V315 complete implementation prepared (not launched): fresh297well encoder/state/FiLM/readout joint training plus bounded conditional allocation versus scalar control, active1/49 disclosed. CPU terminal finalizer prepared; actual GPU max2 remains314 rational/polynomial. V314 snapshot28: rational78.02573963118171 and polynomial78.02454057483888, still below30278.03055279414464, not final gains. Reused research_light_backbones agent for a historically screened larger continuous-representation direction; no new GPU/code task delegated. Strict fifth808/fullOOF82 unmet, retained recipe/delivery unchanged.

V314 live epoch29 update: rationalPOR78.03078642930447, provisional+0.00023363515983 versus302; polynomial78.02847584828407. Not terminal, no robust gain/composition adoption claim, both real processes confirmed live. Continue own whole-target selection and later strata/composition checks.

Evidence-based early-stop requested for both314arms after snapshot31: rational+0.000234whole but-.001316CONT; polynomial negativeboth. Natural epoch-boundary stop and best restoration, no hardkill. Final reports pending.

V314 final POR78.030786429304, gain+0.104405668581, actualTOTAL80.643797118568; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/33, trainingseconds904.568358/peakMiB1814.087402. No otherfolds.

V314 final POR78.028475848284, gain+0.102095087560, actualTOTAL80.643103944262; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/33, trainingseconds904.482853/peakMiB1813.989746. No otherfolds.

Final314 whole scores {'rational': 78.03078642930447, 'polynomial': 78.02847584828407}; rational-polynomial +0.002310581020; CONT paired +0.004656519904; strata {'rational': {'default': 97.4151361319245, 'continuous': 41.673763989730155}, 'polynomial': {'default': 97.41407632714055, 'continuous': 41.66910746982659}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 910.623s; no fullOOF or remaining folds.

V316 prepared/untrained [SW bank feature kernels](results/deep_learning/sw_bank_kernel_v316_20261002/README.md); four rank8 feature matrix bases, actual/reference input routing only, freshjoint80709params/original104only. No GPU launch while315pair live.

V314 closed at33, ownbest29. Rational78.03078642930447/poly78.02847584828407; matched rational whole+.0023105810204/CONT+.00465651990356. Relative retained302whole+.00023363516 butCONT-.00131591734: not adopted, no responsebasis grid. Pairwall910.622802s, each904.568358/904.482853s, concurrent2, peaks1814MiB; sourcefrozen/RAM144665replayexact, no independentdiskreconstruction. Retained optional8TOTAL80.69570795800789 unchanged. V315 actual launched65240/scalar and63596/conditional, initial74.12175268976877 verified, active65266/65314, no thirdGPU.

V315 snapshot16: scalarwhole74.131531/CONT17.703990 (+.201238CONT vs297 butwhole-.028499); conditionalwhole74.124596/CONT17.556323. Conditional earlyboundary best-restoring request; scalar keeps natural14patience. Noadoption; continuous/default tradeoff is retained evidence, not blanketnegative.

V315 conditional: SW74.131849364002, actualTOTAL80.616009205539; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V316 actual launched PID16564 with315scalarPID65240 still live, max2 confirmed. Initial original104SW74.12175268976877/80709params verified in normal trainer; no numerical preflight/review. Reference remains prepared/unlaunched, not autoqueued. Actual overlaps315scalar, pairwall timing will disclose gap. No315repairweights/allocation module used.

V315scalar earlyboundary requested after32; selectedbest26 whole74.14209008848303/CONT17.657627078824376, absolute297whole-.017940/CONT+.154874, strong252CONT-.227241. Retain continuous/default tradeoff positivecontrol, not adoption or allocation grid. Terminal restoration pending.

V315 scalar: SW74.142090088483, actualTOTAL80.619593459108; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final315 whole scores {'scalar': 74.14209008848303, 'conditional': 74.13184936400198}; conditional-scalar -0.010240724481; CONT paired -0.059682158997; strata {'scalar': {'default': 98.73361820152378, 'continuous': 17.657627078824376}, 'conditional': {'default': 98.74490269587389, 'continuous': 17.597944919827714}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 557.720s; no fullOOF or remaining folds.

V315 closed: scalar74.14209008848303/best26/stop34 CONT17.657627078824376, selectedscalaralpha0.045009106; conditional74.13184936400198/best17/stop18 CONT17.597944919827714. Conditional-scalarwhole-.010240724481/CONT-.059682158997. Bothwholebelow297; scalarCONT+.154874 butbelow252. Retain CONTtradeoff evidence, noadoption/allocationgrid. Unequal adaptiveearlyclosure/active1vs49 explicitly notmatchedcapacity proof. Actual6assemblies80.619593459108/80.616009205539 below808. Pairwall557.720221s, individual552.561913/335.255871s/1318MiB; originalsourcefrozen and ownbestRAM144665exact, notindependentdiskreconstruction. V316 actual16564/reference37364 live, original104initialsame/80709, stagger227.14s. Retainedoptional8TOTAL80.695707958008 unchanged.

V316 live pair confirmed: actualPID16564 epoch22/best74.15415795130484@17; referencePID37364 epoch5/best74.14116013736248@4. Both original104init74.12175268976877/80709params verified, no stderr errors, max2realGPUjobs. Reference launched explicitly after315scalarterminal,227.14s gap. Snapshotnotfinal, no matchedgain/adoption claim. CPU finalizer now reconstructs exact seeded U/V/router initialization and measures selectedbest deltas so initially nonzeroV/router hiddenweights are not mistaken for updates. Fullpairfinal/strata/composition pending; retainedoptional8TOTAL80.69570795800789/default5/delivery unchanged, strict808/fullOOF82 unmet.

V317 prepared/untrained [SW originalcurve L/LR embeddings](results/deep_learning/sw_numeric_embedding_v317_20261002/README.md),3600extra/68865params, freshjoint/original104only, no316weights/kernel. No launch while316pair live.

V316 actual: SW74.154157951305, actualTOTAL80.623817211095; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V316 actualfinal74.15415795130484/best17/stop31,CONT17.47294302201801, bothbelow297. Referencebest74.15413935754133@17; earlyboundary request becausecandidateclosed andtinywholepairedbenefit, notgeneralCondConvfailure. V317relu34568explicitlaunched, original9percurveL/LR3600extra/68865params; linearstillprepared, max2.

V316 reference: SW74.154139357541, actualTOTAL80.623810703278; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final316 whole scores {'actual': 74.15415795130484, 'reference': 74.15413935754133}; actual-reference +0.000018593764; CONT paired -0.001590054584; strata {'actual': {'default': 98.83134555555758, 'continuous': 17.47294302201801}, 'reference': {'default': 98.83062660776832, 'continuous': 17.47453307660211}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 656.926s; no fullOOF or remaining folds.

V316 finalclosed: actual74.15415795130484/reference74.15413935754133, bothbest17/stop31 NATURALpatience14; referenceearlyrequesthadnoeffect, notearlyterminated. Actual-referencewhole+.000018593764/CONT-.001590054584, bothwhole/CONTbelow297/252, notadopted/no kernelgrid. AllU/V/routerselectedbestdelta normsfromexactseededinitialization nonzero, notdeadmodule. Pairwall656.925814s, individual497.601626/424.509887s/1284MiB,stagger227.14s/variableoverlap; noisolatedspeedclaim. Sourcefrozen/RAM144665replayexact, notindependentdiskreconstruction. V317 relu34568/linear52776 explicitlylaunched,90.282s gap, original104only/newbody/readoutjoint, no316weights. Retainedoptional8TOTAL80.695707958008 unchanged, nootherfolds.

V317 live verified snapshot: ReLUPID34568 epoch14/best74.15449906167994@13; linearPID52776 epoch6/best74.1476292157653@5. Normal training realgroupgradient checks passed, original104init74.12175268976877/68865sameactiveparams, no stderr error/max2CIMconfirmed. Partial differingprogress, no finalpairedgain or adoption. Finalizer exactinitialparameter deltas prepared. Retainedoptional8TOTAL80.69570795800789 unchanged, fifthstrict808/fullOOF82 unmet; continue ownwholebest14patience and compareCONT/whole atterminal. Atlas refreshed with316closure and317nodes.

V317 snapshot18/9 from savedownbest: LRbest17 whole74.1669496554495 (+.006919vs297), DEFAULT98.85108586849975/CONT17.46977441516435 (-.032978vs297); Lbest5 whole74.1476292157653/CONT17.56512866577589. Progressdiffers, notfinalpairedproof/adoption. LRwholegaincurrentlydefaultdominated; keep originalSW297. A code-evidence hypothesis for future research: oldv275prior-tilt primaryNLLreadsfinalposterior butdirect4branchauxCEreadsstandaloneq; requested history check for prior-aware auxiliary same-capacity control, not assertedbottleneck, no code/GPUlaunch. Existingactive317source/configimmutable; max2/originalfold/Q/80-14/oneextraSWphase unchanged.

V317 relu: SW74.166949655449, actualTOTAL80.628294307546; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V317 LRnatural31final74.1669496554495/best17/CONT17.46977441516435; actualsixTOTAL80.62829430754611. Llatestownbest26 whole74.16204929073999/CONT17.47764285793554 atobserved36, stilllive. LR-minuscurrentLwhole+.004900364710 butCONT-.007868442771: earlierbest17pairedCONTpositive hasnotpersisted; NOTfinalpositive trick/notadopted/no carrynumericLRinto318. Subsequentprior-offsetaux study should usefresh297 only, no317module/weights, pendinghistoryresearch final. Existingbest/default5/optional8TOTAL80.695707958008/delivery unchanged. FullLterminal/finalpair/deltanorms pending; sourcefrozen/RAMLR144665exact, notindependentdiskreconstruction.

V317 linear: SW74.162049290740, actualTOTAL80.626579179898; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final317 whole scores {'linear': 74.16204929073999, 'relu': 74.1669496554495}; relu-linear +0.004900364710; CONT paired -0.007868442771; strata {'linear': {'default': 98.84062637352038, 'continuous': 17.47764285793554}, 'relu': {'default': 98.85108586849975, 'continuous': 17.46977441516435}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 638.151s; no fullOOF or remaining folds.

V318 prepared/untrained [SW auxiliary prior offset](results/deep_learning/sw_prior_offset_aux_v318_20261002/README.md),freshjoint65265params/original104only. Bothsame actualtilt/delta/ramp, auxonlyprioroffset0-vslogprior; notold275rawauxreplication/no317module.

V317 FINAL: LR74.1669496554495/best17/natural31, L74.16204929073999/best26/natural40; LR-Lwhole+.0049003647095/CONT-.0078684427712. BothCONTbelow297/252, notadopted/no widththresholdgrid. Realembeddingweight/bias/outdeltas nonzero (LR.700/.717/2.999,L.260/.324/1.599), notdeadmodule. Pairwall638.151032s, each504.753633/542.533978s,90.282s stagger/peaks1315.602/1297.477MiB/variableoverlap, notisolatedspeedranking. Sourcefrozen/RAM144665exact, noindependentdiskreconstruction. V318neutralPID91028/offsetPID79268explicitpairlaunched, fresh297rawbody/state/FiLM/readoutjoint65265, sameactualdelta/formula/ramp/params; onlyauxprioroffset0-vslogprior. No317module/weights or priorKL. Retainedoptional8TOTAL80.695707958008/default5/delivery unchanged, nootherfolds.

V318 live verified both PID91028neutral/PID79268offset epoch11, sameinitialSW74.12175268976877, ownbeststill0,65265activeparams/realnormaltraining/max2CIMconfirmed, nostderrerrors. Source104frozen, newrawencoder/state/FiLM/densityjoint; nominalmixalpha buffer meansgamma=alpha/.25, not probabilitymixture. Auxinitiallossdropin offset is mathematical baseline difference, not officialgain. CPU terminal evaluator prepared for wholebeststrata/readoutdelta andactualcomposition; await14patience/min10. Atlas refreshed317closure/318nodes; retainedoptional8TOTAL80.69570795800789 unchanged, strict808/fullOOF82unmet.

V318 neutral: SW74.121752689769, actualTOTAL80.612475369558; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

V318 offset: SW74.121752689769, actualTOTAL80.612475369558; fresh65313jointencoder/readout/state, original104prioronly; derivedproxy background, oneaddedSWphase. No fullOOF/otherfolds.

Final318 whole scores {'neutral': 74.12175268976877, 'offset': 74.12175268976877}; offset-neutral +0.000000000000; CONT paired +0.000000000000; strata {'neutral': {'default': 98.71563205981698, 'continuous': 17.63188906048012}, 'offset': {'default': 98.71563205981698, 'continuous': 17.63188906048012}}. Actual six-stage assemblies recorded; optional eight-stage adoption requires whole/continuous assessment, not automatic. Pair wall 244.686s; no fullOOF or remaining folds.

V318 completed: neutral/offset both best0 SW74.12175268976877, learned gain0; paired whole/CONT0; closed, no sweep. V319 prepared native104 whole-body/state/density joint vs +6272 fresh225-code FF adapter; max2 GPU, no other folds.

LIVE LAUNCH verified: control PID440 and adapter PID76864, both initialSW74.12175282815758 exact source replay. Control845913/adapter852185 all trainable, initial training peak1722.19/1723.05MiB. Two GPU jobs; snapshots are not final gain. Atlas curated/built:1167 experiments,231 dead ends;318 closure recorded.

V320 prepared/untrained: PERM fresh291point original154only, joint encoder/readout plus89param ordered-vs-shuffled targetbin Conv. Onephase; no launch while319twoGPU live. Gain unproven.

V319 control: SW74.121752828158, actualTOTAL80.612475417994; native104 whole backbone/state/Gaussian density jointly trainable, with or without fresh6272 FF adapter, oneaddedSWphase. No fullOOF/otherfolds.

V319 adapter: SW74.121752828158, actualTOTAL80.612475417994; native104 whole backbone/state/Gaussian density jointly trainable, with or without fresh6272 FF adapter, oneaddedSWphase. No fullOOF/otherfolds.

FINAL319 CLOSED: control best0/earlystop11, adapter best0/earlystop12; each SW74.12175282815758, actual6TOTAL80.61247541799393. Paired whole/CONT0; selected sourceweight deltas0 because initialization selected, live body/state/density updates verified nonzero. CONT17.63188906048012 is original source, not learned gain. No adoption/LR-layer-grid/fullOOF/otherfolds; no independent diskreload. Individual363.678/379.329s, peak1722.193/1723.052MiB; concurrentpair405.489s/stagger21.527s inclprocess-to-lastterminal, excludeprelaunch/CPUfinalization. Default5/retainedoptional8 unchanged.

LIVE320 confirmed ordered PID17508/shuffled PID82892; same initialPERM89.4055666268939 (small priorprobability flooring numerical difference from15489.40556485691894, not learned gain). New rawencoder/readout jointly training; peak966.646MiB per process. {"ordered": {"status": "running", "pid": 17508, "stage": "ordered", "epoch": 5, "best": 89.50635690124827, "best_epoch": 5, "bad": 0, "peak_MiB": 966.646484375}, "shuffled": {"status": "running", "pid": 82892, "stage": "shuffled", "epoch": 2, "best": 89.4160403193998, "best_epoch": 2, "bad": 0, "peak_MiB": 966.646484375}}; early snapshot only, not mechanism gain/adoption. No fullOOF or otherfolds.

V320 final PERM89.506356901248, gain+0.100792044329, actualTOTAL80.647752633509; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/14, trainingseconds240.581778/peakMiB966.646484. No otherfolds.

V320 final PERM89.511618071525, gain+0.106053214606, actualTOTAL80.649594043106; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/12, trainingseconds222.299530/peakMiB966.646484. No otherfolds.

V320 terminal: ordered89.50635690124827/shuffled89.51161807152464, pairedwhole-.00526117/CONT+.000135714, noadoption. V321 prepared/untrained private151/154inputlatent vs104ownlatent forfreshSWjointbody/readout; oneaddedSWphase, gainunproven.

INITIAL START FAILURE: original321 cross PID86524 failed before initialscore/training due privatecapture method name colliding with inherited fused forwardhook signature. No epoch/checkpoint/score. Originalfailed source/config/log preserved. New321r1 renamesprivate hook factory only, same mechanism/seeds/config, new outputfold5_r1; not a trained repair/additionalphase.

FINAL320 CLOSED: ordered89.50635690124827/shuffled89.51161807152464, both best5; earlystop14/12. Ordered?shuffled whole?0.005261170276/CONT+0.000135714335. Bothwholebelow312 by0.0094475/0.00418633; CONTvs312+.00534634/+.00521063 is small tradeoff, not useful paired topology gain. Allnewmoduleweights/biases updated, not dead; own144665RAM replay/frozenexact, no independentdiskreload. Individual240.582/222.300s peak966.646MiB; concurrentpair252.874s/stagger24.687s. No adoption/kernelwidth/lossweight grid/otherfolds; default5 and optional8 unchanged.

LIVE321r1 confirmed crossPID89936/ownPID75612, both initialSW74.12175268976877. Active86865 matched params, newbody/state/FiLM/readout/fusion jointly train, threeoriginalsourceencodersfrozen. Cross firstmeasuredpeak1560.181MiB; own startedinitialverified. {"own": {"status": "running", "pid": 75612, "stage": "own", "epoch": 1, "best": 74.12175268976877, "best_epoch": 0, "bad": 1, "peak_MiB": 1560.1806640625}, "cross": {"status": "running", "pid": 89936, "stage": "cross", "epoch": 4, "best": 74.12798751189538, "best_epoch": 3, "bad": 1, "peak_MiB": 1560.1806640625}}; live snapshot only, no finalgain/adoption/fullOOF/otherfolds. Atlas updated1174experiments/233deadends including320closure and321startupfailure+r1.

V322 prepared/untrained: fresh302full PORjointbody/readout +162 learnedwindow shift/positiveRMSscale incurve-token side; realvalidwindow vs64trainreference matched. Original151only/onePORphase, nolaunch while321twoGPUlive; gainsunproven.

V321 own: SW74.166030180492, actualTOTAL80.627972491311; fresh297 raw encoder/state/condition/readout +21600 latentfusion joint; original151/154/104 private hidden sources, no donor predictions, oneaddedSWphase. No fullOOF/otherfolds.

V321 cross: SW74.166591356036, actualTOTAL80.628168902751; fresh297 raw encoder/state/condition/readout +21600 latentfusion joint; original151/154/104 private hidden sources, no donor predictions, oneaddedSWphase. No fullOOF/otherfolds.

FINAL321: both best13; early22own/25cross. OwnSW74.166030180492/CONT17.911355427690, gainvs297 whole+.005999957472/CONT+.408602660343, CONTvs252+.026487287780. CrossSW74.166591356036, cross?own whole+.000561175544/CONT?.280224707870; close cross migration, retain own. Actual8 {"POR": 78.03055279414464, "PERM": 89.51580440487884, "SW": 74.16603018049169, "TOTAL": 80.69780794312307}; strict808false/nootherfolds/fullOOFunmet. Newrecipe inthisstudy, old312/default5/deliveriesunchanged. Individual690.157/741.981s peak1560.181MiB, pair746.514s, campaign793.156s inclfailedstartup/fix; no isolatedspeedranking or independentdiskreload. No width/fusiongrid.

LIVE322 both launched adaptive4512/reference73100; initialPOR77.92638076072386 same both, actualactive76754 (inherited302info76592 is parentcount only; newnorm+162). Allfreshrawbody/curveattention/readout/normalizerjoint. Peaks1815.196/1814.883MiB. {"adaptive": {"status": "running", "pid": 4512, "stage": "adaptive", "epoch": 7, "best": 78.01812646762907, "best_epoch": 6, "bad": 1, "peak_MiB": 1815.19580078125}, "reference": {"status": "running", "pid": 73100, "stage": "reference", "epoch": 2, "best": 77.95497893387893, "best_epoch": 2, "bad": 0, "peak_MiB": 1814.88330078125}}; snapshotnotfinalgain/fullOOF. 321ownretainedoptional8TOTAL80.69780794312307, original5/old8/deliveriesunchanged, strict808false.

V323 prepared/untrained: source104only fresh321owncode/seed lateinsteadearly fusion, same86865trainable historical321owncontrol reused; unuseddonorcompute removed, no trajectoryequality claim/no trainedrepair. No GPUlaunchwhile322twojobs.

LIVE322 updated {"adaptive": {"status": "running", "pid": 4512, "stage": "adaptive", "epoch": 36, "best": 78.02922410702564, "best_epoch": 29, "bad": 7, "peak_MiB": 1815.19580078125}, "reference": {"status": "running", "pid": 73100, "stage": "reference", "epoch": 31, "best": 78.02739942868787, "best_epoch": 29, "bad": 2, "peak_MiB": 1814.88330078125}}; reference caughtup laterbest29, earlier+.007907 pairedsnapshot notfinal/robustgain. ExistingPOR30278.030552794145 retained. V323lateSWfreshownfeaturefusion prepared/untrained, historical321own earlycontrol andpredefined22epochcompare/full80candidate; no GPUlaunch whiletwo322 jobs alive. Optional8TOTAL80.69780794312307 preserved, strict808unmet/nootherfolds/fullOOF82unmet.

V322 final POR78.029224107026, gain+0.102843346302, actualTOTAL80.643328421884; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds1000.973573/peakMiB1815.195801. No otherfolds.

V322 final POR78.027399428688, gain+0.101018667964, actualTOTAL80.642781018383; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds998.817949/peakMiB1814.883301. No otherfolds.

V322 final adaptive/reference POR78.029224107026/78.027399428688, paired +0.001824678338/CONT+0.009222531043, below302 whole and CONT; not adopted. Both best29/naturalstop43, 144665 RAM replay/frozen exact, pair1058.971775s/peaks1815.196 and1814.883MiB. V323 late SW launched PID37044, initial74.121752689769, joint encoder/readout, predefined22epoch historical comparison. V324 raw+normalization delta token matched pair prepared,76786active (+32), original151 only/no trainedrepair; gain unproved, max2GPU/no queue/otherfolds.

V324 adaptive launched PID41036 alongside SW323 after fresh actual GPU process check; reference remains unlaunched, no automatic queue. Final322 recorded, absolute replacement closed; existing best recipes/deliveries unchanged.

V323 late: SW74.137935866983, actualTOTAL80.618139481583; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final conclusion
Latefusion SW74.137935866983/CONT17.618651450885, whole-0.028094313508 and CONT-0.292703976805 versus321own. Predefined bestthrough22 exactly same selectedbest17; historicalcontrol/unusedwork/auxgradientpath limitations remain. Evidence-based boundary stop29/bad12 (request submitted after27), not natural14; bestRAM144665 replay/frozenexact, all fusionparameter deltas/gradient nonzero. Actual6TOTAL80.618139481583; below808/no otherfolds, not adopted. Sourceonly previousown positive retained. Close latefusion position, no width/LR/alpha grid. Individual352.155104s/peak1356.091797MiB, process-to-terminal356.243313s incl preparation/initial/train/bestrestore, partly concurrentPOR; not isolatedspeed comparison. No independentdiskreload.

V324 reference explicitly launched PID65712 after authoritativeSW323terminal. AdaptivePID41036 and referencePID65712 are the only2 GPU tasks. No automatic queue/remainingfolds.

V325 PERM matched content-selective/windowmean-gated bidirectional memory prepared,43626active+7056, fresh291/312code original154only, encoder/readoutjoint onephase. No trainedrepair/otherfolds, max2GPU and80-14; unlaunched while324pairactive, gainsunproved.

V324 live adaptive refreshed best20 POR78.026020857869 after best6 plateau; reference best11 POR78.017000201492. Earliernearstop inference superseded by observed20 improvement, not terminalpairedgain. Both PIDs41036/65712 confirmedlive, max2GPU. V325 preparednext, no slotlaunch.

V324 verifiedlive PIDs41036/65712: adaptivebest29 POR78.027986511084, referencebest11 POR78.017000201492, stillbelow302, notfinalpairedgain. Newlate29best reset naturalpat14; no early kill or extraGPUlaunch. V325 atomicterminalpairedfinalizer prepared/syntaxchecked; researchjointselectiveencoder plan unchanged/unlaunched.

V326 SW learnedmax/min latentmorphology prepared fromprimaryresearch, depth/channel matched92289 +5424, fresh321earlycode/original104only, jointencoder/readout inoneSWphase. Unlaunched/no queue while324pairlive/325next; no gain yet.

V324 final POR78.027986511084, gain+0.101605750360, actualTOTAL80.642957143102; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds1046.970454/peakMiB1815.477051. No otherfolds.

V325 selective explicitlylaunched PID76132 afterfreshglobalGPUprocesscheck showedonlyPOR324referencePID65712. Ownjoint43626 active; windowcontrolunlaunched/noautoqueue. V324adaptive naturalstop43/best29 POR78.027986511084, RAM144665replay/frozenexact andrecordactualassembly completed, below302; pairedfinal awaitsreference. SW326preparednotlaunched.

V324 final POR78.027189922468, gain+0.100809161745, actualTOTAL80.642718166517; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds917.975614/peakMiB1815.164551. No otherfolds.

## Final closure
Adaptive/reference bothnaturalstop43/best29: whole78.027986511084/78.027189922468, CONT41.665712583009/41.669385189908. Pairedwhole+0.000796588616 but CONT-0.003672606900; adaptive belowretained302whole-0.002566283060/CONT-0.009367324062. No adoption, close raw+delta token and normalizationextension; no width/bound/LR/windowgrid. Source/bestrecipes/deliveries unchanged. All144665ownbestRAM replay/frozenexact, learnednormalizerupdatesnonzero, no independentdiskreload/fullOOF. Individual1046.970454/917.975614s, pair1179.685143s with256.129325s stagger andinitialSW323overlap, peaks1815.477051/1815.164551MiB; samework wall includesprep/initial/train/bestrestore, excludesimplementation/CPUfinalization, notisolatedspeedranking.

V325 bothinitialPERM89.405566626894/43626active, selectivePID76132/windowPID66616 live. Windowexplicitlylaunched afterfreshCIM confirmedPOR324terminal; max2GPU, no automaticqueue/otherfolds. Currentresultsnotfinalgain.

V325 final PERM89.506957270992, gain+0.101392414073, actualTOTAL80.647962762920; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds258.757945/peakMiB1007.933105. No otherfolds.

V325 final PERM89.509398334584, gain+0.103833477665, actualTOTAL80.648817135177; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds253.656007/peakMiB1007.933594. No otherfolds.

## Final closure
Bothnaturalstop19/best5: selective/window PERM89.506957270992/89.509398334584, CONT74.005909160716/74.008989668121. Pairedwhole-0.002441063592/CONT-0.003080507405, absolute312whole-0.008847133886/-0.006406070294. BothCONTtiny+0.003020756703/+0.006101264108 versus312 but wholeloss; no adoption. Memoryvalue/update/output actualparameterdeltas nonzero, all144665 RAMbestreplay/frozenexact, notindependentdiskreload. Close this contentselective gate memory setting, no timescale/width/rank/LR grid; not allMamba/state-spaceineffective. Individual258.757945/253.656007s, totalpair297.245296s/stagger37.707976s, peak1007.933105/1007.933594MiB; initialPOR324tail/CPUoverlapdisclosed, notisolatedspeedclaim. Bestrecipe80.697807943123/default5/oldsubmission unchanged, strict808/fullOOF82unmet.

V326 depthPID45412/channelPID65636 explicitlylaunched afterfreshactualGPUchecks; only2GPUjobs. Depthordinaryinitial SW74.121752689769/92289active verified, channelinitialpending. Original104only/freshpositiveearlyfusioncode, allfreshbody/readout+morphjoint inoneSWphase, no trainedrepair/otherfolds. Traininggainunproven.

V326 live best13 depth74.169543032075/CONT17.963442088248, channel74.166675514358/CONT17.962394682651. Pairedwhole+0.002867517717/CONT+0.001047405597; absoluteCONTvs321own+0.052086660558/+0.051039254962, most benefit common nonlinearmodule, not bigdepthcausalproof. Bothconfirmedlive45412/65636, snapshotnotfinal/noadoption/recipeunchanged. Currentandbestmorphparameters updated; channelinitialbest0 earlierwasrollback notdead. Conditionalresearchnow explores same5424 module sharedfirst-vsallthreebranches onlyifterminalpositivepersists, no code/newstudy/automaticqueue or trainedrepairweights. Maskedreconstruction remains lowpriorityafterhistoricv45negative.

V326 depth: SW74.169543032075, actualTOTAL80.629201989365; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

V326 channel: SW74.166675514358, actualTOTAL80.628198358164; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

V326 final positive retaineddepthSW74.169543032075/CONT17.963442088248, pairedwhole+.002867518/CONT+.001047406, actual8TOTAL80.699037441177/vs321+0.001229498054. Bothnatural27/best13, RAM144665/frozenexact/no independentdiskreload, nootherfolds. New recipe in326dir; default5/old8/deliveries unchanged. V327sharedall92289 prepared, freshonlyoriginal104/no326weights,27equalepochhistoricalcontrol and80-14fullbudget, no gainyet.

V328 POR latentmorphologytransfer prepared82016(+5424), original151only/fresh302jointcurveattention/body/readout, no302/326weights orSWdonor. Singledepthscreen vs historical302capacitydifferent; freshchannelonlyifwhole/CONTgain, noautomaticqueue/otherfolds. SW327all launchedPID15640, max2GPU.

## Final retained small gain
Bothnaturalstop27/best13. DepthSW74.169543032075/CONT17.963442088248, channel74.166675514358/CONT17.962394682651; pairedwhole+0.002867517717/CONT+0.001047405597. Depthvs321own whole+0.003512851583/CONT+0.052086660558. Most continuousbenefit common nonlinearenvelope, notbigdepthcausalproof; single-seedmicro notrobust. Actual144665fixedtargetcomposition POR302full/PERM312rational/SW326depth TOTAL80.699037441177 (+0.001229498054),8phases/sourcecounted/oneaddedphasehead. NewCURRENT_EIGHT_STAGE_RECIPE.json, oldrecipes/default5/deliveryunchanged; strict808false/nootherfolds/fullOOF82unmet. AllbestRAMreplay/frozenexact, actualmoduleupdates, noindependentdiskreload. Individual576.585568/563.820791s, total595.904282s/stagger30.619320s, peaks1779.212402/1779.157715MiB. Includesprep/initial/train/bestrestore, notimplementation/finalization orisolatedspeedclaim. Conditionalsharedallcoverage327 nowimplementedfreshonly; sourceweightsnotchained.

Explicitlylaunched PID15640: all92289, original104only/freshsharedthreebranches, nominal80/14/savebestthrough27. SW327andPOR328 areonly2GPUjobs afterfreshglobalCIMchecks; traininggainunproved/noautomaticqueue/otherfolds.

Explicitlylaunched PID43860: depth82016, original151only/fresh302curveencoder/readout+morphcode. SW327andPOR328 areonly2GPUjobs afterfreshglobalCIMchecks; traininggainunproved/noautomaticqueue/otherfolds.

V327 terminalCPUfinalizer prepared; actualwhole/default/CONT, same-seed morph parameterdeltas, time/memory/comparisonlimits willrecord once report+assembly terminal. Active source/config/checkpoints unchanged. No extraGPU/preflight/review or promotion.

V328 terminalCPUfinalizer prepared; actualwhole/default/CONT, same-seed morph parameterdeltas, time/memory/comparisonlimits willrecord once report+assembly terminal. Active source/config/checkpoints unchanged. No extraGPU/preflight/review or promotion.

V327livebest13 whole74.161882452445/CONT17.957678430809/default98.631394768297, versus326first whole-0.007660579630/CONT-0.005763657439/default-0.008486438864. Bothstrataandwholeloss, no currentevidenceforcoverage; noadoption/oldrecipeunchanged. Thisis snapshotnotterminal27comparison. PID15640andPOR328PID43860 remainlive/max2GPU. Finalizersprepared forbothstudies, noextraGPUchecks/reviews. ConditionalresearchoneSWfreshbody48->96 expansion preservingpositive326firstcode, original104only, notwidthgrid/checkpointchaining; notimplemented/launched andhistoricalduplicationauditpending.

V329SWonewidth96candidateprepared262017active(+positive326firstcode), frozen104845913/sourceonly104, allfreshencoder/FiLM/state/readout/morphjoint oneSWphase/no trainedrepair. Historical48natural27 comparebestthrough27 versusfull80 separated/capacitydiffers; no widthgrid/automaticqueue/otherfolds. Unlaunchedwhile327/328occupy2GPU.

V327 all: SW74.161882452445, actualTOTAL80.626520786495; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final closure
Naturalstop27/best13 SW74.161882452445, default98.631394768297/CONT17.957678430809. Historical326first same92289/predefined27ceiling deltas whole-0.007660579630/default-0.008486438864/CONT-0.005763657439. Sharedmodules realupdates/gradient nonzero, notdeadinitialization; no adoption, close coverage, nohooksubset/LR/kernelgrid. Capacitysame but compute/sharedgradientdiffers, historicalnot simultaneousreplicate. All144665bestRAMreplay/frozenexact, notindependentdiskreload. Actual6TOTAL80.626520786495/strict808false/nootherfolds; retained8TOTAL80.699037441177 unchanged. Individual634.152107s/peak2452.446777MiB/process638.367898s withPOR328overlap, noisolatedspeedclaim.

V329 wide explicitlylaunched PID75584 afterfreshCIM showedonlyPOR328PID43860; initialSW74.121752689769/actual262017newparameters verifiedduringordinarytraining, no extraGPUpreflight. Bothareonly2GPUtasks, max80/pat14, gainunproved; bestthrough27historical48comparison defined, no326weights/otherfolds.

V328 POR refreshedbest20 78.025030693846, belowretained30278.030552794145; PID43860continues, nochannelcontroloradoption untilwhole/CONTbothpositive.

V330fixedalpha.5 freshpositive32648jointbody/readout prepared92289, original104only/no326329weights orposthocblend. Historicalsamecapacity.25 natural27 reference; prefix27/full80separate. Actual315range0..5 and directCE/RPSnot.25scaled correction recordedresearch. Noalpha grid/automaticqueue/otherfolds. Unlaunchedwhile328/329live.

V328 final POR78.030968049279, gain+0.104587288555, actualTOTAL80.643851604560; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds1045.627869/peakMiB1813.982422. No otherfolds.

V329 wide: SW74.131552954773, actualTOTAL80.615905462309; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Batch328-330
POR328 closed: whole +0.000415255 but CONT -0.002781409 versus302; no adoption/channel followup. SW329 width96 closed: SW74.131552955, -0.037990077 versus326; CONT -0.418465523, actual262017 new params, early latent fusion (recorder inherited late-fusion wording corrected). Retained8-phase fixed fifth TOTAL80.6990374411772 unchanged, default5 unchanged, fullOOF>82 unmet. SW330 half PID76420 started alone, initialSW74.121752690/92289 new params, joint encoder/readout, fixed train/inference alpha.5; one replacement SW phase, max8/max2. No other folds or queue.

V330 half: SW74.142277545469, actualTOTAL80.619659069053; fresh297encoder/state/readout +21600 own104norm128 latentlatefusion, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Final outcome
Natural stop27/best13, restored SW74.1422775454687. Same92289-capacity historical326 alpha0.25 versus fixed0.5 at predefined27: whole -0.027265486606; default98.5086222741472 vs98.63988120716091 (-0.131258933014); continuous18.175039918709572 vs17.96344208824835 (+0.211597830461). Whole target best only, not CONT epoch selection. Useful diagnostic tradeoff, no whole gain/adoption; close fixed-half setting without alpha/LR grid. Keep326 current8 TOTAL80.6990374411772 and default5 unchanged; no remainingfolds, fullOOF82 unmet. All joint encoder/readout groups received gradients, all144665 best RAM replay/frozen exact, not independent disk reload. Individual255.737770s/process260.306205s, solo GPU peak1779.212MiB; prep/initial/train/restore included, implementation/CPUfinalization excluded. Historical326 ran concurrently, so wall-times do not establish speed gain. Recorder inherited late-latent-fusion prose is incorrect: actual326 morphology/ownlatentEARLY fresh architecture per CONFIG/source.

Prepared331: latent task-specific vs shared expert routing,870new params/93159joint, original104only fresh326positive architecture, oneSWphase/max8. Not launched/no result; default and retained recipes unchanged.

331 live: shared PID5788/task PID26068 launched with global GPU cap2; initial/ordinary training evidence pending. Prepared paired CPU finalizer, no numerical preflight or separate review. OneSWphase with870 latent routers/allfreshencoderreadoutjoint, no trainedrepair weights.

331 ordinary training: shared/task live e4, both initial best74.12175268976877/93159trainable, group gradients verified by training, each peak1822.855MiB, no learned gain yet. Research primary source/history limits complete. Max2 actualjobs, no followup queue/otherfolds; retained8TOTAL80.699037441 unchanged.

Prepared332 POR numeric token RBF/cumulative256params/76848joint, original151only fresh302code. Train-inputcenters only; unlaunched/no gain while331twoGPUs live.

V331 shared: SW74.152268696139, actualTOTAL80.623155971788; original104only/fresh326earlylatent+morph joint encoder/readout with870 latent task-router parameters,93159 active, oneaddedSWphase. No fullOOF/otherfolds.

V331 task: SW74.151660618453, actualTOTAL80.622943144597; original104only/fresh326earlylatent+morph joint encoder/readout with870 latent task-router parameters,93159 active, oneaddedSWphase. No fullOOF/otherfolds.

## Final closure
Natural27/best13 both: sharedSW74.15226869613926/CONT18.016340491051338; taskSW74.15166061845294/CONT18.039633514083548. Task minus shared whole -0.000608077686/CONT +0.023293023032. Relative326 sharedwhole -0.017274335936/CONT +0.052898402803, taskwhole -0.017882413622/CONT +0.076191425835. No adoption, close latent task-routing with no router/LR/width scan. Routers actually changed; all144665 RAM bestreplay/frozenexact, not independent diskreload; same93159 physicalparams but shared lower effective capacity, zero state-gate delayed private gradients. Pairwall638.154386s/stagger1.195s, individuals633.705506/630.088574s, each1822.855MiB, prep initial train bestrestore included/implementation finalization excluded. Retained8TOTAL80.699037441/default5 unchanged, strict808false/nootherfolds/fullOOF82 unmet.

332 LIVE: rbf PID76224/cumulative PID42448 launched after331 terminal, globalmax2. OrdinaryinitialPOR77.92638076072386/76848trainable both, no learned gain yet. Only151 trainedcheckpoint +fresh302code+256 residual numeric basis; source/config frozen once launched, allnewbodyattention/readout joint, onePORphase/max8. Attention warn-only nondeterminism as historic302; single-seed tiny delta not robustness proof. Train-onlyq fit included in timing.

Prepared333PERM nonlinear local edge message,3528params/40098joint fromoriginal154only fresh312code, max/mean controls; unlaunched while332twoGPUs live/no gain/noqueue.

V332 final POR78.020299603815, gain+0.093918843091, actualTOTAL80.640651070921; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds515.142976/peakMiB1819.070801. No otherfolds.

V332 final POR78.022463922177, gain+0.096083161454, actualTOTAL80.641300366430; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop6/20, trainingseconds513.957182/peakMiB1819.070801. No otherfolds.

## Final closure
Both naturalstop20/best6. RBF POR78.02029960381482/CONT41.60782861942168; cumulative78.0224639221774/CONT41.618027814697186. RBF minus cumulative whole -0.002164318363/CONT -0.010199195276. Bothbelow302: whole -0.010253190330/-0.008088871967, CONT -0.067251287650/-0.057052092374. Projectionnorms0.782424/0.712187 confirm nonzerolearned numeric residual, not dead initialization. No adoption or bandwidth/count/LR scan; close this numeric-token setting. Equal nominal80-14 but historical302 ran longer, no equalwork causal claim; CPUfinalizer inherited predefined27 fields are null/notapplicable (332trainer neverpredeclared27). Best144665 RAM replay/frozenexact, not independentdiskreload. Pairwall520.725597s/stagger1.166s, individual515.142976/513.957182s, peaks1819.071MiB; inputstatistics/prep/initial/train/restore included, implementation/finalization excluded. Retained8TOTAL80.699037441/default5 unchanged/strict808false/fullOOF82 unmet/nootherfolds.

333 LIVE: maximumPID59720/meanPID80892 after332terminal, GPUcap2. OrdinaryinitialPERM89.4055666268939/40098trainable both. Original154only+fresh312code+3528edge-message, allnewbody/readout/shapes/messages joint, onePERMphase/max8. No learnedgain yet/otherfolds. [Primarysource/history/interface research](../cross_domain_research_20261001/PERM_EDGE_MESSAGES_RESEARCH_20261002.md).

V333 final PERM89.499348140436, gain+0.093783283517, actualTOTAL80.645299567225; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds268.503931/peakMiB964.092773. No otherfolds.

V333 final PERM89.505470296241, gain+0.099905439322, actualTOTAL80.647442321757; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds266.493704/peakMiB946.983398. No otherfolds.

Prepared334SW meanpreserving variance-onlyboundedDSU p.5/std?10%,92289jointfresh326params/original104only; identity/noise sameRNGcontrols, no newphase or repairweights. PreviousMixStyle negatives audited, gain unproved/noqueue.

## Final closure
Both natural19/best5: maxPERM89.49934814043613/CONT74.00681575433087; mean89.50547029624123/CONT74.01617967397496. Maxminusmean whole -0.006122155805/CONT -0.009363919644. Versus312 maxwhole -0.016456264443/CONT +0.003927350318, meanwhole -0.010334108638/CONT +0.013291269962. Noadoption, close nonlinear-edge setting without radius/messagewidth/LRscan. Zerooutputactuallylearned normsmax1.631627/.187094 vsmean1.905923/.283305; all144665 RAMbest/frozenexact, notindependentdiskreload. Pair274.030683s/stagger1.119s, individuals268.503931/266.493704s, peaks964.093/946.983MiB, prepinitialtrainrestoreincluded/implementationfinalizationexcluded. Nominalsame80-14, effective max/meanfunction/gradient sparsity different, inheritednull27fields notapplicable. Retain8TOTAL80.699037441/default5/strict808false/nootherfolds/fullOOF82 unmet.

334 launched identityPID47368/noisePID36948 after333 terminal, actualmax2GPU globaladmissions. Normalenvironment80pat14/ownwholetargetbest/oneSWphase/jointfreshbodyreadout preserved; not gainyet.

Prepared335 POR positive302curveattentiontoken32?64,70768attention/128336jointactive, original151only, onewidthcandidate/historical32 reference43 cap. Unlaunched while334twoGPU active/no result/noqueue.

V334 noise: SW74.163475724720, actualTOTAL80.627078431791; original104only/fresh326earlylatent+morph joint encoder/readout training-only mean-preserving boundedvarianceDSU,92289 active, oneaddedSWphase. No fullOOF/otherfolds.

V334 identity: SW74.169543032075, actualTOTAL80.629201989365; original104only/fresh326earlylatent+morph joint encoder/readout training-only mean-preserving boundedvarianceDSU,92289 active, oneaddedSWphase. No fullOOF/otherfolds.

## Final closure
Both explicit evidence-based earlystop25/best13 (not natural27). Identitywhole74.1695430320749/default98.63988120716091/CONT17.96344208824835, exactly matches historical326. Noisewhole74.16347572471956/default98.63999486415061/CONT17.943177678122417. Pairedwhole -0.006067307355/CONT -0.020264410126, no adoption. Stoprequests recorded at24 because bothbest13/11nonimproving and noise loseswholeCONT; source/config unchanged and samePIDsrestore/RAM144665exact/frozenexact, notindependentdiskreload. Bothactual25 fair paired scope, predeclared27 snapshots not reached/no27gain claim. Close varianceDSU without strength/probability/meanvariance/LRgrid. Pairwall589.518556s/stagger1.102s, individuals585.006699/583.524759s, peaks1779.212/1779.998MiB, prepinitialtrainrestoreincluded/implementationCPUfinalizationexcluded. Identitypreserved retained positive control/reproducibility, not anothergain. Retain8TOTAL80.699037441/default5/fullOOF82unmet/nootherfolds.

335 LIVE: widePID42384 launched SOLO after334 bothterminal, freshglobalGPU check. OrdinaryinitialPOR77.92638076072386/128336active/70768attention confirmed; initialoriginalpredictorretained. Onewidth64candidate only, no secondjob/width-LRqueue, actualcost/memory ongoing. Attention backward warns nondeterminism as302; tiny single-seed difference notrobust gain. Source/config frozen once launched.

Prepared336 PERM transfer of POR306 locallypositive lowrankfeaturecross,816new/37386joint/original154only fresh312code. Cross/additive controls, no trainedrepairweights/phasechain/queue. Notresultyet.

336 LIVE crossPID80612, launchalongsidePOR335 with globalmax2. Additive remains UNLAUNCHED/noqueue; do not claimpairedgain until bothterminal. Only154trainedprior+fresh312body/readout/shape/816adapterjoint/onePERMphase/max8. [Primary and duplicate research](../cross_domain_research_20261001/PERM_LOW_RANK_FEATURE_CROSS_RESEARCH_20261002.md): POR306 localpositive, notnewDCNclass ortrainedweighttransfer.

335 LIVE ownbest20 snapshot: POR78.03415534328305 (+0.003602549138 vs302), default97.40771749843678, CONT41.69736588089467 (+0.022285973823). Bothpreliminarypositive, currentepoch21; no adoption/qualification/diskreload/fullOOFclaim yet, historicalcapacity/init/compute differ and single-seed tiny. PERM336cross nowconcurrent, soloearlycost distinct fromlater overlap.

336 cross LIVE best5 {'epoch': 11, 'best_epoch': 5, 'whole': 89.51838939879066, 'default': 97.49443975594247, 'continuous': 74.00963708319706, 'scope': 'Live own whole-target best snapshot, additive control unlaunched/no paired gain or adoption'}; no terminal/adoption/pairedgain until additivecontrol completes, otherfolds notstarted.

V336 final PERM89.518389398791, gain+0.112824541872, actualTOTAL80.651964007649; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds335.842096/peakMiB896.343750. No otherfolds.

336 cross TERMINAL natural19/best5 PERM89.51838939879066, CONT74.00963708319706, own144665RAMreplay/frozenexact, notindependentdiskreload. Individual335.842096s/peak896.344MiB, concurrentPOR335; no partialpairedgainclaim. Additivecontrol explicitlaunchPID16068 afterfreshglobalcapcheck, max2withPOR335; currentqueue empty. Pairedfinalpending, no adoption/otherfolds.

335 ownbest28 LIVE POR78.04221263500206 (+0.011659840857 vs302), CONT41.724510830138236 (+0.049430923067), default97.40559788886887; notterminal/no adoption/otherfolds. Historicalcapacity/initdifferent, tiny single-seed positive notrobustgain. Stilllive samePID42384, nowconcurrentPERM336additive aftercross terminal.

V335 final POR78.042212635002, gain+0.115831874278, actualTOTAL80.647224980277; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds715.343113/peakMiB2074.647949. No otherfolds.

V336 final PERM89.515909221499, gain+0.110344364580, actualTOTAL80.651095945597; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds203.706321/peakMiB893.296875. No otherfolds.

## Final adoption (fixed fifth only)
Natural42/best28, POR78.04221263500206/CONT41.724510830138236, vs302whole+0.011659840857/CONT+0.049430923067. Actual8TOTAL80.70253539343442 with312/326, +0.003497952257 vs prior326recipe. Best144665 RAMreplay/frozenexact, source checkpoint/pred SHAs in recipe, NOT independentdiskreload/fullOOF. Predeclared43notreached (actual42 vs historical43), no equalworkscaling/causalcapacity claim. Zero bridge actuallylearned. Individual715.343113s/process720.709706s/peak2074.648MiB; soloearly then PERM336cross/additive concurrency, no isolated comparison. Adopt only one replacementPORphase, default5/oldrecipes/deliveries untouched.

## Final adoption (fixed fifth only)
Bothnatural19/best5. Cross89.51838939879066/CONT74.00963708319706 vs additive89.51590922149897/CONT74.01434038347962. Pairedwhole+0.002480177292 but CONT -0.004703300283; do not claim continuous dominance over additive. Cross vs retained312 whole+0.002584993912/CONT+0.006748679184, choose cross by wholePERM while preserving CONT vs predeclaredbaseline. Actual8TOTAL80.70344014130356 with335POR/326SW, +0.004402700126 vs prior326recipe. Same37386physicalparams/init but functionfamily differs/zero-updowngrad delay, single seed notrobust. Ups actuallychanged; RAM144665replay/frozenexact notindependentdiskreload. Pair591.187799s/stagger382.007s includes delayed explicitcontrol start; individuals335.842096/203.706321s, peaks896.344/893.297MiB, mixed POR335overlap thenadditivesolo, notspeedranking. Adopt onePERMreplacementphase, allmethod8, strict808false/nootherfolds/fullOOF82unmet.

- [v337 POR curve-token128](results/deep_learning/por_curve_token_width_v337_20261002/README.md): single fresh joint encoder/readout width follow-up to final335, prepared; no score/no launch yet.

Explicitly launched v337 token128 single candidate; TIMING.json records PID/start, no automatic queue. Active no result yet; retained8 TOTAL80.703440 unchanged.

Live snapshot: PID55704, epoch1, bestPOR77.944334530954 (not final). First observed epoch 14.413s, peak 2599.449MiB, single GPU process at launch; no OOM observed so far, not final peak/speed conclusion. Full atomic state saved, actual initialized score77.926380760724 and330128 trainableparameters. Atlas curate/build completed; no independent disk reload or remaining folds.

Completed one CPU diagnostic [continuous bottlenecks](BOTTLENECK_DIAGNOSTIC.md). Compared with default5: PORwhole+.115831874 butCONT-.229450225; PERMwhole+.112824542 butCONT-.087274861; SWwhole+.047790204 andCONT+.331553028. Existing335/336 local-control CONT positives remain valid, but do not mean original151/154 CONT improved. Fixed fifth diagnostic only, no oracle/model-selection/reweighting, no new phase. TOTAL gap to80.8=.096559859; no cause-of-backbone claim. Planned next research depth/curve coupling,337 still active unchanged.

- [v338 POR depth-curve coupling](results/deep_learning/por_depth_curve_coupling_v338_20261002/README.md): fresh64 curveattention interleaved lowrank17tap temporal residual, predefined depth/control. Prepared, no score/no GPU yet.

Explicit depth launch recorded in depth_TIMING.json, overlaps337; control prepared and not launched/queued. No score yet; retained8 recipe unchanged.

V337 final POR78.031174140712, gain+0.104793379988, actualTOTAL80.643913431990; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop10/24, trainingseconds394.302705/peakMiB2599.449219. No otherfolds.

Closed v337 width128: natural24/best10 POR78.031174140712, vs335whole-.011038494290/CONT-.055595209042 (CONT41.668915621097). No adoption/adjacentwidth-LRgrid. Predefined42notreached, smallerwork not equal42gain. RAM144665replay/frozenexact,394.303s inclprep/warmup/bestrestore,peak2599.449MiB;23/24 overlap338depth, no isolated speedranking. Keep335 and retained8 TOTAL80.703440. Negative checkpoints not deleted: earlier native cleanup policy rejection still unresolved; do not bypass.

Explicit control launch after337 authoritative terminal; actual timing/PID in control_TIMING.json. Now two338GPUarms, staggered not synchronous, no automaticqueue. Research completed. Terminal CPU finalize_pair prepared, not executed against live jobs.

Verified-live snapshot {"depth": {"status": "running", "pid": 66772, "stage": "depth", "epoch": 9, "best": 78.02393658914868, "best_epoch": 6, "bad": 3, "peak_MiB": 2212.4833984375}, "control": {"status": "running", "pid": 86424, "stage": "control", "epoch": 1, "best": 77.94462573495032, "best_epoch": 1, "bad": 0, "peak_MiB": 2212.4833984375}}. Both initializePOR77.926380760724 with134752activeparams; depth observedpeak2212.483MiB, noOOM observed so far. This is preliminary, no whole/CONTgain yet. Paired process walltime must include explicit stagger and337overlap for earlydepth; same code/params does not imply isolated speed ranking. Atlas curated/built:337deadend and338live pair.

- [v339 POR conditional-onlyRPS](results/deep_learning/por_conditional_rps_v339_20261002/README.md): samefresh335encoder/readout jointly trained, globalrowmean conditionalRPS vs historicaljoint; prepared/untrained/researchpending, no queue or launch.

Latest live paired snapshot: {"depth": {"status": "running", "pid": 66772, "stage": "depth", "epoch": 19, "best": 78.02495795438341, "best_epoch": 11, "bad": 8, "peak_MiB": 2212.4833984375}, "control": {"status": "running", "pid": 86424, "stage": "control", "epoch": 11, "best": 78.025972197231, "best_epoch": 11, "bad": 0, "peak_MiB": 2212.4833984375}}. Both below historical335whole; no paired or absolute gain established. Control best11 is slightly above depthbest11, preliminary only. Native PIDs re-observed live, no restart or source changes.339conditionalRPS source/PLAN prepared pending final research document, noGPU/queue. Atlas rebuilt thisbatch.

Historical research completed: [conditionalRPS evidence](../cross_domain_research_20261001/POR_CONDITIONAL_RPS_RESEARCH_20261002.md), no exact335/287 negative found in limited lookup. Generic analyticCONTCRPS already existed includingPOR; not firstconditional supervision. Implementation retains globalrowmean and original mixedF; source syntax prepared only, UNTRAINED/noGPU/queue. Planned explicit solo launch after338slots clear: observed338depth solo~12.6s vs dual~30s per epoch suggests this workload benefits little from2concurrency; not isolated complete-work benchmark. No changes to active338source/budget.

Observed timing decision: depth epochs2-8 (after337exit, beforecontroltraining overlap) mostly12.6-14.2s; sustained paired depth epochs9 onward roughly30s/control~30s. Two same-length concurrent epochs ~30s exceeds two representative solo12.6s epochs ~25.2s; observational not clean fixed-work benchmark, warmup/CPU/power/persistence overlap differ. Do not assume max2 means always2; let active pair finish unchanged and prefer next339 solo. Deeper20best78.034091141181 remains below33578.042212635002, no adoption. Next gate/body joint historical research started, idea only/no code/GPU/queue.

- [v340 POR fresh-gate joint](results/deep_learning/por_fresh_gate_joint_v340_20261002/README.md): +49 boundedgate/fresh335encoder-density jointly, original151only. Prepared/untrained/notqueued,339priority.

Saved-best prediction diagnostic (not modeldiskreload): {"arms": {"depth": {"observed_through_epoch": 33, "best_epoch": 28, "whole": 78.0458220539967, "strata": {"default": 97.40559788886887, "continuous": 41.73489002626708}, "whole_vs335": 0.0036094189946425104, "CONT_vs335": 0.01037919612884508, "CONT_vs_original151": -0.21907102863698213}, "control": {"observed_through_epoch": 25, "best_epoch": 20, "whole": 78.03125037336724, "strata": {"default": 97.40771749843678, "continuous": 41.68901238894759}, "whole_vs335": -0.010962261634816173, "CONT_vs335": -0.035498441190647156, "CONT_vs_original151": -0.26494866595647437}}, "preliminary": true, "not_final_pair_or_gain": true, "saved_best_predictions_only": true, "no_independent_model_disk_reload": true, "no_epoch_selection_by_CONT": true}. Initial localdepth positivewhole/CONT vs335 now observed, but unfinished pair/unequal currentepochs so no finalgain/adoption. Original151 CONT remains better; DEFAULT currentlyequal335 for depth. Active normal80/14 continues, no epoch chosen by CONT/no source changes. Prepared339/340 remain untrained; do not trainchain338weights.

Terminal CPU actual8 combination assembler prepared, NOTexecuted: requires completedPAIRED_FINAL, depthwhole+CONT vs335 AND completedcontrol positive. Only then actualfixedPOR/PERM336/SW326columns, newrecipe artifact/nooverwrite/nootherfolds. Livecontrolhasnewbest28near335, so partialpairedmargin narrowed; no currentadoption.

V338 final POR78.045822053997, gain+0.119441293273, actualTOTAL80.648307805976; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds1171.925845/peakMiB2212.483398. No otherfolds.

Depth terminal natural42/best28 POR78.045822053997; predefined42 snapshot saved, versus335whole+.003609418995 andCONT+.010379196129 (CONT41.734890026267), original151CONT-.219071028637. RAM144665best replay/frozenexact;1171.926s/peak2212.483MiB mixed337early/solo/controloverlap. No finalpair/adoption untilcontrolterminal and actual8combination.339/340prepared/noGPUqueue, next339solo preferred.

V338 final POR78.042133067874, gain+0.115752307150, actualTOTAL80.647201110139; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds1140.112847/peakMiB2212.483398. No otherfolds.

V338 final positive retained: bothnatural42/best28, depthwhole78.045822053997/CONT41.734890026267; pairedwhole+.003688986123/CONT+.008620242854, versus335whole+.003609418995/CONT+.010379196129; native151CONTstill-.219071028637. Actual8TOTAL80.704522967002, gainvs336+0.001082825698, default5/oldrecipes/deliveries intact; strict808false/nootherfolds/fullOOF82unmet. RAM144665replay/frozenexact notmodeldiskreload. Pairwall1279.131s/stagger133.536s/mixed337overlap, individual1171.926/1140.113s/peak2212.483MiB. Next339explicit solo launch recorded,340preparednotqueued.

Verifiedlive PID36304: {"status": "running", "pid": 36304, "stage": "conditional", "epoch": 8, "best": 78.02258667724234, "best_epoch": 6, "bad": 2, "peak_MiB": 2074.64794921875}. Initialized77.926380760724/128336active, solo firstepochs11.9-12.1s/peak2074.648MiB/noOOMobserved. No finalgain/CONTclaim yet. Currentretained8 actual338recipe80.704522967002, source/control335preserved; candidateisolatesRPS and doesnotload338weights. Atlasadoption/prepared340 updated and built.

V339 final POR78.042499459539, gain+0.116118698815, actualTOTAL80.647311027638; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds506.354363/peakMiB2074.647949. No otherfolds.

V339 terminal natural42/best28 whole78.042499459539/CONT41.725335618872, DEFAULT97.405597888869 unchanged335. Samecapacitymechanismdelta vs335whole+.000286824537/CONT+.000824788733, native151CONTstill-.228625436033; lowerthanretained338whole-.003322594458, no currentrecipe replacement. Tiny single-seed/matchedcapacity historicalcomparison notrobustgain; sourcecode/positivecheckpoint preserved, maycarrycode only intoisolatedjointfollowup nottrainedweights. RAM144665replay/frozenexact, solo506.354s/peak2074.648MiB inclprep/warmup/restore, actual42prefix saved. Inheritedsnapshot phrasecapacitydiffers isnottruefor339/335; humanFINALclarifies. Nootherfolds.

Explicit solo340launch after339authoritative terminal, TIMINGrecordsPID/start. Original151only/fresh335encoder-readout+zero49gate trainjoint inonephase; no339trainedweights/losscombination. Retain3388TOTAL80.704522967002 whileunscored/nootherfolds. Continuous-supportoffsetresearchclosedbeforetraining duegradient/probability/costscope, no OOM ornegative modelscoreclaimed.

Verifiedlive340snapshot {"status": "running", "pid": 25556, "stage": "gate", "epoch": 5, "best": 77.93029485497402, "best_epoch": 5, "bad": 0, "peak_MiB": 2074.65185546875}. Earlywholebelowinitial; no benefit proven, no prematureclaimnewgatehelps. Original151sourceinit77.926380760724/128385active, 49gateparameters plusjointfreshencoder/density, ~12s/epoch/peak2074.652MiB solo/noOOMobserved. Atminimum10 comparetrajectory and savedbest before evidence-basedclosure; nosource/budgetchange orautoqueue. Atlas339localpositive andsupportresearchclosednottrained curated/built. Currentretained8TOTAL80.704522967002 unchanged.

- [v341 POR positive-code combination](results/deep_learning/por_coupling_conditional_rps_v341_20261002/README.md): fresh338depth +339conditionalRPS, same134752capacityloss-onlyvs338, fromoriginal151only/onePORphase. Prepared/untrained/noGPUqueue;340continues.

Live340epoch11/best6POR77.970000337129 vsretained33878.045822053997, noadvantageyet. Prepared341samecapacitypositivecodecombination, onePORphase/from151only, noGPUorqueue. Current340source/budgetunchanged, monitorfirstbeforeexplicitnextlaunch.

Saved-best diagnostic at27/best20: {"epoch": 27, "best_epoch": 20, "whole": 77.97763069712732, "strata": {"default": 97.12792903547167, "continuous": 42.05959181044599}, "whole_vs335_final": -0.06458193787473476, "whole_vs335_best_through_same_epoch": -0.05652464615572228, "CONT_vs335": 0.335080980307751, "DEFAULT_vs335": -0.27766885339720204, "saved_predictions_only_not_modeldiskreload": true, "preliminary": true}. CONTpositive(+.335080980vs335, +.105630756vs151) whileDEFAULT-.277668853/whole-.064581938, so conditionalearly-stoprule requiringCONTnegative wasNOTmet/no stoprequestwritten. ActivePID25556 continuesnormal80/14; do notassumeterminalfrombad13asbest20reset. One earlierCPUterminalfinalizer invocation beforeactualterminal failed missingreport, wrote nofinalresult/noGPUrestart; futurefinalizationonlyafterreport+CIMterminal. Newtrain-onlyauxstate/frozeninferencegate researchidea requested, no code/GPU/weightsreuse.341prepared/untrained/noqueue.

V340 final POR77.992995342042, gain+0.066614581318, actualTOTAL80.632459792389; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop29/43, trainingseconds523.074109/peakMiB2074.651855. No otherfolds.

V340 closeddeploymentgatebound.5 setting: natural43/best29POR77.992995342042, DEFAULT97.227550685164/CONT41.916925164121. Vs335whole-.049217292960/DEFAULT-.178047203705/CONT+.192414333982, versusnative151CONT-.037035890784. Earlierbest20CONT42.059591810446 wasnotfinalwhole-selected checkpoint; no CONT-best epoch selection/adoption. Positivecontinuousrepresentation tradeoff motivates exactlyone auxstate/frozenmain-gate researchcompleted, no bound/LR/BCEgrid. RAM144665best replay/frozenexact, predefined42samewhole-selectedbest29;523.074s/peak2074.652MiB solo. Nootherfolds/recipechange: retain3388TOTAL80.704522967002.

Explicit solo341launch after340report/CIMterminal, PID/start TIMING. Fresh338depthencoder/readout+339conditionalRPS CODEonly fromoriginal151, nottrainedrepairchain.134752samecapacitycontrol338, onePORphase/max8, unscored/no queue/nootherfolds; retainedrecipeunchanged.

Verifiedlive341 {"status": "running", "pid": 82236, "stage": "conditional", "epoch": 6, "best": 78.02412321285436, "best_epoch": 6, "bad": 0, "peak_MiB": 2212.4833984375}. Solo~12.5-12.8s/epoch, peak2212.483MiB/noOOMobserved, original151initial77.926380760724/134752active. No finalgain, current338recipeunchanged; no preparedauxstateGPU orqueue.340deploymentgateclosed/CONTtradeoff and341live atlascurated/built. Auxstate research final340corrected, earlierCONTbest20notchosenafterfinalbest29.

- [v342 POR auxstate/frozen main gate](results/deep_learning/por_aux_state_frozen_gate_v342_20261002/README.md): freshencoder/density+49stateBCEonly, original151gateforjointRPS andprediction; same128385capacityrolechangevs340. Prepared/untrained/noqueue,341active.

Verifiedlive341snapshot {"status": "running", "pid": 82236, "stage": "conditional", "epoch": 21, "best": 78.03310518112255, "best_epoch": 20, "bad": 1, "peak_MiB": 2212.4833984375}.342auxstate source/PLAN/localsyntax prepared andatlasbuilt, noGPU/queue. One SWcurvetoken32 transfer research started only, noimplementation/modelscore/launch. Current3388TOTAL80.704522967002 unchanged/strict808notmet/nootherfolds,341samePIDcontinues.

V341 final POR78.046430717932, gain+0.120049957208, actualTOTAL80.648490405156; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds519.848364/peakMiB2212.483398. No otherfolds.

2026-10-02 v341 terminal42/best28 POR78.046430717932/CONT41.736640291995; tiny whole+.000608664/CONT+.001750266 vs338. User dropout: v343 p.1 FFN, joint encoder/readout, original151 only,134752 unchanged params, prepared/score unproven;342 held. See results/deep_learning/por_ffn_dropout_v343_20261002/README.md.

2026-10-02 terminal341: whole78.046430717932/CONT41.736640291995, vs338+.000608664/+.001750266; native151 CONT still-.217320763. Natural42/best28,519.848s/2212.483MiBsolo. Actual144665 fixed columns341/336/326 TOTAL80.704705566183; optional8 retained, single microgain not robustness, RAMreplay not independent disk reload. Strict808false, no otherfolds/fullOOF82.

Explicit solo launch PID4160 after341terminal/globalGPU0. Observedinitial77.926380760724/134752 params and epoch1 POR77.944651857837, peak2229.881MiB; running snapshot not finalgain.

V343 final POR78.044847767677, gain+0.118467006953, actualTOTAL80.648015520080; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds487.441191/peakMiB2229.881348. No otherfolds.

## Terminal conclusion
42 epochs/best28, POR78.044847767677, delta341-0.001582950256; CONT41.732088381788, delta-0.004551910207; DEFAULT unchanged97.405597888869. Actual144665 columns343POR/336PERM/326SW TOTAL80.704230681106, delta retained-0.000474885077, diagnostic only/not adopted. Same historical341 parameter init/capacity/loss and actual42 budget; single seed, stochastic trajectories differ, no robustness claim.487.441191s individual/491.917613s process wall solo includes prep/warmup/restoration,2229.881MiB peak. RAMbest replay/frozenexact, not independent disk reload. Close this FFN p.1 setting, no dropout grid. Retain341 optional8 TOTAL80.704705566183/default5 and prior deliveries. No otherfolds/fullOOF82 unmet. No new GPU launch.

2026-10-02 previousgoalturn=progress:343dropout completed negative, preserves341TOTAL80.704705566183.342auxstate explicitlylaunchedPID50104 solo after343terminal; original151/freshencoderreadout/allmodulesjoint.344SWcurve interaction prepared full/grouped111313params/original104only, no GPU yet/noqueue. See studies README.

V342 final POR78.041027224873, gain+0.114646464150, actualTOTAL80.646869357239; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds497.890525/peakMiB2074.651855. No otherfolds.

2026-10-02 terminal342 natural42/best28 POR78.041027224873, CONT41.721102080908, DEFAULT97.405597888869. Vs335 whole-.001185410129/CONT-.003408749230; vs340 whole+.048031882831 butCONT-.195823083212, native151CONT-.232858973996. Neither absolute whole+CONT condition holds, no adoption/auxrole setting closed/no grid.497.890525s individual/502.480048s process wall solo includes prep/warmup/restoration,2074.652MiB; all144665RAMbest replay/frozenexact notmodeldiskreload/fullOOF. Current341 eight-stage TOTAL80.704705566183 retained.

2026-10-02 SW344 full explicitlysolo launched PID43324 after342terminal/globalGPU0; grouped not launched/noautoqueue. Gain unproven. Independent adaptive-depth/orderednumeric research completed; research not training.

Observed actual training: PID43324 live, activeparameters111313, latestepoch7, bestSW74.128225851893/epoch4, peak2080.373MiB. Initial original10474.121752689769 matched within prescribed tolerance. No gain claim from this running snapshot; grouped still unlaunched.

2026-10-02 v345 sparse adaptive depth reader prepared adaptive/fixed136626params/original151only/fresh341code; no GPU or score/no queue. See study README.344full remainsactive.

V344 full: SW74.147729339521, actualTOTAL80.621567196971; fresh326encoder/state/readout/ownlatent/morph plus19024 curve interaction, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

344full terminal27/best13 SW74.147729339521, CONT17.904849573953, DEFAULT98.634079836945; vs326 whole-0.021813692554/CONT-0.058592514295, noadoption/paired scope evidence pending.306.951190s/2080.373MiB solo inclprep/warmup/restoration. Explicitgrouped PID9332 afterfullterminal/globalGPU0, same111313physicalparams/init; not equal effective functions. Noqueue/otherfolds.

Grouped ordinary run observed PID9332live, actualactive111313, epoch6, best74.129472610215/epoch4, peak2080.374MiB, stillsnapshot/unproven. Original104initial74.121752689769 matched. Fullcurve delta326whole-.021813692554/CONT-.058592514295 means notadopted even beforepaired scope evidence. Next345prepared only/noGPUqueue.

V344 grouped: SW74.147460525283, actualTOTAL80.621473111988; fresh326encoder/state/readout/ownlatent/morph plus19024 curve interaction, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

## Terminal paired conclusion
Bothnatural27/best13, fullSW74.147729339521/CONT17.904849573953 vs groupedSW74.147460525283/CONT17.900824215758. Full-minus-grouped whole+.000268814237/CONT+.004025358195: real local single-seed scope-positive, not robustness. However full vs326 whole-.021813692554/CONT-.058592514295, grouped-.022082506792/-.062617872490, so both not adopted/no width-LRgrid. Close this current fresh326 intersection as replacement, preserve local fullcommunication code idea; not all curve methods invalid. Both111313physicalparams/init but differing communication/functioncapacity, historical32692289 capacity differs. Individual306.951190/302.731841s solo includingprep/warmup/restoration; process pair657.270827s including serial launchgap/stagger349.212, peaks2080.373/2080.374MiB. All144665 RAMbestreplay/frozenexact, notmodeldiskreload/fullOOF; strict808/fullOOF82 unmet. Current3418TOTAL80.704705566183 remains.

2026-10-02 adaptive explicitlysolo launched PID76472 after344pairterminal/globalGPU0; fixed arm not launched/noautoqueue. Source/config unchanged, gain unproven untilterminal.

Ordinary live training confirmed PID76472, initialPOR15177.926380760724, actualactive136626, epoch4, best77.977395185364/epoch4, peak2437.805MiB, around13s/epoch. Encoder/readout/interaction group gradients observed positive. Snapshot notgain, fixedcontrol notlaunched.

2026-10-02 v346 PERM orderedknots prepared learned/fixed fresh336code original154only, <=40914active, onePERMstage/max8; noGPU/gain/noautoqueue.345adaptive remainsactive.

V345 final POR78.045290238763, gain+0.118909478039, actualTOTAL80.648148261406; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds573.113275/peakMiB2437.804688. No otherfolds.

Adaptive natural42/best28 POR78.045290238763/CONT41.737336256473/DEFAULT97.403478279301; vs341 whole-0.001140479170/CONT+0.000695964478, native151CONT-0.216624798431. Selectoroffset weightnorm0.998149335/bias0.176569641, tapweightnorm1.067079663, readerparameters genuinelychanged, not merely whole-groupgradient; norms do notprove meaningful positions/causal gain.573.113275s/2437.805MiBsolo inclprep/warmup/restoration,144665RAMbest replay/frozenexact notmodeldiskreload. Noadoption/fixedcontrolpending. Fixed explicitPID18156 afteradaptive terminal/globalGPU0.346orderednumeric remainsprepared/untrained.

Fixed actualtraining observed PID18156live/activeparams136626, epoch8, best78.023776976369/epoch6, peak2364.837MiB around13s/epoch. OrdinaryinitialPOR151 matched. AdaptivefinalCONT+.000695964478 vs341 butDEFAULT-.002119609568 leadswhole-.001140479170; no absoluteadoption, fixedpairpending.346 preparedonly/untrained.

V345 final POR78.043240611423, gain+0.116859850699, actualTOTAL80.647533373204; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds577.917894/peakMiB2364.836914. No otherfolds.

## Terminal paired conclusion
Bothnatural42/best28, adaptivePOR78.045290238763/CONT41.737336256473 vsfixedPOR78.043240611423/CONT41.729454620568: pairwhole+.002049627340/CONT+.007881635905. Localmechanism-positive underphysicalmatchedrolesdifferentcontrol/singleseednotrobustness. Adaptivevs341whole-.001140479170/CONT+.000695964478/DEFAULT-.002119609568 andnative151CONT-.216624798431 => noabsoluteadoption. Retainadaptivecodeaslocalpositiveidea, do notloadtrained345repair intoextra phase/nooffsetrange-LRgrid; fixedreplacement notworthy. Offsetweights genuinelychanged0.998149norm/bias.176570, up.963980, not justgroupgrad. Individual573.113275/577.917894s solo inclprep/warmup/restoration; pair1192.352305s includes explicitserialgap/stagger609.035, peaks2437.805/2364.837MiB.144665RAMbestreplay/frozenexact notmodeldiskreload/fullOOF,42prefixsamework. Current341TOTAL80.704705566183 retained, strict808/fullOOF82unmet/nootherfolds.

2026-10-02 learned explicitsolePID50464after345pairterminal/globalGPU0; ordinary INITIALPERM89.405566626894 actual40914params, epoch1normal/914.309MiB. Inputboundsfit9columns then64trainanchors avoids206wide temporary; sameinputpopulation. Fixedarmnotlaunched/noautoqueue; gainunproven. SPO+ alternative research completed, notimplemented/noGPUqueue.

V346 final PERM89.506387553016, gain+0.100822696097, actualTOTAL80.647763361628; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds153.535463/peakMiB914.309082. No otherfolds.

Learned natural19/best5 PERM89.506387553016/CONT74.005935936341/DEFAULT97.478168914750; vs336 whole-0.012001845775/CONT-0.003701146856, native154CONT-0.090976007476. Actualknots logitnorm0.760082901/projection2.225332022, intervalfractionmin0.107098334/max0.142446488, activeS72; learning occurred, no causalbenefitclaim.153.535463s/914.309MiBsolo incltraininputfit/prep/warmup/restoration,144665RAMbestreplay/frozenexact notdiskreload. Noadoption/fixedpending. Fixed explicitPID56572afterlearnedterminal/globalGPU0; noautoqueue.

2026-10-02 v347 PERM finalmixture SPO+ prepared reference/lambda.1 same37386fresh336/original154only; actualowntrainingtruth/oldEvaluatorK/actions, no newparameters/decoder orGPU/score/noqueue.346fixed remainsactive. See studyREADME.

V346 final PERM89.506393762633, gain+0.100828905714, actualTOTAL80.647765534994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds151.729385/peakMiB903.087891. No otherfolds.

## Terminal paired conclusion
Bothnatural19/best5, learnedPERM89.506387553016/CONT74.005935936341 vsfixed89.506393762633/CONT74.006486486651; learned-fixedwhole-.000006209617/CONT-.000550550310. Vs336learnedwhole-.012001845775/CONT-.003701146856, fixed-.011995636158/-.003150596546 => no adoption/no knotsgain, close current8uniform orderedknot/stemsetting, no K/bounds/family/LRgrid. Not allnumericembeddinginvalid. Learnedlogits changednorm.760083/projection2.225332/widthfractions.107098-.142446, no minfloorcollapse; fixedzero logits/equal.125 verified. Learned40914 vsfixed40842/72trainablecapacitydifference disclosed. Individual153.535463/151.729385s solo includestraininputfit/prep/warmup/restoration; pair398.630086s includes explicitserialgap/stagger241.804, peaks914.309/903.088MiB.144665RAMbestreplay/frozenexact notdiskreload/fullOOF/single seed,19prefixsamework. Current341TOTAL80.704705566183 retained/strict808fullOOF82unmet/nootherfolds.

2026-10-02 reference explicitlysolo launchedPID70228 after346pairterminal/globalGPU0; spo armnotlaunched/noautoqueue. Zero-coeffreferencecomputesSPO diagnostics butno addedloss, samefresh33637386; no oldtrainedrepairweights. In modeldiskreload construct/configure originaldecision buffersbeforeloadingstate, originaldecoder unchanged. Gainunproven.

V347 final PERM89.518389398791, gain+0.112824541872, actualTOTAL80.651964007649; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds169.793728/peakMiB907.652832. No otherfolds.

Reference natural19/best5 PERM89.518389398791/CONT74.009637083197/DEFAULT97.494439755942; original336 predictions exactlysame=True, changedrows0; no baseline alteration.169.793729s/907.653MiBsolo inclprep/traintruthtransfer/warmup/restoration. ReferencecomputedSPO0coefficient extra graph notnewloss.144665RAMbest replay/frozenexact notdiskreload/fullOOF. SPOexplicitPID58256afterreferenceterminal/globalGPU0, same37386params/init andK/actions, gainunproven/pairedpending. Current341TOTAL80.704705566183 remains.

V347 final PERM89.518347404301, gain+0.112782547382, actualTOTAL80.651949309578; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop5/19, trainingseconds171.090512/peakMiB907.652832. No otherfolds.

## Terminal paired conclusion
Bothnatural19/best5, referencePERM89.518389398791/CONT74.009637083197/DEFAULT97.494439755942 exactlyreproduceshistorical336predictions, versusSPO89.518347404301/CONT74.008876185089/DEFAULT97.494767488904. SPO-reference whole-.000041994490/CONT-.000760898108/DEFAULT+.000327732962 => no whole+CONTgain/no adoption, close fixedlambda.1SPO+ setting/no lambda-temperature-decoder-LRgrid, not all decisionlossinvalid. Native154CONT still-.088035758727. Same37386params/init/K/actions800/fullprobabilitysupport257, actualowntrainingtruth578891rows/heldouttruth NaN, no validationoracle/decoderchange, allnewencoderreadoutjointonePERMphase/max8/no repairweights. Individual169.793728/171.090512s solo includestraintruthtransfer/prep/warmup/restoration, pair398.121619s includes explicitserialgap/stagger221.925, peaks907.653MiBboth.144665RAMbestreplay/frozenexact notmodeldiskreload/single seed,19prefixsamework. Current341TOTAL80.704705566183 remains/strict808fullOOF82unmet/nootherfolds. After terminal, auxiliary documentation snapshot writer initially expected epoch field andfailedKeyError; no fileswritten/noGPUfailure/no restart, corrected by reading terminalreport. Nativeexpert earlyLoRA research inprogress, no source/GPUyet.

V348 prepared: [native encoder rank4](results/deep_learning/perm_native_encoder_lora_v348_20261002/README.md); original154 warm native enc2+readout85792 active, no score/no new head-only job, no otherfolds.

348 terminal failed initial dimension assertion/no first batch/no score. [349 corrected native rank4](results/deep_learning/perm_native_encoder_lora_v349_20261002/README.md): actual128to128,2048new/84768active, same hypothesis/original154warm/nativeforward, no newheadonlyjob/nootherfolds.

349 runtime snapshot: initialized exact native154 score89.405564856919; real enc2LoRA/readout gradient groups recorded in GRADIENT_SNAPSHOT, A/B current norms separate, livePID88616, gainunproven.849.643MiB peak so far, solo, no OOM/no otherfolds.

V349 final PERM89.405564856919, gain+0.000000000000, actualTOTAL80.612475417994; sixphases inclsource/one addedPERMstage, strict808=False; all144665replay/frozenexact, best/stop0/14, trainingseconds119.130165/peakMiB849.642578. No otherfolds.

V349 terminal: natural14/best0 PERM89.405564856919, CONT74.09691194381553, exactnative154predictions changedrows0. No gain/adoption; whole vs336-0.11282454187171709. Encoder/readout gradientsnonzero, terminal currentBnorms changed but best0 restoreszeroB, notgradientblock. Solo119.130s/849.643MiB inclprep/warmup/restoration; RAM144665replay/frozenexact notdiskreload. Closedenc2rank4/nativewarmsetting/no sweep, fullOOF82 andfifth808 unmet; current341TOTAL80.704705566183 unchanged.

[350 parallel depth reading](results/deep_learning/por_parallel_depth_read_v350_20261002/README.md) prepared/no score: same345positivecode/init136626params, parallel residual connection only, original151only/allnewencoderreadoutjoint/onePORphase/max8, no otherfolds orautomatic queue.

350 adaptive live PID69740 confirmed by CIM/record at epoch3, bestPOR77.97621358057867, best_epoch3, peakMiB2401.8046875. Initial exactly15177.926380760724. Preliminary only/no gainclaim, soleGPUtask, fixedcontrol notlaunched/noautomaticqueue, original341current recipe retained80.704705566183.

[351 dynamic cross-spectrum](results/deep_learning/sw_dynamic_cross_spectrum_v351_20261002/README.md) prepared/no GPU/no score, aligned/permuted92841freshjoint/original104only/oneSWphase/max8; current350 running.

350 live snapshot PID69740 confirmed: epoch23, bestPOR78.03311426692649/bestepoch20, peak2401.805MiB, soleGPUtask. Continued improvement at20 meansno evidenceearlyclosure now. No terminal/positive claim.351SWdynamiccross-spectrum prepared552new/92841joint, notlaunched; prefer serial based previous measuredPORsolo12.6s vs dual~30s/epoch each, no new current351speed claim. No queue/otherfolds.

V350 final POR78.045046771530, gain+0.118666010806, actualTOTAL80.648075221236; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds591.607444/peakMiB2401.804688. No otherfolds.

V350 terminal POR78.04504677153032/CONT41.7366361454133, best/stop28/42, parallel minusserial345 whole-0.0002434672324937992/CONT-0.0007001110596434046, vs341whole-0.0013839464020009018. Diagnostic8TOTAL80.70429038226204; adopted=False. parallel connection only, no offset-width-LR grid; does not invalidate345localadaptivepositive. Fixedarm notlaunched/no matchedgainclaim. Solo individual591.6074435999617s/peak2401.8046875MiB includesprep/warmup/restoration;144665RAMbestreplay/frozenexact/notdiskreload/fullOOF82unmet/nootherfolds.

351 aligned explicitsolo launch PID72608 after350terminal confirmed/globalGPU0. Only original104/fresh326code92841joint/552spectrumparams, no gainclaim untilterminal. Permutedcontrol prepared/notlaunched/noautomaticqueue. Source/config frozen fromlaunch; current341TOTAL80.704705566183 unchanged/nootherfolds.

351 aligned running PID72608, epoch16, bestSW74.17361205223327/bestepoch13, peak1800.100MiB/soleGPUtask. Initial74.121752689769 same fresh326function within historicnumericvariation; no gainclaim/matchedpermuted notlaunched.350completeclosed; retained341TOTAL80.704705566183 unchanged.

351 live best snapshot epoch21/best13 SW74.17361205223327, vs326whole0.00406902015836863/CONT0.015957061798282268. Nonterminal/no matchedcausalgain/adoption; LIVE_BEST_DIAGNOSTIC recordsgroups andlearnedmixing/bridgenorms.

V351 aligned: SW74.173612052233, actualTOTAL80.630626146420; fresh326encoder/state/readout/ownlatent/morph plus19024 curve interaction, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

V351 aligned terminal SW74.17361205223327/CONT17.979399150046632, vs326whole0.00406902015836863/CONT0.015957061798282268, best/stop13/27. Diagnostic8TOTAL80.70612972323806 (notadopted untilpair); spectrum mixing/bridgenorms recorded/266.364304799994s solo/1800.099609375MiB inclprep/warmup/restoration;27prefix/True/144665RAMbestreplay/frozenexact/notdiskreload. Strict808=False/fullOOF82unmet/nootherfolds.

351 aligned terminal verified positive vs326 whole+.004069020158/CONT+.015957061798, actualfixed341/336/351TOTAL80.706129723238/retained+.001424157055, noadoptionbeforepairedcontrol. Permuted same92841params/init explicitsoleGPUlaunch PID74664 afteralignedterminal/globalGPU0, noautoqueue/otherfolds;strict808false/fullOOF82unmet. Individual266.364s/1800.100MiB inclprep/warmup/restoration/process270.527s,27prefix best13.

V351 permuted: SW74.168936197103, actualTOTAL80.628989597125; fresh326encoder/state/readout/ownlatent/morph plus19024 curve interaction, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

V351 permuted terminal SW74.16893619710333/CONT17.95482079589234, vs326whole-0.0006068349715633303/CONT-0.008621292356011168, best/stop13/27. Diagnostic8TOTAL80.7044931739426 (notadopted untilpair); spectrum mixing/bridgenorms recorded/246.76499830000103s solo/1800.099609375MiB inclprep/warmup/restoration;27prefix/True/144665RAMbestreplay/frozenexact/notdiskreload. Strict808=False/fullOOF82unmet/nootherfolds.

V351 pairterminal aligned-minus-permuted whole0.004675855129931961/CONT0.024578354154293436, mechanismpositive=True, adopted=True. Candidate actual341/336/351TOTAL80.70612972323806/delta0.0014241570554247573; eightphases/oneadditionperhead/default5 andoldrecipes/deliveriesunchanged. Pairwall582.017707824707s inclserialgap/stagger330.95099997520447s,individual/peaks inarms.144665RAMreplay/frozenexact/notdiskreload/singleseed, strict808false/fullOOF82unmet/nootherfolds. Positive finite mechanism; eligible for distinct isolated extension using positive code/init only, never trainedrepairchain

[352 phase coherency](results/deep_learning/sw_phase_coherency_v352_20261002/README.md) prepared/no score;351pairedpositive adoptedoptional8TOTAL80.706129723238,352positivecodeinitonly original104freshjoint92841, no newparams/no trainingchain/otherfolds.

352 phase explicitsoleGPUlaunch PID68788 after351pairedterminal/globalGPU0; magnitudeprepared/noautomaticqueue. Original104only/same92841/freshencoderreadoutjoint/noextraSWstage. Currentretained351optional8TOTAL80.706129723238;351gainconfirmedfinitepair notrobustness/fullOOF/strict808.

[353 POR dynamiccross-spectrum transfer](results/deep_learning/por_dynamic_cross_spectrum_v353_20261002/README.md) prepared/nottrained/noGPU,135304joint/552module/original151only/codepositive351notweights, onePORphase/max8, no queue/otherfolds.352SWstillrunning.

V352 phase: SW74.172212414540, actualTOTAL80.630136273228; fresh326encoder/state/readout/ownlatent/morph plus19024 curve interaction, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

V352 phase terminal SW74.1722124145395/CONT17.970803388760466, vs326whole0.0026693824646031317/CONT0.007361300512116031, best/stop13/27. Diagnostic8TOTAL80.70563985004524 (notadopted untilphase-magnitude pair); spectrum mixing/bridgenorms recorded/250.45614449999994s solo/1803.537109375MiB inclprep/warmup/restoration;27prefix/True/144665RAMbestreplay/frozenexact/notdiskreload. Strict808=False/fullOOF82unmet/nootherfolds.

352 phase terminal natural27/best13 SW74.172212414540/CONT17.970803388760, vs351whole-.001399637694/CONT-.008595761286; noadoption. Same92841magnitudecontrol explicitsoloPID41428 afterphaseterminal/globalGPU0; no automaticqueue/otherfolds.353PORtransferprepared135304/original151only/notlaunched. Current351optional8TOTAL80.706129723238 unchanged/strict808false/fullOOF82unmet.

V352 magnitude: SW74.172948855703, actualTOTAL80.630394027635; fresh326encoder/state/readout/ownlatent/morph plus19024 curve interaction, original104only; no trained321repairweights, oneaddedSWphase. No fullOOF/otherfolds.

V352 magnitude terminal SW74.17294885570288/CONT17.978131308398886, vs326whole0.0034058236279861376/CONT0.014689220150536642, best/stop13/27. Diagnostic8TOTAL80.70589760445243 (notadopted untilphase-magnitude pair); spectrum mixing/bridgenorms recorded/245.60118850000435s solo/1803.037109375MiB inclprep/warmup/restoration;27prefix/True/144665RAMbestreplay/frozenexact/notdiskreload. Strict808=False/fullOOF82unmet/nootherfolds.

V352 pair terminal realrho-minus-absrho whole-0.0007364411633830059/CONT-0.00732791963842061; bothwhole below351, noadoption/Both realrho/absrho current5frequency/.25descriptor below351whole; close descriptor setting/no smoothing-coefficient-LR grid. Not all phase/spectrum invalid; positive351 retained.. Pairwall664.4868988990784s/stagger414.7520000934601s includes serialgaps; all144665RAMbestreplay/frozenexact/notdiskreload/nootherfolds/fullOOF82unmet. Current351TOTAL80.706129723238 unchanged.


V353 explicit soloPID30716 after352pair terminal; snapshot {"status": "running", "pid": 30716, "stage": "aligned", "epoch": 37, "best": 78.04357779977842, "best_epoch": 28, "bad": 9, "peak_MiB": 2222.54248046875}; no final gain yet. V343 FFN dropout p.1 completed negative POR-.0015829502555/CONT-.0045519102074 vs341, not adopted. Retainedoptional8 fifthTOTAL80.706129723238; strict80.8/fullOOF82 unmet/no otherfolds.

V354 prepared finite confusing-continuous representation: [study](results/deep_learning/sw_confusing_representation_v354_20261002/README.md). Original104only/fresh351/alljoint92841params/0new/oneSWphase; joint vsdetached aux, no training yet/queue/claimedgain.

V353 terminal POR78.04357779977842/CONT41.72843647938606, best/stop28/42, diagnostic vsserial345 whole-0.0017124389843985455/CONT-0.0088997770868815, vs341whole-0.002852918153905648. Diagnostic8TOTAL80.7052738477919; adopted=False. POR351 amplitude-spectrum transfer below retained341whole; permuted not trained/no matchedgain claim, close transfer configuration no grid; SW351positive preserved. Permutedarm notlaunched/no matchedgainclaim. Solo individual523.3101248000166s/peak2222.54248046875MiB includesprep/warmup/restoration;144665RAMbestreplay/frozenexact/notdiskreload/fullOOF82unmet/nootherfolds.

V353 completed42/best28 POR78.043577800/CONT41.728436479, vs341-.002852918/-.008203813; no adoption/permutedclosednottrained/no matchedclaim. V354 joint explicitlysoloPID57064 afterterminal, original104/fresh351 jointencoderreadout92841parameters/0new; gainunproven/nootherfolds/currentTOTAL80.706129723238.

V354 failed initialevaluation hookoverride/no train/no score; corrected [V355](results/deep_learning/sw_confusing_representation_v355_20261002/README.md), originals frozen.

V355 corrected auxiliaryhook capture_conr preserves native282fuse capture/state_feature. Explicit soloPID56612, source104only/allnewencoderreadoutjoint92841/no additionalparameters/oneSWphase. Firstscorepending.

Actual355joint training verified livePID56612, epoch5/bestSW74.13839885318127; laststep-mean auxstats{"samples": 70.25, "eligible": 66.625, "positive_pairs": 2910.234375, "confusing_pairs": 781.96875, "aux": 3.9902289658784866}. Auxiliary eligible nonzero, no empty-sampler failure. ~9.3s/epoch/1835.185MiB; stillbelowretained351SW74.173612052/notfinalgain. Atomiclaststate active, no detachedlaunch/otherfolds.

V355 live interim best fromatomiclaststate (not independentdiskreload/finalgain): {"epoch": 15, "best_epoch": 13, "SW": 74.17708613558165, "strata": {"default": 98.7593258306848, "continuous": 17.713957761858413}, "diagnostic_scores": {"POR": 78.04643071793232, "PERM": 89.51838939879066, "SW": 74.17708613558165, "TOTAL": 80.70734565241}, "SW_vs351": 0.0034740833483795086, "CONT_vs351": -0.2654413881882185, "interim": true, "adopted": false, "complete_five_folds": false, "remaining_folds_started": false, "independent_disk_reload": false}. Matchedauxdetached pending explicitlaunch afterjointterminal, no currentrecipe change.

Latest user residualbackbone/predictionbranch: [V356](results/deep_learning/perm_parallel_residual_v356_20261002/README.md) prepared native154fullypreserved plusfresh22592localencoder+zerodistributionresidual readout; onePERMphase/no repairchain/GPUyet.

V355 joint terminal SW74.17708613558165/CONT17.713957761858413, vs326whole0.007543103506748139/CONT-0.24948432638993623, best/stop13/27. Diagnostic8TOTAL80.70734565241 (notadopted untilpair); spectrum mixing/bridgenorms recorded/280.2747422999819s solo/1835.18505859375MiB inclprep/warmup/restoration;27prefix/True/144665RAMbestreplay/frozenexact/notdiskreload. Strict808=False/fullOOF82unmet/nootherfolds.

V355 natural27/best13 SW74.177086136(+.003474083) butCONT17.713957762(-.265441388), DEFAULT98.759325831; diagnosticTOTAL80.707345652 notadopted because continuousgoalnotmet. Existing351TOTAL80.706129723238 remainscurrent. No detachedGPU/no matchedcausal gainclaim; close testedT.1/coeff.1over3/no grid. V356latestuserparallelresidualjointencoder/readout launchedsoloPID90652/original154only/onePERMphase/max8.

[357 backbone residual](results/deep_learning/perm_backbone_residual_v357_20261002/README.md) prepared preservingnative154enc2branch plusnewnonlinear17featurepatch andwarmreadoutjoint, 104928active/onePERMphase/no356trainedweights.

V356 natural14/best0 {"report": {"stage": "local", "target": "PERM", "target_score": 89.40556485691894, "initial_score": 89.40556485691894, "against_initial": 0.0, "against_v97_target": 0.00884080755312766, "best_epoch": 0, "stopped_epoch": 14, "validation_rows": 144665, "predictions_reproduced_exactly": true, "frozen_parameters_exact": true, "gradient_groups": {"PERM_parallel_residual_feature_encoder": 2.9405848181340843e-05, "PERM_new_distribution_residual_readout": 0.0036415986251085997}, "residual_parameter_norms": {"residual_encoder.stem.weight": 3.9845287799835205, "residual_encoder.stem.bias": 0.2519529461860657, "residual_encoder.norm.weight": 6.928203105926514, "residual_encoder.norm.bias": 0.0, "residual_encoder.depth.weight": 4.029961109161377, "residual_encoder.depth.bias": 1.0131064653396606, "residual_encoder.expand.weight": 5.741763114929199, "residual_encoder.expand.bias": 0.8430691957473755, "residual_readout.weight": 0.0, "residual_readout.bias": 0.0}, "seconds": 101.61426489998121, "peak_MiB": 786.46533203125, "complete_five_folds": false, "early_terminated": false}, "strata": {"default": 97.27870539397452, "continuous": 74.09691194381553}, "current_readout_norms": {"residual_readout.weight": 1.040154218673706, "residual_readout.bias": 0.059016741812229156}, "whole_vs154": 0.0, "whole_vs336": -0.11282454187171709, "CONT_vs336": 0.08727486061846434, "adopted": false, "point_trained": false, "independent_disk_reload": false, "complete_five_folds": false, "remaining_folds_started": false, "closed_setting": "parallel native319-to48 local17 predictor distribution residual .25; natural14/best0, no parameter grid"}. Backbone/readout gradients positive and currentreadoutnonzero; restoredzero isbest0, not gradientblocking. Pointnottrained/no matchedclaim. Close outputresidualsetting; 357newbonebranch original154only/not356weights.

V357 explicitsoloPID46896 after356terminal/globalGPU0; original154only, no trained356weights. Two22208freshparallelenc2branches andwarm82720native154readoutjoint, onePERMphase/max8. Pointnotlaunched/noqueue; initialscorepending.


## Latest user: conditional early-stopping patience
All future NEW training: max80/min10, patience10 until strictcurrentSOTA is broken, then latchpatience14 for rest ofsame run. OfficialwholeTOTAL isSOTAcriterion; for an independenttarget with otherretainedheads fixed, use mathematically equivalent targetthreshold (POR34178.04643071793232/PERM33689.51838939879066/SW35174.17361205223327; assembledTOTAL80.70612972323806). Freeze reference atlaunch, do not useCONTbest/oracle. Active357 frozen trainer/config remainshistoricalpat14; external existingepoch-boundary stoprequest enforces10untilPERMSOTA then14, no restart. Sourceimmutability/explicitresume/strictfifth80.8beforeotherfolds/max2/<=8phases unchanged.

V357 terminal13/best0 {"report": {"stage": "local", "target": "PERM", "target_score": 89.40556485691894, "initial_score": 89.40556485691894, "against_initial": 0.0, "against_v97_target": 0.00884080755312766, "best_epoch": 0, "stopped_epoch": 13, "validation_rows": 144665, "predictions_reproduced_exactly": true, "frozen_parameters_exact": true, "gradient_groups": {"PERM_preserved_warm154_readout": 0.46561193466186523, "PERM_parallel_enc2_feature_residual": 0.002559244865551591}, "residual_parameter_norms": {"residual_branches.0.down.weight": 3.2706642150878906, "residual_branches.0.down.bias": 0.30207735300064087, "residual_branches.0.norm.weight": 5.656854152679443, "residual_branches.0.norm.bias": 0.0, "residual_branches.0.depth.weight": 3.183403253555298, "residual_branches.0.depth.bias": 0.6948413252830505, "residual_branches.0.expand.weight": 4.552340984344482, "residual_branches.0.expand.bias": 0.8035740852355957, "residual_branches.0.up.weight": 0.0, "residual_branches.0.up.bias": 0.0, "residual_branches.1.down.weight": 3.293905019760132, "residual_branches.1.down.bias": 0.28711530566215515, "residual_branches.1.norm.weight": 5.656854152679443, "residual_branches.1.norm.bias": 0.0, "residual_branches.1.depth.weight": 3.17657732963562, "residual_branches.1.depth.bias": 0.7844030261039734, "residual_branches.1.expand.weight": 4.648634910583496, "residual_branches.1.expand.bias": 0.7975549101829529, "residual_branches.1.up.weight": 0.0, "residual_branches.1.up.bias": 0.0}, "seconds": 111.41831959999399, "peak_MiB": 854.9150390625, "complete_five_folds": false, "early_terminated": true}, "strata": {"default": 97.27870539397452, "continuous": 74.09691194381553}, "current_exit_norms": {"residual_branches.0.up.weight": 1.7500327825546265, "residual_branches.0.up.bias": 0.2155779004096985, "residual_branches.1.up.weight": 1.8259357213974, "residual_branches.1.up.bias": 0.22815413773059845}, "adopted": false, "point_trained": false, "independent_disk_reload": false, "complete_five_folds": false, "remaining_folds_started": false, "patience_control": {"guard_pid": 63112, "training_pid": 46896, "training_live": true, "threshold_target_SOTA": 89.51838939879066, "threshold_basis": "PERM336 with fixedPOR341/SW351, equivalent to retainedassembledTOTAL80.706129723238", "strict": true, "SOTA_broken": false, "effective_patience": 10, "snapshot": {"status": "running", "pid": 46896, "stage": "local", "epoch": 12, "best": 89.40556485691894, "best_epoch": 0, "bad": 12, "peak_MiB": 854.9150390625}, "reason": "Latestuser patience10 untilstrictSOTA, then14; existing epoch-boundary request/bestrestoration", "stop_request_written": true}, "actual_stop_note": "Userpolicy arrived after bad10; guardobserved bad12, existingepoch-boundary stop restoredbest after13, not claim10epoch run", "closed_setting": "native154 enc2 nonlinear32width17tap parallelfeaturebranch pluswarmreadout no gain; no widthkernelLRgrid"}. Bothgradientgroupspositive and currentzeroinitialexitnormsnonzero; restoredzero isbest0 notgradientblock. Latestuserpatience10untilstrictSOTA thenlatch14 recorded; activelegacyguardobservedbad12/stopped13 because instructionarrivedafter10, not10epoch claim. No pointGPU/no matchedgainclaim; existingbest80.706129723238/deliveries preserved.

[V358 localpath residual](results/deep_learning/perm_path_residual_v358_20261002/README.md) preparedfreshpositive336code/original154prior/39162jointparams(+1776)/onePERMphase, area vsvariation. Newnativepatience10untilstrictcurrentSOTAthen14 implemented, noGPUyet.

V358 explicitarea soloPID63544 after357terminal/globalGPU0. Newnativepat10 beforestrictPERM336SOTA then14, referencefixed89.51838939879066. Variationpreparednotlaunched/noqueue; firstscorepending/unproven. Research: results/deep_learning/cross_domain_research_20261001/PERM_LOCAL_PATH_AREA_RESEARCH_20261002.md; implementedfixedtanh/16/full17valid differs researchTVnorm(notrun), no newgrid.

Actual358 area training verified livePID63544: {"status": "running", "pid": 63544, "stage": "area", "epoch": 13, "best": 89.51746356358993, "best_epoch": 5, "bad": 8, "effective_patience": 10, "SOTA_broken": false, "peak_MiB": 903.2548828125}. Nativeinitial154ordinaryassert passed/39162active/1776newpath. Realnewpat10/14cfg andruntimeeffectivepatience verified. ~7.5s/epoch/903.255MiB. Stillnotfinalgain/currentrecipeunchanged; no variationlaunch/otherfolds. Researchfinalpath ../cross_domain_research_20261001/PERM_LOCAL_PATH_AREA_RESEARCH_20261002.md synchronized fixedtanh/16/fullvalid variant, noTVtrial.

V358 area terminal PERM89.51746356358993/CONT74.01331830014448, vs336whole-0.0009258352007321946/CONT0.0036812169474131906, best/stop5/15, effectivepatience10/SOTAbrokenFalse. Actualdiagnostic8TOTAL80.7058056809178; notadopted beforepaireddecision. 120.20548730000155s solo/903.2548828125MiB inclprep/warmup/restoration/144665RAMbestreplay/frozenexact/notdiskreload. Strict808false/fullOOF82unmet/nootherfolds.

V358 variation explicit soloPID67172 afterarea terminal/globalGPU0. Same39162physicalparams/init/newpat10untilstrictSOTAthen14. Area whole-.000925835/CONT+.003681217 vshistorical336; matchedorderbenefit unproven untilcontrol, noadoption/recipechange/otherfolds.

V358 variation terminal PERM89.51701907785088/CONT74.01187788532252, vs336whole-0.0013703209397846194/CONT0.002240802125456298, best/stop5/15, effectivepatience10/SOTAbrokenFalse. Actualdiagnostic8TOTAL80.70565011090915; notadopted beforepaireddecision. 112.8456646000268s solo/903.9736328125MiB inclprep/warmup/restoration/144665RAMbestreplay/frozenexact/notdiskreload. Strict808false/fullOOF82unmet/nootherfolds.

V358 paired terminal area-minusvariation whole0.00044448573905242483/CONT0.0014404148219568924/DEFAULT-6.771410521366761e-05; mechanismpositive=True; noadoption/bothbelowretained336whole/currentTOTAL80.706129723238 unchanged. Individual120.205/112.846s serial/peak903.255/903.974MiB inclprep/warmup/restoration, pairwall322.1232771873474s inclserialgaps/stagger203.84299993515015. Both below retained336 whole; noadoption. Same39162params/init/pat10to14, actualbest5/stop15, descriptor scales/functions differ. Single-seed microdelta not robustness; preserve positive code lesson only if both whole/CONT pairedpositive; no window/LR/normalization grid.

[V359 learned depth residual](results/deep_learning/perm_depth_path_residual_v359_20261002/README.md) preparedpositive336+358code/all45274encoderreadoutjoint, original154only/+6112zeroexittemporalbranch/onePERMphase/no trainedrepairweights. Pat10strictSOTAthen14.

V359 depth explicitlysoloPID78800 after358pair terminal/globalGPU0. Original154only/fresh336358positivecode+6112temporalbranch(all45274encoderreadoutjoint)/onePERMphase. Newnativepat10 untilstrictSOTAthen14. Controlpreparednotlaunched/noautoqueue. Gainunproven/nootherfolds/currentTOTAL80.706129723238unchanged.

Actual359depth verified livePID78800: {"status": "running", "pid": 78800, "stage": "depth", "epoch": 9, "best": 89.50038441492794, "best_epoch": 5, "bad": 4, "effective_patience": 10, "SOTA_broken": false, "peak_MiB": 916.0927734375}. All45274encoder/readout/path/temporal initializedoriginal154+freshcode only; ordinaryinitialassert passed. ~7.2s/epoch/916.093MiB. Newpat10/14effective10, no SOTAbreak/finalgain/currentrecipechange/controlqueue/otherfolds.

V359 depth terminal PERM89.50038441492794/CONT74.01426148184036, vs336whole-0.01800498386272409/CONT0.00462439864330122, best/stop5/15, effectivepatience10/SOTAbrokenFalse. Actualdiagnostic8TOTAL80.69982797888612; notadopted beforepaireddecision. 131.43910970003344s solo/916.0927734375MiB inclprep/warmup/restoration/144665RAMbestreplay/frozenexact/notdiskreload. Strict808false/fullOOF82unmet/nootherfolds.

[V360 SWstate range](results/deep_learning/sw_state_range_v360_20261002/README.md) preparedfinitebound8vsretained351bound4, positive2892to4evidence/full92841joint/0newparams/oneSWphase/original104only. Pat10strictSOTAthen14/noGPUyet/nogrid.

V359 closed: depth best5/stop15, PERM89.50038441492794 vs336 -0.0180049838627241, CONT74.01426148184036 (+0.00462439864330122). Actual TOTAL80.69982797888612 below retained80.70612972323806. Control closed untrained/no matched mechanism claim; no width/range/LR grid. Small CONT gain does not offset whole score loss; not all residual networks invalid. No other folds.

V360 bound8 explicitly launched solo PID23528 after359terminal/global GPU0; ordinary initial evaluation and training actual running. Snapshot epoch16/best13 SW74.17170342746022, below35174.17361205223327; effective patience10/SOTA false. Last-batch raw patch >4 fraction0 is a snapshot, not population saturation proof. No adoption/otherfolds.

V361 prepared: joint fresh336 PERM encoder/readout with finite channel covariance auxiliary .01/3 vs0, zero new parameters/37386joint/one PERM phase/max8. Untrained/no score/no queue; original154 only/current recipe unchanged. See results/deep_learning/perm_joint_decov_v361_20261002/README.md.

V360 terminal best13/stop23/pat10/SOTAnotbroken. SW74.17170342746022 vs351 -0.001908624773051315, CONT17.96807878588985 vs351 -0.011320364156780727. Actual TOTAL80.7054617045675; notadopted, close8/no rangeLRgrid. 252.23016620002454s solo/1800.099609375MiB includingprep/warmup/restoration; RAMbest144665/frozenexact/notdiskreload. Common prefix23 epochs: new74.17170342746022/old74.17361205223327. Raw patch snapshots {"last_batch_max_over_observed_epochs": 1.3359375, "last_batch_gt4_fraction_max_over_observed_epochs": 0.0, "limits": "last minibatch/epoch snapshots only; not whole population/saturation proof"}. Current80.70612972323806 unchanged/strict808false/fullOOF82unmet/no otherfolds.

V361 decorrelation explicitsolo launch PID604 after360terminal/globalGPU0; identity preparednotlaunched/noqueue. Sourcecompiled/ordinaryinitialevaluation and training pending actual readback; no gain yet.

V361 decorrelation terminal PERM89.51434795830305/CONT73.997176916398, vs336whole-0.0040414404876116805/CONT-0.012460166799058925, best/stop5/15, effectivepatience10/SOTAbrokenFalse. Actualdiagnostic8TOTAL80.70471521906741; notadopted beforepaireddecision. 116.3372768999543s solo/897.4375MiB inclprep/warmup/restoration/144665RAMbestreplay/frozenexact/notdiskreload. Strict808false/fullOOF82unmet/nootherfolds.

User switches to five complete seeds of80.706130 eight-stage recipe. V361 decorrelation terminal; identity remains untrained/held, no paired gain or broad negative claim. Preserved results/currentrecipe.

Latest user explicitly authorizes five COMPLETE models of80.706130 eight-stage recipe: seeds20261010..20261014, newSOURCE and all7downstream stages perseed, fixedoriginal fifth, no otherfolds. Finite serial8x5 dependency batch authorized/no unrelatedqueue/no retry/autoresume, max2. Latest80/10->14 budget common to allnewseeds; historical budget differs. Eachseed own train-onlyQ/weights, no cross-seed head splice. Study results/deep_learning/complete_seed_compare_v362_20261002; old best/deliveries preserved.

V362 actual batch launched controllerPID56316/firstSOURCEPID24596, seed20261010, globalGPU0 beforelaunch; live admissionfree10.804GiB. Source train-only preprocessing actualstarted, not yet a completedmodel/no seedscore. Finite serial5x8 authorized; no auto resume/retry/otherfolds. Complete workers+CLI import compile verified.

V362 actualSOURCE initial readback: seed20261010/originalfold5/64train16heldout/features206/max80/pat10->14. Latest sourceepoch6, sourcebest77.6419287354647; prerequisite only, notcomplete model score. Prepared downstreamworker now persists actual guard-resolved checkpoint path/SHA trace in config; not yet launched, source/controller and historicaltrained files unchanged.

V362 monitored actualliveSOURCEPID24596/controller56316; snapshotepoch23/sourcebest79.17320054732483/bestepoch22/bad1/pat10. Completed wholemodels0/5; pending all7downstream phases of firstseed and4newseedmodels. Guard-resolved dependency trace now included in unlaunched stageworker config. Source/controller unchanged; no SOTA/strict808/fullOOF claim. Goal remainsfullOOF>82 afterqualified fifth, not source-level score.

V362 actual live check: SOURCE seed20261010 PID24596, epoch45, best79.90066192223925 at44, bad1/pat10. Still0/5 COMPLETE_MODEL files; source-only score cannot rank complete models. Current80.706129723238 retained/no otherfolds.

V362 seed20261010 SOURCE terminal: best52/stop62/POR77.09029723727674/PERM89.35243145677488/SW73.65984026430232/TOTAL80.18138427356004 (prerequisite only, not full8stage model). 1299.184347100032s training/prep/restoration, processwall1310.2443144321442s solo/peak815.75390625MiB. Best predictions actualreproduced/all144665once. OwnsourceSHA7ce54c5a19dcdd61ef89ce21b2aaac621ee171e51ad7250059addc95f6230556; ownQ50/1736 downweighted/f75294cae5747e11a7f72dbb2879b4fcc9d394a8b8a5a587bf25cba02c639e4a. POR113 actualPID77552 launched; cfg actualweighttrace onlysame seed. Complete models0/5; source/controller frozen; no otherfolds/no SOTA/fullOOF claim.

V362 firstseed completes2/8 phases. POR113 best18/stop28/POR77.42544665121915 vsowninitial+0.3349088674011682;181.394s solo920.848MiB/RAMbest144665/frozenexact, notindependentdiskreload/fullmodelscore. POR151 actualPID39756/cfg reads ownnewSOURCE and ownPOR113 checkpoint, parentSHA58ed13b4f8d60b8d7e1e26a16aaca8c4fac3c09db315a836f61b78aa05b54861. No crossseed/historicalweights; old recipes unchanged. Complete models0/5, remaining6phases thenother4seeds, no otherfolds.

V362 firstseed3/8 phases terminal. POR151 best0/stop10/POR77.42544665121915, deltaowninitial0; nonzero newheadgrad.2969729/104.960s solo924.547MiB/RAMbest144665/frozenexact. Valid no-gain stage/current10patience, not general151negative/no skip or historicalweight replacement. PERM154 actualPID18924/cfg ownSOURCEonly. Wholemodels0/5; completeall8 thenrank/not crossseedheadsplice/no otherfolds; old80.706129723238 preserved.

V362 firstseed4/8 phases terminal. PERM154 best0/stop10/PERM89.35243145677488, deltaowninitial0; actualgrad.2127831/87.182s solo786.465MiB/RAMbest144665/frozenexact. Zero-gain under current10patience/newSOURCE; not general154negative. SW104 actualPID79660, random845913body parameters/SAM.1/utility1/fromowntrain-onlymetadata andownQ50 only, no historical104 neuralweights. Wholemodels0/5; later341336351repairs still pending, no otherfolds/oldrecipechanges.

V362 firstseed5/8 phases complete. SW104 best34/stop44/SW74.03612875776867,638.196s solo1722.197MiB/RAMbest144665/frozenexact. Actualsame-seed five-stage prefix TOTAL80.41363007045598 vsoldfive -0.19884534753795435; notfinal8stage score/notseedranking. POR341 actualPID14768/joint134752freshencoderattentiontemporalreadout; cfg trace ownSOURCE/POR113/POR151 only/no old341weights. Three finalrepairs pending/wholemodels0/5/no otherfolds/current80.706129723238 unchanged.

V362 seed20261010 completed POR341: target77.5520561792274, best6/stop16, owninitialdelta0.12660952800825953; 248.978s solo/2212.483MiB. RAMbest144665/frozenexact, not independent diskreload. 6/8 stages; next observed 20261010/PERM336/PID13648. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 isolated manual interface correction: SW351 failed at epoch0 before first optimizer update; missing model argument now matches original351 trainer. Failed source/log/config/laststate and all seven completed stages preserved. Corrected r1 worker/model binding/controller use SW351_fixed directory. This is replacement of untrained failed phase, not ninth learned stage; no extra seed or old-stage rerun. Finite remaining1+4x8 dependency launches only, no automatic retries/stop resumption. Full elapsed time includes failed attempt and repair gap.

V362 seed20261010 completed PERM336: target89.49625487509155, best5/stop15, owninitialdelta0.14382341831664291; 157.014s solo/896.344MiB. RAMbest144665/frozenexact, not independent diskreload. 7/8 stages; next observed 20261010/SW351/PID47572. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 complete seed 20261010: {"POR": 77.5520561792274, "PERM": 89.49625487509155, "SW": 74.03612889675152, "TOTAL": 80.5019511739133}. Completed 1/5; complete8stage source retrained. Ranking on fixed fifth only, no other folds or independent-generalization claim.

V362 seed20261010 completed SW351: target74.03612889675152, best0/stop10, owninitialdelta0.0; 111.105s solo/1800.100MiB. RAMbest144665/frozenexact, not independent diskreload. 8/8 stages; next observed 20261011/SOURCE/PID32596. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 first complete model seed20261010 official fixed-fifth TOTAL80.5019511739133, vs historical80.70612972323806 -0.20417854932476. Same-seed five-stage prefix80.41363007045598; added POR341/PERM336/SW351 phases +0.08832110345731792. CONT same whole-target checkpoint: POR39.98306844175757/PERM73.73543850514635/SW16.966455441053746; all below historical CONT references. Full serial model wall3098.6025s (51.6434min), includes preprocessing/warmup/training/restoration/failed interface attempt/repair gap/CPUassembly; detailed CPU_DIAGNOSTIC.json separates successfulphase sum and failedattempt. Not independent diskreload/fullOOF/seed robustness; historical budget differs. Completed1/5, second SOURCE20261011 actualPID32596; remaining4 complete models pending. No otherfolds/historicalreplacement.

Terminal fixed1:1 distribution mixture: {"POR": 77.93964020763568, "PERM": 89.56537181009944, "SW": 74.17086138314515, "TOTAL": 80.68957367992631}; vs original -0.016556043312, vsnew +0.187622506013. Both models independently reloaded/native predictions exact on all144665 rows. CONT={"POR": 41.22282838987248, "PERM": 74.01305415036849, "SW": 17.52288214309329}. Channel evaluations sum37.560s/peak1362.866MiB; sequential channels concurrently with existing SOURCE32596, no training interruption/max2. No weight grid or followup launch; original retained/notadopted. Two models8stages each/16historical phases/noaddedtraining. Five-seed study continues; no otherfolds/fullOOF claim. Final artifact FINAL.json.

Terminal101 common weights: best original0.81/new0.19, TOTAL80.724892908977/originaldelta+0.018763185739. Coarse0.05 best0.85/80.723424811469; native0/1/0.5 andsame144665coverage exact. CONT samegridselectedprediction {"POR": 41.64952908696266, "PERM": 74.01991053428122, "SW": 17.84395496760571}. Channel evaluations sum46.256s/peak1362.866MiB, studywall121.382s inclprep/warmup/grid/CPUassembly/implementation/gaps/plot; serial targets concurrent with existing SOURCE32596/max2, notsolo benchmark. Same-fifth validation-tuned maximum within0.01grid only; no finergrid/otherfolds/newtraining/adoption as single8stage model. Two models8each/16 prior phases. Full table WEIGHT_SCORES.csv, curve weight_curve.png, FINAL.json. Five-seed training continues.

Terminal userauthorized per-head fixed ratios {"POR": {"original_weight": 0.93, "new_seed_weight": 0.06999999999999995, "score": 78.06314784005319, "selected_grid_index": 93}, "PERM": {"original_weight": 0.49, "new_seed_weight": 0.51, "score": 89.56546548645483, "selected_grid_index": 49}, "SW": {"original_weight": 0.71, "new_seed_weight": 0.29000000000000004, "score": 74.1951684727142, "selected_grid_index": 71}}. ActualTOTAL80.735166237725/originaldelta+0.029036514487/common81:19delta+0.010273328748. CONTsame selectedratios {"POR": 41.733030179719194, "PERM": 74.01018085296752, "SW": 17.72871820182738}; CONTdeltaoriginal {"POR": -0.003610112275950428, "PERM": 0.0005437697704593347, "SW": -0.2506809482192516}. CPU0.742s/nativeverifiedcachedgrids/noextraGPU/training. Same-fifth tuned maximum0.01grid/fixedperchannel ratios, notroworacle/CONToracle/generalization/OOF. Two8stage models/16prior phases/noadoption as one8stage model/nootherfolds. Existingfive-seedtraining continues.

V362 seed20261011 SOURCE prerequisite completed: scores{"POR": 75.83717325309883, "PERM": 89.27064029934503, "SW": 73.37300880368564, "TOTAL": 79.67642916199037}, best42/stop52, 1040.496s solo/815.754MiB, processwall1051.897s, bestRAMreplay/allvalidationonce. Not complete8stage score. Next observed 20261011/POR151/PID83852; own SOURCE weights/Q only, no historical/otherseed learnedweights. Five-seed batch pending/nootherfolds/noindependentdiskreload claim.

V362 seed20261011 completed POR113: target75.9509669518149, best2/stop12, owninitialdelta0.02771350351585511; 77.638s solo/920.848MiB. RAMbest144665/frozenexact, not independent diskreload. 2/8 stages; next observed 20261011/POR151/PID83852. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 timing clarification seed20261011SOURCE PID32596: sole training task, but V363/V364 read-only native evaluations overlapped (max2totalGPU). Original controller TIMING solo is training-count shorthand, not isolatedGPU timing; SOURCEprocess1051.897459s includes this overlap. SEED20261011_SOURCE_CONCURRENCY_ADDENDUM.json supplies actual admission evidence without modifying frozen timing/checkpoints.
V362 firstseed gain-source CPU decomposition: three final phases +0.088321103457TOTAL; POR default contributes+0.062627449625 but CONT-0.024644591223, PERM default+0.036476135335/CONT+0.013862061076, SW effectively unchanged. Thus firstseedPOR341wholegain does not preserve CONT; do not generalize the historical341 mechanism or skip phases from this single run. Weight81:19 +0.018763185739 similarly combines default+0.041008768152 and CONT-0.022245582413. Evidence PHASE_GAIN_SOURCES.json and WEIGHT81_19_GAIN_SOURCES.json; no training change.

V366 original/new firstseed stage gain and leaf-deletion analysis complete, CPU only. OriginalSW351 least damaging tail removal; firstnewSW351 score-neutral within1.4e-7 target/lastbest0. Original151/154 positive vs newfirstbest0; second151alreadypositive, no universal pruning. Original V201 directskip113 lower0.511035POR. Intermediate151/154 removal needs untrained-state equivalence or fresh downstream training, cannot retain learned descendants and eraseancestor cost. Recommended optional next mergePERM154+336 fromSOURCE, untested; original8/newfullseed batch unchanged. Evidence results/deep_learning/stage_gain_analysis_v366_20261002/FINAL.json.

V362 seed20261011 completed POR151: target76.59083162289573, best22/stop32, owninitialdelta0.6398646710808293; 284.396s solo/924.547MiB. RAMbest144665/frozenexact, not independent diskreload. 3/8 stages; next observed 20261011/SW104/PID87280. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261011 completed PERM154: target89.27064029934503, best0/stop10, owninitialdelta0.0; 79.000s solo/786.465MiB. RAMbest144665/frozenexact, not independent diskreload. 4/8 stages; next observed 20261011/SW104/PID87280. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261011 completed SW104: target73.74686202568397, best27/stop37, owninitialdelta2.549746944212302; 505.855s solo/1722.197MiB. RAMbest144665/frozenexact, not independent diskreload. 5/8 stages; next observed 20261011/POR341/PID82544. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

2026-10-02 stage-gain analysis readback completed: original8=80.706129723238/newseed20261010=80.501951173913; new SW351 leaf deletion effectively neutral, original SW351 deletion costsTOTAL0.018150728426 andSWCONT0.347510089567. Secondseed151 gain0.639864671081 rules out universal151 pruning. Added CPU arithmetic gap decomposition; no new training/weight deletion or recipe change. Five-seed batch remains active; secondseed5/8, POR341 actualPID82544 at readback. See results/deep_learning/stage_gain_analysis_v366_20261002/README.md.

2026-10-02 fixed explicit curated-node indexing for nested5seed roots/interfacefailure and363..366 result studies. Opt-in virtual records require existing evidence files; only completed complete-model scores get fifth-fold primary, weight search marked diagnostic, partial seeds unscored. Rebuilt curation/data/offlineHTML; 10 new nodes present and scoped real Edge offline layout verified finite/no backward edges/no pageerrors (v362_v366_index_check.json). Active training/source/hash/config unchanged. Existing unrelated unknown-category records were not reaudited in this scoped change.

V362 seed20261011 completed POR341: target76.72694269202935, best6/stop16, owninitialdelta0.13611106913361937; 207.560s solo/2212.483MiB. RAMbest144665/frozenexact, not independent diskreload. 6/8 stages; next observed 20261011/PERM336/PID73068. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

2026-10-02 V366 secondseed20261011 completed6stage CPU artifact addendum:151 wholePOR+0.639864671081/CONT+2.233561712687;341 wholePOR+0.136111069134 butCONT-0.304315261664, matching default-driven341 behavior in original andfirstseed.154 secondbest0/nativepredidenticalSOURCE. Actual144665 prefix only, notcomplete-model ranking; nextPERM336 confirmedlivePID73068. No new GPU task/recipe/source changes. Evidence stage_gain_analysis_v366_20261002/SEED20261011_STAGE_SNAPSHOT.json.

V362 seed20261011 completed PERM336: target89.38299330747085, best6/stop16, owninitialdelta0.11236178586572976; 130.898s solo/896.344MiB. RAMbest144665/frozenexact, not independent diskreload. 7/8 stages; next observed 20261011/SW351/PID87196. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 complete seed 20261011: {"POR": 76.72694269202935, "PERM": 89.38299330747085, "SW": 73.74686202568397, "TOTAL": 80.11353217421299}. Completed 2/5; complete8stage source retrained. Ranking on fixed fifth only, no other folds or independent-generalization claim.

V362 seed20261011 completed SW351: target73.74686202568397, best0/stop10, owninitialdelta0.0; 104.440s solo/1800.100MiB. RAMbest144665/frozenexact, not independent diskreload. 8/8 stages; next observed 20261012/SOURCE/PID79292. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

2026-10-02 complete seed20261011 eighth phase finished:TOTAL80.113532174213 vsoriginal-0.592597549025; sourcegap-0.545431921049/downstreamgap-0.047165627976 (see actual decomposition JSON). Own5prefix80.033375300629/addedthree+0.080156873584. SW351 best0/predictions identical104, valid7phase tail deletion same score. CPU native144665 report/population/strata evidence saved; no independentGPU modeldiskreload claim.2/5 modelscomplete/seed20261012 SOURCE actualPID79292 continues. No otherfolds/old source or recipe changes; originalOOF82 unmet.

2026-10-02 V366 SOURCE saved-learning-curve analysis added (CPU logs only, no model replay): first20261010 shares originalSOURCE actualfoldseed, agreesepochs1/2 then divergesepoch3; bothbest52, so its observed deficit is not a missingbest caused by62vs70 stopping. SecondSOURCEearlydifferent/best42, poststop52unobserved; no patiencecausalclaim. Third20261012 snapshot10epochs/best78.121890688. Source seed configbase+4 confirmed; same-seed nondeterminism warnings retained. Five-seed batch unchanged/activeSOURCEPID79292. Evidence stage_gain_analysis_v366_20261002/SOURCE_CURVE_ANALYSIS.json andplot.

2026-10-02 verified live wait: controller74228 and seed20261012/SOURCE actualPID79292 confirmedcmdline/create_time, currentSOURCEepoch21/bestepoch20/bestTOTAL79.267685402423; same-observed-horizon originalSOURCEbest79.017542279546, diagnosticdelta+0.250143122877. Snapshot only/notcomplete8stage/notOOF. Complete models2/5; no restart/new GPU/recipechange. Live observation includes process evidence anddiskfree; existing source continues.

2026-10-02 V366 CPU best0 initializer equivalence proved for seed20261010 PERM154 (323tensors), seed20261011 PERM154 (323), seed20261010 POR151 (335): ownSOURCE/POR113 dependencies only beforecomparison, reconstructed andsavedstate hashesexact/noGPU/forward/optimizer. Future shorter learned-phase candidates5(firstseed)/6(secondseed), untrained; historical actual8stage count unchanged/no model orrecipe deletion. Originalpositive151/154 cannot universallyprune. Summary INITIALIZER_EQUIVALENCE_SUMMARY.json; existing reportedphasecompute sums303.247s/183.440s are work estimates, not new measuredpipeline walltime. ThirdSOURCE stillactive/5seedbatch unchanged.

V362 seed20261012 SOURCE prerequisite completed: scores{"POR": 76.20128636423783, "PERM": 89.22126653579818, "SW": 73.63502228726165, "TOTAL": 79.86008699734228}, best38/stop48, 950.112s solo/815.754MiB, processwall958.651s, bestRAMreplay/allvalidationonce. Not complete8stage score. Next observed 20261012/POR113/PID87528; own SOURCE weights/Q only, no historical/otherseed learnedweights. Five-seed batch pending/nootherfolds/noindependentdiskreload claim.

2026-10-02 thirdSOURCE20261012 terminal best38/stop48/TOTAL79.860086997342 vsoriginalSOURCE-0.361774085697; earlycurve advantage notfinalSOURCEgain. Native bestRAMpredictions/all144665 validated in trainer, PID79292 exited/TIMINGexit0; process958.651s/report950.112s/815.754MiB solo. POR113 continuationPID87528 actualcfg loads ownSOURCE/ownprepared_data only. Sources3completed butwhole8stagemodels2/5; nootherfolds/restart/extraexperiment. Curveplot/analysis updated, historical/interim snapshots preserved in README.

V362 seed20261012 completed POR113: target76.53985411899569, best6/stop16, owninitialdelta0.31470845350737875; 99.114s solo/920.848MiB. RAMbest144665/frozenexact, not independent diskreload. 2/8 stages; next observed 20261012/POR151/PID17960. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261012 completed POR151: target76.59180097270605, best6/stop16, owninitialdelta0.051946853710362006; 136.823s solo/924.547MiB. RAMbest144665/frozenexact, not independent diskreload. 3/8 stages; next observed 20261012/PERM154/PID63116. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261012 completed PERM154: target89.29139192003413, best8/stop18, owninitialdelta0.0701253842359506; 127.224s solo/786.465MiB. RAMbest144665/frozenexact, not independent diskreload. 4/8 stages; next observed 20261012/SW104/PID53552. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

2026-10-02 V366 thirdseed20261012 completed4/8 CPU native stage analysis: POR151+0.051946853710/CONT+0.266655235588; PERM154+0.070125384236/CONT+0.154125062520/DEFAULT+0.026924896515. Firsttwo154best0 initializer-equivalence is seed-specific, thirdpositive forbids universalpruning. Added STAGE_GAIN_COMPARISON.md original/three seeds withunfinished explicit; CPU prefix80.001785264365 notcompletemodel/ranking/qualification. SW104 actualPID53552 continues, noextraGPUtasks/otherfolds.

V362 seed20261012 completed SW104: target73.84783722739398, best27/stop37, owninitialdelta2.650722145922316; 502.024s solo/1722.197MiB. RAMbest144665/frozenexact, not independent diskreload. 5/8 stages; next observed 20261012/POR341/PID41656. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

2026-10-02 thirdseed20261012 SW104 terminal best27/stop37/SW73.847837227394, actualSOURCEgain0.212814940132; DEFAULT+0.418793000238/CONT-0.260297591277. Native144665 fiveprefixTOTAL80.076270493412, notcomplete8stage. Phase502.024s/process509.497s/1722.197MiB solo; POR341 actual41656 continues. Added bounded CPU observer (observe_batch.py; neverstart/stop/resume) verified live andhandoff; no extraGPU. Held continuousauxreadout mechanism proposal only, noimplementation/training/scores orqueuedlaunch; four evidence fields andEXPLORATIONS/atlas record. Five-seed batch unchanged, wholemodels2/5/nootherfolds.

V362 seed20261012 completed POR341: target76.81520301795747, best18/stop28, owninitialdelta0.2234020452514187; 351.846s solo/2212.483MiB. RAMbest144665/frozenexact, not independent diskreload. 6/8 stages; next observed 20261012/SW351/PID89048. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261012 completed PERM336: target89.46897371901133, best7/stop17, owninitialdelta0.1775861878471261; 148.410s solo/896.344MiB. RAMbest144665/frozenexact, not independent diskreload. 7/8 stages; next observed 20261012/SW351/PID89048. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 complete seed 20261012: {"POR": 76.81520301795747, "PERM": 89.46897371901133, "SW": 73.84783722739398, "TOTAL": 80.2054447366291}. Completed 3/5; complete8stage source retrained. Ranking on fixed fifth only, no other folds or independent-generalization claim.

V366 stage-removal decision refreshed using third seed terminal POR341 (+0.223402) and PERM336 (+0.177582), original144665 saved-pred CPU readback. Third prefix7/8 TOTAL80.205444736629; SW351 pending, only2 complete models. Prioritize fixed SW351 leaf ablation, retain positive POR113/151/341 and PERM336; propose SOURCE-direct154/336 merge only, unimplemented/untrained. No universal lossless removal verified; current finite five-seed batch unchanged. Evidence: results/deep_learning/stage_gain_analysis_v366_20261002/STAGE_REMOVAL_DECISION.md.

V362 seed20261012 completed SW351: target73.84783722739398, best0/stop10, owninitialdelta0.0; 108.352s solo/1800.100MiB. RAMbest144665/frozenexact, not independent diskreload. 8/8 stages; next observed 20261013/SOURCE/PID86204. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V366 terminal update: third seed20261012 all8 complete TOTAL80.205444736629; SW351 best0/stop10 and predictions byte-equal to SW104, so this leaf removal preserves saved prediction score. Full model wall2487.882s serial soloGPU (prep/warmup/restoration/CPU assembly included), SW351 process115.726s/report108.352s/peak1800.100MiB. Three completed models of5; seed20261013 SOURCE actualPID86204 active. Decision table updated; no new compressed training or other folds.

V367 prepared, not launched: fresh49parameter training-only continuous auxiliary readout on retained POR151/fresh341 code, matchedaux0/aux01 coefficient0/.1. Encoder/main readout jointly trainable; auxiliary never enters inference. New model/trainer syntax compiled only, no forward/GPU preflight/training/checkpoint load; gain unproven. Atomic resume and80/pat10→14 maintained, overall8stage replacement (no trained341 warmstart). PLAN launch_enabled=false plus entry requires finite five-seed batch terminal complete. Current fourthSOURCE actualPID86204 keeps running, no unrelated GPU route queued. Evidence results/deep_learning/por_continuous_aux_v367_20261002.

Verified wait: actual controllerPID74228/sourcePID86204 live with matching cmdline/create_time, seed20261013 SOURCE epoch28, best79.388158620347@27, bad1/pat10. Three complete models of5; this is SOURCE snapshot, not full-model gain. V367 implementation prepared/held with syntax only, atlas unscored preparation readback passed, no extra GPU launch.

V367 terminal CPU recorder prepared/syntax compiled, not executed: fixed original336/351 with checksum/report/prediction checks, all144665 actual columns, DEFAULT/CONT, same-capacity configuration match after excluding only auxiliary coefficient/stage, cached immutable outputs, no independent model disk reload or complete-recipe speed claim. Unlike original341 recorder, this uses8stage fixed heads, not6stage154/104. No GPU preflight or training launch.

Verified continued training: seed20261013/SOURCE actualPID86204, epoch48, best79.79912095227387, bestepoch40, bad8; confirmed actual controller and child cmdlines/create_time. Three complete seeds of5, no restart/aux launch. Prepared V367 terminal CPU assembly now distinguishes original336/351 eight-stage heads, same-capacity coefficient-only config comparison, and default/continuous strata; syntax only/no executed score. Goal strictfifth80.8/fullOOF82 still unmet.

V362 seed20261013 SOURCE prerequisite completed: scores{"POR": 75.9912319074892, "PERM": 89.28218349252357, "SW": 73.57996330755388, "TOTAL": 79.79912095227387}, best40/stop50, 887.663s solo/815.754MiB, processwall896.496s, bestRAMreplay/allvalidationonce. Not complete8stage score. Next observed 20261013/POR113/PID49676; own SOURCE weights/Q only, no historical/otherseed learnedweights. Five-seed batch pending/nootherfolds/noindependentdiskreload claim.

V362 seed20261013 completed POR113: target76.57342888482353, best2/stop12, owninitialdelta0.5734083660107814; 67.806s solo/920.848MiB. RAMbest144665/frozenexact, not independent diskreload. 2/8 stages; next observed 20261013/POR151/PID37384. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V366 quality provenance clarification: all4newSOURCEs have own train-only prior-best Q snapshots50/40/36/38, downweighted1736/1394/1233/1459 vs historicalQ50/1755. Source records every2epochs (alsofirst), not onlyat50; next-epoch application, downstream chooses strictbeforebest, heldoutQ remains1. Saved config/source interpretation only/no new Q calculation/forward/training/evaluation. Does not establish Q as cause of score differences; current finite seed batch unchanged. Evidence QUALITY_PROVENANCE.md/json.

V362 seed20261013 completed POR151: target77.27542879912328, best15/stop25, owninitialdelta0.7019999142997477; 220.583s solo/924.547MiB. RAMbest144665/frozenexact, not independent diskreload. 3/8 stages; next observed 20261013/PERM154/PID52000. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261013 completed PERM154: target89.29058665900838, best8/stop18, owninitialdelta0.008403166484811209; 124.732s solo/786.465MiB. RAMbest144665/frozenexact, not independent diskreload. 4/8 stages; next observed 20261013/SW104/PID58220. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

Fourthseed terminal POR151/PERM154 saved-pred readback: POR151 +0.701999914300 channel/+0.210599974290 TOTAL, CONT+2.402298195162/DEFAULT-0.204542323304, best15/stop25,220.583report/227.439processs/solo924.547MiB. PERM154 +0.008403166485 channel/+0.002941108270 TOTAL, CONT+0.154093345540/DEFAULT-0.066524344191, best8/stop18,124.732report/132.076processs/solo786.465MiB. PERM weak whole-target gain coexists with continuous-value gain and default-action loss; not evidence that continuous learning did nothing. Three-head four-stage prefixTOTAL80.187321128034, not full8stage ranking. CurrentSW104 actualPID58220 continues; complete models3/5; no new GPU route. Stage table renderer now uses only already-verified JSON (no model imports/label scoring), preserves incomplete-vs-zero distinction.

V368 user-authorized six-pair native distribution fusion: four completed eight-stage models original/seed10/11/12, per-head fixed0.01weight grid, first pair cached, five fresh pairs share native forwards. Pending actual assembly; no new training, no adoption/fullOOF/remaining folds. GPU evaluator yields before trainer handoffs; maximum two GPU tasks.

V362 complete seed 20261013: {"POR": 77.38833330639497, "PERM": 89.52127968635101, "SW": 73.63533098562274, "TOTAL": 80.3213137271093}. Completed 4/5; complete8stage source retrained. Ranking on fixed fifth only, no other folds or independent-generalization claim.

范围增补：seed20261013 在评估期间完成八阶段，TOTAL80.3213137271093。保留最初6对，新增它与原模型/seed10/11/12的4对，共5个完整模型10对；seed14仍训练中，排除。无新训练阶段。

V362 seed20261013 completed SW104: target73.63533098562274, best23/stop33, owninitialdelta2.438215904151079; 426.114s solo/1722.197MiB. RAMbest144665/frozenexact, not independent diskreload. 5/8 stages; next observed 20261014/SOURCE/PID39412. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261013 completed POR341: target77.38833330639497, best11/stop21, owninitialdelta0.11290450727169343; 263.228s solo/2212.483MiB. RAMbest144665/frozenexact, not independent diskreload. 6/8 stages; next observed 20261014/SOURCE/PID39412. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261013 completed PERM336: target89.52127968635101, best5/stop19, owninitialdelta0.2307281383021973; 158.101s solo/896.344MiB. RAMbest144665/frozenexact, not independent diskreload. 7/8 stages; next observed 20261014/SOURCE/PID39412. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261013 completed SW351: target73.63533098562274, best0/stop10, owninitialdelta0.0; 128.447s solo/1800.100MiB. RAMbest144665/frozenexact, not independent diskreload. 8/8 stages; next observed 20261014/SOURCE/PID39412. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V368 concurrency correction: seed20261013/SW351_fixed overlaps one read-only GPU evaluation; frozen timing remains, but must not be interpreted as solo speed. Maximum two GPU tasks preserved.

V368 concurrency correction: seed20261014/SOURCE overlaps one read-only GPU evaluation; frozen timing remains, but must not be interpreted as solo speed. Maximum two GPU tasks preserved.

Completed 10-pair native distribution diagnostic over 5 completed models / 144665 original fifth-fold rows. Best original__seed_20261013 TOTAL=80.863457557382; delta80.74=+0.123457557382; delta prior80.735166=+0.128291319656. All new native model predictions and grid endpoints exact, each row covered once; first pair reuses verified cache. Fixed whole-head ratios, grid0.01; no row/CONT-specific selection. No new training stages, 16 preexisting stages per pair, no other folds/adoption/fullOOF claim. Training continues. Timing concurrent GPU sum 246.635s, CPU37.745s, studywall 657.116s including implementation/wait; no solo throughput claim.

V368 completed10pair diagnostic TOTAL80.863457557382/original+seed13, equal50:50=80.840351605081, >80.74 +0.123457557382. Fixed per-head weights56/44,44/56,68/32. CONTdeltaoriginal POR-0.248553617806/PERM+0.168665049892/SW-0.480118282476; not all-continuous improvement. No new training/fullOOF/adoption/otherfolds. Atlas updated with diagnostic scope; fourth new complete, last seed14 continues.

V369 complete model/trainer/terminal CPU assembly entries prepared and syntax checked; no forward/numeric preflight/training/score. Same1260new/38646active parameters point/local connectivity pair, fresh336design/old154only, jointly train new backbone and categorical readout; fixed originalPOR341/SW351, one replacement PERM phase/8total. Point effective kernel redundant, not equal-function-capacity claim. Launch disabled/no queue until finite five-seed batch completes.

V369 preparation complete: model/trainer/CPU recorder syntax passed, no GPU/forward/training. Scoped offline atlas verification3nodes/3finite rightward edges, prepared node has no score; normalized two affected legacy lane labels and deduplicated their identical edges, no broad legacy audit. Current5seed batch unchanged, launch_enabled remainsfalse.

V362 seed20261014 SOURCE prerequisite completed: scores{"POR": 76.85409064868203, "PERM": 89.27383271035303, "SW": 73.33354533853603, "TOTAL": 79.96880951171578}, best51/stop61, 1107.616s solo/815.754MiB, processwall1117.917s, bestRAMreplay/allvalidationonce. Not complete8stage score. Next observed 20261014/POR113/PID15260; own SOURCE weights/Q only, no historical/otherseed learnedweights. Five-seed batch pending/nootherfolds/noindependentdiskreload claim.

V362 lastseed14 SOURCE terminal best51/stop61/TOTAL79.968809511716, historical sourcegap-0.253051571324; all144665 restored-best exact/one coverage, own Q50 strictlybefore51 with1761downweighted training rows, no Q refit/causal claim. ActualPOR113 ownSOURCE+prepareddata load traces confirmed. Source processwall1117.917s includes V368 one-trainer/one-eval overlap; frozen TIMING solo field not isolated throughput. Complete newmodels4/5, last model prerequisite1/8 only. Fixed source curves and stage snapshot refreshed CPU-only, no model forward or repeated labels evaluation; final8stage still pending.

V362 seed20261014 completed POR113: target77.28236341028625, best45/stop55, owninitialdelta0.3788061003721168; 274.481s solo/920.848MiB. RAMbest144665/frozenexact, not independent diskreload. 2/8 stages; next observed 20261014/PERM154/PID4196. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261014 completed POR151: target77.29412438388195, best7/stop17, owninitialdelta0.011760973595698943; 146.137s solo/924.547MiB. RAMbest144665/frozenexact, not independent diskreload. 3/8 stages; next observed 20261014/PERM154/PID4196. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

Lastseed14 POR113/151 terminal saved-pred diagnosis: POR113 actualSOURCEgain+0.428272761604/optimizergain+0.378806100372, CONT+1.245449611541/DEFAULT-0.007418633488, best45/stop55. POR151 +0.011760973596 target/+0.003528292079TOTAL, CONT+0.165011553734/DEFAULT-0.069947115741, best7/stop17. Small whole-target151gain coexists with positive continuous gain and default loss; not universal basis to delete151. Three-stage prefixTOTAL80.100819632276, NOT full8stage ranking. Source14 ownQ50/1761trainingrows, no Q refit. ActivePERM154 actual next PID4196, no otherfolds/new training routes.

V362 seed20261014 completed PERM154: target89.30441589327997, best12/stop22, owninitialdelta0.030583182926932295; 154.957s solo/786.465MiB. RAMbest144665/frozenexact, not independent diskreload. 4/8 stages; next observed 20261014/SW104/PID91072. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

Lastseed14 PERM154 terminal actualtargetgain+0.030583182927/TOTALcontribution+0.010704114024, DEFAULTdelta+0.001251378256/CONTdelta+0.087616385232, best12/stop22. Four-stage prefixTOTAL80.111523746300, not full8model or seed ranking. SW104 actualPID91072 continues; complete newmodels4/5, no newqueue/otherfolds.

One-shot manual post-seed launch entry prepared/syntax checked, not run. Initial two routes aux01/local, same-capacity aux0/point controls still held. Guard requires all5complete/current-controller exit/no existing GPUjobs/no stop marker; if an8stage seed passes80.8, promotion takes priority. No watcher, background queue, automatic retry/resume. Current launch_enabled remainsfalse and seed training unchanged.

V362 seed20261014 completed SW104: target73.79497059371414, best42/stop52, owninitialdelta2.597855512242475; 699.125s solo/1722.197MiB. RAMbest144665/frozenexact, not independent diskreload. 5/8 stages; next observed 20261014/SW351/PID28352. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261014 completed POR341: target77.42032846923811, best6/stop16, owninitialdelta0.12620408535616434; 207.485s solo/2212.483MiB. RAMbest144665/frozenexact, not independent diskreload. 6/8 stages; next observed 20261014/SW351/PID28352. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 complete seed 20261014: {"POR": 77.42032846923811, "PERM": 89.44922714148387, "SW": 73.79607662428037, "TOTAL": 80.36195485878892}. Completed 5/5; complete8stage source retrained. Ranking on fixed fifth only, no other folds or independent-generalization claim.

[V371 原模型+最后种子20261014融合](results/deep_learning/original_last_seed_fusion_v371_20261002/README.md)：用户明确追加，只作第五折诊断，V370补折配对不变。

V362 seed20261014 completed PERM336: target89.44922714148387, best5/stop15, owninitialdelta0.1447541928945668; 128.345s solo/896.344MiB. RAMbest144665/frozenexact, not independent diskreload. 7/8 stages; next observed None/None/PIDNone. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

V362 seed20261014 completed SW351: target73.79607662428037, best2/stop12, owninitialdelta0.0011060305662340397; 120.511s solo/1800.100MiB. RAMbest144665/frozenexact, not independent diskreload. 8/8 stages; next observed None/None/PIDNone. Whole-model ranking awaits all8; no otherfolds/current historical recipe unchanged.

[V370 两种子补折](results/deep_learning/two_seed_fullcv_v370_20261002/README.md)：用户授权原模型+20261013补原1–4折、每头每折扫比重，同时保留固定比重OOF。8阶段/模型、16历史阶段/配对，当前入口准备和语法检查通过；先完成V371追加诊断。V367/V369退役未启动。

V370 fold5 complete, 144665 rows. Fixed TOTAL80.863457557382; foldwise tuned diagnostic TOTAL80.863457557382; weights{'POR': {'weight_original': 0.56, 'weight_seed13': 0.43999999999999995, 'score': 78.23303953551347, 'fixed_weight_score': 78.23303953551347}, 'PERM': {'weight_original': 0.44, 'weight_seed13': 0.56, 'score': 89.74367648664987, 'fixed_weight_score': 89.74367648664987}, 'SW': {'weight_original': 0.68, 'weight_seed13': 0.31999999999999995, 'score': 74.23788264685729, 'fixed_weight_score': 74.23788264685729}}. Exact native disk replay and output readback; no rowwise/stratum choice.

V371 final per-head diagnostic TOTAL80.849582637629; delta-original+0.143452914391; delta-original+seed13-0.013874919753. Native disk replay, endpoints, coverage and actual three-head assembly/readback exact. No extra training.

V370已按最新授权启动有限补折批次。控制器30132，第一折原模型SOURCE进程27184；V362完成5/5且旧控制器已退出，V371诊断已完成。固定比例OOF与逐折最优比例诊断分开记录；当前没有新完整OOF成绩。

V370新增有限CPU存储维护（PID6196），最多处理本批8个完成SOURCE，无GPU/无训练启动。仅在SOURCE报告、退出码0和实际旧PID消失后，较早观测保留原epoch+逐点逐通道Q；大量NLL/熵/尾概率/CDF变化数组记录形状/哈希/统计后压缩为摘要。最新观测保留全部，完整导出Q轨迹/最佳与最后checkpoint/预测/缓存/日志哈希不变，因此Q导出和最后状态恢复依赖保留。每次实际压缩写SOURCE_DIAGNOSTIC_COMPACTION.json。历史输出与模型文件不动，不绕过此前被拒绝的历史checkpoint删除。当前等待首SOURCE结束，尚未压缩。

V370续盯快照：实际SOURCE进程27184已核验，原模型原第1折第28轮，最佳看到第27轮TOTAL81.653359695217，pat14。SOURCE仍活跃，未恢复终点，不是完整八阶段模型或OOF成绩。180秒观察到期后再次核实同一PID/创建时间/命令仍活跃，未重启。CPU维护6196等待8个完成SOURCE，未修改活跃训练输出；当前空闲约4.208GiB，暂时无磁盘失败。下一步继续现有批次，在SOURCE结束后核对本折缓存/Q/幅值与首个POR113衔接。

V370接续调度已准备并实测接管：仅替换CPU控制器30132→86104，实际GPU SOURCE27184的PID/创建时间/命令不变，未停止/重启/恢复训练，原七个训练上下文入口哈希不变。SOURCE结束后允许下一SOURCE与本模型头链重叠，最多2GPU，最多1个SOURCE/1个SW/1个扫描；仍有限8个model-fold、同8阶段/损失/预算/分头选模。当前首SOURCE仍活跃，因此尚未发生训练重叠、尚不能声称提速。每个阶段记录overlapped/solo及全批实际墙钟，最终按相同工作量计时评价。代码与接管证据：experiments/pure_dl/run_overlap_pair_fullcv_v370r1.py、OVERLAP_HANDOFF.json、OVERLAP_START_VERIFICATION.json。

V370 original/fold1 completed-source diagnostic compaction reclaimed 702.466MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_1/original/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370原模型原第1折SOURCE终点：TOTAL82.496037758876，POR79.979927065565/PERM90.989052707754/SW76.159689118551；最佳55/停止69，源模型RAM最佳预测精确复现，全部144621验证行覆盖一次。本折训练578935行，Q54严格早于最佳55，1670点降权，验证不降权；POR113实际载入本折private descriptor/shared input/own SOURCE三条路径，无历史第五折权重。SOURCE solo墙钟1187.908s含预处理/预热/训练/恢复/缓存写盘。诊断精简回收702.466MiB，保留权重/Q/恢复依赖SHA精确未变。已接管非父子SOURCE进程的OS退出码不可读，不能写exit0；实际终点source_report+fold_report/cache/无Traceback与PID消失已核实。尚无完整8阶段或全OOF成绩。


## 2026-10-02 obsolete experiment cleanup

User explicitly authorized removal of useless attempts and unused seeds. Deleted 30.588 GiB (13147 files); compact results/summary retained in 1105 artifact directories. Retain original + seed20261013, entire active V370 tree and historical-model dependencies; remove seeds20261010/11/12/14 weights/caches/logs. Data, delivered packages and reusable source unchanged. Verification passed for 282 retained files. Training continues; no stop/restart. Manifest and summary: D:/match_predict/maintenance_cleanup_20261002/README.md.

V370 original/fold1/POR113 terminal. Completed prefix2/8 scores{'POR': 80.11389573158694, 'PERM': 90.98905270775431, 'SW': 76.15968911855128, 'TOTAL': 82.53622835868303}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_POR113_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing465.380s (overlapped).

V370 original/fold1/POR151 terminal. Completed prefix3/8 scores{'POR': 80.25441098530335, 'PERM': 90.98905270775431, 'SW': 76.15968911855128, 'TOTAL': 82.57838293479796}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_POR151_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing338.353s (overlapped).

V370 original/fold1/PERM154 terminal. Completed prefix4/8 scores{'POR': 80.25441098530335, 'PERM': 90.98905270775431, 'SW': 76.15968911855128, 'TOTAL': 82.57838293479796}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_PERM154_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing177.226s (overlapped).

V370 seed_20261013/fold1 completed-source diagnostic compaction reclaimed 578.850MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_1/seed_20261013/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 seed_20261013/fold1/SOURCE terminal. Completed prefix1/8 scores{'POR': 79.9621869163639, 'PERM': 90.82341132483818, 'SW': 76.27531211748415, 'TOTAL': 82.47320927972198}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_SOURCE_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing1625.620s (overlapped).

V370 seed20261013/fold1 SOURCE terminal best42/stop56/TOTAL82.473209279722; restored144621 predictions exact, process exit0/PID33188 gone. Own source checkpoint differs from original; shared input path/SHA identical, own Q metadata retained. SOURCE difference -0.022828479154, not full model comparison. Wall1625.620s overlapped with head training (not solo throughput). Actual next POR113 PID46888 overlaps original SW104 PID50344; full8 and OOF remain unfinished.

V370 original/fold1/SW104 terminal. Completed prefix5/8 scores{'POR': 80.25441098530335, 'PERM': 90.98905270775431, 'SW': 76.87512613235297, 'TOTAL': 82.82878588962856}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_SW104_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing823.678s (overlapped).

V370 latest terminal milestone: original/fold1 prefix5/8 TOTAL82.828785889629 (POR80.254410985303/PERM90.989052707754/SW76.875126132353). SW104 best25/stop39, actual delta vs own SOURCE +0.715437013802, TOTAL contribution +0.250402954831; DEFAULT +0.535845326923, CONT +1.210251469116. This differs from +2.209069808162 optimizer gain against cold SW104 initialization. Old delivered full fold1 TOTAL82.823076877363 is comparison only, not a matched budget ablation. New seed13 own SOURCE Q40<best42, original Q54<best55; both normalizations exactly match shared input cache, no learned weights shared. Source2 compaction completed without deleting required checkpoints/Q. Actual POR341 PID77132 and seed13 POR113 PID46888 now run, maximum2; full8/fivefold fixed and diagnostic OOF unfinished.

V370 seed_20261013/fold1/POR113 terminal. Completed prefix2/8 scores{'POR': 80.00977386449392, 'PERM': 90.82341132483818, 'SW': 76.27531211748415, 'TOTAL': 82.48748536416099}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_POR113_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing297.989s (overlapped).

V370 original/fold1/POR341 terminal. Completed prefix6/8 scores{'POR': 80.25441098530335, 'PERM': 90.98905270775431, 'SW': 76.87512613235297, 'TOTAL': 82.82878588962856}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_POR341_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing334.452s (overlapped).

V370 seed_20261013/fold1/POR151 terminal. Completed prefix3/8 scores{'POR': 80.00977386449392, 'PERM': 90.82341132483818, 'SW': 76.27531211748415, 'TOTAL': 82.48748536416099}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_POR151_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing233.836s (overlapped).

## V370 verified CUDA OOM failure and prepared recovery

Two first-fold PERM workers failed during validation with CUDA OOM: original PERM336 PID18788 (last complete epoch9), seed20261013 PERM154 PID49708 (last complete epoch7). Controller86104 exited; no GPU training currently running. Best states, optimizer/scheduler/RNG and window states retained. Both saved best predictions were reloaded sequentially using original validation batch32 and reproduced exactly on144621 rows; current frozen weights exact. Read-only peak reserved812MiB each, not a training-memory benchmark. No failed trial was resumed. New CPU controller resume_pair_serial_v370r2.py is prepared/syntaxchecked, maximum1 GPU worker with driver free-memory7GiB gate; unchanged source/config/BS64/validationBS32/80-epoch budget/pat10->14. It reuses completed stages, resumes only the two exact authorized last states once, and stops on another failure. Previous failures/checkpoints copied and SHA verified. Explicit user recovery confirmation required by study AGENTS.md: ??/???????????. FullOOF remains unproven; goal remains active. See OOM_RECOVERY_PREPARED.json and both RECOVERY_READBACK files.

2026-10-02 explicit recovery activated: user confirmed checkpoint serial recovery. Controller27968, maximum1 GPU. Original fold1 PERM336 resumed from complete epoch9, first new epoch10, stopped14/best0, target90.989045182657; disk predictions and frozen weights exact. This stage adds no optimizer gain, and is -0.000007525098 versus prior PERM154 due to initial projection. SW351 now active. Seed13 PERM154 epoch7 recovery remains queued. Same8-stage private models, fixed and foldwise-tuned scans unchanged; no fullOOF yet. Old OOM evidence retained; another failure stops further launches. RECOVERY_ACTIVATION_VERIFIED.json preserves actual Chinese user reply (older launch files encoded that display field as question marks).

V370 original/fold1/PERM336 terminal. Completed prefix7/8 scores{'POR': 80.25441098530335, 'PERM': 90.98904518265655, 'SW': 76.87512613235297, 'TOTAL': 82.82878325584434}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_PERM336_TERMINAL.json. Best restored exactly. This is a prefix only, not complete model/fullOOF. Timing191.171s (failed overlap plus explicit serial recovery).

2026-10-02 serial recovery scoped verification passed: all7 frozen training-code hashes unchanged; atlas reflects explicit approval, running status and no fullOOF result. CPU-only maintenance45756 continues exactly6 held completed-source compactions, preserving Q/weights/predictions/input dependencies; no GPU launches or active-source edits. Existing successful replay is not repeated.

V370 original/fold1/SW351 terminal. Completed stages8/8 scores{'POR': 80.25441098530335, 'PERM': 90.98904518265655, 'SW': 76.9677842865412, 'TOTAL': 82.86121360981022}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/original_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing748.770s (solo).

V370 seed_20261013/fold1/PERM154 terminal. Completed stages4/8 scores{'POR': 80.00977386449392, 'PERM': 90.82341132483818, 'SW': 76.27531211748415, 'TOTAL': 82.48748536416099}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing175.014s (failed overlap plus explicit serial recovery).

2026-10-02 recovery terminal milestone: BOTH authorized PERM states resumed at next epochs10/8 and completed normally at14, best0, disk predictions/frozen weights exact. No new PERM optimizer gain. Original fold1 complete8 stages TOTAL82.861213609810; SW351 actual +0.092658154188 (CONT+1.150843178975, DEFAULT-0.291407492219), best62/stop76. New seed13 SW104 now active, serial maximum1 GPU, controller27968. Remaining private folds1-4 and native fixed/tuned scans continue; complete original five-fold OOF not yet available. Evidence PAIR_RECOVERY_TERMINAL_VERIFIED.json.

2026-10-02 compact adjacent-stage analysis: STAGE_GAINS.md/CSV summarize10 verified terminal non-SOURCE stages on fold1. Original SW104 +0.715437014 and SW351 +0.092658154; latter CONT+1.150843179 versus DEFAULT-0.291407492. Original POR113/POR151 +0.133968666/+0.140515254; PERM154 and POR341 best0, PERM336 initial projection -0.000007525. New seed13 POR113 +0.047586948, POR151/PERM154 best0. These are stage trajectory deltas, not causal stage-removal/retraining ablations or fullOOF gains. No recipe change or new GPU experiment. Pending: finish current SW104 and following stages, refresh table after terminal events; then all native per-fold scans and723556-row OOF.

V370 seed_20261013/fold1/SW104 terminal. Completed stages5/8 scores{'POR': 80.00977386449392, 'PERM': 90.82341132483818, 'SW': 76.93384368027814, 'TOTAL': 82.71797141113889}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing600.302s (solo).

V370 seed_20261013/fold1/POR341 terminal. Completed stages6/8 scores{'POR': 80.07054527363273, 'PERM': 90.82341132483818, 'SW': 76.93384368027814, 'TOTAL': 82.73620283388053}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing240.876s (solo).

V370 seed_20261013/fold1/PERM336 terminal. Completed stages7/8 scores{'POR': 80.07054527363273, 'PERM': 90.82622318971067, 'SW': 76.93384368027814, 'TOTAL': 82.73718698658591}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing132.527s (solo).

V370 seed_20261013/fold1/SW351 terminal. Completed stages8/8 scores{'POR': 80.07054527363273, 'PERM': 90.82622318971067, 'SW': 76.93384368027814, 'TOTAL': 82.73718698658591}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_1/seed_20261013_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing152.635s (solo).

V370 fold1 complete, 144621 rows. Fixed TOTAL82.976656765718; foldwise tuned diagnostic TOTAL82.993534618515; weights{'POR': {'weight_original': 0.53, 'weight_seed13': 0.47, 'score': 80.53637241360478, 'fixed_weight_score': 80.53030337518445}, 'PERM': {'weight_original': 0.7, 'weight_seed13': 0.30000000000000004, 'score': 91.06448763386392, 'fixed_weight_score': 91.02663620495527}, 'SW': {'weight_original': 0.69, 'weight_seed13': 0.31000000000000005, 'score': 77.02872063594612, 'fixed_weight_score': 77.02355166122501}}. Exact native disk replay and output readback; no rowwise/stratum choice.

2026-10-02 first paired fold complete: original8-stage TOTAL82.861213609810; seed13 complete8-stage82.737186986586. All3 native scan endpoints, coverage and disk replay exact144621 rows; scan stage counts0, serial wall about70s. Fifth-fixed ratio TOTAL82.976656765718 (+0.115443155908 versus original), per-head tuned diagnostic82.993534618515; weights original/new POR53/47, PERM70/30, SW69/31. Fixed ratios remain56/44,44/56,68/32. FOLD_RESULTS.md and each FUSION_STRATA.json preserve results: fixed POR DEFAULT+0.227751/CONT+0.381138, PERM DEFAULT-0.026168/CONT+0.183875, SW DEFAULT+0.157499/CONT-0.224527. One uniform ratio per target, no stratum/rowwise model choice. Stage gain table now14 completed non-SOURCE stages: new SW104+0.658532, POR341+0.060771, PERM336+0.002812, SW3510. Source52616 original/fold2 verified live, maximum1 GPU; no new training route or fullOOF achievement. Pending: all folds2-4 for both models, uniform native scans and723556-row OOF. Pair/ratios selected on fifth disclosed; foldwise tuned max diagnostic only.

2026-10-02 verified wait on original fold2 SOURCE52616: configuration seed20261006 + zero-based fold1 equals actual command/main initialization20261007, no double fold offset; max80, active patience14. Latest snapshot is SOURCE only, not completed8-stage or fullOOF. Previous goal turn was progress (first paired fold native scans/report completed); this turn verifies the running handle and changed-fold seed contract. Training continues unchanged, no new route or automatic retry. Evidence FOLD2_SOURCE_LIVE_SEED_VERIFICATION.json.

V370 original/fold2 completed-source diagnostic compaction reclaimed 597.380MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_2/original/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 original/fold2/SOURCE terminal. Completed stages1/8 scores{'POR': 78.90804266508842, 'PERM': 90.34162334254212, 'SW': 74.29579105693773, 'TOTAL': 81.29550783934447}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_SOURCE_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing1017.644s (solo).

V370 original/fold2/POR113 terminal. Completed stages2/8 scores{'POR': 79.18536012198616, 'PERM': 90.34162334254212, 'SW': 74.29579105693773, 'TOTAL': 81.3787030764138}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_POR113_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing130.522s (solo).

V370 original/fold2/POR151 terminal. Completed stages3/8 scores{'POR': 79.18536012198616, 'PERM': 90.34162334254212, 'SW': 74.29579105693773, 'TOTAL': 81.3787030764138}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_POR151_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing144.577s (solo).

V370 original/fold2/PERM154 terminal. Completed stages4/8 scores{'POR': 79.18536012198616, 'PERM': 90.41513889231291, 'SW': 74.29579105693773, 'TOTAL': 81.40443351883357}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing131.883s (solo then safe overlap).

2026-10-02 latest explicit user authorization restores maximum2 GPU workers for the unchanged original+seed13 fullCV batch. CPU-only handoff from27968 to77612 preserved original/fold2 PERM154 PID74592 without training restart; it completed and advanced normally. Actual process-verified activation jobs: [{"identity": "seed_20261013", "fold": 2, "stage": "SOURCE", "pid": 75188, "create_time": 1790934993.989125, "process_identity_exact": true}, {"identity": "original", "fold": 2, "stage": "SW104", "pid": 91432, "create_time": 1790935073.1618612, "process_identity_exact": true}]. At most one SOURCE/SW/scan/PERM each; launch requires7GiB driver free. All frozen training code hashes unchanged; any new failure stops future launches without automatic retry. Evidence SAFE_OVERLAP_ACTIVATION_QA.json. Earlier serial recovery remains historical, current cap2. Actual overlap and complete same-work batch wall remain to be measured; memory wait is allowed. Full OOF not completed. Pending: finish both seeds folds2-4, uniform native scans, fixed-ratio OOF and diagnostic OOF; possible later stage simplification requires evidence.

2026-10-02 bounded verified wait after double-worker activation. Previous goal turn made progress through authorized CPU handoff, preserved GPU child and actual second worker launch. Current observation verifies controller and both exact OS process handles: [{"identity":"seed_20261013","fold":2,"stage":"SOURCE","pid":75188,"alive":true,"epoch":14},{"identity":"original","fold":2,"stage":"SW104","pid":91432,"alive":true,"epoch":19}]. Fold2 original terminal actual gains: POR113 +0.277317456898 (DEFAULT +0.019556982872, CONT +0.804349879765); POR151 0; PERM154 +0.073515549771 (DEFAULT +0.006615535105, CONT +0.217723058755). STAGE_GAINS has17 completed non-SOURCE rows. These adjacent gains are not causal stage-removal ablations. Stage efficacy varies across seeds/folds; do not delete stage from the fixed current recipe based on one zero-gain result. Pending same fullCV completion, native scans and canonical OOF validation; no final achievement claimed. Driver memory launch gate and maximum2 remain; observation expiry never restarts training. Evidence LIVE_OBSERVATION.json and fold_2/original_PERM154_TERMINAL.json.



2026-10-02 observational parallel timing: fold2 SOURCE epoch5->15 serial original 162.89999999999998s versus overlapping seed13 347.90000000000003s (ratio 2.135666052793125). SW104 epoch6-15 mean seed13/fold1 proven solo 12.60708385001053s versus original/fold2 actual overlap 21.704543180007022s. Original/fold1 SW104 was overlapped, not a serial baseline; that interim label corrected by its terminal timing receipt. Different seeds/folds/Q; no controlled same-seed speed claim. Individual time is distinct from complete same-work total wall; overall acceleration remains unproven. No extra GPU benchmark or training change. Evidence PARALLEL_TIMING_SNAPSHOT.json. Pending full batch timings and OOF; preserve max2 with headroom/PERM exclusions.

V370 original/fold2/SW104 terminal. Completed stages5/8 scores{'POR': 79.18536012198616, 'PERM': 90.41513889231291, 'SW': 74.82606500865796, 'TOTAL': 81.59002940193565}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing839.829s (safe overlap).

2026-10-02 latest direct user instruction: existing two workers may finish, thereafter strictly serial. CPU-only handoff77612->68704 preserved exact SOURCE75188 and POR34156476; no GPU worker stopped/restarted. New dispatcher cannot launch while ANY worker is active, maximum future GPU jobs1. All frozen training/source hashes unchanged. Any failure blocks new launches without retry. Current completed fold2 original prefix 5/8 TOTAL 81.59002940193565 is not complete model/fullOOF. Evidence SERIAL_DRAIN_ACTIVATION_QA.json; pending finish original+seed13 folds2-4/native scans/full OOF. Subsequent speed decisions respect this explicit serial instruction.

V370 original/fold2/POR341 terminal. Completed stages6/8 scores{'POR': 79.2222399867954, 'PERM': 90.41513889231291, 'SW': 74.82606500865796, 'TOTAL': 81.60109336137842}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing414.439s (existing safe overlap then serial drain).

2026-10-02 verified actual transition to one worker: original/fold2 POR34156476 is terminal; SOURCE75188 remains live with exact PID/create/command. CPU controller68704 running, no other GPU training/evaluation process, original PERM336 config absent and not launched while SOURCE runs. Future maximum1 enforced. Previous goal turn made progress by CPU-only serial handoff; this turn completes POR341 reporting and verifies real serialized wait. Pending both full8-stage fold2 models and native scan, then folds3-4/full OOF. Evidence SERIAL_DRAIN_FIRST_SINGLE_WORKER.json.

V370 seed_20261013/fold2 completed-source diagnostic compaction reclaimed 576.273MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_2/seed_20261013/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 seed_20261013/fold2/SOURCE terminal. Completed stages1/8 scores{'POR': 79.05605643154057, 'PERM': 90.52438800229126, 'SW': 74.94652012820983, 'TOTAL': 81.63163477513754}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_SOURCE_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing1710.158s (existing safe overlap then serial drain).

V370 fold2 source-pair provenance complete: same raw training/validation rows, disjoint training versus validation; identical train-only normalization and shared input hash, different own-source weights and private Q strictly before each source best epoch. Original SOURCE 81.29550783934447, seed13 SOURCE 81.63163477513754. These are prerequisites, not full8-stage models/fullOOF. Evidence FOLD2_SOURCE_PAIR_PROVENANCE.json. Pending downstream heads, uniform native fusion and full original OOF.

V370 original/fold2/PERM336 terminal. Completed stages7/8 scores{'POR': 79.2222399867954, 'PERM': 90.53761710575662, 'SW': 74.82606500865796, 'TOTAL': 81.64396073608373}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing182.516s (solo after user requested serial).

V370 original/fold2/SW351 terminal. Completed stages8/8 scores{'POR': 79.2222399867954, 'PERM': 90.53761710575662, 'SW': 74.8706765086893, 'TOTAL': 81.65957476109469}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/original_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing239.446s (solo after user requested serial).

V370 seed_20261013/fold2/POR113 terminal. Completed stages2/8 scores{'POR': 79.28172969620036, 'PERM': 90.52438800229126, 'SW': 74.94652012820983, 'TOTAL': 81.69933675453548}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_POR113_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing128.206s (solo after user requested serial).

V370 seed_20261013/fold2/POR151 terminal. Completed stages3/8 scores{'POR': 79.31187842855309, 'PERM': 90.52438800229126, 'SW': 74.94652012820983, 'TOTAL': 81.7083813742413}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_POR151_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing181.223s (solo after user requested serial).

V370 seed_20261013/fold2/PERM154 terminal. Completed stages4/8 scores{'POR': 79.31187842855309, 'PERM': 90.56670516460275, 'SW': 74.94652012820983, 'TOTAL': 81.72319238105032}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing148.785s (solo after user requested serial).

V370 seed_20261013/fold2/SW104 terminal. Completed stages5/8 scores{'POR': 79.31187842855309, 'PERM': 90.56670516460275, 'SW': 74.75889434054632, 'TOTAL': 81.6575233553681}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing601.319s (solo after user requested serial).

2026-10-02 fold2 seed13 SW104 terminal: restored epoch30, stopped44. Actual predecessor SW74.946520128210 ->74.758894340546, delta -0.187625787664; TOTAL contribution -0.065669025682, prefix5/8 TOTAL81.657523355368. DEFAULT -0.184567390634 and CONT -0.195229072194. Optimizer gain +2.128566494318 is against cold initial72.630327846229, not SOURCE gain. Existing other SW104 comparisons positive (fold1 original +0.715437, fold1 seed13 +0.658532, fold2 original +0.530274); this is a scoped seed/fold regression, not a method-wide dead end. Fixed8-stage fullCV recipe stays unchanged; no validation-driven per-fold SOURCE/head substitution. After fullCV, a possible fixed-fifth study can test preserving SOURCE predictive function via additive repair in an existing stage slot; hypothesis untested, no new task/weights launched. Previous goal turn made progress by terminal POR151/PERM154 reports; this turn records the negative SW result. Pending remaining3 stages, pair scans and folds3-4 OOF. Evidence fold_2/seed_20261013_SW104_TERMINAL.json and STAGE_GAINS.csv.


V370 seed_20261013/fold2/POR341 terminal. Completed stages6/8 scores{'POR': 79.3197410309422, 'PERM': 90.56670516460275, 'SW': 74.75889434054632, 'TOTAL': 81.65988213608483}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing288.305s (solo after user requested serial).

V370 seed_20261013/fold2/PERM336 terminal. Completed stages7/8 scores{'POR': 79.3197410309422, 'PERM': 90.66686308090588, 'SW': 74.75889434054632, 'TOTAL': 81.69493740679093}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing180.817s (solo after user requested serial).

V370 seed_20261013/fold2/SW351 terminal. Completed stages8/8 scores{'POR': 79.3197410309422, 'PERM': 90.66686308090588, 'SW': 74.76940252532535, 'TOTAL': 81.69861527146358}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_2/seed_20261013_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing170.785s (solo after user requested serial).

V370 fold2 complete, 144667 rows. Fixed TOTAL81.886326454841; foldwise tuned diagnostic TOTAL81.914219881560; weights{'POR': {'weight_original': 0.45, 'weight_seed13': 0.55, 'score': 79.59191148520766, 'fixed_weight_score': 79.53302174261225}, 'PERM': {'weight_original': 0.42, 'weight_seed13': 0.5800000000000001, 'score': 90.83816680404728, 'fixed_weight_score': 90.83305623013669}, 'SW': {'weight_original': 0.62, 'weight_seed13': 0.38, 'score': 74.98082301308975, 'fixed_weight_score': 74.95671500431293}}. Exact native disk replay and output readback; no rowwise/stratum choice.

V370 completed paired fold2: 144667 rows; original 81.65957476109469, seed13 81.69861527146358, fifth-fixed ratios 81.88632645484104, fold-tuned diagnostic 81.91421988156026. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall 63.05961203575134s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs [{"identity": "original", "fold": 3, "stage": "SOURCE", "pid": 41088, "alive": true}], future cap1. Full OOF still pending; fifth selection boundary disclosed. Evidence FOLD2_PAIRED_NATIVE_QA.json, FOLD_RESULTS.md and FUSION_STRATA.json. Pending both remaining model pairs/full8-stage chains, native scans and723556-row OOF; existing deliveries unchanged.

2026-10-02 completed-stage compression analysis: 4 completed8-stage model/fold cases, actual adjacent changes only. POR113 continuous gains positive in all4 current cases; POR151 positive2/4; POR341 positive3/4 but CONT drops in those3; PERM336 fold2 useful, first-fold projection near-zero separated; SW104 one SOURCE regression and3 positive cases, SW351 positive3/4. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Previous goal turn completed fold2 native scans/readback; this turn adds stage-removal priorities and verifies running SOURCE3. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 original/fold3/SOURCE terminal. Completed stages1/8 scores{'POR': 80.42572005691339, 'PERM': 90.62186752824304, 'SW': 76.60895040981836, 'TOTAL': 82.65850229539551}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_SOURCE_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing1157.688s (solo after user requested serial).

V370 original/fold3 completed-source diagnostic compaction reclaimed 641.241MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_3/original/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 original/fold3/POR113 terminal. Completed stages2/8 scores{'POR': 80.60605445308767, 'PERM': 90.62186752824304, 'SW': 76.60895040981836, 'TOTAL': 82.71260261424779}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_POR113_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing154.593s (solo after user requested serial).

2026-10-02 final-OOF audit prepared, not executed: audit_final_oof.py checks canonical723556-row truth/raw mapping/grouped splits, exactly-once coverage, all four OOF arrays against five fold files, actual uniform native-grid columns for fixed and diagnostic weights, recomputed official scores and frozen code hashes. CPU only; no training or ratio changes. Syntax compilation passed; real audit awaits all five paired folds and FINAL.json. Training/source/Q provenance remains a separate evidence requirement. Fifth selection and per-fold tuning scopes stay explicit. Previous goal turn completed SOURCE3 readback; this turn completes POR1133 (real POR +0.180334396174, CONT +0.550068400159, prefix2/8 TOTAL82.712602614248) and prepares final canonical audit. Full8-stage folds3/4 and fullOOF still pending; latest max1 remains active.

V370 original/fold3/POR151 terminal. Completed stages3/8 scores{'POR': 80.60605445308767, 'PERM': 90.62186752824304, 'SW': 76.60895040981836, 'TOTAL': 82.71260261424779}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_POR151_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing140.944s (solo after user requested serial).

V370 original/fold3/PERM154 terminal. Completed stages4/8 scores{'POR': 80.60605445308767, 'PERM': 90.62186752824304, 'SW': 76.60895040981836, 'TOTAL': 82.71260261424779}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing124.786s (solo after user requested serial).

2026-10-02 completed original/fold3 POR151 and PERM154: both best_epoch0/stop14, exact initial-function restoration and actual target/TOTAL gain0. Prefix4/8 TOTAL82.712602614248; not final8-stage model or OOF. Sequential process wall140.944323+124.786292=265.730615s (load/cache/warmup/train/restore included); not a benchmark of a retrained shorter chain. Each of these stages now has2 positive and3 zero completed-stage cases across original/new folds1/2 and original3. That does not establish causal removal safety: POR341/PERM336 still need controlled downstream retraining after omission; uniform fixed-fifth ablation remains future work after current fullCV. Current recipe unchanged, SW104 now live in max1 CPU-dispatched GPU batch. Previous goal turn completed POR113 and prepared final OOF audit; this turn completes two zero-gain reports and native stage-gain summary (31 non-SOURCE rows). Original/new folds3-4, paired native scans and full723556-row audit remain pending.

2026-10-02 verified wait on original/fold3 SW104: actual PID52588/create/command and CPU controller68704 confirmed alive, exactly one GPU job and future cap1. Observed epoch24, best SW76.91075380516953, provisional delta versus SOURCE SW76.60895040981836=0.30180339535117184. Training not terminal, no restored-checkpoint gain or final prefix/OOF claimed. Bounded observer completed normally; no restart/stop/new route. Previous goal turn made progress by terminal POR151/PERM154 reports; this turn is a verified live wait. Pending SW104 terminal restore, three downstream stages, second seed, fold4 and complete canonical OOF audit. Evidence OBSERVATIONS.jsonl and LIVE_OBSERVATION.json.

V370 original/fold3/SW104 terminal. Completed stages5/8 scores{'POR': 80.60605445308767, 'PERM': 90.62186752824304, 'SW': 76.93400078207954, 'TOTAL': 82.82637024453919}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing554.775s (solo after user requested serial).

2026-10-02 original/fold3 SW104 terminal: best26/stop40, exact restored disk replay and frozen-parameter check. SOURCE SW76.608950409818 ->76.934000782080; actual +0.325050372261, TOTAL contribution+0.113767630291. DEFAULT +0.187798371043, CONT +0.708654512512. Prefix5/8 TOTAL82.826370244539, not a completed8-stage model/fullOOF/pruning experiment. Optimizer +2.156856266504 is against cold initialization and is not the SOURCE gain. Single-worker wall554.774703s includes loading/cache/warmup/train/restore. SW104 now has4 positive completed-stage cases and one seed13/fold2 SOURCE regression; preserve the fixed recipe, avoid validation-driven per-fold head substitution. Previous goal turn was a verified live wait; this turn completes SW104 reporting, updates32 stage-gain rows and verifies subsequent POR341. Pending final3 stages, second seed fold3, both fold4 models, native pair scans and complete canonical OOF audit; latest max1 continues.

V370 original/fold3/POR341 terminal. Completed stages6/8 scores{'POR': 80.69242876306755, 'PERM': 90.62186752824304, 'SW': 76.93400078207954, 'TOTAL': 82.85228253753316}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing490.035s (solo after user requested serial).

2026-10-02 original/fold3 POR341 terminal: best24/stop38, exact restored disk replay, all frozen parameters exact. Actual POR80.606054453088 ->80.692428763068 (+0.086374309980), TOTAL contribution+0.025912292994; prefix6/8 TOTAL82.852282537533. DEFAULT +0.224177769038 / CONT -0.236670539702. Prior first-two-fold POR341 positive cases also had CONT regression; now4 positive target-score cases all show CONT regression, one unchanged case. Scoped finding: this module mainly improves DEFAULT in these completed-stage cases, and is not evidence of improved continuous-value readout. Future same-stage continuous-preserving/omission controls require real fixed-fifth downstream retraining; no new stage/route or post-hoc fold substitution. Process wall490.034948s solo, includes load/cache/warmup/train/restore; not a speed benchmark. Previous goal turn completed SW104; this turn completes POR341 and verifies sequential PERM336. Remaining PERM336/SW351, second-seed fold3, both fold4 models, pair scans and723556-row OOF. Current eight-stage recipe stays unchanged, future max1, old deliveries intact.

V370 original/fold3/PERM336 terminal. Completed stages7/8 scores{'POR': 80.69242876306755, 'PERM': 90.67689121621815, 'SW': 76.93400078207954, 'TOTAL': 82.87154082832446}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing176.166s (solo after user requested serial).

V370 original/fold3/SW351 terminal. Completed stages8/8 scores{'POR': 80.69242876306755, 'PERM': 90.67689121621815, 'SW': 76.9340006440961, 'TOTAL': 82.87154078003024}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/original_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing149.206s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 5 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 5, "near_zero": 0, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}, "POR151": {"positive": 2, "near_zero": 3, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 3, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 4, "near_zero": 0, "negative": 1, "CONT_positive": 4, "CONT_negative": 1}, "POR341": {"positive": 4, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 4}, "PERM336": {"positive": 4, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 4}, "SW351": {"positive": 3, "near_zero": 2, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

2026-10-02 completed-stage compression analysis: 5 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 5, "near_zero": 0, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}, "POR151": {"positive": 2, "near_zero": 3, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 3, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 4, "near_zero": 0, "negative": 1, "CONT_positive": 4, "CONT_negative": 1}, "POR341": {"positive": 4, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 4}, "PERM336": {"positive": 4, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 4}, "SW351": {"positive": 3, "near_zero": 2, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

2026-10-02 original/fold3 complete8-stage model TOTAL82.871540780030 (145090 rows), actual assembled best-head prediction replay checked. PERM336 best5/stop19: target+0.055023687975, DEFAULT+0.183450621481, CONT-0.258582490025; wall176.166404s solo. SW351 best0/stop14: optimizer0, actual predecessor delta-0.000000137983 from initialization numeric projection, CONT0; wall149.206094s solo. No lossless pruning claim. Updated35 stage-gain rows and compression analysis to5 completed model/fold cases; dynamic SW counts corrected in narrative. New seed13/fold3 SOURCE now runs serially (actual fold seed20261011, label seed20261013); pair fusion/fullOOF not yet complete.

CPU-only saved-action diagnosis completed10 late-head contrasts, no new model/phase or rowwise choice. CONTINUOUS_ACTION_DIAGNOSIS.md and .json: each four-way predicted-action transition sum exactly matches recorded CONT delta. Third-fold POR has400 true-CONT rows switching CONT->DEFAULT (weighted CONT loss-0.379706206644), while CONT->CONT rows contribute+0.138293931049. PERM has195 switching to DEFAULT (-0.265274232626), CONT->CONT nearly unchanged (+0.000206280663). In all8 meaningful positive late-head cases, more true-CONT rows receive DEFAULT actions. This supports a future hypothesis about default-action pressure; internal gate causality or weak-backbone causality is not proved. Future train-only, same-stage controls remain unlaunched until fullCV; current recipe and uniform prediction rule preserved. Previous goal turn completed SW104; this turn completes final3 stages and action-transition evidence. Pending new seed3, both seed4 models, uniform native scans and full723556-row canonical audit; objective remains unmet.

2026-10-02 verified wait: seed13/fold3 SOURCE PID86108 and controller68704 OS identity/alive confirmed; epoch32, provisional best TOTAL82.46179095355426, future max1. Not terminal/complete8-stage/OOF. Previous goal turn completed original3 and action diagnosis; this turn waits on actual live training. Bounded observer timeout completed normally, no restart/stop/new route. Pending source best restore/provenance, remaining7 heads, paired scans and fold4/full OOF. Evidence OBSERVATIONS.jsonl and LIVE_OBSERVATION.json.

2026-10-02 verified wait on seed13/fold3 SOURCE actual PID86108/create/cmd and controller68704 alive, exactly one GPU job; epoch51, provisional best TOTAL82.7083048119633, snapshot delta versus original SOURCE 0.04980251656779444. Not terminal, no completed8-stage/new gain proof or OOF claim. Previous goal turn was verified wait; this turn is another verified wait with actual improving live training. Bounded observer ended normally without restart/stop/new launch. Source pair normalization/privateQ audit awaits best restore; seven heads, paired native scans, fold4 and canonical OOF remain. Evidence OBSERVATIONS.jsonl and LIVE_OBSERVATION.json.

V370 seed_20261013/fold3 completed-source diagnostic compaction reclaimed 828.361MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_3/seed_20261013/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 seed_20261013/fold3/SOURCE terminal. Completed stages1/8 scores{'POR': 80.68744991196792, 'PERM': 90.36959787239834, 'SW': 76.89370537474923, 'TOTAL': 82.74839111009202}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_SOURCE_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing1350.496s (solo after user requested serial).

V370 fold3 source-pair provenance complete: same raw training/validation rows, disjoint training versus validation; identical train-only normalization and shared input hash, different own-source weights and private Q strictly before each source best epoch. Original SOURCE 82.65850229539551, seed13 SOURCE 82.74839111009202. These are prerequisites, not full8-stage models/fullOOF. Evidence FOLD3_SOURCE_PAIR_PROVENANCE.json. Pending downstream heads, uniform native fusion and full original OOF.

2026-10-02 seed13/fold3 SOURCE terminal: best71/max80, exact best-checkpoint prediction restore. POR80.687449911968/PERM90.369597872398/SW76.893705374749/TOTAL82.748391110092. Source-only delta versus original3+0.089888814697, not complete8-stage/fusion/OOF gain. Source process wall1350.495653s solo; original source1157.688186s stopped62 versus80, unequal epoch work so no throughput winner claim. FOLD3_SOURCE_PAIR_PROVENANCE.json verifies145090 heldout rows/578466 training rows, disjoint own raw populations, identical seven normalization fields/shared input bytes, distinct source checkpoint files, private Q46<best48 original and Q70<best71 new. Q downweighted counts1466/2596 are descriptive, not causal gain evidence. Previous goal turn was verified wait; this turn completes SOURCE3 restore/provenance and verifies new POR113 process. Remaining7 new-seed stages, paired native scan, both fold4 models and canonical723556-row OOF; latest max1 persists.

V370 seed_20261013/fold3/POR113 terminal. Completed stages2/8 scores{'POR': 80.67555741502757, 'PERM': 90.36959787239834, 'SW': 76.89370537474923, 'TOTAL': 82.74482336100992}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_POR113_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing83.642s (solo after user requested serial).

V370 seed_20261013/fold3/POR151 terminal. Completed stages3/8 scores{'POR': 80.67555741502757, 'PERM': 90.36959787239834, 'SW': 76.89370537474923, 'TOTAL': 82.74482336100992}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_POR151_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing133.186s (solo after user requested serial).

V370 seed_20261013/fold3/PERM154 terminal. Completed stages4/8 scores{'POR': 80.67555741502757, 'PERM': 90.36959787239834, 'SW': 76.89370537474923, 'TOTAL': 82.74482336100992}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing110.298s (solo after user requested serial).

2026-10-02 seed13/fold3 first three heads terminal: POR113 best0/stop14, optimizer gain0, actual SOURCE delta-0.011892496940 (DEFAULT+0.042279140652/CONT-0.138884000947); initial function change is not a learned gain, wall83.642077s solo. POR151 best0/stop14 target gain0, wall133.186470s; PERM154 best0/stop14 target gain0, wall110.297528s. Exact restored predictions/frozen parameters verified. Prefix4/8 TOTAL82.744823361010, not final model/fusion/OOF. This is the first completed POR113 SOURCE regression among the currently recorded trials; prior five positive cases remain evidence, not universal benefit. Do not substitute SOURCE by fold/label type in the fixed recipe. Future source-preserving initialization within the existing POR113 stage is a hypothesis only, after fullCV. SW104 now live; last4 stages, native pair scan and both fold4 models/full OOF remain. Previous goal turn was verified wait; this turn completes SOURCE/private-Q provenance and three head records,38 non-SOURCE stage rows. Latest max1 maintained; no new optimization route or retry.

V370 seed_20261013/fold3/SW104 terminal. Completed stages5/8 scores{'POR': 80.67555741502757, 'PERM': 90.36959787239834, 'SW': 76.8191159902716, 'TOTAL': 82.71871707644274}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing580.897s (solo after user requested serial).

2026-10-02 seed13/fold3 SW104 terminal: best29/stop43, actual source SW76.893705374749 ->76.819115990272, delta-0.074589384478 and TOTAL contribution-0.026106284567. DEFAULT-0.154197081089/CONT+0.147905309550: tradeoff differs from late POR/PERM default-action shifts. Optimizer+2.041971474696 is against cold initialization, not SOURCE improvement. Exact restored predictions/frozen parameters verified, wall580.897496s solo including load/cache/warmup/train/restore. Prefix5/8 TOTAL82.718717076443, not completed model/pair/OOF. This is second completed SW104 source regression (new seed fold2/fold3), with four earlier positive cases; scope is this recipe/seed/fold evidence, not all SW backbones. Future preserving the SOURCE predictive distribution via additive correction within the existing SW104 slot is still an untested hypothesis, no new GPU route or fold-dependent substitution. Previous goal turn completed source and first3 head reports; this turn completes SW104 and39 stage-gain rows, actual POR34191308 live. Remaining last3 stages, native pair scan, both fold4 models and full canonical OOF; latest max1 unchanged.

V370 seed_20261013/fold3/POR341 terminal. Completed stages6/8 scores{'POR': 80.67912955281808, 'PERM': 90.36959787239834, 'SW': 76.8191159902716, 'TOTAL': 82.71978871777989}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing306.735s (solo after user requested serial).

V370 seed_20261013/fold3/PERM336 terminal. Completed stages7/8 scores{'POR': 80.67912955281808, 'PERM': 90.4980173719235, 'SW': 76.8191159902716, 'TOTAL': 82.7647355426137}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing200.448s (solo after user requested serial).

2026-10-02 seed13/fold3 late POR/PERM terminal: POR341 best9/stop23 actual+0.003572137791, DEFAULT+0.177965685070/CONT-0.405248784786, TOTAL+0.001071641337; wall306.735001s solo. PERM336 best10/stop24 actual+0.128419499525, DEFAULT+0.087013835351/CONT+0.229528135658, TOTAL+0.044946824834; wall200.448211s solo. PERM optimizer gain+0.128432627567 differs from actual predecessor gain by small initial projection, kept separate. Both disk restores/frozen parameters exact. Prefix7/8 TOTAL82.764735542614; source/model/pair scope preserved. PERM now has a positive continuous-readout case, so earlier four CONT-regression cases do not justify a universal default-only gain explanation. SW35188804 live (snapshot epoch14, provisional target+0.002720808977, no terminal gain claim). Previous goal turn completed source/first3 heads; this turn completes SW104 and late POR/PERM with41 non-SOURCE stage records. Existing completed-model compression/action analysis still excludes incomplete new seed3 and will refresh after stage8. Pending last stage, native pair scan, both fold4 models and723556-row OOF; max1 and fixed8-stage recipe unchanged.

V370 seed_20261013/fold3/SW351 terminal. Completed stages8/8 scores{'POR': 80.67912955281808, 'PERM': 90.4980173719235, 'SW': 76.82183679924867, 'TOTAL': 82.76568782575568}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_3/seed_20261013_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing274.465s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 6 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 5, "near_zero": 0, "negative": 1, "CONT_positive": 5, "CONT_negative": 1}, "POR151": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 4, "near_zero": 0, "negative": 2, "CONT_positive": 5, "CONT_negative": 1}, "POR341": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 5}, "PERM336": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 4, "near_zero": 2, "negative": 0, "CONT_positive": 4, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 fold3 complete, 145090 rows. Fixed TOTAL82.956087179217; foldwise tuned diagnostic TOTAL82.979094780384; weights{'POR': {'weight_original': 0.51, 'weight_seed13': 0.49, 'score': 80.9492883435713, 'fixed_weight_score': 80.91853978695657}, 'PERM': {'weight_original': 0.52, 'weight_seed13': 0.48, 'score': 90.70964780647334, 'fixed_weight_score': 90.68740044499263}, 'SW': {'weight_original': 0.44, 'weight_seed13': 0.56, 'score': 76.98837584299105, 'fixed_weight_score': 76.97124310680644}}. Exact native disk replay and output readback; no rowwise/stratum choice.

V370 completed paired fold3: 145090 rows; original 82.87154078003024, seed13 82.76568782575568, fifth-fixed ratios 82.95608717921664, fold-tuned diagnostic 82.97909478038392. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall 63.32169055938721s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs [{"identity": "original", "fold": 4, "stage": "SOURCE", "pid": 76620, "alive": true}], future cap1. This verifies one fold, not full OOF; fifth selection boundary disclosed. Evidence FOLD3_PAIRED_NATIVE_QA.json, FOLD_RESULTS.md and FUSION_STRATA.json. Remaining paired folds [4]; final723556-row OOF audit remains separate; existing deliveries unchanged.

2026-10-02 paired fold3 complete145090 rows: original8-stage82.871540780030, seed13 eight-stage82.765687825756, fifth-fixed native fusion82.956087179217 (+0.084546399186 versus original), per-head tuned diagnostic82.979094780384. Diagnostic weights original/new POR51/49, PERM52/48, SW44/56; fixed weights remain56/44,44/56,68/32. All3 native scan exits0, exact disk replay/coverage/endpoints and standalone arrays equal assembled complete models; frozen7 code hashes unchanged. Serial scan invocation wall63.321690559387s includes load/cache/warmup/forward/101-grid/write, no extra training stages. FOLD3_PAIRED_NATIVE_QA.json verifies new original/fold4 SOURCE76620 as sole actual GPU job, cap1. Only paired fold4 remains; no FINAL/full-OOF claim.

Seed13/fold3 SW351 best13/stop27 restored exactly: actual+0.002720808977, DEFAULT-0.216329456527/CONT+0.614942037263, wall274.464567s solo. All42 non-SOURCE stage rows, six completed model/fold compression cases,12 late-head action contrasts updated. New PERM CONT gain+0.229528135658 is explained descriptively by CONT->CONT contribution+0.376163287141 outweighing CONT->DEFAULT loss-0.158416759900 (plus reverse+0.011781608416); internal gate causality still unproven. Original/new POR3 action pressure remains a useful future same-stage hypothesis. Previous goal turn completed source/first3 heads; this turn completes final4 stages, paired native scan/readback and conditional analyses. Full original OOF>82 remains unresolved until both model4 chains and canonical723556-row score audit; original deliveries untouched.

V370 original/fold4 completed-source diagnostic compaction reclaimed 595.880MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_4/original/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 original/fold4/SOURCE terminal. Completed stages1/8 scores{'POR': 79.23922786171148, 'PERM': 91.20589315227929, 'SW': 74.00334279523662, 'TOTAL': 81.595000940144}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_SOURCE_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing1038.047s (solo after user requested serial).

2026-10-02 original/fold4 SOURCE terminal: best45/stop59, exact restored source scores{"POR": 79.23922786171148, "PERM": 91.20589315227929, "SW": 74.00334279523662, "TOTAL": 81.595000940144}. Source-only, not completed8-stage/pair/OOF. Own training Q epoch44 strictly before best, actual source seed20261009 (base06+3). Solo process wall1038.04656457901s includes preparation/load/warmup/train/restore. Source pair4 provenance waits for new seed SOURCE. Previous goal turn completed paired fold3 native readback; this turn completes original4 SOURCE after a verified live wait and confirms subsequent POR11372496. Remaining seven original heads, new seed4 complete chain, native scans and full723556-row canonical OOF audit; latest max1 maintained.

V370 original/fold4/POR113 terminal. Completed stages2/8 scores{'POR': 79.29833330914761, 'PERM': 91.20589315227929, 'SW': 74.00334279523662, 'TOTAL': 81.61273257437485}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_POR113_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing137.159s (solo after user requested serial).

V370 original/fold4/POR151 terminal. Completed stages3/8 scores{'POR': 79.47038584739558, 'PERM': 91.20589315227929, 'SW': 74.00334279523662, 'TOTAL': 81.66434833584924}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_POR151_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing251.250s (solo after user requested serial).

V370 original/fold4/PERM154 terminal. Completed stages4/8 scores{'POR': 79.47038584739558, 'PERM': 91.20589315227929, 'SW': 74.00334279523662, 'TOTAL': 81.66434833584924}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing133.823s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 6 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 5, "near_zero": 0, "negative": 1, "CONT_positive": 5, "CONT_negative": 1}, "POR151": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 4, "near_zero": 0, "negative": 2, "CONT_positive": 5, "CONT_negative": 1}, "POR341": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 5}, "PERM336": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 4, "near_zero": 2, "negative": 0, "CONT_positive": 4, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 final remaining-fold training audit prepared in audit_remaining_training.py. Syntax compiled; not executed because fold4 and canonical final OOF are still incomplete. It checks actual checkpoint/config hashes, own-source and parent load traces, original source train/validation raw-row metadata, source-private Q before selected epoch and eight-stage/80-epoch budgets for all eight remaining-fold models. Scope excludes historical fifth provenance and unbiased accuracy; no new GPU job or training-code change. Previous goal turn recorded original4 POR113/POR151/PERM154 and revised compression evidence; current verified SW104 wait remains live.

V370 final remaining-fold training audit prepared in audit_remaining_training.py. Syntax compiled; not executed because fold4 and canonical final OOF are still incomplete. It checks actual checkpoint/config hashes, own-source and parent load traces, original source train/validation raw-row metadata, source-private Q before selected epoch and eight-stage/80-epoch budgets for all eight remaining-fold models. Scope excludes historical fifth provenance and unbiased accuracy; no new GPU job or training-code change. Previous goal turn recorded original4 POR113/POR151/PERM154 and revised compression evidence; current verified SW104 wait remains live.

V370 original/fold4/SW104 terminal. Completed stages5/8 scores{'POR': 79.47038584739558, 'PERM': 91.20589315227929, 'SW': 74.4724953639945, 'TOTAL': 81.8285517349145}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing486.528s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 6 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 5, "near_zero": 0, "negative": 1, "CONT_positive": 5, "CONT_negative": 1}, "POR151": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 4, "near_zero": 0, "negative": 2, "CONT_positive": 5, "CONT_negative": 1}, "POR341": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 5}, "PERM336": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 4, "near_zero": 2, "negative": 0, "CONT_positive": 4, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 prepared final canonical OOF audit now also checks the six currently retained fifth-fold endpoint head checkpoint hashes against the original+seed20261013 native-scan AUGMENT_PLAN, and records the hashes. Syntax compiled; full audit still not run until FINAL exists. This is preserved-artifact identity evidence, not historical training-provenance or unbiased-accuracy proof. No training/evaluation recipe changed.

V370 prepared final canonical OOF audit now also checks the six currently retained fifth-fold endpoint head checkpoint hashes against the original+seed20261013 native-scan AUGMENT_PLAN, and records the hashes. Syntax compiled; full audit still not run until FINAL exists. This is preserved-artifact identity evidence, not historical training-provenance or unbiased-accuracy proof. No training/evaluation recipe changed.

V370 original/fold4/POR341 terminal. Completed stages6/8 scores{'POR': 79.49289205568876, 'PERM': 91.20589315227929, 'SW': 74.4724953639945, 'TOTAL': 81.83530359740244}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing288.511s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 6 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 5, "near_zero": 0, "negative": 1, "CONT_positive": 5, "CONT_negative": 1}, "POR151": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 4, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 4, "near_zero": 0, "negative": 2, "CONT_positive": 5, "CONT_negative": 1}, "POR341": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 5}, "PERM336": {"positive": 5, "near_zero": 1, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 4, "near_zero": 2, "negative": 0, "CONT_positive": 4, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 original/fold4/PERM336 terminal. Completed stages7/8 scores{'POR': 79.49289205568876, 'PERM': 91.20591317052484, 'SW': 74.4724953639945, 'TOTAL': 81.83531060378839}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing212.009s (solo after user requested serial).

V370 original/fold4/SW351 terminal. Completed stages8/8 scores{'POR': 79.49289205568876, 'PERM': 91.20591317052484, 'SW': 74.50542147402476, 'TOTAL': 81.84683474229898}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/original_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing177.012s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 7 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 6, "near_zero": 0, "negative": 1, "CONT_positive": 6, "CONT_negative": 1}, "POR151": {"positive": 3, "near_zero": 4, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 5, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 5, "near_zero": 0, "negative": 2, "CONT_positive": 6, "CONT_negative": 1}, "POR341": {"positive": 6, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 6}, "PERM336": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

2026-10-02 original/fold4 complete eight-stage model: TOTAL 81.846834742299, old submitted fold4 81.857560312022, delta -0.010725569723; paired fixed/diagnostic fusion still pending new seed4. Eight-vs-five observed increment +0.018283007384, not a causal pruning ablation. Sequential eight process invocation wall 2724.339222s includes each load/preparation/warmup/training/best restore/write, excludes between-stage controller/CPU waiting; not the whole two-seed fold total. All49 terminal head rows/seven completed model cases/fourteen late-head action contrasts refreshed. Original4 POR341 continuous loss -.347109579177 corresponds descriptively to CONT-to-DEFAULT -.412368671 plus CONT-to-CONT +.065259092; predicted action changes do not prove internal gate causality. PERM336 optimizer gain0, initial projection +.000020018246 kept separate. SW351 true predecessor gain+.032926110030 with CONT+.466931015120/DEFAULT-.152005875255, best2/stop16. New seed4 SOURCE PID27140 was verified alive; actual seed20261012/base20261009, serial cap1, source-private training/Q. Source pair4 proof/native scans/canonical723556-row OOF and final training evidence remain pending; no >82 achievement or additional trial.

2026-10-02 original/fold4 complete eight-stage model: TOTAL 81.846834742299, old submitted fold4 81.857560312022, delta -0.010725569723; paired fixed/diagnostic fusion still pending new seed4. Eight-vs-five observed increment +0.018283007384, not a causal pruning ablation. Sequential eight process invocation wall 2724.339222s includes each load/preparation/warmup/training/best restore/write, excludes between-stage controller/CPU waiting; not the whole two-seed fold total. All49 terminal head rows/seven completed model cases/fourteen late-head action contrasts refreshed. Original4 POR341 continuous loss -.347109579177 corresponds descriptively to CONT-to-DEFAULT -.412368671 plus CONT-to-CONT +.065259092; predicted action changes do not prove internal gate causality. PERM336 optimizer gain0, initial projection +.000020018246 kept separate. SW351 true predecessor gain+.032926110030 with CONT+.466931015120/DEFAULT-.152005875255, best2/stop16. New seed4 SOURCE PID27140 was verified alive; actual seed20261012/base20261009, serial cap1, source-private training/Q. Source pair4 proof/native scans/canonical723556-row OOF and final training evidence remain pending; no >82 achievement or additional trial.

2026-10-02 verified live wait: previous goal turn completed all original/fold4 stages (TOTAL81.846834742299) and handoff to new seed4 SOURCE. Current source PID27140/create_time1790947694.162996 and command are reverified with OS, controller alive, serial cap1; latest history epoch20, bestTOTAL80.430154099283, bad0. Observation timeout only closes CPU observer, never stops/resumes training. This is a running SOURCE snapshot, not complete8-stage/fused/fullOOF score. Source-pair4 provenance waits for terminal checkpoint; new seven head stages, three native scans and canonical723556-row final audits remain pending. No new experiment, retry or achievement claim.

2026-10-02 verified live wait: previous goal turn completed all original/fold4 stages (TOTAL81.846834742299) and handoff to new seed4 SOURCE. Current source PID27140/create_time1790947694.162996 and command are reverified with OS, controller alive, serial cap1; latest history epoch20, bestTOTAL80.430154099283, bad0. Observation timeout only closes CPU observer, never stops/resumes training. This is a running SOURCE snapshot, not complete8-stage/fused/fullOOF score. Source-pair4 provenance waits for terminal checkpoint; new seven head stages, three native scans and canonical723556-row final audits remain pending. No new experiment, retry or achievement claim.

V370 verified live wait continued on the same new seed/fold4 SOURCE PID27140/create_time1790947694.162996: latest epoch34, bestTOTAL81.125720395839, bad2, effective patience14. Actual source/controller OS commands verified; serial cap1. This is a running SOURCE snapshot, not full model/fusion/OOF. Prior turn was also verified wait; no blocker, stop, retry or new trial. Source pair4 provenance, seven heads, native scan and canonical final audits remain pending.

V370 verified live wait continued on the same new seed/fold4 SOURCE PID27140/create_time1790947694.162996: latest epoch34, bestTOTAL81.125720395839, bad2, effective patience14. Actual source/controller OS commands verified; serial cap1. This is a running SOURCE snapshot, not full model/fusion/OOF. Prior turn was also verified wait; no blocker, stop, retry or new trial. Source pair4 provenance, seven heads, native scan and canonical final audits remain pending.

V370 seed_20261013/fold4 completed-source diagnostic compaction reclaimed 719.995MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: outputs/pure_dl_two_seed_fullcv_v370/fold_4/seed_20261013/SOURCE/SOURCE_DIAGNOSTIC_COMPACTION.json.

V370 seed_20261013/fold4/SOURCE terminal. Completed stages1/8 scores{'POR': 79.78916816460293, 'PERM': 91.27156260971567, 'SW': 74.04333352827172, 'TOTAL': 81.79696409767647}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_SOURCE_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing1351.536s (solo after user requested serial).

V370 fold4 source-pair provenance complete: same raw training/validation rows, disjoint training versus validation; identical train-only normalization and shared input hash, different own-source weights and private Q strictly before each source best epoch. Original SOURCE 81.595000940144, seed13 SOURCE 81.79696409767647. These are prerequisites, not full8-stage models/fullOOF. Evidence FOLD4_SOURCE_PAIR_PROVENANCE.json. Pending downstream heads, uniform native fusion and full original OOF.

2026-10-02 new seed/fold4 SOURCE terminal best56/stop70: exact restored TOTAL81.796964097676, versus original SOURCE+0.201963157532 (SOURCE comparison only). Actual seed20261012/base09; private Q epoch54<56,1945 downweighted training rows versus original1577 does not establish Q causality. Both sources exact same seven normalization fields/shared seed-invariant input cache, same579043 training and144513 held-out rows, disjoint original raw populations, different source hashes. Solo invocation wall1351.535722s includes preparation/load/warmup/train/best restore/write; not whole two-model chain timing. FOLD4_SOURCE_PAIR_PROVENANCE.json records prerequisites only. Previous goal turn was verified wait; this turn completes SOURCE after verified wait and confirms POR113 PID64956. Seven head stages, three native scans, complete723556-row OOF and final provenance audits remain. No new optimization route or >82 claim; serial cap1 unchanged.

2026-10-02 new seed/fold4 SOURCE terminal best56/stop70: exact restored TOTAL81.796964097676, versus original SOURCE+0.201963157532 (SOURCE comparison only). Actual seed20261012/base09; private Q epoch54<56,1945 downweighted training rows versus original1577 does not establish Q causality. Both sources exact same seven normalization fields/shared seed-invariant input cache, same579043 training and144513 held-out rows, disjoint original raw populations, different source hashes. Solo invocation wall1351.535722s includes preparation/load/warmup/train/best restore/write; not whole two-model chain timing. FOLD4_SOURCE_PAIR_PROVENANCE.json records prerequisites only. Previous goal turn was verified wait; this turn completes SOURCE after verified wait and confirms POR113 PID64956. Seven head stages, three native scans, complete723556-row OOF and final provenance audits remain. No new optimization route or >82 claim; serial cap1 unchanged.

V370 seed_20261013/fold4/POR113 terminal. Completed stages2/8 scores{'POR': 79.80273286971332, 'PERM': 91.27156260971567, 'SW': 74.04333352827172, 'TOTAL': 81.80103350920957}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_POR113_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing92.600s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 7 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 6, "near_zero": 0, "negative": 1, "CONT_positive": 6, "CONT_negative": 1}, "POR151": {"positive": 3, "near_zero": 4, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 5, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 5, "near_zero": 0, "negative": 2, "CONT_positive": 6, "CONT_negative": 1}, "POR341": {"positive": 6, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 6}, "PERM336": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 seed_20261013/fold4/POR151 terminal. Completed stages3/8 scores{'POR': 79.80273286971332, 'PERM': 91.27156260971567, 'SW': 74.04333352827172, 'TOTAL': 81.80103350920957}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_POR151_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing142.227s (solo after user requested serial).

V370 seed_20261013/fold4/PERM154 terminal. Completed stages4/8 scores{'POR': 79.80273286971332, 'PERM': 91.30387186022239, 'SW': 74.04333352827172, 'TOTAL': 81.81234174688693}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_PERM154_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing148.006s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 7 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 6, "near_zero": 0, "negative": 1, "CONT_positive": 6, "CONT_negative": 1}, "POR151": {"positive": 3, "near_zero": 4, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 5, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 5, "near_zero": 0, "negative": 2, "CONT_positive": 6, "CONT_negative": 1}, "POR341": {"positive": 6, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 6}, "PERM336": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 new seed/fold4 POR151 terminal best0/stop14 gain0; PERM154 best4/stop18 actual+.032309250507 (DEFAULT+.028789448367, CONT+.039402392085),148.006489s solo. Exact restored four-stage prefixTOTAL81.812341746887, not full model or fused fold. All52 terminal head-stage rows updated; seven fully completed model cases stay separate from this incomplete model. Current SW104 PID24812/create_time1790949429.500398 is OS-process verified live, epoch20 bestSW74.321920636934, serial cap1; this is a running snapshot, not terminal gain. Previous turn completed SOURCE and POR113; current turn completes POR151/PERM154 and verifies continuing SW104. Remaining SW104 plus three final repairs/native scans/canonical723556-row OOF and final lineage audits; no extra experiment, retry or >82 claim.

V370 new seed/fold4 POR151 terminal best0/stop14 gain0; PERM154 best4/stop18 actual+.032309250507 (DEFAULT+.028789448367, CONT+.039402392085),148.006489s solo. Exact restored four-stage prefixTOTAL81.812341746887, not full model or fused fold. All52 terminal head-stage rows updated; seven fully completed model cases stay separate from this incomplete model. Current SW104 PID24812/create_time1790949429.500398 is OS-process verified live, epoch20 bestSW74.321920636934, serial cap1; this is a running snapshot, not terminal gain. Previous turn completed SOURCE and POR113; current turn completes POR151/PERM154 and verifies continuing SW104. Remaining SW104 plus three final repairs/native scans/canonical723556-row OOF and final lineage audits; no extra experiment, retry or >82 claim.

V370 final-only canonical OOF audit preparation expanded: retained fifth source checkpoints both contain original training/validation raw-row metadata and206-feature contract. Prepared audit checks both canonical fold5 source populations, actual base+4 seeds, last source-private Q epoch strictly before selected SOURCE best and each retained head source/Q-trace SHA, in addition to preserved head weights and uniform native fusion. Syntax compiled; full audit not executed before FINAL exists. Historical budget differences remain disclosed; these metadata checks do not establish unbiased accuracy. No active/historical training code or recipe changed.

V370 final-only canonical OOF audit preparation expanded: retained fifth source checkpoints both contain original training/validation raw-row metadata and206-feature contract. Prepared audit checks both canonical fold5 source populations, actual base+4 seeds, last source-private Q epoch strictly before selected SOURCE best and each retained head source/Q-trace SHA, in addition to preserved head weights and uniform native fusion. Syntax compiled; full audit not executed before FINAL exists. Historical budget differences remain disclosed; these metadata checks do not establish unbiased accuracy. No active/historical training code or recipe changed.

V370 seed_20261013/fold4/SW104 terminal. Completed stages5/8 scores{'POR': 79.80273286971332, 'PERM': 91.30387186022239, 'SW': 74.67057243794171, 'TOTAL': 82.03187536527143}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_SW104_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing701.778s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 7 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 6, "near_zero": 0, "negative": 1, "CONT_positive": 6, "CONT_negative": 1}, "POR151": {"positive": 3, "near_zero": 4, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 5, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 5, "near_zero": 0, "negative": 2, "CONT_positive": 6, "CONT_negative": 1}, "POR341": {"positive": 6, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 6}, "PERM336": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

2026-10-02 new seed/fold4 SW104 terminal best36/stop50: SW74.670572437942, actual SOURCE gain+.627238909670, DEFAULT-.046019408560/CONT+2.207265274791, solo process wall701.778236s. Cold-start optimizer gain+3.216708321061 is separate from SOURCE gain. Actually assembled five-stage prefix TOTAL82.031875365271, not full8-stage, paired fusion or original fullOOF. All53 terminal head-stage rows updated, incomplete model excluded from seven-case completed-model compression summary. Previous goal turn completed POR151/PERM154; current turn completes SW104 after verified live wait and confirms POR341 PID78160. Remaining three repairs, native scans and canonical723556-row OOF/final provenance audit; serial cap1 and eight-stage recipe unchanged. No per-row/stratum prediction selection or new GPU trial.

2026-10-02 new seed/fold4 SW104 terminal best36/stop50: SW74.670572437942, actual SOURCE gain+.627238909670, DEFAULT-.046019408560/CONT+2.207265274791, solo process wall701.778236s. Cold-start optimizer gain+3.216708321061 is separate from SOURCE gain. Actually assembled five-stage prefix TOTAL82.031875365271, not full8-stage, paired fusion or original fullOOF. All53 terminal head-stage rows updated, incomplete model excluded from seven-case completed-model compression summary. Previous goal turn completed POR151/PERM154; current turn completes SW104 after verified live wait and confirms POR341 PID78160. Remaining three repairs, native scans and canonical723556-row OOF/final provenance audit; serial cap1 and eight-stage recipe unchanged. No per-row/stratum prediction selection or new GPU trial.

V370 seed_20261013/fold4/POR341 terminal. Completed stages6/8 scores{'POR': 79.83274986803075, 'PERM': 91.30387186022239, 'SW': 74.67057243794171, 'TOTAL': 82.04088046476666}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_POR341_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing540.849s (solo after user requested serial).

V370 seed_20261013/fold4/PERM336 terminal. Completed stages7/8 scores{'POR': 79.83274986803075, 'PERM': 91.30387108958806, 'SW': 74.67057243794171, 'TOTAL': 82.04088019504465}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_PERM336_TERMINAL.json. Best restored exactly. Scope: actually assembled completed-stage prefix; not complete8stage model or fullOOF. Timing131.906s (solo after user requested serial).

2026-10-02 completed-stage compression analysis: 7 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 6, "near_zero": 0, "negative": 1, "CONT_positive": 6, "CONT_negative": 1}, "POR151": {"positive": 3, "near_zero": 4, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "PERM154": {"positive": 2, "near_zero": 5, "negative": 0, "CONT_positive": 2, "CONT_negative": 0}, "SW104": {"positive": 5, "near_zero": 0, "negative": 2, "CONT_positive": 6, "CONT_negative": 1}, "POR341": {"positive": 6, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 6}, "PERM336": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 5, "near_zero": 2, "negative": 0, "CONT_positive": 5, "CONT_negative": 0}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 new seed/fold4 POR341 best29/stop43 actual+.030016998317, DEFAULT+.210278408613/CONT-.317041224310,540.848758s solo; PERM336 best0/stop14 optimizer0, actual-.000000770634 initial projection (near-zero),131.906410s solo. Seven-stage prefixTOTAL82.040880195045. SW351 then failed CUDA OOM during epoch12 forward, exit1, solo invocation140.466178s; full checkpoint epoch11/best3/SW74.674643672407/bad8 preserved. Source controller68704 and worker60132 confirmed exited; later launches stopped. Read-only disk best replay144513 exact, current frozen signature exact, original validation batch16, zero training steps,543.277MiB allocated/636MiB reserved, unchanged last-state SHA. Current driver/RAM snapshots cannot establish failure-time cause; no other project trainer. OOM_SW351_FOLD4_PREPARED.json and new guarded r5 serial recovery controller prepared/NOT launched; approval absent, old specific PERM recovery authorization not reused. Resume would preserve logical batch64/model/loss/80-pat10->14/RNG/optimizer/scheduler and max1; new failure still stops. All55 terminal head-stage rows remain separate from unfinished eighth stage. Original paired4 native scans and full723556-row OOF/final audit still pending; no full-goal achievement.


2026-10-02 user explicitly approved "重试断点续训就好了". Final seed13/fold4 SW351 recovery launched from completed epoch11 with unchanged logical batch64/config/optimizer/scheduler/RNG, serial maximum1 GPU. Controller PID89216, worker PID52244 with --resume were OS-verified; seven frozen training/evaluation source hashes unchanged. Evidence SW351_FOLD4_RECOVERY_ACTIVATION_QA.json and OOM_SW351_FOLD4_AUTHORIZATION.json. This authorizes the single failed stage and dependent native scans/final OOF only; future failure still stops, no automatic retry. Terminal score, paired4 and fullOOF remain pending.

V370 seed_20261013/fold4/SW351 terminal. Completed stages8/8 scores{'POR': 79.83274986803075, 'PERM': 91.30387108958806, 'SW': 74.67464367240689, 'TOTAL': 82.04230512710745}; real source/predecessor gains recorded in results/deep_learning/two_seed_fullcv_v370_20261002/fold_4/seed_20261013_SW351_TERMINAL.json. Best restored exactly. Scope: actually assembled complete8stage single-fold model; not fullOOF. Timing206.750s (failed solo invocation plus approved serial recovery).

2026-10-02 completed-stage compression analysis: 8 completed8-stage model/fold cases, actual adjacent changes only. Stage counts {"POR113": {"positive": 7, "near_zero": 0, "negative": 1, "CONT_positive": 6, "CONT_negative": 2}, "POR151": {"positive": 3, "near_zero": 5, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "PERM154": {"positive": 3, "near_zero": 5, "negative": 0, "CONT_positive": 3, "CONT_negative": 0}, "SW104": {"positive": 6, "near_zero": 0, "negative": 2, "CONT_positive": 7, "CONT_negative": 1}, "POR341": {"positive": 7, "near_zero": 1, "negative": 0, "CONT_positive": 0, "CONT_negative": 7}, "PERM336": {"positive": 5, "near_zero": 3, "negative": 0, "CONT_positive": 1, "CONT_negative": 4}, "SW351": {"positive": 6, "near_zero": 2, "negative": 0, "CONT_positive": 5, "CONT_negative": 1}}. Historical fifth not pooled due differing budgets. No causal lossless8->5 proof or new training. Future controlled fifth priority documented in STAGE_COMPRESSION_ANALYSIS.md; current original+seed13 fullCV continues serially. Pending full grouped OOF and actual uniform pruning trials afterwards.

V370 fold4 complete, 144513 rows. Fixed TOTAL82.152477252078; foldwise tuned diagnostic TOTAL82.220920308922; weights{'POR': {'weight_original': 0.32, 'weight_seed13': 0.6799999999999999, 'score': 80.07835869745401, 'fixed_weight_score': 79.92732342144706}, 'PERM': {'weight_original': 0.42, 'weight_seed13': 0.5800000000000001, 'score': 91.46634406482754, 'fixed_weight_score': 91.46634043927916}, 'SW': {'weight_original': 0.45, 'weight_seed13': 0.55, 'score': 74.81197793427448, 'fixed_weight_score': 74.74588877684619}}. Exact native disk replay and output readback; no rowwise/stratum choice.

V370 completed paired fold4: 144513 rows; original 81.84683474229898, seed13 82.04230512710745, fifth-fixed ratios 82.15247725207799, fold-tuned diagnostic 82.2209203089219. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall 60.21754193305969s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs [], future cap1. This verifies one fold, not full OOF; fifth selection boundary disclosed. Evidence FOLD4_PAIRED_NATIVE_QA.json, FOLD_RESULTS.md and FUSION_STRATA.json. Remaining paired folds []; final723556-row OOF audit remains separate; existing deliveries unchanged.


2026-10-02 V370 final verified result: fixed original grouped723556-row OOF 82.16741757218335 (POR79.829021599653/PERM90.751217660625/SW75.587956888768), old submitted81.89101929722543 delta+0.27639827495792. Fold4 fixed82.152477252078, tuned82.220920308922; complete fold-tuned diagnostic82.194651448538. Canonical coverage/score/native uniform grid audit and all remaining actual checkpoints/configs/load traces/Q/source hashes passed; direct training-source inspection recorded. Final SW351 explicit epoch11 recovery completed next12 through17, best3, exit0; no current trainer/controller. Fifth-selected pair/fixed ratios remain disclosed; tuned OOF is diagnostic only. Each model8 stages; ensemble16 historical stages (accepted two-seed scope), no8-total claim. Old deliveries untouched, new submission package not prepared in this batch. Result and limitations: results/deep_learning/two_seed_fullcv_v370_20261002/FINAL_SUMMARY.md. Future optional independent pruning/generalization work remains unstarted.


2026-10-02 submission导出10套完整图完成；跨井批次组织修正后，全部723556行三个通道预测逐元素复现82.19465144853783。模型参数及配比未改变。当前生成测试10井结果，待独立解压CLI读取隔离及最终ZIP验证。


2026-10-02 用户指定82.194651诊断版本submission已交付到新目录 `submission_pure_dl_v370_82_194651_20261002`。完整推理图、各折逐头比例、result.zip及submission_code.zip；完整OOF模型重放和独立解压10井95948行推理/字节一致/读取隔离/ZIP核验通过。旧交付不改，每模型8阶段、配对16历史阶段，测试精度未知。结果见新目录delivery_validation.json。


2026-10-02???????????/????50%????????????723556?OOF82.174383260017???????82.167417572183?0.006965687834????80.840351605081???????101?????0.50?????????????????????OOF????????????????????submission_pure_dl_v370_equal_82_174383_20261002???????????fixed??????????????????????????????

2026-10-02 ????????submission_pure_dl_v370_equal_82_174383_20261002?OOF82.174383?10?95948????????/??/??/ZIP/manifest????????

2026-10-02 A?????v97 OOF81.891019?A?83.93????v370 OOF82.174383?A?83.61???????A???0.32?????30???????state_dict?????POR341/PERM336/SW351??checkpoint?????8?????best_epoch=0?????????ZIP/??ZIP SHA???????????10?95948??????result.json??????????OOF????A???????????v97?????A???????????????????????????????TERMINAL_MODEL_AUDIT.json?

2026-10-02 ???????????????submission_pure_dl_v370_original_81_989553_20261002???8??????????15????????????20%???OOF723556???????????81.989553???10?95948????????/??/??/ZIP/manifest???A???????????????18??????????

2026-10-02 ???????v97?????v370???????/?bank???????/feature/utils/window???????SHA?????256/256/1024?stride128?800???????????????????????????????15?????????JSON????{'rows': 95948, 'changed': [11, 19, 0], 'max_abs': [4.560707000000001, 0.030282999999999838, 0.0], 'targets': ['POR', 'PERM', 'SW']}????????????????????????????????A?0.32??????????????_process_comparison?

2026-10-02 ???submission_pure_dl_v370_original_perwell_20261002???v97????????????????????????20%?????????????10?95948???CLI??????????????/??/??/ZIP/CRC/SHA?????????OOF81.989553??????????OOF????A???????

2026-10-02 base-only????????/???../submission_pure_dl_v370_base_ablation_20261002/README.md???????15base???????OOF????

Base-only??????81.937732???81.989553?-0.051821???????????????OOF???????A?????????submission_pure_dl_v370_base_ablation_20261002?????15base??????POR??t3/Cauchy?????10?95948?????/ZIP?????????results/deep_learning/base_only_interface_ablation_20261003/README.md?A???????

2026-10-07 GitHub交付：更新README、七版逐折成绩及A榜反例、六版推理源码镜像；保留远程审计，排除权重、逐行数据及可重建图谱。待完成：合并/提交/推送/远程核验。中文输出已显式UTF-8。
