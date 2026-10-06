# POR conditional-RPS v339

## Exploration focus
Hypothesis: joint atom+conditional CDF supervision on trueDEFAULT rows may shift fresh density toward default decisions at the expense of continuous numeric precision. Original151 gate/prior remain frozen, fresh33564curveattention/raw48/readout all jointly train. No trained335/302 repair weights, no new parameters (128336 active), onePORaddedphase/max8. This changes learning of encoder together with readout, not head-only tuning.

## Predefined control and verification
Single candidate replaces ONLY mixed-prediction jointRPS with conditionalCDF RPS on truecontinuous rows. Original finite255edges, Qweighted all-row mean/3 unchanged. Mixed likelihoodCE, directcomponent hard/softCE, losscoefficients, .25alpha/5ramp, data/seed/decoder unchanged. Default rows zero in RPS, not fake continuous labels. ConditionalRPS removes frozen gate attenuation; its effective continuous gradient differs even with same global normalization. This is not a clean pure default-gradient-only ablation or normalization-matched gradient comparison. Historical335 same architecture/init control, best28/natural42; predefine bestthrough42 and separately terminal80/14. No repeated baseline fullrun/preflight/review; ordinary initial eval/checkpoint selection/replay checks.

## Gains
Prepared, UNTRAINED, no gain. Historical335 whole78.042212635/CONT41.724510830; original151CONT41.953961055 higher, report both. Retained8 TOTAL80.703440 unchanged. A positive CONT alone is diagnostic; adoption requires ownwhole-best simultaneously whole andCONT positive vs335 and actual fixed-target combination. Single-seed microgain not robust/generalization. Strict808/fullOOF82 unmet.

## Limits and closure
Default Bayes decisions can change despite frozen gate: conditional density affects joint decoder utility, so no defaultperformance guarantee. SW252 has conditional-vs-jointRPS but different target/architecture; originalPOR151jointRPS was positive on another head. Historical exactsame POR mechanism lookup is pending; if exact negative found, hold/close rather than repeat. No RPScoefficient/LR grid, no mixed two-checkpoint selection or roworacle, no validation-label reweighting. No newmodule/chaining/teacher/external data.

## Completed, planned and not done
Source/PLAN/local syntax prepared. Research pending. No GPU launch or queue;338depth/control occupy slots. If historical evidence allows, explicit later launch on a free slot; record trajectories/wholeDEFAULTCONT/timing and terminal best. No otherfolds/delivery edits.

Historical research completed: [conditionalRPS evidence](../cross_domain_research_20261001/POR_CONDITIONAL_RPS_RESEARCH_20261002.md), no exact335/287 negative found in limited lookup. Generic analyticCONTCRPS already existed includingPOR; not firstconditional supervision. Implementation retains globalrowmean and original mixedF; source syntax prepared only, UNTRAINED/noGPU/queue. Planned explicit solo launch after338slots clear: observed338depth solo~12.6s vs dual~30s per epoch suggests this workload benefits little from2concurrency; not isolated complete-work benchmark. No changes to active338source/budget.

Explicit solo v339 launch after both338terminal, PID/start inTIMING.json. Initial candidate unscored, no340queue/nootherfolds. Historical335 jointRPS control retained; newoptional8 uses338depth but thisloss-only isolated candidate stillfromfresh335/original151, notrained338weights.

Verifiedlive PID36304: {"status": "running", "pid": 36304, "stage": "conditional", "epoch": 8, "best": 78.02258667724234, "best_epoch": 6, "bad": 2, "peak_MiB": 2074.64794921875}. Initialized77.926380760724/128336active, solo firstepochs11.9-12.1s/peak2074.648MiB/noOOMobserved. No finalgain/CONTclaim yet. Currentretained8 actual338recipe80.704522967002, source/control335preserved; candidateisolatesRPS and doesnotload338weights. Atlasadoption/prepared340 updated and built.

V339 final POR78.042499459539, gain+0.116118698815, actualTOTAL80.647311027638; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds506.354363/peakMiB2074.647949. No otherfolds.

V339 terminal natural42/best28 whole78.042499459539/CONT41.725335618872, DEFAULT97.405597888869 unchanged335. Samecapacitymechanismdelta vs335whole+.000286824537/CONT+.000824788733, native151CONTstill-.228625436033; lowerthanretained338whole-.003322594458, no currentrecipe replacement. Tiny single-seed/matchedcapacity historicalcomparison notrobustgain; sourcecode/positivecheckpoint preserved, maycarrycode only intoisolatedjointfollowup nottrainedweights. RAM144665replay/frozenexact, solo506.354s/peak2074.648MiB inclprep/warmup/restore, actual42prefix saved. Inheritedsnapshot phrasecapacitydiffers isnottruefor339/335; humanFINALclarifies. Nootherfolds.
