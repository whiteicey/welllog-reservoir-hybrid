# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# SW latent task routing v331

## Focus and control
Fresh326 positive architecture from original104ONLY; three48d branches already exist. Add TWO Linear144to3 routers (870 parameters) before shared144to48 fuse. Zero logits, softmax times3 preserves original concat. Shared arm averages both logits and uses same routing for state/density; task arm uses separate logits. State path reuses SAME fuse weights by functional linear/GELU; no additional experts/private encoders. Density auxiliary four features retained. All fresh body/readout/state/FiLM/ownlatentearly/morph/routers train jointly,93159 parameters, alpha.25 unchanged. OneSWphase/6standalone/max8.

## Explored / validation
Hypothesis: separate latent branch selection can allow state and conditional-density representations to coexist. Distinct from11twoTCNfixedshare/split,315predictionallocation,301gradientdetach,304projection; those failures limit expectations. Shared control has lower effective routing capacity despite identical parameter counts; positive delta is mechanism evidence, not equal effective-function-capacity proof. No per-trial review/preflight; ordinary training initial/frozen/best RAM replay, own whole-target checkpoint. Original64/16fixedfifth/seed20261010/9curves206relative/Q50/b64x64/80pat14/cos80 unchanged, train-only context, no teacher/pseudo/testlabels/repairchain. Source104SHA retained PLAN.

## Gains

efined plus absolute comparison; save bestthrough27 during normal training, historical326natural27; full later best not equal-work evidence. Do not select by CONT epoch.

## Closure / costs
Only two named arms. If paired negative close latent task-routing setting, no widths/router/LR scan. Absolute adoption requires whole and CONT improvement. Preserve old deliveries/default5/retained8TOTAL80.6990374411772; fullOOF82 unmet. Record actual individual and pair wall-time/concurrency/memory, prep initial train restore included, implementation/finalization excluded. Max2 globalGPU at launch; explicit resume only.

331 live: shared PID5788/task PID26068 launched with global GPU cap2; initial/ordinary training evidence pending. Prepared paired CPU finalizer, no numerical preflight or separate review. OneSWphase with870 latent routers/allfreshencoderreadoutjoint, no trainedrepair weights.

Research: [MMoE-style latent routing and historical limits](../cross_domain_research_20261001/SW_LATENT_TASK_ROUTING_RESEARCH_20261002.md). This is a small task-routing adaptation, not full MMoE/PLE reproduction. Old29 includes near-concept predictive split with SWnegative; old experts froze,331latentbody joint. Shared routing effective capacity lower, shared fuse/aux density constraints still allow conflict, soft routing may collapse.

V331 shared: SW74.152268696139, actualTOTAL80.623155971788; original104only/fresh326earlylatent+morph joint encoder/readout with870 latent task-router parameters,93159 active, oneaddedSWphase. No fullOOF/otherfolds.

V331 task: SW74.151660618453, actualTOTAL80.622943144597; original104only/fresh326earlylatent+morph joint encoder/readout with870 latent task-router parameters,93159 active, oneaddedSWphase. No fullOOF/otherfolds.

## Final closure
Natural27/best13 both: sharedSW74.15226869613926/CONT18.016340491051338; taskSW74.15166061845294/CONT18.039633514083548. Task minus shared whole -0.000608077686/CONT +0.023293023032. Relative326 sharedwhole -0.017274335936/CONT +0.052898402803, taskwhole -0.017882413622/CONT +0.076191425835. No adoption, close latent task-routing with no router/LR/width scan. Routers actually changed; all144665 RAM bestreplay/frozenexact, not independent diskreload; same93159 physicalparams but shared lower effective capacity, zero state-gate delayed private gradients. Pairwall638.154386s/stagger1.195s, individuals633.705506/630.088574s, each1822.855MiB, prep initial train bestrestore included/implementation finalization excluded. Retained8TOTAL80.699037441/default5 unchanged, strict808false/nootherfolds/fullOOF82 unmet.


