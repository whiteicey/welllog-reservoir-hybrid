# Input-context cross-attention repair
Focus: learned154 readout is pointwise (stored depthkernel collapses to one tap); add pointqueries over16 pooled same-well input-feature memory tokens, retaining original learned prediction. Inspired by CATS cross-attention idea, not its horizonquery model reproduction: https://arxiv.org/abs/2405.16877.
Explored: prepared/untrained; only new LayerNorm/QKV16/output32 trains, all154/source weights frozen. Zero output preserves initial function. Modify bounded location0.1referenceSD and logweight0.1 only; gate/scale retained. No historical target labels or predicted teacher targets; only original frozen input hidden128. Exact masks exclude invalid pooled tokens.
Gain: unproved. Primary control original154 PERM89.40556485691894, actualTOTAL80.61247541799393. Single-fold diagnostic only. Not a matched-capacity attention-versus-MLP claim.
Not worth continuing:197curvecore failed and no further core sweep;204 tests position-to-context connectivity rather than same core mechanism. No new fullbackbone or oldprediction voting. Alternatives202/204 both warm154 and replace each other; never stack, one additional PERMstage, six total alone/seven with POR203; default remains5.
Budget:80/14/min10/b64/64steps ownPERM NLL/BCE/jointRPS1,LR1e-4 normalGPU max2. Atomicstates/frozen signatures/144665 replay and actualassembly.
Timing: record loading/training/eval and peak, real overlap/totalbatchwall; no whole-recipe speed claim from isolated patch timing.

V204 failed after GPU admission during model construction due inherited capture method name collision, before epochs/initial score; no efficacy result. Preserve source/traceback. V205 fresh source only renames capture_context, same planned module/loss/data, explicit launch, not queue/retry of frozen file.

