# Historical summary

Focus and explored conditions: see retained README/FINAL and SUMMARY_RESULTS.json.

Gain: scores and original comparison deltas are retained; incomplete folds remain incomplete.

Not worth extending: original negative conclusions are retained below where available.

Detailed logs, weights and caches were removed by the user request; rerunning requires retraining.

## README.md

# V350 parallel depth reading

## Focus
Carry the local-positive345 adaptive-offset mechanism into one small wiring change: same temporal17tap and sparse5tap modules read the same firstcurve-block h; sum their residuals rather than temporal(reader(h)). No new weights or new stage. Preserve positive338 temporal mechanism,341 conditionalRPS,345 sparse reader code/init, only original151 checkpoint.

## Exploration and verification
Prepared, no score. Fresh encoder/readout/temporal/reader jointly train136626 parameters; original151 frozen, no trained341/345 repairs. All module objects and initialization match345, connection alone changes. Historical serial-adaptive345 whole78.045290238763/CONT41.737336256473 vs serialfixed78.043240611423/CONT41.729454620568: local+.002049627/+ .007881636 but below retained341 whole78.046430717932/CONT41.736640291995. Hypothesis: serial residual interaction erodes existing evidence; parallel may preserve it. This is unproven, not expected-large-gain evidence. Nonconcurrent historical control and single-seed limits explicit. Secondary parallel adaptive/fixed comparison uses same136626 physicalparams/init but selector roles/effectivecapacity differ.

## Gain
Unproven. Default5 unchanged80.612475417994; retained8 unchanged80.704705566183. Whole official target best selects checkpoint; CONT/DEFAULT reported at same best, not oracle epochs. New80epoch ho

Whole official target best selects checkpoint; CONT/DEFAULT reported at same best, not oracle epochs. New80epoch horizon/pat14/min10/b64/64steps/originalQ50/fifth split/9curves206+relative unchanged. 42prefix recorded for equal-work historical comparison, no mandatory42 limit. Source151 CONT41.953961054904 also reported. One addedPORstage, six with original PERM/SW, at most8 with retained alternatives. No otherfolds unless actual fusedTOTALstrict80.8+usefulgain.

## Not worth extending
345 absolute replacement failed but localadaptive advantage preserved; no offsetrange/width/LR sweep. This tests connection only. If whole/CONT do not offer useful same-capacity control advantage, close parallel wiring, not all sparse sampling. No repair checkpoint chains, teachers/KL, data leakage, external prediction voting, old queues.

## Timing
Explicit solo adaptive launch first; fixed prepared but no automatic queue. Max2 cap, solo preferred measured contention. Record individual preprocessing/warmup/training/restoration and process/pairwall including serial gaps separately. RAM best replay is not independent diskreload.

350 adaptive live PID69740 confirmed by CIM/record at epoch3, bestPOR77.97621358057867, best_epoch3, peakMiB2401.8046875. Initial exactly15177.926380760724. Preliminary only/no gainclaim, soleGPUtask, fixedcontrol notlaunched/noautomaticqueue, original341current recipe retained80.704705566183.

350 live snapshot PID69740 confirmed: epoch23, bestPOR78.03311426692649/bestepoch20, peak2401.805MiB, soleGPUtask. Continued improvement at20 meansno evidenceearlyclosure now. No terminal/positive claim.351SWdynamiccross-spectrum prepared552new/92841joint, notlaunched; prefer serial based previous measuredPORsolo12.6s vs dual~30s/epoch each, no new current351speed claim. No queue/otherfolds.

V350 final POR78.045046771530, gain+0.118666010806, actualTOTAL80.648075221236; sixphases inclsource/one addedPORstage, strict808=False; all144665replay/frozenexact, best/stop28/42, trainingseconds591.607444/peakMiB2401.804688. No otherfolds.

V350 terminal POR78.04504677153032/CONT41.7366361454133, best/stop28/42, parallel minusserial345 whole-0.0002434672324937992/CONT-0.0007001110596434046, vs341whole-0.0013839464020009018. Diagnostic8TOTAL80.70429038226204; adopted=False. parallel connection only, no offset-width-LR grid; does not invalidate345localadaptivepositive. Fixedarm notlaunched/no matchedgainclaim. Solo individual591.6074435999617s/peak2401.8046875MiB includesprep/warmup/restoration;144665RAMbestreplay/frozenexact/notdiskreload/fullOOF82unmet/nootherfolds.


