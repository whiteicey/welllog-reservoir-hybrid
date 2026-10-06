"""Prepare one bounded gradient-routing transfer without re-running valid control."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/sw_density_priority_projection_v304_20261002';STUDY.mkdir(parents=True,exist_ok=True)
path=CODE/'train_sw_density_priority_projection_v304.py'
if path.exists():raise RuntimeError('Refuse overwrite')
train=(CODE/'train_sw_state_bound_v289.py').read_text(encoding='utf-8').replace('sw_state_bound_v289','sw_density_priority_projection_v304')
train=train.replace("'sw_joint_share_v285.py'","'sw_state_bound_v289.py','sw_joint_share_v285.py'")
train=train.replace('from sw_density_priority_projection_v304 import channel_objective','from sw_density_priority_projection_v304 import channel_objective,backward_with_density_priority')
train=train.replace('best_pred,best=evaluator(model);initial=best;', 'best_pred,best=evaluator(model);assert abs(best-74.12175268976877)<1e-3;initial=best;')
train=train.replace('tick=time.perf_counter();model.set_epoch(epoch+1);model.train();total=0.', 'tick=time.perf_counter();model.set_epoch(epoch+1);model.train();total=0.;projection_cosines=[];projection_conflicts=[]')
old='loss=loss_fn(output,ix,mask);assert bool(torch.isfinite(loss));loss.backward()'
new='loss=loss_fn(output,ix,mask);assert bool(torch.isfinite(loss));cosine,conflict=backward_with_density_priority(loss,output,data[\'labels\'][ix],data[\'default\'][ix],mask,data[\'quality\'][ix],groups[0][\'params\'],criterion);projection_cosines.append(cosine);projection_conflicts.append(conflict)'
# Only the ordinary first pass changes; SAM remains disabled in every stage.
train=train.replace(old,new,1)
train=train.replace('training_direct_CE=model.last_aux_ce,','training_state_density_cosine=float(np.mean(projection_cosines)),training_conflict_fraction=float(np.mean(projection_conflicts)),training_direct_CE=model.last_aux_ce,')
path.write_text(train,encoding='utf-8')
record=(CODE/'record_sw_state_bound_v289.py').read_text(encoding='utf-8').replace('sw_state_bound_v289','sw_density_priority_projection_v304').replace('V289','V304')
(CODE/'record_sw_density_priority_projection_v304.py').write_text(record,encoding='utf-8')
plan=json.loads((ROOT/'results/deep_learning/sw_state_bound_v289_20261002/PLAN.json').read_text(encoding='utf-8-sig'))
plan.update(candidates=['density_priority'],primary_control='historical289 exact original forward/init/loss/80-14; newbody stategradient projected only when conflicts with density',trainable_parameters=57617,new_parameters=0,no289_or301_trained_weights=True,only_original104_checkpoint=True,prepared_not_launched=True,reference_SW289=74.1537264289993,reference_CONT289=17.72012680313723,reference_CONT252=17.88486813991017)
(STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(STUDY/'AGENTS.md').write_text('Original104 checkpoint only; freshrawencoder/state/density jointonephase, same289forward/capacity/loss. Project only conflicting stategradient on sharednewbody, densitypriority/privateheads unchanged. No validation gradients, teachers/extrainputs/roworacle. Original9curves206+relative/Q50/split/score/seed/80-14/min10/max2GPU/max8. Historical289control reused; document oldPCGrad negatives and extraautogradcost. Maintain README/PROJECTS/EXPLORATIONS/atlas; no frozen sources/deliveries/otherfolds edits.\n',encoding='utf-8')
(STUDY/'README.md').write_text('''# SW density-priority shared gradient projection v304

## Focus / prior evidence
301 detached all state gradients into newrawbody and lost SW-.004822/continuous-.225620 versus289. Instead preserve aligned state supervision and project only its conflicting component: gs'=gs-min(dot(gs,gd)/||gd||²,0)*gd; sharedgradient=gd+gs'. Densitytask is mixedCE+conditionalRPS+directcomponentCE, statetask exactlytrainQcalibratedbalancedBCE/3. Private gate/density heads retain original total-loss gradients, original forward/mainlossvalues/optimizer/schedule/inference unchanged. Allnew encoder/readout/state jointlytrain in ONE addedSWphase from ORIGINAL104, no289/301weights, same57617capacity/init/ .25share/statebound4/80-14.

Not first PCGrad: posterior_pcgrad_loss projected three expert targets randomly; PERMv68 BCE/NLL symmetricprojection had about43.4% conflict but fullscore88.827250 unchanged/failedadoption. CurrentSW275->289 uses summedloss; this is a single asymmetric state->density gradient-priority migration, not standard random/symmetric PCGrad reproduction, not more expressive network. Historical289 completeconfiguration reused as control, not concurrentlyrerun. A graph-gradient decomposition includes all losses; projection restricted to group0 newrawstem/branches/fuse, no frozenprior parameters.

## Explored and validation
Prepared/untrained. Ordinaryinitial104 replay; original losses finite, separatebody gradnorm/dot finite, allnewgroups active/frozenchecks/atomicstates; training64steps records mean cosine/conflictfraction (trainlabels only). Extra two autograd.grad calls retain graph before ordinarybackward, then sharedgrad replacement before sameclip2/AdamW. Adam preconditioning means Euclidean nonnegative taskdot is not a guarantee on actualparameter-update task progress. No additionalGPUgradientpreflight, no validationlabel gradient or conflict-based sample selection. OfficialwholeSW-best restored; actual144665 assembly/strata afterterminal, no stratumcheckpointselection. RAMreplay not independentdiskreload.

## Gain
Unproved. WholeSW-minus289 and continuous-minus289/252, absolute297overall74.16003022301975. If conflict rare, mechanism scarcelytested, report uninformative ratherthan tuningthreshold to invent activation. Currentoptional8/default5/oldfullOOFdelivery intact. StrictfifthTOTAL>80.8 usefulgain before originalremainingfolds, OOF82 unmet.

## Unworthy followup
If conflictprojection materiallyactivates yet continuous/overall fail, close this directedprojection, no priority/threshold/BCEweight grid. State features can help density even with locally opposed gradients; perbatch noise versus generalization conflict differ. Negative oldPCGrad/currenttrial not family-wide proof. Preserve active/key/control weights; redundantnegative cleanup presentlypolicyblocked.

## Timing
Measure individualdata/initial/train/finalrestore and peak, including extraautogradwork; record epoch time against historical289 with contention difference, no speedranking or claimedfree projection. Max2GPU; codeprep/CPUassemblyexcluded.
''',encoding='utf-8')
(STUDY/'PROJECTS.md').write_text('Goal: useful continuous gain preserving alignedstate supervision in onejointstage. Done: oldPCGrad/301 evidence and asymmetricprotocol/source. Pending: explicitlaunch afterGPUslot, actualgradient/conflict/wholebestassembly/strata/time, atlas. After useful808 only: qualify otherfolds towardOOF82.\n',encoding='utf-8')
print(str(STUDY))
