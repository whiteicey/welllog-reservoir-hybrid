"""Prepare isolated gradient-ownership transfer, reuse valid historical control."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/sw_density_owned_encoder_v301_20261002';STUDY.mkdir(parents=True,exist_ok=True)
path=CODE/'train_sw_density_owned_encoder_v301.py'
if path.exists():raise RuntimeError('Refuse overwrite')
train=(CODE/'train_sw_state_bound_v289.py').read_text(encoding='utf-8').replace('sw_state_bound_v289','sw_density_owned_encoder_v301')
train=train.replace("'sw_joint_share_v285.py'","'sw_state_bound_v289.py','sw_joint_share_v285.py'")
train=train.replace('best_pred,best=evaluator(model);initial=best;', 'best_pred,best=evaluator(model);assert abs(best-74.12175268976877)<1e-3;initial=best;')
path.write_text(train,encoding='utf-8')
record=(CODE/'record_sw_state_bound_v289.py').read_text(encoding='utf-8').replace('sw_state_bound_v289','sw_density_owned_encoder_v301').replace('V289','V301')
(CODE/'record_sw_density_owned_encoder_v301.py').write_text(record,encoding='utf-8')
plan=json.loads((ROOT/'results/deep_learning/sw_state_bound_v289_20261002/PLAN.json').read_text(encoding='utf-8-sig'))
plan.update(candidates=['density_owned'],primary_control='historical289 exact same architecture/init/loss/budget but stategradient flows versus detached newbody stateinput',trainable_parameters=57617,reference_SW289=74.1537264289993,reference_CONT289=17.72012680313723,reference_CONT252=17.88486813991017,forward_capacity_same=True,new_state_input_detached=True,only_original104_checkpoint=True,prepared_not_launched=True)
(STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(STUDY/'AGENTS.md').write_text('Original104 frozenprior only; fresh newraw encoder/readout/state all train in oneSWstage. Detach only state-input gradient into newencoder, statehead trainnormal; continuousCE/RPS still jointlytrain encoder and density. Same original9curves206+relative/Q50/split/score/actions/80-14/seed/max2GPU/max8phases. Reuse complete289 historical control, no new warmrepairweights or redundant preflight. Maintain README/PROJECTS/EXPLORATIONS/atlas, preserve launched source/deliveries.\n',encoding='utf-8')
(STUDY/'README.md').write_text('''# SW density-owned raw representation v301

## Focus / evidence
Recent newraw joint state/density encoders mainly increase default score while continuous deteriorates: 289continuous17.72013, 297well17.50275, 296pooled17.46917, 299hierarchical17.40412, versus frozen-feature25217.88487. This does not prove gradient conflict, but motivates isolating gradient ownership rather than more width/normalizer sweeps.

EXACT original289 code/body/readout/init/loss/share.25ramp5/statebound4/balancedBCE/directcontinuousCE1/conditionalRPS1/80-14, ORIGINAL104 checkpoint only. Fresh207->48 rawencoder, three pointMLP branches/fuse, direct256density and49state residual train in ONE addedSWstage. Only change: gate_patch receives h.detach(). Stateweights still learn BCE on continually updated rawfeatures; encoder and density still train jointly via continuous mixtureCE/RPS/directcomponentCE. No losses/gates/actions removed or frozen, no inference function/value change from detach alone, no newparams. This can reduce representation competition, but does not guarantee actual gradients conflicted or defaultprediction remains unchanged after training.

## Explored and validation
Prepared/untrained. Historical completed289 is same-capacity/configuration control, not concurrentpaired rerun; original286/284 failures not controls. Only old104 weights loaded, no289/297/299repair weights. Ordinary initialsourcefunction check, allnew optimizergroup gradients/frozenproof/atomic laststate; ownWHOLESW officialbest, not continuous-selected. Existing frozen helper hashes included. Actual144665 composition and strata after terminal; RAMbest replay not independentdiskreload. Prior v11 splitTCN differs from this sharedfeature gradient-routing intervention.

## Gain
Unproved. Report wholeSW-minus289 and continuous-minus289/252, and absolute297overall74.16003022301975. Positive local gain may be carried into a fresh isolated jointstage; only complete useful actualcombination strictly>80.8 qualifies otherfolds. Current optional8/default5/oldfullOOFdelivery retained. Max8wholemethod, oneaddedSWphase, no teacher/testlabels/extra inputs/oracle.

## Unworthy followup
If no meaningful continuous benefit and overall no gain, close this detached-state representation setting, no detachratio/LR/BCEweight grid. Detached changing features can harm state fit, and encoder may still lack discriminative continuous inputs. A failed trial cannot prove all multitask methods fail. Frozen-feature252 is an absolute reference, not samearchitecture fairness control.

## Timing
Individual data/initial/train/finalrestore wall+peak; historical289 wall measuredunder different contention, no speedranking. Codeprep/CPUassemblyexcluded; maximum2 not assumedoptimalparallelism.
''',encoding='utf-8')
(STUDY/'PROJECTS.md').write_text('Goal: real continuous representation gain with same forwardcapacity and onejointstage. Done: evidence-based gradient-routing protocol/source. Pending: explicit GPUlaunch after slot; officialbest assembly/strata/timing and atlas. OriginalfullOOF82 unmet; usefulstrict808 before otherfolds.\n',encoding='utf-8')
print(str(STUDY))
