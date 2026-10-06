"""Prepare matched coarse/fine training without touching historical sources."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/sw_coarse_fine_v299_20261002';STUDY.mkdir(parents=True,exist_ok=True)
train=(CODE/'train_sw_state_bound_v289.py').read_text(encoding='utf-8').replace('sw_state_bound_v289','sw_coarse_fine_v299')
train=train.replace("'sw_joint_share_v285.py'","'sw_state_bound_v289.py','sw_joint_share_v285.py'")
train=train.replace('best_pred,best=evaluator(model);initial=best;', 'best_pred,best=evaluator(model);assert abs(best-74.12175268976877)<1e-3;initial=best;')
path=CODE/'train_sw_coarse_fine_v299.py'
if path.exists():raise RuntimeError('Refuse overwrite')
path.write_text(train,encoding='utf-8')
record=(CODE/'record_sw_state_bound_v289.py').read_text(encoding='utf-8').replace('sw_state_bound_v289','sw_coarse_fine_v299').replace('V289','V299')
(CODE/'record_sw_coarse_fine_v299.py').write_text(record,encoding='utf-8')
plan=json.loads((ROOT/'results/deep_learning/sw_state_bound_v289_20261002/PLAN.json').read_text(encoding='utf-8-sig'))
plan.update(candidates=['flat','hierarchical'],primary_control='same trained newrawbody/groupFiLM/coarse+fine readout, initial tensors; only normalization flat vs conditional',groups=8,training_group_fit='Q-weighted original64trainingwell continuous bins; enforce nonempty contiguous existingbin groups',same_original256grid_800actions=True,trainable_parameters=58777,no289_or297_trained_weights=True,prepared_not_launched=True,reference_SW289=74.1537264289993,reference_continuous252=17.88486813991017)
(STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(STUDY/'AGENTS.md').write_text('Original104 frozen prior only; fresh jointencoder/state/coarsefine readout. Trainonly Q-weighted256bin groupfit; no teacher/pretrained/testlabels/pseudolabels. Flat/hierarchical sameinitialization/params, change normalization only. Max2GPU/80-14/min10/originalfold5/9curves206+relative/Q50. OneSWaddedphase/max8, no repairweights or extra folds. Maintain README/PROJECTS/EXPLORATIONS/atlas; freeze launched source/config.\n',encoding='utf-8')
(STUDY/'README.md').write_text('''# SW coarse-to-fine continuous density v299

## Focus
New raw encoder jointly learns broad continuous regime and within-regime detail. Eight contiguous groups of the EXISTING256bins, training64wells only/Q-weighted continuous-label quantiles. Quantile cuts clamp to nonempty groups; no labels/IDs as inference features, no changed actions or groupwise oracle. Fresh48rawbody inherited289code plus49statehead, original104 prior frozen; no289/297trained weights. Protected .25mix/ramp5, statebound4/balancedBCE/directcontinuousCE1/conditionalRPS unchanged.

The shared48features receive per-group learnable48gamma/48beta FiLM bounded.25 (zero init). Shared fine256 linear outputs, coarse8 linear outputs. Hierarchical q_j=softmax(c)_group*softmax(v_within_group)_j. Flat control q=softmax(c_group+v_j-log(group_size)). Same body, registered and active parameters, complete initial tensors; change only normalization. Coarse initialweights0/biaslog(group_size), fineweights/bias0 -> uniform q in BOTH; alpha0 original104 prediction. This initialization differs289 randomreadout, so historical289 is absolute control, not strict architecture ablation. At arbitrary free logits both express a full categorical distribution; no claim hierarchy adds information or universally greater capacity. Its hypothesis is separating coarse mass competition from fine gradients, with identical group-specific feature processing in both arms.

## Explored and validation
Prepared/not launched. Ordinary initialeval checks sourcefunction, actual allnew groups trained, atomic laststate/best and groupboundaries. Boundaries saved as buffer; restored Python slices synchronized on load. Directcomponent auxiliary uses same normalized q for branch and finalfeature reads, no synthetic label signals. Each ownbest selected by wholeSW, default/continuous scores diagnostic only. Actual144665 composition after terminal; RAM replay not independent diskreload.

## Gain
Unproved. Primary hierarchical-minus-flat final wholeSW AND continuous stratum; absolute28974.1537264289993 and252continuous17.88486813991017. Preserve retained positive models/candidate29780.6935127, never adopt a worse replacement. One newSWphase from104, max8 wholemethod; strict80.8/usefulgain before remainingfolds, OOF82 unmet.

## Unworthy followup / failure conditions
If only default improves or hierarchy fails flat/retainedcontinuous, stop this8group setting, no groupcount/quantile/FiLMwidth grids. Groups can separate training modes that do not generalize across wells. Narrow groups may impede flow of softtolerance supervision across boundaries; counts may concentrate into a few bins, clamp disclosed. Original rawbody stillpointwise; hierarchy cannot restore absent input information. This is cross-domain inspired, not HCA teacher/range-preserving distillation or full original-paper reproduction.

## Timing
Individual data/initial/train/bestRAMrestore wall and peakMiB; same-work pairedwall with actual overlap/gaps; codeprep/CPUassembly excluded. Max2GPU only after authoritative admission, no redundant preflight.
''',encoding='utf-8')
(STUDY/'PROJECTS.md').write_text('Goal: substantive continuous readout gain through joint newrawencoder and coarse/fine competition. Done: matched twoarm sources/protocol. Pending: launch after GPUslot, official ownbest+actualcomposition/pairedstrata/timing, cleanup/finalatlas. Optional after useful strict808: qualify originalremainingfolds, fullOOF>82.\n',encoding='utf-8')
print(str(STUDY))
