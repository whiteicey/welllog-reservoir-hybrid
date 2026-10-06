"""Prepare scoped joint state/density control after v298 measured tradeoff."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];CODE=Path(__file__).resolve().parent
STUDY=ROOT/'results/deep_learning/perm_joint_state_share_v300_20261002';STUDY.mkdir(parents=True,exist_ok=True)
path=CODE/'train_perm_joint_state_share_v300.py'
if path.exists():raise RuntimeError('Refuse overwrite')
train=(CODE/'train_perm_adaptive_share_v298.py').read_text(encoding='utf-8').replace('perm_adaptive_share_v298','perm_joint_state_share_v300')
train=train.replace("'perm_sparse_reacquisition_v291.py'","'perm_adaptive_share_v298.py','perm_sparse_reacquisition_v291.py'")
path.write_text(train,encoding='utf-8')
record=(CODE/'record_perm_adaptive_share_v298.py').read_text(encoding='utf-8').replace('perm_adaptive_share_v298','perm_joint_state_share_v300').replace('V298','V300')
(CODE/'record_perm_joint_state_share_v300.py').write_text(record,encoding='utf-8')
plan=json.loads((ROOT/'results/deep_learning/perm_adaptive_share_v298_20261002/PLAN.json').read_text(encoding='utf-8-sig'))
plan.update(candidates=['fixed_gate','joint_gate'],primary_control='registered zero49param statehead frozen versus trained, samebody/readout/share init; activecapacitydifference49 is deliberate update scope',trainable_parameters={'fixed_gate':36515,'joint_gate':36564},extra_active_parameters={'fixed_gate':0,'joint_gate':49},new_state_bound=2.,no298trained_weights=True,prepared_not_launched=True)
(STUDY/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
(STUDY/'AGENTS.md').write_text('Original154 checkpoint only; no298trainedweights. Fresh sparsebody/readout/conditionalshare plus statehead trainjointly in ONE addedPERMstage. Fixed0 vs jointstate update49paramdifference disclosed, same original ownBCE/NLL/RPS; no changedscore/decoder/defaultthreshold. Max2GPU/normal/80-14/min10/original9curves206+relative/Q50/fixed5, max8whole. Maintain README/PROJECTS/EXPLORATIONS/atlas; do not mutate frozen sources/deliveries or start otherfolds.\n',encoding='utf-8')
(STUDY/'README.md').write_text('''# PERM continuous/state coupling v300

## Focus / evidence
V298 conditional allocation produced continuous74.091162808666 vs scalar74.037494684582 (+.053668) and retained29174.001066339761 (+.090096), but default97.287981640665 vs29197.494555208792 lost, wholePERM89.409738 below29189.515555. Gate network frozen did not fix official action because utility combines state probability and changed continuous mass. Test whether jointly adapting state representation can retain the continuous signal while recovering this loss.

Fresh new291 sparsebody/readout and298conditional allocation code only, ORIGINAL154 checkpoint frozen only, no trained291/298repair weights. Add zero48->1 state residual2*tanh(raw/2) from same newfeatures. Fixed control registers statehead but freezes it0; joint arm trains49 additional parameters; samebody/readout/share/initialtensors/loss/budget and decoder. Active36515vs36564 difference49 disclosed as intentional update-scope comparison, not exact active-capacity ablation. Allnew body/readout/share joint3e-4 in both, optionalstate3e-4 in joint. Same own trainingBCE/Q50/equal-trainwell continuousNLL/jointRPS1/directCE1; no changed defaultthreshold/score/actions or posthoc perrow winner. Newstate gradients also shape newrepresentation.

## Explored and verification
Prepared/untrained. Ordinary initial154 replay, actual encoder/readout/share/state gradients, original154 frozen checks, atomic current/best/optimizer/RNG. Own wholePERM-best, then actual144665 assembly/default-continuous diagnostics; no stratum-specific checkpoint selection. Fixed arm should replay298conditional trajectory if deterministic same inputs, but requires actual evidence; do not assert equivalence from code alone.

## Gain
Unproved. Primary joint-minus-fixed wholePERM and stratumtradeoff; absolute retained29189.5155547932572/continuous74.001066339761 and v298continuous74.091162808666. Preserve oldmodels, only adopt complete channel candidate after actualcomposition. ONE PERMaddedphase from154, max8whole, not alpha-trained then gate-trained phases. Strictfifth>80.8/usefulgain before otherfolds; fullOOF>82 unmet.

## Not worth following
If joint state fails to recover overall score or destroys continuousgain, close this directcoupling setting rather than sweep statebounds/LR or trainingweights. Default-state and continuous objectives may interfere in the samefeatures; positive continuous diagnostics are mechanism evidence, not deliverable improvement. Prior input may still lack discriminative information. No claim all adaptive allocation fails.

## Timing
Record individual and paired totalwall incl data/initial/train/finalRAMrestore; codeprep/CPUassemblyexcluded, actual serial/overlap/gap and peakMiB, max2GPU. No redundant preflight/review.
''',encoding='utf-8')
(STUDY/'PROJECTS.md').write_text('Goal: recover defaultactionloss while retaining conditional continuousgain in onejointstage. Done: measured298tradeoff and paired fresh protocol/sources. Pending: explicit GPUlaunch once slot, ownbest actualcomposition/strata/timing; currentchampions intact. After useful808 candidate only: qualify and originalotherfolds toward OOF82.\n',encoding='utf-8')
print(str(STUDY))
