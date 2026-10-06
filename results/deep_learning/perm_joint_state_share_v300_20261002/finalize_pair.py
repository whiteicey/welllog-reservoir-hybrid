"""CPU paired whole/stratum evidence for joint state learning."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
truth=np.asarray(data['truth'][data['validation']]);default=base.dmask(truth)[:,1]
reports={};strata={};preds={}
for arm in ('fixed_gate','joint_gate'):
    out=ROOT/f'outputs/pure_dl_perm_joint_state_share_v300_fold5/{arm}'
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=read(out/'report.json');assert reports[arm]['predictions_reproduced_exactly'] and reports[arm]['frozen_parameters_exact']
    preds[arm]=np.load(out/'channel_pred.npy');pred=truth.copy();pred[:,1]=preds[arm]
    strata[arm]=dict(PERM=base.metric(truth,pred)['PERM'],default_PERM=base.metric(truth[default],pred[default])['PERM'],continuous_PERM=base.metric(truth[~default],pred[~default])['PERM'])
for label,path in [('retained291',ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5/point/channel_pred.npy'),('conditional298',ROOT/'outputs/pure_dl_perm_adaptive_share_v298_fold5/conditional/channel_pred.npy')]:
    pred=truth.copy();pred[:,1]=np.load(path)
    strata[label]=dict(PERM=base.metric(truth,pred)['PERM'],default_PERM=base.metric(truth[default],pred[default])['PERM'],continuous_PERM=base.metric(truth[~default],pred[~default])['PERM'])
exact_fixed_replay=np.array_equal(preds['fixed_gate'],np.load(ROOT/'outputs/pure_dl_perm_adaptive_share_v298_fold5/conditional/channel_pred.npy'))
timing=read(STUDY/'BATCH_TIMING.json')
finish=max((ROOT/f'outputs/pure_dl_perm_joint_state_share_v300_fold5/{a}/report.json').stat().st_mtime for a in reports)
timing.update(terminal=True,total_paired_wall_seconds=finish-min(timing['joint_gate_start_epoch'],timing['fixed_gate_start_epoch']),individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},scope='Partialoverlap/stagger/otherjob contention; data/initial/train/finalrestore incl, codeprep/CPUassembly excluded; not isolatedspeed')
result=dict(reports=reports,strata=strata,fixed_reproduces298conditional_predictions_exactly=bool(exact_fixed_replay),joint_minus_fixed=reports['joint_gate']['target_score']-reports['fixed_gate']['target_score'],joint_minus_fixed_continuous=strata['joint_gate']['continuous_PERM']-strata['fixed_gate']['continuous_PERM'],joint_minus_retained291=reports['joint_gate']['target_score']-89.5155547932572,active_capacity_difference=49,adopted=False,remaining_folds_started=False,complete_five_folds=False,timing=timing)
result['conclusion']='Small wholePERM benefit from trainedstate still below retained291; assess continuous tradeoff rather than claiming recovery. No statebound/LR grid; preserve measured local benefit as mechanism control, not adopted recipe.'
for name,value in [('PAIRED_FINAL.json',result),('STRATA_DIAGNOSTIC.json',strata),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\n\n## Final paired outcome\nFixed PERM{strata['fixed_gate']['PERM']:.12f}, joint{strata['joint_gate']['PERM']:.12f}, pairedwhole{result['joint_minus_fixed']:+.12f}/pairedcontinuous{result['joint_minus_fixed_continuous']:+.12f}, joint-minus291{result['joint_minus_retained291']:+.12f}. Strata recorded separately; no adoption or statebound/LR grid. Fixed control exactlyreplays298conditional144665preds={exact_fixed_replay}; ownbestRAM/frozen/actualassembly, no independentdiskreload. Active49parameterdifference intendedupdate-scope. Individual joint{reports['joint_gate']['seconds']:.3f}s/fixed{reports['fixed_gate']['seconds']:.3f}s, pairwall{timing['total_paired_wall_seconds']:.3f}s partialoverlap/no speedranking. Localpositivecontrols retained; checkpointcleanup for unpromising redundantstates remains policyblocked, no deletion claim. Currentoptional8/default5 intact, strict808/OOF82 unmet.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({k:v for k,v in result.items() if k not in ('reports','timing')}))
