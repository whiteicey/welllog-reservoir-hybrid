"""CPU terminal paired evidence; retain predictions before checkpoint cleanup."""
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
reports={};strata={};selected={}
for arm in ('scalar','conditional'):
    out=ROOT/f'outputs/pure_dl_perm_adaptive_share_v298_fold5/{arm}'
    assert (STUDY/f'{arm}_assembled.json').exists()
    reports[arm]=read(out/'report.json');assert reports[arm]['predictions_reproduced_exactly'] and reports[arm]['frozen_parameters_exact']
    pred=truth.copy();pred[:,1]=np.load(out/'channel_pred.npy')
    strata[arm]=dict(PERM=base.metric(truth,pred)['PERM'],default_PERM=base.metric(truth[default],pred[default])['PERM'],continuous_PERM=base.metric(truth[~default],pred[~default])['PERM'])
    saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)['state_dict']
    selected[arm]=dict(nominal_ramp=float(saved['mix_alpha']),share_bias=float(saved['share.bias'][0]),share_weight_norm=float(saved['share.weight'].float().norm()))
    if arm=='scalar':selected[arm]['actual_selected_scalar_alpha']=float(saved['mix_alpha']*(1+torch.tanh(saved['share.bias'][0]/.3)))
control=ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5/point/channel_pred.npy'
pred=truth.copy();pred[:,1]=np.load(control)
strata['retained291']=dict(PERM=base.metric(truth,pred)['PERM'],default_PERM=base.metric(truth[default],pred[default])['PERM'],continuous_PERM=base.metric(truth[~default],pred[~default])['PERM'])
timing=read(STUDY/'BATCH_TIMING.json')
finish=max((ROOT/f'outputs/pure_dl_perm_adaptive_share_v298_fold5/{a}/report.json').stat().st_mtime for a in reports)
timing.update(terminal=True,total_paired_wall_seconds=finish-min(timing['scalar_start_epoch'],timing['conditional_start_epoch']),individual_seconds={a:r['seconds'] for a,r in reports.items()},peak_MiB={a:r['peak_MiB'] for a,r in reports.items()},scope='Partialoverlap/stagger with other tasks; includes data/initial/train/finalrestore, excludes codeprep/CPUassembly; not speed benchmark')
result=dict(reports=reports,strata=strata,selected_share=selected,conditional_minus_scalar=reports['conditional']['target_score']-reports['scalar']['target_score'],conditional_minus_scalar_continuous=strata['conditional']['continuous_PERM']-strata['scalar']['continuous_PERM'],conditional_minus_retained291_continuous=strata['conditional']['continuous_PERM']-strata['retained291']['continuous_PERM'],scalar_minus_retained291=reports['scalar']['target_score']-89.5155547932572,active_capacity_difference=48,adopted=False,remaining_folds_started=False,complete_five_folds=False,timing=timing,conclusion='Overall underperformance prevents adoption, but conditional continuous stratum improves versus scalar and retained291; default action loss offsets it despite frozen gate. Preserve conditional as continuous-positive mechanism control. Close allocation-bound/LR grids; future fresh joint state+density encoder may test this tradeoff, not automatically launched.')
for name,value in [('PAIRED_FINAL.json',result),('STRATA_DIAGNOSTIC.json',strata),('BATCH_TIMING.json',timing)]:
    (STUDY/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
note=f"\n\n## Final paired negative\nScalar PERM{strata['scalar']['PERM']:.12f}, conditional{strata['conditional']['PERM']:.12f}; conditional-minus-scalar{result['conditional_minus_scalar']:+.12f}, scalar-minus291{result['scalar_minus_retained291']:+.12f}. Continuous scalar{strata['scalar']['continuous_PERM']:.12f}/conditional{strata['conditional']['continuous_PERM']:.12f} versus291{strata['retained291']['continuous_PERM']:.12f}. Actual selected scalarshare{selected['scalar']['actual_selected_scalar_alpha']:.12f}, nominal .3 is not actual alpha. Newbody/readout/allocation grads nonzero, original154 frozen/RAMbest replay/144665actualassembly. No adoption: close this early adaptive allocation setting, retain original291 and earlier candidates. Pairwall{timing['total_paired_wall_seconds']:.3f}s, individual scalar{reports['scalar']['seconds']:.3f}s/conditional{reports['conditional']['seconds']:.3f}s; partial overlap, no speedclaim. Redundant negative weights eligible for manifest cleanup after this record; compact predictions/source/config evidence retained.\n"
for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps({k:v for k,v in result.items() if k not in ('reports','timing')}))
