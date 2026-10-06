"""CPU-only whole-seed diagnostics and equal-work wall times."""
from pathlib import Path
import json, sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask
STUDY=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'
STAGES=('SOURCE','POR113','POR151','PERM154','SW104','POR341','PERM336','SW351')
REFERENCE={'POR':41.736640291995144,'PERM':74.00963708319706,'SW':17.979399150046632}
values=[]
for complete_path in sorted(OUT.glob('seed_*/COMPLETE_MODEL.json')):
    own=complete_path.parent
    result=json.loads(complete_path.read_text(encoding='utf-8-sig'))
    existing=own/'CPU_DIAGNOSTIC.json'
    if existing.exists():
        values.append(json.loads(existing.read_text(encoding='utf-8-sig')))
        continue
    data=torch.load(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=np.asarray(data['truth'][data['validation']]);del data
    pred=np.load(own/'complete_model_pred.npy')
    assert pred.shape==truth.shape==(144665,3) and np.isfinite(pred).all()
    scores=metric(truth,pred)
    assert all(abs(scores[k]-result['scores'][k])<1e-9 for k in scores)
    default=dmask(truth)
    strata={}
    for c,target in enumerate(('POR','PERM','SW')):
        strata[target]={}
        for label,mask in [('DEFAULT',default[:,c]),('CONT',~default[:,c])]:
            strata[target][label]=dict(rows=int(mask.sum()),score=metric(truth[mask],pred[mask])[target])
        strata[target]['CONT_delta_vs_historical']=strata[target]['CONT']['score']-REFERENCE[target]
    def phase(s):return own/('SW351_fixed' if s=='SW351' else s)
    timings={s:json.loads((phase(s)/'TIMING.json').read_text(encoding='utf-8-sig')) for s in STAGES}
    failed=own/'SW351/TIMING.json'
    failed_seconds=json.loads(failed.read_text())['process_wall_seconds'] if failed.exists() else 0.
    full_wall=complete_path.stat().st_mtime-timings['SOURCE']['started_epoch']
    successful_sum=sum(v['process_wall_seconds'] for v in timings.values())
    prefix=np.column_stack([np.load(own/s/'channel_pred.npy') for s in ('POR151','PERM154','SW104')])
    prefix_scores=metric(truth,prefix)
    value=dict(seed=result['seed'],scores=scores,strata=strata,
               own_five_stage_prefix_scores=prefix_scores,
               added_three_phases_gain=scores['TOTAL']-prefix_scores['TOTAL'],
               full_model_wall_seconds=full_wall,
               sum_successful_phase_process_seconds=successful_sum,
               failed_attempt_seconds=failed_seconds,
               gaps_and_assembly_seconds=full_wall-successful_sum-failed_seconds,
               stage_timings=timings,concurrency='serial seed-stage execution; temporary concurrent read-only GPU evaluations if listed in addenda',
               concurrency_addenda=[json.loads(p.read_text(encoding='utf-8-sig')) for p in STUDY.glob(f'SEED{result["seed"]}_*_CONCURRENCY_ADDENDUM.json')],
               includes='train-only preprocessing, warmup, training, restoration, failure/repair gaps, complete-model CPU assembly',
               no_other_seed_weights=True,total_training_stages=8,
               complete_five_folds=False,remaining_folds_started=False,
               independent_disk_reload=False,
               limits='Same fifth used for selecting five whole-model seeds; historical budget differs. CONT diagnostics use the SAME selected whole-target checkpoints, no CONT epoch/head oracle.')
    existing.write_text(json.dumps(value,indent=2),encoding='utf-8')
    values.append(value)
rank=sorted(values,key=lambda x:x['scores']['TOTAL'],reverse=True)
summary=dict(completed_seeds=len(values),planned_seeds=5,complete=len(values)==5,
             ranking=[dict(seed=v['seed'],scores=v['scores'],strata=v['strata'],
                           own_five_stage_prefix_scores=v['own_five_stage_prefix_scores'],
                           added_three_phases_gain=v['added_three_phases_gain'],
                           full_model_wall_seconds=v['full_model_wall_seconds']) for v in rank],
             complete_five_folds=False,remaining_folds_started=False)
(STUDY/'SEED_DIAGNOSTICS.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
