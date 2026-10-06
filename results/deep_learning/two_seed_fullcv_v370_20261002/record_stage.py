"""CPU-only newly completed stage gains; no incomplete model/OOF claim."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import psutil
import torch

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from fullcv_pair_context_v370 import STAGES,BASE_SEEDS,own_dir,write,sha
from train_invariant import metric,TS,dmask

identity,fold,stage=sys.argv[1],int(sys.argv[2]),sys.argv[3]
assert identity in BASE_SEEDS and fold in range(1,5) and stage in STAGES
own=own_dir(identity,fold);out=own/stage;destination=STUDY/f'fold_{fold}'/(identity+'_'+stage+'_TERMINAL.json')
if destination.exists():raise RuntimeError('Already recorded unchanged terminal stage')
report_path=out/(f'fold_{fold}_report.json' if stage=='SOURCE' else 'report.json')
report=json.loads(report_path.read_text());timing=json.loads((out/(stage+'_TIMING.json')).read_text())
assert timing['terminal'] and timing['exit_code'] in (0,None) and not psutil.pid_exists(timing['pid'])
if stage=='SOURCE':assert (out/'source_report.json').exists() and report['best_checkpoint_predictions_reproduced']
else:assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
descriptor=torch.load(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
shared=torch.load(descriptor['shared_input_path'],map_location='cpu',weights_only=False)
data=shared['data'];truth=data['truth'][data['validation']];rows=len(truth)
assert rows==report.get('validation_rows',rows)
pred=np.load(own/'SOURCE'/f'fold_{fold}_pred.npy');assert pred.shape==truth.shape
before=metric(truth,pred);stage_gains=[];completed=[]
for s in STAGES:
    parent=own/s;rp=parent/(f'fold_{fold}_report.json' if s=='SOURCE' else 'report.json')
    tp=parent/(s+'_TIMING.json')
    if not rp.exists() or not tp.exists():break
    terminal=json.loads(tp.read_text());assert terminal['terminal'] and terminal['exit_code'] in (0,None)
    completed.append(s)
    if s=='SOURCE':
        if stage=='SOURCE':break
        continue
    rr=json.loads(rp.read_text());c=TS.index(rr['target']);old=pred[:,c].copy()
    channel=np.load(parent/'channel_pred.npy');assert channel.shape==(rows,) and np.isfinite(channel).all()
    previous=metric(truth,pred)[TS[c]];pred[:,c]=channel;current=metric(truth,pred)[TS[c]]
    assert abs(current-rr['target_score'])<1e-9
    default=dmask(truth)[:,c];strata={}
    for name,mask in [('DEFAULT',default),('CONT',~default)]:
        yy=truth[mask];p1=yy.copy();p2=yy.copy();p1[:,c]=old[mask];p2[:,c]=channel[mask]
        a=metric(yy,p1)[TS[c]];b=metric(yy,p2)[TS[c]]
        strata[name]=dict(rows=int(mask.sum()),before=a,after=b,delta=b-a)
    stage_gains.append(dict(stage=s,target=TS[c],actual_target_gain=current-previous,
         actual_TOTAL_contribution=(.30,.35,.35)[c]*(current-previous),previous_target_score=previous,
         selected_target_score=current,optimizer_gain=rr['against_initial'],best_epoch=rr['best_epoch'],
         stopped_epoch=rr['stopped_epoch'],strata=strata,process_wall_seconds=terminal['process_wall_seconds'],
         concurrency=terminal['concurrency']))
    if s==stage:break
assert stage in completed
scores=metric(truth,pred);assert np.isfinite(pred).all()
checkpoint=out/(f'fold_{fold}.pt' if stage=='SOURCE' else 'model_best.pt')
complete_model=len(completed)==len(STAGES)
scope='actually assembled complete8stage single-fold model; not fullOOF' if complete_model else 'actually assembled completed-stage prefix; not complete8stage model or fullOOF'
result=dict(identity=identity,fold=fold,stage=stage,observed_epoch=time.time(),report=report,timing=timing,
      rows=rows,source_scores=before,source_sha256=descriptor['source_sha256'],own_source_Q=descriptor['qinfo'],
      completed_prefix=completed,prefix_scores=scores,stage_gains=stage_gains,checkpoint_sha256=sha(checkpoint),
      score_scope=scope,complete8stage_single_fold=complete_model,CPU_only=True)
write(destination,result);write(STUDY/f'fold_{fold}'/(identity+'_PREFIX_SNAPSHOT.json'),result)
note=f'\nV370 {identity}/fold{fold}/{stage} terminal. Completed stages{len(completed)}/8 scores{scores}; real source/predecessor gains recorded in {destination.relative_to(ROOT).as_posix()}. Best restored exactly. Scope: {scope}. Timing{timing["process_wall_seconds"]:.3f}s ({timing["concurrency"]}).\n'
for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
    with p.open('a',encoding='utf-8') as stream:stream.write(note)
print(dict(identity=identity,fold=fold,stage=stage,completed=completed,prefix_scores=scores,latest_gain=stage_gains[-1] if stage_gains else None))
