"""User-authorized finite five-seed complete recipe, serial, no retries/resume."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np
import psutil
import torch
from train_invariant import metric

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/complete_seed_compare_v362_20261002'
OUT=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'
STAGES=('SOURCE','POR113','POR151','PERM154','SW104','POR341','PERM336','SW351')
def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu_jobs():
 jobs=[]
 for p in psutil.process_iter(['pid','name','cmdline']):
  try:
   cmd=' '.join(p.info['cmdline'] or [])
   if (p.info['name'] or '').lower().startswith('python') and any(s in cmd for s in ('train_','benchmark','evaluate')):jobs.append(p.info['pid'])
  except psutil.Error:pass
 return jobs
def record_seed(seed):
 own=OUT/('seed_'+str(seed))
 data=torch.load(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
 y=data['truth'][data['validation']]
 preds=np.column_stack([np.load(own/s/'channel_pred.npy') for s in ('POR341','PERM336','SW351')])
 assert preds.shape==(144665,3) and np.isfinite(preds).all()
 scores=metric(y,preds);np.save(own/'complete_model_pred.npy',preds)
 reports={s:json.loads((own/s/('fold_5_report.json' if s=='SOURCE' else 'report.json')).read_text(encoding='utf-8-sig')) for s in STAGES}
 for i,s in enumerate(('POR341','PERM336','SW351')):
  assert reports[s]['predictions_reproduced_exactly'] and abs(scores[('POR','PERM','SW')[i]]-reports[s]['target_score'])<1e-9
 result=dict(seed=seed,scores=scores,delta_vs_historical_80706130=scores['TOTAL']-80.70612972323806,stages=list(STAGES),total_training_stages=8,screening_fold=5,validation_rows=144665,complete_five_folds=False,remaining_folds_started=False,source_fully_retrained=True,other_seed_weights_loaded=False,independent_disk_reload=False,strict_808_qualified=scores['TOTAL']>80.8,checkpoint_sha256={s:sha(own/s/('fold_5.pt' if s=='SOURCE' else 'model_best.pt')) for s in STAGES},prediction_sha256=sha(own/'complete_model_pred.npy'),reports=reports)
 write(own/'COMPLETE_MODEL.json',result)
 results=[json.loads(p.read_text()) for p in OUT.glob('seed_*/COMPLETE_MODEL.json')]
 rank=sorted(results,key=lambda x:x['scores']['TOTAL'],reverse=True)
 totals=[v['scores']['TOTAL'] for v in results]
 summary=dict(completed_seeds=len(results),planned_seeds=5,ranking=[dict(seed=v['seed'],scores=v['scores'],delta_vs_historical=v['delta_vs_historical_80706130']) for v in rank],mean_TOTAL=float(np.mean(totals)),std_TOTAL_population=float(np.std(totals)),range_TOTAL=[min(totals),max(totals)],screening_fold=5,complete_five_folds=False,remaining_folds_started=False,complete=len(results)==5,selection_limits='Same held-out fifth used for ranking seeds; selection can be optimistic, not an independent generalization/fullOOF claim. No cross-seed head splice/ensemble/row choice.')
 write(STUDY/'SEED_COMPARISON.json',summary)
 text='\nV362 complete seed '+str(seed)+': '+json.dumps(scores)+'. Completed '+str(len(results))+'/5; complete8stage source retrained. Ranking on fixed fifth only, no other folds or independent-generalization claim.\n'
 for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
  with p.open('a',encoding='utf-8') as h:h.write(text)
 return scores
def main():
 plan=json.loads((STUDY/'PLAN.json').read_text());assert plan['user_authorized']
 if (STUDY/'BATCH_STATE.json').exists():raise RuntimeError('Existing batch requires explicit scoped resumption; no auto restart')
 OUT.mkdir(exist_ok=True);started=time.time()
 for seed in plan['seeds']:
  own=OUT/('seed_'+str(seed))
  for stage in STAGES:
   if (OUT/'STOP_BATCH.json').exists():write(STUDY/'BATCH_STATE.json',dict(status='stopped',explicit_resume_only=True,seed=seed,stage=stage));return
   if gpu_jobs():raise RuntimeError('GPU process present before explicit serial stage launch: '+str(gpu_jobs()))
   dest=own/stage;dest.mkdir(parents=True,exist_ok=True)
   env=os.environ.copy();env.update(WELLLOG_RUN_SEED=str(seed),PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
   if stage=='SOURCE':cmd=[sys.executable,'-u','experiments/pure_dl/train_complete_seed_source_v362.py','--out',str(dest),'--learning-rate','.0006','--schedule','cosine','--fold','5','--seed',str(seed)]
   else:cmd=[sys.executable,'-u','experiments/pure_dl/train_complete_seed_stage_v362.py','--stage',stage]
   tick=time.time()
   with (dest/'stdout.log').open('w',encoding='utf-8') as stdout,(dest/'stderr.log').open('w',encoding='utf-8') as stderr:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    write(STUDY/'BATCH_STATE.json',dict(status='running',controller_pid=os.getpid(),active_pid=child.pid,seed=seed,stage=stage,started_epoch=tick,batch_started_epoch=started,concurrency='serial',maximum_GPU_jobs=2,completed_models=len(list(OUT.glob('seed_*/COMPLETE_MODEL.json')))))
    print('EXPLICIT_SEED_STAGE',seed,stage,child.pid,flush=True)
    code=child.wait()
   write(dest/'TIMING.json',dict(seed=seed,stage=stage,pid=child.pid,started_epoch=tick,terminal=True,exit_code=code,process_wall_seconds=time.time()-tick,concurrency='solo',includes='preprocessing warmup training best restoration'))
   report=dest/('fold_5_report.json' if stage=='SOURCE' else 'report.json')
   if code or not report.exists():
    write(STUDY/'BATCH_STATE.json',dict(status='failed_or_stopped',seed=seed,stage=stage,exit_code=code,explicit_resume_only=True,no_retry=True));return
  print('COMPLETE_SEED',seed,json.dumps(record_seed(seed)),flush=True)
 write(STUDY/'BATCH_STATE.json',dict(status='complete',completed_models=5,total_wall_seconds=time.time()-started,complete_five_folds=False,remaining_folds_started=False))
 print('FIVE_SEEDS_COMPLETE',flush=True)
if __name__=='__main__':main()
