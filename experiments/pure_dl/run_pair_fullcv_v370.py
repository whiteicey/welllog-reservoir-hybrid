"""Finite authorized eight model-folds, serial, no retry or implicit resume."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from fullcv_pair_context_v370 import ROOT,STUDY,OUTPUT,STAGES,BASE_SEEDS,own_dir,require_authorization,read,write,sha

def gpu_jobs():
    jobs=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or [])
            if p.pid!=os.getpid() and (p.info['name'] or '').lower().startswith('python') and any(s in cmd for s in ('train_','benchmark','evaluate')):jobs.append(p.pid)
        except psutil.Error:pass
    return jobs

def main():
    plan=require_authorization();receipt=STUDY/'LAUNCH.json'
    if receipt.exists() or OUTPUT.exists():raise RuntimeError('Existing fullCV launch/state; explicit inspection/resume only')
    if gpu_jobs():raise RuntimeError('Other GPU tasks present: '+str(gpu_jobs()))
    preparation=read(STUDY/'PREPARATION.json')
    for path,expected in preparation['code_sha256'].items():assert sha(ROOT/path)==expected,path
    OUTPUT.mkdir();started=time.time()
    write(receipt,dict(controller_pid=os.getpid(),started_epoch=started,plan_sha256=sha(STUDY/'PLAN.json'),
          status='running',concurrency='serial',maximum_GPU_jobs=2,finite_model_folds=8,retry=False,implicit_resume=False))
    def run(script,args,dest,identity,fold,stage,expected):
        if (OUTPUT/'STOP_BATCH.json').exists():raise RuntimeError('Explicit stop; no queued launch')
        if gpu_jobs():raise RuntimeError('Other GPU task before serial handoff')
        if Path(expected).exists():raise RuntimeError('Unexpected completed output; no silent restart/reuse')
        for path,digest in preparation['code_sha256'].items():assert sha(ROOT/path)==digest,path
        dest.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env.update(WELLLOG_MODEL_ID=identity,WELLLOG_FOLD=str(fold),PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command=[sys.executable,'-u','experiments/pure_dl/'+script,*args];tick=time.time()
        with (dest/(stage+'_stdout.log')).open('w',encoding='utf-8') as out,(dest/(stage+'_stderr.log')).open('w',encoding='utf-8') as err:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            write(STUDY/'BATCH_STATE.json',dict(status='running',controller_pid=os.getpid(),active_pid=child.pid,
                  identity=identity,fold=fold,stage=stage,started_epoch=tick,batch_started_epoch=started,concurrency='serial',maximum_GPU_jobs=2))
            print('AUTHORIZED_FULLCV_CHILD',identity,fold,stage,child.pid,flush=True)
            code=child.wait()
        timing=dict(identity=identity,fold=fold,stage=stage,pid=child.pid,terminal=True,exit_code=code,
                    process_wall_seconds=time.time()-tick,concurrency='solo',includes='process/preprocessing/warmup/training or scan/best restoration/write')
        write(dest/(stage+'_TIMING.json'),timing)
        if code or not Path(expected).exists():raise RuntimeError(f'{identity}/fold{fold}/{stage}: exit{code}, expected={expected}')
    from record_pair_fullcv_v370 import record_model,record_fold,record_oof
    if not (STUDY/'fold_5/FOLD_REPORT.json').exists():record_fold(5)
    for fold in plan['folds_to_train']:
        for identity,base_seed in BASE_SEEDS.items():
            own=own_dir(identity,fold)
            for stage in STAGES:
                dest=own/stage
                if stage=='SOURCE':
                    script='train_pair_fullcv_source_v370.py'
                    args=['--out',str(dest),'--learning-rate','.0006','--schedule','cosine','--fold',str(fold),'--seed',str(base_seed+fold-1)]
                    expected=dest/f'fold_{fold}_report.json'
                else:
                    script='train_pair_fullcv_stage_v370.py';args=['--stage',stage];expected=dest/'report.json'
                run(script,args,dest,identity,fold,stage,expected)
            record_model(identity,fold)
        for target in ('POR','PERM','SW'):
            dest=STUDY/f'fold_{fold}'
            run('evaluate_pair_fullcv_v370.py',['--fold',str(fold),'--target',target],dest,'original',fold,'SCAN_'+target,dest/(target+'_GPU_REPORT.json'))
        record_fold(fold)
    result=record_oof()
    write(STUDY/'BATCH_STATE.json',dict(status='complete',controller_pid=os.getpid(),total_wall_seconds=time.time()-started,
          concurrency='serial',maximum_GPU_jobs=2,completed_model_folds=8,complete_original_five_fold_OOF=True,scores=result['scores']))
    print('AUTHORIZED_FULLCV_COMPLETE',json.dumps(result['scores']),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(STUDY/'BATCH_STATE.json',dict(status='failed_or_stopped',controller_pid=os.getpid(),error=str(exc),
              traceback=traceback.format_exc(),explicit_resume_only=True,no_retry=True))
        raise
