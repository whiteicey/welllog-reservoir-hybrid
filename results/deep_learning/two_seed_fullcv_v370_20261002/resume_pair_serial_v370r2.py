"""Explicitly approved OOM recovery; serialize the unchanged finite recipe."""
import os,sys,time,subprocess,json
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from fullcv_pair_context_v370 import OUTPUT,STAGES,BASE_SEEDS,own_dir,read,write,sha,require_authorization
from run_overlap_pair_fullcv_v370r1 import task,validate_terminal,MODEL_FOLDS,actual_gpu_pids
from record_pair_fullcv_v370 import record_model,record_fold,record_oof

def main():
    require_authorization();prepared=read(STUDY/'OOM_RECOVERY_PREPARED.json')
    # Only the root may write this AFTER a direct user answer to the recovery question.
    approval=read(STUDY/'OOM_RECOVERY_AUTHORIZATION.json')
    assert approval['explicit_user_approved'] and approval['maximum_GPU_jobs']==1
    assert approval['prepared_sha256']==sha(STUDY/'OOM_RECOVERY_PREPARED.json')
    assert approval['controller_sha256']==sha(Path(__file__))
    if (STUDY/'SERIAL_RECOVERY_LAUNCH.json').exists():raise RuntimeError('No retry or implicit resume')
    assert not actual_gpu_pids() and not (OUTPUT/'STOP_BATCH.json').exists()
    prior=read(STUDY/'BATCH_STATE.json');assert prior['status']=='failed_or_stopped'
    assert not psutil.pid_exists(prior['controller_pid'])
    allowed={(j['fold'],j['identity'],j['stage']):j for j in prepared['failed_jobs']}
    for entry in allowed.values():
        assert sha(Path(entry['last_state_path']))==entry['last_state_sha256']
        assert (STUDY/(entry['identity']+'_'+entry['stage']+'_RECOVERY_READBACK.json')).is_file()
    prep=read(STUDY/'PREPARATION.json');began=time.time();batch_start=read(STUDY/'LAUNCH.json')['started_epoch']
    used=set();events=[];active=None
    def verify_code():
        for name,digest in prep['code_sha256'].items():assert sha(ROOT/name)==digest,name
        assert sha(Path(__file__))==approval['controller_sha256']
    def state(status='running',**extra):
        jobs=[active] if active else [];primary=active or {}
        write(STUDY/'BATCH_STATE.json',dict(status=status,controller_pid=os.getpid(),active_pid=primary.get('pid'),
            identity=primary.get('identity'),fold=primary.get('fold'),stage=primary.get('stage'),started_epoch=primary.get('started_epoch'),
            batch_started_epoch=batch_start,concurrency='serial after explicit OOM recovery',maximum_GPU_jobs=1,
            active_jobs=jobs,completed_model_folds=sum((own_dir(i,f)/'COMPLETE_MODEL.json').exists() for f,i in MODEL_FOLDS),
            events=events[-8:],**extra))
    verify_code();write(STUDY/'SERIAL_RECOVERY_LAUNCH.json',dict(controller_pid=os.getpid(),started_epoch=began,
        maximum_GPU_jobs=1,authorization=approval,unchanged_training_recipe=True,no_retry=True))
    try:
        while True:
            if (OUTPUT/'STOP_BATCH.json').exists():state('stopped',explicit_resume_only=True);return
            for fold,identity in MODEL_FOLDS:
                own=own_dir(identity,fold)
                if not (own/'COMPLETE_MODEL.json').exists() and all((own/s/(f'fold_{fold}_report.json' if s=='SOURCE' else 'report.json')).exists() for s in STAGES):record_model(identity,fold)
            for fold in range(1,5):
                dest=STUDY/f'fold_{fold}'
                if not (dest/'FOLD_REPORT.json').exists() and all((dest/(t+'_GPU_REPORT.json')).exists() for t in ('POR','PERM','SW')):record_fold(fold)
            if all((STUDY/f'fold_{f}/FOLD_REPORT.json').exists() for f in range(1,6)):
                result=record_oof();state('complete',total_wall_seconds=time.time()-batch_start,serial_recovery_seconds=time.time()-began,
                    complete_original_five_fold_OOF=True,scores=result['scores']);return
            ready=[]
            for fold,identity in MODEL_FOLDS:
                for stage in STAGES:
                    job=task(fold,identity,stage)
                    if Path(job['expected']).exists():
                        timing=read(Path(job['dest'])/(stage+'_TIMING.json'))
                        assert timing['terminal'] and timing['exit_code'] in (0,None)
                        continue
                    ready.append(job);break
            for fold in range(1,5):
                if all((own_dir(i,fold)/'COMPLETE_MODEL.json').exists() for i in BASE_SEEDS):
                    for target in ('POR','PERM','SW'):
                        job=task(fold,'original','SCAN_'+target)
                        if not Path(job['expected']).exists():ready.append(job)
            assert ready,'Incomplete fullCV without runnable job'
            ready.sort(key=lambda j:(j['fold'],j['role']!='scan',list(BASE_SEEDS).index(j['identity'])))
            job=ready[0];dest=Path(job['dest']);dest.mkdir(parents=True,exist_ok=True)
            key=(job['fold'],job['identity'],job['stage']);resume=False;previous_time=0.
            if job['role']!='scan' and (dest/'config.json').exists():
                assert key in allowed and key not in used,'Unapproved existing checkpoint'
                item=allowed[key];assert sha(dest/'last_state.pt')==item['last_state_sha256']
                assert sha(dest/'config.json')==item['config_sha256']
                job['command'].append('--resume');resume=True
                previous_time=read(dest/(job['stage']+'_TIMING.json'))['process_wall_seconds']
            state();verify_code();assert not actual_gpu_pids()
            # Query the driver without creating an extra controller CUDA context.
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0].strip())
            if free<7*1024:
                events.append(dict(event='waiting_for_memory',free_MiB=free,required_MiB=7*1024));state();time.sleep(30);continue
            if resume:used.add(key)
            env=os.environ.copy();env.update(WELLLOG_MODEL_ID=job['identity'],WELLLOG_FOLD=str(job['fold']),PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            suffix='serial_resume' if resume else 'serial';tick=time.time()
            with (dest/(job['stage']+'_'+suffix+'_stdout.log')).open('w',encoding='utf-8') as out,(dest/(job['stage']+'_'+suffix+'_stderr.log')).open('w',encoding='utf-8') as err:
                child=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            job.update(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),started_epoch=tick,resumed=resume,attached=False)
            active=job;events.append(dict(event='explicit_resume' if resume else 'launch',**{k:job[k] for k in ('pid','identity','fold','stage')}));state()
            print('SERIAL_CHILD',job,flush=True)
            while child.poll() is None:time.sleep(2)
            code=child.returncode;wall=time.time()-tick
            timing=dict(identity=job['identity'],fold=job['fold'],stage=job['stage'],pid=child.pid,terminal=True,exit_code=code,
                process_wall_seconds=previous_time+wall,serial_invocation_seconds=wall,previous_failed_invocation_seconds=previous_time,
                concurrency='failed overlap plus explicit serial recovery' if resume else 'solo',resumed=resume,
                includes='load/preprocess/warmup/training or scan/restore/write; sums invocations, excludes approval waiting; whole batch wall includes waiting')
            write(dest/(job['stage']+'_TIMING.json'),timing);active=None
            events.append(dict(event='terminal',pid=child.pid,identity=job['identity'],fold=job['fold'],stage=job['stage'],exit_code=code))
            validate_terminal(job,code);state()
    except BaseException as exc:
        state('failed_or_stopped',error=str(exc),explicit_resume_only=True,no_retry=True);raise

if __name__=='__main__':main()
