"""Same finite model-fold recipe; one SOURCE overlaps one downstream worker.

Attach to an existing exact child after explicit root controller handoff. No
GPU worker is stopped, repeated or resumed. Maximum two GPU children, one
SOURCE, one SW stage and one native scan. Failures stop all further launches.
"""
import os,subprocess,sys,time,traceback
from pathlib import Path
import psutil
from fullcv_pair_context_v370 import ROOT,STUDY,OUTPUT,STAGES,BASE_SEEDS,own_dir,require_authorization,read,write,sha

MODEL_FOLDS=[(fold,identity) for fold in range(1,5) for identity in BASE_SEEDS]

def is_gpu(process):
    command=' '.join(process.info['cmdline'] or [])
    return (process.info['name'] or '').lower().startswith('python') and any(s in command for s in ('train_','benchmark','evaluate'))

def actual_gpu_pids():
    result=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if process.pid!=os.getpid() and is_gpu(process):result.append(process.pid)
        except psutil.Error:pass
    return result

def task(fold,identity,stage):
    own=own_dir(identity,fold)
    if stage in STAGES:
        dest=own/stage;role='SOURCE' if stage=='SOURCE' else 'SW' if stage.startswith('SW') else 'head'
        if stage=='SOURCE':
            script='train_pair_fullcv_source_v370.py'
            args=['--out',str(dest),'--learning-rate','.0006','--schedule','cosine','--fold',str(fold),'--seed',str(BASE_SEEDS[identity]+fold-1)]
            expected=dest/f'fold_{fold}_report.json'
        else:script='train_pair_fullcv_stage_v370.py';args=['--stage',stage];expected=dest/'report.json'
    else:
        assert stage.startswith('SCAN_');target=stage[5:];assert target in ('POR','PERM','SW')
        role='scan';dest=STUDY/f'fold_{fold}';expected=dest/(target+'_GPU_REPORT.json')
        script='evaluate_pair_fullcv_v370.py';args=['--fold',str(fold),'--target',target]
    return dict(fold=fold,identity=identity,stage=stage,role=role,dest=str(dest),expected=str(expected),
                command=[sys.executable,'-u','experiments/pure_dl/'+script,*args])

def validate_terminal(job,code):
    expected=Path(job['expected']);assert code in (0,None) and expected.is_file(),(job,code)
    r=read(expected)
    if job['role']=='SOURCE':assert r['best_checkpoint_predictions_reproduced'] and r['all_validation_rows_covered_once'] and r['fold']==job['fold']
    elif job['role']=='scan':assert r['native_disk_replay_exact'] and r['coverage_once'] and r['endpoints_exact'] and r['fold']==job['fold']
    else:assert r['predictions_reproduced_exactly'] and r['frozen_parameters_exact']

def main():
    require_authorization();handoff=read(STUDY/'OVERLAP_HANDOFF.json')
    assert handoff['GPU_child_preserved'] and handoff['no_model_training_changes']
    if (STUDY/'OVERLAP_LAUNCH.json').exists():raise RuntimeError('Already launched: no retry or implicit resume')
    old=handoff['old_controller'];assert not psutil.pid_exists(old['pid'])
    job=task(handoff['batch']['fold'],handoff['batch']['identity'],handoff['batch']['stage'])
    attach=handoff['attached_child'];process=psutil.Process(attach['pid'])
    assert process.create_time()==attach['create_time'] and process.cmdline()==attach['cmdline']
    assert actual_gpu_pids()==[process.pid] or sorted(actual_gpu_pids())==[process.pid]
    preparation=read(STUDY/'PREPARATION.json')
    def verify_code():
        for path,digest in preparation['code_sha256'].items():assert sha(ROOT/path)==digest,path
        assert sha(Path(__file__))==handoff['overlap_controller_sha256']
    verify_code();began=time.time();whole_started=read(STUDY/'LAUNCH.json')['started_epoch']
    active={};handles={};counter=0;events=[];failure=None
    job.update(pid=process.pid,create_time=process.create_time(),started_epoch=handoff['batch']['started_epoch'],
               attached=True,overlap_observed=False);active[process.pid]=job;handles[process.pid]=process
    write(STUDY/'OVERLAP_LAUNCH.json',dict(controller_pid=os.getpid(),started_epoch=began,whole_batch_started_epoch=whole_started,
          attached_child=job,maximum_GPU_jobs=2,maximum_SOURCE_jobs=1,maximum_SW_jobs=1,maximum_scan_jobs=1,
          no_model_training_changes=True,no_retry=True,concurrency='one SOURCE plus one downstream when ready'))
    from record_pair_fullcv_v370 import record_model,record_fold,record_oof

    def snapshot(status='running',**extra):
        jobs=list(active.values());primary=next((j for j in jobs if j['role']=='SOURCE'),jobs[0] if jobs else {})
        write(STUDY/'BATCH_STATE.json',dict(status=status,controller_pid=os.getpid(),active_pid=primary.get('pid'),
              identity=primary.get('identity'),fold=primary.get('fold'),stage=primary.get('stage'),
              started_epoch=primary.get('started_epoch'),batch_started_epoch=whole_started,
              concurrency='overlap up to2; maximum one SOURCE/SW/scan each',maximum_GPU_jobs=2,
              active_jobs=jobs,completed_model_folds=sum((own_dir(i,f)/'COMPLETE_MODEL.json').exists() for f,i in MODEL_FOLDS),
              events=events[-8:],**extra))

    def eligible(job):
        if len(active)>=2:return False
        if job['role'] in ('SOURCE','SW','scan') and any(j['role']==job['role'] for j in active.values()):return False
        # SOURCE+downstream keeps at most one 1.8GiB temporary training CDF.
        return True

    def launch(job):
        nonlocal counter
        verify_code();external=set(actual_gpu_pids())-set(active)
        if external:raise RuntimeError('Unmanaged GPU tasks: '+str(external))
        assert len(active)<2 and eligible(job)
        dest=Path(job['dest']);dest.mkdir(parents=True,exist_ok=True)
        assert not Path(job['expected']).exists()
        if job['role']!='scan' and (dest/'config.json').exists():raise RuntimeError('Unmanaged existing state; explicit inspection required')
        environment=os.environ.copy();environment.update(WELLLOG_MODEL_ID=job['identity'],WELLLOG_FOLD=str(job['fold']),
                    PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        tick=time.time()
        with (dest/(job['stage']+'_stdout.log')).open('w',encoding='utf-8') as out,(dest/(job['stage']+'_stderr.log')).open('w',encoding='utf-8') as err:
            child=subprocess.Popen(job['command'],cwd=ROOT,env=environment,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        job.update(pid=child.pid,create_time=psutil.Process(child.pid).create_time(),started_epoch=tick,attached=False,
                   overlap_observed=bool(active),other_GPU_pids_at_launch=list(active))
        for other in active.values():other['overlap_observed']=True
        active[child.pid]=job;handles[child.pid]=child;counter+=1
        events.append(dict(event='launch',pid=child.pid,identity=job['identity'],fold=job['fold'],stage=job['stage'],epoch=tick))
        snapshot();print('OVERLAP_CHILD',job,flush=True)

    try:
        snapshot()
        while True:
            if (OUTPUT/'STOP_BATCH.json').exists():failure='Explicit stop marker; no further launches'
            for pid,job in list(active.items()):
                child=handles[pid]
                if job['attached']:
                    try:code=child.wait(timeout=0)
                    except psutil.TimeoutExpired:continue
                else:
                    code=child.poll()
                    if code is None:continue
                # No success is inferred from a vanished handle alone.
                try:validate_terminal(job,code)
                except BaseException as exc:failure=str(exc)
                timing=dict(identity=job['identity'],fold=job['fold'],stage=job['stage'],pid=pid,terminal=True,
                    exit_code=code,process_wall_seconds=time.time()-job['started_epoch'],
                    concurrency='overlapped' if job['overlap_observed'] else 'solo',
                    includes='load/preprocess/warmup/training or scan/restore/write',attached_without_training_restart=job['attached'])
                write(Path(job['dest'])/(job['stage']+'_TIMING.json'),timing)
                events.append(dict(event='terminal',pid=pid,stage=job['stage'],fold=job['fold'],identity=job['identity'],epoch=time.time(),exit_code=code))
                del active[pid],handles[pid]
            if failure:
                snapshot('failed_or_stopped',error=failure,explicit_resume_only=True,no_retry=True)
                if active:time.sleep(2);continue
                return
            # Advance completed chains in CPU without inventing untrained scores.
            for fold,identity in MODEL_FOLDS:
                own=own_dir(identity,fold)
                if (own/'COMPLETE_MODEL.json').exists():continue
                if any(j['fold']==fold and j['identity']==identity and j['role']!='scan' for j in active.values()):continue
                if all((own/s/(f'fold_{fold}_report.json' if s=='SOURCE' else 'report.json')).exists() for s in STAGES):record_model(identity,fold)
            for fold in range(1,5):
                dest=STUDY/f'fold_{fold}'
                if not (dest/'FOLD_REPORT.json').exists() and all((dest/(t+'_GPU_REPORT.json')).exists() for t in ('POR','PERM','SW')):record_fold(fold)
            if all((STUDY/f'fold_{f}/FOLD_REPORT.json').exists() for f in range(1,6)) and not active:
                result=record_oof();snapshot('complete',total_wall_seconds=time.time()-whole_started,
                    overlap_controller_seconds=time.time()-began,complete_original_five_fold_OOF=True,scores=result['scores'])
                print('TWO_SEED_FULLCV_COMPLETE',result['scores'],flush=True);return
            ready=[]
            for fold,identity in MODEL_FOLDS:
                if any(j['fold']==fold and j['identity']==identity and j['role']!='scan' for j in active.values()):continue
                own=own_dir(identity,fold)
                for stage in STAGES:
                    j=task(fold,identity,stage)
                    if Path(j['expected']).exists():
                        timing=Path(j['dest'])/(stage+'_TIMING.json')
                        assert timing.exists() and read(timing)['terminal'] and read(timing)['exit_code'] in (0,None)
                        continue
                    ready.append(j);break
            for fold in range(1,5):
                if not all((own_dir(i,fold)/'COMPLETE_MODEL.json').exists() for i in BASE_SEEDS):continue
                for target in ('POR','PERM','SW'):
                    j=task(fold,'original','SCAN_'+target)
                    if not Path(j['expected']).exists() and not any(a['fold']==fold and a['stage']==j['stage'] for a in active.values()):ready.append(j)
            # Preserve contiguous model chains; fill idle slot with the next SOURCE.
            ready.sort(key=lambda j:(j['fold'],j['role']=='SOURCE',j['role']=='scan',list(BASE_SEEDS).index(j['identity'])))
            for j in ready:
                if eligible(j):launch(j)
                if len(active)>=2:break
            snapshot()
            if not active:raise RuntimeError('No runnable task but fullCV incomplete; explicit inspection required')
            time.sleep(2)
    except BaseException as exc:
        snapshot('failed_or_stopped',error=str(exc),traceback=traceback.format_exc(),explicit_resume_only=True,no_retry=True)
        # Existing children are left to complete; never kill a training worker.
        raise

if __name__=='__main__':main()
