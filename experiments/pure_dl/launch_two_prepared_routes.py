"""Explicit one-shot launch after finite seed batch; no watcher, queue, or retry."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil

# Latest explicit user plan replaces both prepared optimization launches.
raise RuntimeError('Retired: finish V362, then two-seed fullCV V370; V367/V369 remain held')

ROOT=Path(__file__).resolve().parents[2]
SEED_STUDY=ROOT/'results/deep_learning/complete_seed_compare_v362_20261002'
RECEIPT=ROOT/'results/deep_learning/perm_depth_residual_v369_20261002/TWO_ROUTE_LAUNCH.json'
JOBS=(
 ('por_continuous_aux_v367_20261002','pure_dl_por_continuous_aux_v367_fold5','train_por_continuous_aux_v367.py','aux01'),
 ('perm_depth_residual_v369_20261002','pure_dl_perm_depth_residual_v369_fold5','train_perm_depth_residual_v369.py','local'))

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,value):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gpu_jobs():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=' '.join(p.info['cmdline'] or [])
            if p.pid!=os.getpid() and (p.info['name'] or '').lower().startswith('python') and any(s in cmd for s in ('train_','benchmark','evaluate')):
                result.append(dict(pid=p.pid,cmdline=p.info['cmdline']))
        except psutil.Error:pass
    return result

def main():
    # This function is called only by an explicit root tool command after
    # inspecting actual complete scores, never from the old batch controller.
    batch=read(SEED_STUDY/'BATCH_STATE.json')
    if batch.get('status')!='complete' or batch.get('completed_models')!=5:
        raise RuntimeError('Finish finite five-complete-seed batch before launching any new route')
    if (ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/STOP_BATCH.json').exists():
        raise RuntimeError('Stop marker present; user resumption must be resolved first')
    receipt_early=SEED_STUDY/'INTERFACE_FIX_LAUNCHED.json'
    controller=read(receipt_early)['controller_pid']
    if psutil.pid_exists(controller):
        p=psutil.Process(controller)
        if 'run_complete_seeds_v362r1.py' in ' '.join(p.cmdline()):
            raise RuntimeError('Original controller has not exited yet')
    if RECEIPT.exists():raise RuntimeError('Existing launch receipt: no automatic retry or resume')
    comparison=read(SEED_STUDY/'SEED_COMPARISON.json')
    if comparison['completed_seeds']!=5 or not comparison['complete']:
        raise RuntimeError('Complete seed comparison missing')
    if any(row['scores']['TOTAL']>80.8 for row in comparison['ranking']):
        raise RuntimeError('Complete8stage seed passes80.8; assess promotion before starting new candidates')
    if gpu_jobs():raise RuntimeError('Other GPU task present before explicit two-route launch')
    prepared=[]
    for study_name,out_name,script,stage in JOBS:
        study=ROOT/'results/deep_learning'/study_name;plan=read(study/'PLAN.json');prep=read(study/'PREPARATION.json')
        assert plan['total_training_stages']==8 and all(n<=1 for n in plan['added_head_stages'].values())
        assert plan['maximum_GPU_jobs']==2 and plan['epochs']==80
        assert plan['patience']==10 and plan['patience_after_strict_SOTA']==14
        hashes=prep.get('code_sha256',prep.get('sha256'))
        if not hashes:raise RuntimeError('Prepared source hashes missing')
        for relative,expected in hashes.items():
            if sha(ROOT/relative)!=expected:raise RuntimeError('Prepared code changed: '+relative)
        out=ROOT/'outputs'/out_name/stage
        if out.exists():raise RuntimeError('Existing candidate output; explicit state inspection required: '+str(out))
        prepared.append((study,plan,out,script,stage))
    receipt=dict(status='launching',observed_epoch=time.time(),root_explicit_one_shot=True,
                 previous_batch=batch,complete_seed_ranking=comparison['ranking'],maximum_GPU_jobs=2,
                 choice='First test hypotheses aux01/local on separate routes; matched aux0/point controls held for subsequent explicit decisions',
                 no_queue=True,no_retry=True,no_autoresume=True,remaining_folds_started=False,jobs=[])
    write(RECEIPT,receipt)
    try:
        for study,plan,out,script,stage in prepared:
            if len(gpu_jobs())>=2:raise RuntimeError('Two GPU tasks already present')
            plan.update(launch_enabled=True,prepared_not_launched=False,current_status='explicit_initial_candidate_launch',
                        explicit_root_launch_epoch=time.time(),selected_first_candidate=stage,
                        launch_policy='Finite seed batch complete; this candidate explicitly launched by root, no queue/retry')
            write(study/'PLAN.json',plan);out.mkdir(parents=True)
            env=os.environ.copy();env.update(PYTHONIOENCODING='utf-8',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            cmd=[sys.executable,'-u','experiments/pure_dl/'+script,'--stage',stage]
            tick=time.time()
            with (out/'stdout.log').open('w',encoding='utf-8') as stdout,(out/'stderr.log').open('w',encoding='utf-8') as stderr:
                child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            job=dict(study=study.name,stage=stage,pid=child.pid,started_epoch=tick,cmdline=cmd,out=str(out),
                     concurrency='two independent GPU candidates, not solo benchmark')
            receipt['jobs'].append(job);write(RECEIPT,receipt)
            print('EXPLICIT_ROUTE_LAUNCHED',json.dumps(job),flush=True)
            note=f'\nExplicit initial candidate {stage} launched PID{child.pid}, after all5 complete seeds and actual old-controller exit. Two independent routes share max2GPU; no retry/autoresume/queue, no otherfolds. First hypothesis only, paired control still held. Single-phase and whole-recipe timing remain distinct.\n'
            for p in (study/'README.md',study/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with p.open('a',encoding='utf-8') as h:h.write(note)
        receipt['status']='launched';write(RECEIPT,receipt)
    except BaseException as exc:
        receipt.update(status='partial_launch_or_failed',error=str(exc),explicit_inspection_required=True)
        write(RECEIPT,receipt);raise

if __name__=='__main__':main()
