"""Keep the fixed recipe; overlap only future root pretraining with main stages.

Maximum two GPU workers, at most one SW stage. The immutable training scripts
and existing child process are retained. Failed workers are never retried.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from fullcv_context_v97 import ROOT, STUDY, RECIPE, sha
from fullcv_models_v97 import STAGES, fold_dir, root_dir, stage_dir

CODE=Path(__file__).resolve().parent
SCRIPT_ROLES={
    'train_root_fullcv_v97.py':'root','train_stage_fullcv_v97.py':'stage',
    'prepare_fold_fullcv_v97.py':'data','assemble_sources_fullcv_v97.py':'assemble',
    'record_fullcv_v97.py':'records','record_overlap_fullcv_v97.py':'records',
}


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def task(fold,role,stage=None,action=None):
    label=f'fold{fold}_{stage or action or role}'
    if role=='root':
        out=root_dir(fold);expected=out/f'fold_{fold}_report.json'
        command=[sys.executable,'-u',str(CODE/'train_root_fullcv_v97.py'),'--fold',str(fold),'--out',str(out),'--learning-rate','.0006','--schedule','cosine']
    elif role=='stage':
        out=stage_dir(fold,stage);expected=out/'report.json'
        command=[sys.executable,'-u',str(CODE/'train_stage_fullcv_v97.py'),'--fold',str(fold),'--stage',stage]
    elif role=='data':
        out=fold_dir(fold);expected=out/'data.pt'
        command=[sys.executable,'-u',str(CODE/'prepare_fold_fullcv_v97.py'),'--fold',str(fold)]
    elif role=='assemble':
        out=fold_dir(fold);expected=out/({'factors':'selected_factors.pt','private':'private_PERM.pt','final':'fold_report.json'}[action])
        command=[sys.executable,'-u',str(CODE/'assemble_sources_fullcv_v97.py'),'--fold',str(fold),'--action',action]
    else:raise ValueError(role)
    return dict(fold=fold,role=role,stage=stage,action=action,label=label,out=str(out),expected=str(expected),command=command,
                gpu=role in ('root','stage','data'),started_at=None)


def frontier():
    for f in (1,2,3,4):
        if (fold_dir(f)/'fold_report.json').exists():continue
        if not (root_dir(f)/f'fold_{f}_report.json').exists():return task(f,'root')
        if not (fold_dir(f)/'data.pt').exists():return task(f,'data')
        for stage,*_ in STAGES:
            if (stage_dir(f,stage)/'report.json').exists():continue
            if stage=='por_mean' and not (fold_dir(f)/'selected_factors.pt').exists():return task(f,'assemble',action='factors')
            if stage=='por_amplitude' and not (fold_dir(f)/'private_PERM.pt').exists():return task(f,'assemble',action='private')
            return task(f,'stage',stage=stage)
        return task(f,'assemble',action='final')
    return None


def workers():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        try:
            if 'python' not in (p.info['name'] or '').lower():continue
            args=p.info['cmdline'] or []
            script=next((Path(a).name for a in args if Path(a).name in SCRIPT_ROLES),None)
            if not script:continue
            role=SCRIPT_ROLES[script]
            if '--fold' not in args:continue
            f=int(args[args.index('--fold')+1]);stage=args[args.index('--stage')+1] if '--stage' in args else None
            action=args[args.index('--action')+1] if '--action' in args else None
            if role=='records':continue
            desc=task(f,role,stage,action);desc.update(pid=p.pid,creation_time=p.info['create_time'],adopted=True)
            result.append(desc)
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    return result


def alive(desc):
    try:
        p=psutil.Process(desc['pid'])
        return p.is_running() and abs(p.create_time()-desc['creation_time'])<.001
    except psutil.NoSuchProcess:return False


def free_gib():
    value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return float(value.splitlines()[0].strip())/1024


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--takeover',action='store_true');ap.add_argument('--preview',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    auth=json.loads((STUDY/'authorization.json').read_text(encoding='utf-8'));assert auth['remaining_folds_authorized'] and auth['recipe_sha256']==sha(RECIPE)
    manifest=json.loads((STUDY/'training_entry_manifest.json').read_text())
    assert all(sha(ROOT/p)==digest for p,digest in manifest['source_sha256'].items()),'Historical training sources changed'
    status_path=STUDY/'controller_status.json';old=json.loads(status_path.read_text())
    existing=workers()
    preview=dict(old_controller=old,live_workers=existing,next_task=frontier(),free_gpu_gib=free_gib(),
        maximum_gpu_workers=2,maximum_SW_stages=1,prefetch_root_admission_gib=6.8,main_stage_admission_gib=4.,unchanged_training_source_manifest=True)
    if args.preview:
        write(STUDY/'overlap_preview.json',preview);print('OVERLAP_PREVIEW',json.dumps(preview));return
    if args.takeover:
        assert old['status']=='running'
        controller=psutil.Process(old['pid']);cmd=controller.cmdline()
        assert any(Path(a).name=='run_fullcv_v97.py' for a in cmd),'Only the old owned CPU controller may be replaced'
        write(STUDY/'controller_takeover_plan.json',dict(recorded_at=datetime.now().astimezone().isoformat(),
            old_controller_pid=old['pid'],command=cmd,workers_before=existing,training_sources_unchanged=True,terminate_training=False))
        # Windows terminates only this CPU parent. Child training stays alive.
        controller.terminate();controller.wait(timeout=10)
        adopted=workers()
        for w in existing:
            assert alive(w) or Path(w['expected']).exists(),'An old worker ended without its completion artifact'
        write(STUDY/'controller_takeover_done.json',dict(recorded_at=datetime.now().astimezone().isoformat(),old_controller_terminated=True,
            training_workers_adopted=adopted,training_termination_requested=False))
    else:
        if old.get('pid') and psutil.pid_exists(old['pid']):raise RuntimeError('Existing controller is live')
        if not args.resume:raise RuntimeError('Explicit --resume is required after a failed/stopped controller')
        adopted=workers()
    jobs={w['label']:dict(w,handle=None) for w in adopted};events=[]
    event_path=STUDY/'overlap_events.jsonl'
    def event(kind,**values):
        row=dict(at=datetime.now().astimezone().isoformat(),kind=kind,**values);events.append(row)
        with event_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
        print('EVENT',json.dumps(row),flush=True)
    def update(reason=None):
        public=[{k:v for k,v in j.items() if k!='handle'} for j in jobs.values()]
        assert sum(j['gpu'] for j in public)<=2
        assert sum(j['role']=='stage' and (j.get('stage') or '').startswith('sw_') for j in public)<=1
        next_task=frontier()
        write(status_path,dict(pid=os.getpid(),status='running',updated_at=datetime.now().astimezone().isoformat(),
            maximum_gpu_jobs=2,maximum_SW_stages=1,execution='main stages serial; next-fold root pretraining overlaps',
            current=next_task['label'] if next_task else 'final_verification',active_jobs=public,waiting_reason=reason))
    def start(desc):
        out=Path(desc['out']);out.mkdir(parents=True,exist_ok=True)
        if desc['role'] in ('root','stage') and (out/'config.json').exists():
            last=out/(f"fold_{desc['fold']}_last_state.pt" if desc['role']=='root' else 'last_state.pt')
            if not args.resume or not last.exists():raise RuntimeError(f"Unfinished {desc['label']} requires explicit recovery, no retry")
            desc['command'].append('--resume')
        with (out/'overlap_execution.log').open('ab') as stream:
            p=subprocess.Popen(desc['command'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        desc.update(pid=p.pid,creation_time=psutil.Process(p.pid).create_time(),handle=p,adopted=False,started_at=datetime.now().astimezone().isoformat())
        jobs[desc['label']]=desc;event('launch',label=desc['label'],pid=p.pid,role=desc['role'],free_gpu_gib=free_gib());update()
    def reap():
        for label,j in list(jobs.items()):
            if j['handle'] is not None:
                code=j['handle'].poll()
                if code is None:continue
                if code:raise RuntimeError(f'{label} exited {code}; no retry')
            elif alive(j):continue
            if not Path(j['expected']).exists():raise RuntimeError(f'{label} ended without completion artifact; no retry')
            event('complete',label=label,pid=j['pid'],role=j['role']);del jobs[label]
            if j['role']=='assemble' and j['action']=='final':
                subprocess.run([sys.executable,str(CODE/'record_overlap_fullcv_v97.py')],cwd=ROOT,check=True)
    try:
        event('takeover' if args.takeover else 'resume',adopted_labels=list(jobs),training_sources_unchanged=True)
        update()
        while True:
            reap();next_task=frontier()
            if next_task is None and not jobs:break
            unknown=[w for w in workers() if w['label'] not in jobs]
            if unknown:raise RuntimeError(f'Untracked worker prevents launch: {unknown}')
            gpu_count=sum(j['gpu'] for j in jobs.values())
            # Reserve the second slot for a future root. At most one source
            # prefetch and one main worker; two SW training workers never run.
            if next_task and gpu_count<2 and not any(j['role']=='root' for j in jobs.values()):
                future=next((f for f in range(next_task['fold']+1,5)
                    if not (root_dir(f)/f'fold_{f}_report.json').exists()),None)
                if future and free_gib()>=6.8:start(task(future,'root'));gpu_count+=1
            if next_task and next_task['label'] not in jobs:
                # Only the prefetch may run ahead of the main DAG.
                main_busy=any(j['role']!='root' or j['fold']==next_task['fold'] for j in jobs.values())
                if not main_busy:
                    minimum=6.8 if next_task['role']=='root' else 4.
                    if not next_task['gpu'] or (sum(j['gpu'] for j in jobs.values())<2 and free_gib()>=minimum):start(next_task)
            update(reason='worker still running or conservative GPU admission')
            time.sleep(10)
        subprocess.run([sys.executable,str(CODE/'verify_fullcv_v97.py'),'--final'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(CODE/'cleanup_fullcv_v97.py'),'--apply'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(CODE/'record_overlap_fullcv_v97.py')],cwd=ROOT,check=True)
        final=json.loads((STUDY/'FINAL_REPORT.json').read_text())
        final.update(total_authorized_batch_wall_seconds=(datetime.now().astimezone()-datetime.fromisoformat(auth['authorized_at'])).total_seconds(),
            timing_scope='authorization through preparation, main-stage/root overlap, checkpoint IO, records, full disk verification and cleanup',overlap_events=str(event_path.relative_to(ROOT)))
        write(STUDY/'FINAL_REPORT.json',final)
        write(status_path,dict(pid=os.getpid(),status='complete',updated_at=datetime.now().astimezone().isoformat(),
            oof_scores=final['oof_scores'],strict_oof82_passed=final['strict_oof82_passed'],all_four_disk_replays_verified=True,
            maximum_gpu_jobs=2,maximum_SW_stages=1,execution='main stages serial; root pretraining overlap'))
    except BaseException as exc:
        write(status_path,dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),
            automatic_retry=False,explicit_resume_required=True,live_children=[{k:v for k,v in j.items() if k!='handle'} for j in jobs.values()]))
        raise


if __name__=='__main__':main()
