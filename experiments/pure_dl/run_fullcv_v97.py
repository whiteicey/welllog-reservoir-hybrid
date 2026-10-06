"""Authorized serial completion of the fixed v96 recipe on folds1..4.

No retries or implicit recovery of failed/stopped stages. Explicit --resume
continues only the recorded stage last state, after any implementation fix.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil
from fullcv_context_v97 import ROOT, STUDY, RECIPE, sha
from fullcv_models_v97 import STAGES, fold_dir, root_dir, root_path, stage_dir


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def live_roots():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cmd=p.info.get('cmdline') or []
            if any(Path(a).name=='train_root_fullcv_v97.py' for a in cmd):result.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    authorization=json.loads((STUDY/'authorization.json').read_text(encoding='utf-8'));assert authorization['remaining_folds_authorized'] and authorization['recipe_sha256']==sha(RECIPE)
    assert (STUDY/'decoder_migration_proof.json').exists() and (STUDY/'identity_stage_proof.json').exists()
    source_manifest=json.loads((STUDY/'training_entry_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==digest for p,digest in source_manifest['source_sha256'].items())
    status_path=STUDY/'controller_status.json'
    if status_path.exists() and not args.resume:raise RuntimeError('Controller already started; explicit --resume required')
    previous=json.loads(status_path.read_text()) if status_path.exists() else {}
    if previous.get('pid') and psutil.pid_exists(previous['pid']):raise RuntimeError('Controller is already live')
    owned=None
    def update(**more):
        status=dict(pid=__import__('os').getpid(),status='running',updated_at=datetime.now().astimezone().isoformat(),maximum_gpu_jobs=2,execution='serial',**more)
        write(status_path,status)
    def run(script,argv,out,label,resume=False):
        nonlocal owned
        owned=out;out.mkdir(parents=True,exist_ok=True);update(current=label,artifact=str(out.relative_to(ROOT)))
        command=[sys.executable,'-u',str(Path(__file__).with_name(script)),*argv]
        if resume:command.append('--resume')
        log=out/('resume.log' if resume else 'execution.log')
        with log.open('ab') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            update(current=label,child_pid=proc.pid,command=command,artifact=str(out.relative_to(ROOT)))
            code=proc.wait()
        if code:raise RuntimeError(f'{label} exited {code}; see {log}')
    try:
        for f in (1,2,3,4):
            if (fold_dir(f)/'fold_report.json').exists():continue
            rdir=root_dir(f)
            if not (rdir/f'fold_{f}_report.json').exists():
                if f==1 and live_roots():
                    update(current='fold1_root_external',child_pids=live_roots())
                    while not (rdir/f'fold_{f}_report.json').exists():
                        if not live_roots():raise RuntimeError('External first-fold source ended without report; no automatic resume')
                        time.sleep(15)
                else:
                    resume=args.resume and (rdir/f'fold_{f}_last_state.pt').exists()
                    if (rdir/'config.json').exists() and not resume:raise RuntimeError(f'Unfinished root fold{f}; explicit resume required')
                    run('train_root_fullcv_v97.py',['--fold',str(f),'--out',str(rdir),'--learning-rate','.0006','--schedule','cosine'],rdir,f'fold{f}_root',resume)
            assert root_path(f).exists()
            if not (fold_dir(f)/'data.pt').exists():run('prepare_fold_fullcv_v97.py',['--fold',str(f)],fold_dir(f),f'fold{f}_data')
            for stage,*_ in STAGES:
                out=stage_dir(f,stage)
                if (out/'report.json').exists():continue
                if stage=='por_mean' and not (fold_dir(f)/'selected_factors.pt').exists():run('assemble_sources_fullcv_v97.py',['--fold',str(f),'--action','factors'],fold_dir(f),f'fold{f}_factor_sources')
                if stage=='por_amplitude' and not (fold_dir(f)/'private_POR.pt').exists():run('assemble_sources_fullcv_v97.py',['--fold',str(f),'--action','private'],fold_dir(f),f'fold{f}_private_extraction')
                resume=args.resume and (out/'last_state.pt').exists()
                if (out/'config.json').exists() and not resume:raise RuntimeError(f'Unfinished {f}/{stage}; explicit resume required')
                run('train_stage_fullcv_v97.py',['--fold',str(f),'--stage',stage],out,f'fold{f}_{stage}',resume)
            run('assemble_sources_fullcv_v97.py',['--fold',str(f),'--action','final'],fold_dir(f),f'fold{f}_assembled')
            run('record_fullcv_v97.py',[],STUDY,f'fold{f}_records')
        final=json.loads((STUDY/'FINAL_REPORT.json').read_text())
        final['total_authorized_batch_wall_seconds']=(datetime.now().astimezone()-datetime.fromisoformat(authorization['authorized_at'])).total_seconds()
        final['timing_scope']='from explicit authorization record; includes preparation, initial external root, serial training, validation, checkpoint IO and record gaps'
        write(STUDY/'FINAL_REPORT.json',final)
        update(status_final='complete',current='complete',oof_scores=final['oof_scores'],strict_oof82_passed=final['strict_oof82_passed'])
        status=json.loads(status_path.read_text());status['status']='complete';write(status_path,status)
    except BaseException as exc:
        import traceback
        write(status_path,dict(status='failed',pid=__import__('os').getpid(),error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False,explicit_resume_required=True,artifact=str(owned) if owned else None))
        raise


if __name__=='__main__':main()
