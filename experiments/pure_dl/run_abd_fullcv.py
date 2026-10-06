"""Two-slot dependency queue: B_fold -> A_fold, reuse verified fifth and D."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[2]


def active_gpu_jobs():
    jobs=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        info=proc.info
        if (info['name'] or '').lower()!='python.exe':continue
        for argument in info['cmdline'] or []:
            name=Path(argument.replace('\\','/')).name
            if 'pure_dl/' in argument.replace('\\','/') and name.startswith(('train_','evaluate_','benchmark_')) and name.endswith('.py'):
                jobs.append(info['pid']);break
    return jobs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();proof_path=out/'promotion.json'
    proof=json.loads(proof_path.read_text(encoding='utf-8'))
    assert proof['passed'] and proof['score']>80.4 and proof['score']>proof['baseline_total']
    fifth=Path(proof['fifth_folder'])
    assert hashlib.sha256((fifth/'fold_5.pt').read_bytes()).hexdigest()==proof['fifth_checkpoint_sha256']
    queue=[]
    for f in range(1,5):
        queue.extend([('B',f),('A',f)])
    running={};finished=set();failed=[];started=time.perf_counter()
    status_path=out/'queue_status.json'
    for role,f in queue:
        folder=out/role/f'fold_{f}'
        result=folder/f'fold_{f}_report.json'
        if result.exists():
            item=json.loads(result.read_text(encoding='utf-8'))
            assert item['best_checkpoint_predictions_reproduced'] and (folder/f'fold_{f}.pt').exists()
            finished.add((role,f))
    while len(finished)<len(queue):
        for task,(proc,log) in list(running.items()):
            exit_code=proc.poll()
            if exit_code is None:continue
            log.close();del running[task]
            role,f=task;folder=out/role/f'fold_{f}'
            report_path=folder/f'fold_{f}_report.json'
            if exit_code!=0 or not report_path.exists():
                failed.append({'task':task,'exit_code':exit_code})
                print('JOB_FAILED',task,exit_code,flush=True)
            else:
                report=json.loads(report_path.read_text(encoding='utf-8'))
                assert report['best_checkpoint_predictions_reproduced'] and report['fold']==f
                finished.add(task);print('JOB_COMPLETE',task,report['scores'],flush=True)
        # Stop launching after failure; let any already-running independent job finish.
        if failed and not running:break
        if not failed:
            available=2-len(active_gpu_jobs())
            candidates=[task for task in queue if task not in finished and task not in running and
                        (task[0]=='B' or ('B',task[1]) in finished)]
            candidates.sort(key=lambda task:(0 if task[0]=='A' else 1,task[1]))
            for role,f in candidates[:max(0,available)]:
                folder=out/role/f'fold_{f}';folder.mkdir(parents=True,exist_ok=True)
                common=['--out',str(folder),'--learning-rate','0.0006','--schedule','cosine','--fold',str(f)]
                if role=='B':
                    command=[sys.executable,'-u','experiments/pure_dl/train_abd_member_b_fold80.py',*common,
                             '--seed-base','20261008','--ensemble-proof',str(proof_path)]
                else:
                    d=next(item for item in proof['d_checkpoints'] if item['fold']==f)
                    assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
                    command=[sys.executable,'-u','experiments/pure_dl/train_abd_continuation_fold80.py',*common,
                             '--ablation',proof['ablation'],'--proof',str(proof_path),
                             '--b-checkpoint',str(out/'B'/f'fold_{f}'/f'fold_{f}.pt'),'--d-checkpoint',d['path']]
                log=(folder/'runner.log').open('a',encoding='utf-8')
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                running[(role,f)]=(proc,log)
                print('JOB_STARTED',role,f,proc.pid,flush=True)
        status={'status':'failed_draining' if failed else 'running','finished':[list(t) for t in sorted(finished)],
                'running':[{'task':list(task),'pid':proc.pid} for task,(proc,_) in running.items()],
                'failures':failed,'seconds':time.perf_counter()-started,'max_gpu_jobs':2}
        status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
        if len(finished)<len(queue):time.sleep(20)
    status={'status':'failed' if failed else 'training_complete_needs_oof_verification',
            'finished':[list(t) for t in sorted(finished)],'failures':failed,'seconds':time.perf_counter()-started}
    status_path.write_text(json.dumps(status,indent=2),encoding='utf-8')
    print('QUEUE_FINAL',json.dumps(status),flush=True)
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
