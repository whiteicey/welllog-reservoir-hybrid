"""CPU-only one-shot handoff; never starts a third GPU job or restarts a failure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'outputs/pure_dl_abd_b_scratch80_fold5'
OUT=ROOT/'outputs/pure_dl_frozen_backbone_channel_heads'


def write_state(status, **extra):
    state={'status':status,'queue_pid':os.getpid(),'gpu_job':False,**extra}
    (OUT/'handoff_status.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    print(json.dumps(state),flush=True)


def gpu_jobs():
    jobs=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        if proc.info['name']!='python.exe':continue
        for argument in proc.info['cmdline'] or []:
            value=argument.replace('\\','/');name=Path(value).name
            if 'pure_dl/' in value and name.endswith('.py') and name.startswith(('train_','benchmark_','evaluate_')):
                jobs.append({'pid':proc.pid,'script':name});break
    return jobs


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int,required=True);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    try:
        watched=psutil.Process(args.wait_pid)
        if not any('train_abd_b_scratch80.py' in part for part in watched.cmdline()):
            raise RuntimeError('Requested PID is not the expected control trainer')
    except psutil.NoSuchProcess:
        watched=None
    write_state('waiting_control',wait_pid=args.wait_pid)
    while watched is not None and watched.is_running():
        time.sleep(20)
    report_path=CONTROL/'fold_5_report.json'
    state=json.loads((CONTROL/'runtime_status.json').read_text())
    if not report_path.exists() or state['status']!='complete':
        write_state('not_launched_control_incomplete');return
    report=json.loads(report_path.read_text())
    if not report.get('best_checkpoint_predictions_reproduced'):
        write_state('not_launched_control_unverified');return
    write_state('waiting_capacity',control_scores=report['scores'])
    while True:
        jobs=gpu_jobs()
        query=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        free_mib=int(query.stdout.strip().splitlines()[0])
        if len(jobs)<2 and free_mib>=3072:break
        time.sleep(20)
    if (OUT/'config.json').exists():
        write_state('not_launched_existing_run');return
    command=[sys.executable,'-u','experiments/pure_dl/train_abd_channel_heads80.py',
             '--out','outputs/pure_dl_frozen_backbone_channel_heads']
    child=subprocess.Popen(command,cwd=ROOT,creationflags=subprocess.CREATE_NO_WINDOW)
    write_state('child_running',child_pid=child.pid,other_gpu_jobs=jobs)
    result=child.wait()
    write_state('child_complete' if result==0 else 'child_failed',child_pid=child.pid,exit_code=result)


if __name__=='__main__':main()
