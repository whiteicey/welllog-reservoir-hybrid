"""Explicit one-at-a-time recovery after recorded CUDA OOM; no automatic queue."""
from datetime import datetime
import argparse,json,subprocess,sys,shutil
from sw_density_private_v62 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs,append
from train_dual8222_heads80 import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('Recovery is serial; another trainer is active')
    out=ROOT/f'outputs/pure_dl_sw_density_private_v62_SW_{a.mode}_fold5'
    if (out/'fold_5_report.json').exists():raise RuntimeError('Already complete')
    if not (out/'last_state.pt').exists():raise RuntimeError('Missing full resume state')
    if (out/'resume_1_launch.json').exists():raise RuntimeError('Do not repeat recovery automatically')
    failure=json.loads((out/'failure.json').read_text(encoding='utf-8'));assert failure['status']=='failed'
    shutil.copyfile(out/'runtime_status.json',out/'runtime_failure_01.json')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_density_private_v62.py'),'--mode',a.mode,'--out',str(out),'--resume']
    with (out/'resume_1.log').open('wb') as log,(out/'resume_1.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    event=dict(pid=proc.pid,started_at=datetime.now().astimezone().isoformat(),mode=a.mode,command=cmd,serial_recovery=True,automatic_queue=False,
        reason='Both initial jobs failed with CUDA OOM after completed epoch7; source/config unchanged; resume atomic optimizer/scheduler/RNG state')
    atomic_json(out/'resume_1_launch.json',event)
    marker='## v62 '+a.mode+' explicit serial recovery'
    note=marker+'\n'+json.dumps(event,ensure_ascii=False)
    for path in (STUDY/'README.md',STUDY/'PROJECTS.md',out/'README.md',out/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(path,marker,note)
    print(json.dumps(event),flush=True)
if __name__=='__main__':main()
