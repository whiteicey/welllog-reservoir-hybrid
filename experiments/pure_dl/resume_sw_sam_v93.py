"""Explicit serial recovery of confirmed failed v93 jobs, no queue."""
import argparse,json,subprocess,sys
from datetime import datetime
from sw_sam_v93 import ROOT,STUDY,MODES
from run_retrieval_smooth_v36 import jobs
from train_dual8222_heads80 import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True);a=ap.parse_args()
    if jobs():raise RuntimeError('Serial recovery requires no active workspace trainers')
    out=ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{a.mode}_fold5'
    status=json.loads((out/'runtime_status.json').read_text(encoding='utf-8'))
    if status['status']!='failed':raise RuntimeError('Only confirmed failed v93 trial allowed')
    if not (out/'last_state.pt').is_file():raise RuntimeError('Missing atomic recovery state')
    cmd=[sys.executable,'-X','utf8','-u',str(ROOT/'experiments/pure_dl/train_sw_sam_v93.py'),'--mode',a.mode,'--out',str(out),'--resume']
    with (out/'train.resume1.log').open('wb') as log,(out/'train.resume1.err').open('wb') as err:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    launch=dict(mode=a.mode,pid=proc.pid,start=datetime.now().astimezone().isoformat(),status='running',serial_recovery=True,command=cmd)
    atomic_json(out/'resume1_launch.json',launch)
    path=STUDY/'execution_segments.json';segments=json.loads(path.read_text(encoding='utf-8'));segments.append(launch);atomic_json(path,segments)
    print(json.dumps(launch),flush=True)
if __name__=='__main__':main()
