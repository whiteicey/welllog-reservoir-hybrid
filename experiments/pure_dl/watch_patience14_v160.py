"""One authorized live trainer: request existing boundary stop after bad14."""
import json,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/pure_dl_por_rps_utility_v160_fold5/joint_rps'
STUDY=ROOT/'results/deep_learning/por_rps_utility_v160_20261001'
PID=62172
while True:
    try:
        cmd=psutil.Process(PID).cmdline()
        if not any(Path(a).name=='train_por_rps_utility_v160.py' for a in cmd):break
    except psutil.Error:break
    p=OUT/'runtime_status.json'
    if p.exists():
        try:state=json.loads(p.read_text(encoding='utf-8'))
        except (OSError,ValueError):time.sleep(1);continue
        if state.get('status')!='running':break
        if state.get('epoch',0)>=10 and state.get('bad',0)>=14:
            request=STUDY/'joint_rps_early_stop_request.json'
            tmp=request.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(reason='explicit user patience14 override; frozen trainer unchanged',patience=14,observed=state)),encoding='utf-8')
            tmp.replace(request)
            print('REQUESTED',state,flush=True);break
    time.sleep(1)
