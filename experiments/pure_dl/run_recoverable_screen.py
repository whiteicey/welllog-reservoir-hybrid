"""OBSOLETE_NOT_TRAINED: controller disabled following updated user instruction."""
raise RuntimeError("OBSOLETE_NOT_TRAINED: user withdrew this policy; no GPU may be started")
import subprocess,sys,json,time
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'outputs/pure_dl_recoverable';out.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(out).free>2*1024**3,'Require at least 2 GiB free before training'
started=time.time()
with (out/'train.log').open('w',encoding='utf-8') as log,(out/'train.err').open('w',encoding='utf-8') as err:
    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'experiments/pure_dl/train_recoverable_sequence.py'),'--out',str(out),'--learning-rate','.001','--schedule','cosine','--fold-count','1'],cwd=ROOT,stdout=log,stderr=err)
    print('TRAINING_PID',process.pid,flush=True);code=process.wait()
result={'training_pid':process.pid,'exit_code':code,'controller_wall_seconds':time.time()-started,'independent_verification':'pending separate parent audit'}
(out/'controller_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
if code:raise SystemExit(code)
