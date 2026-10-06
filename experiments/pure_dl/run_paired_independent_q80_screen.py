"""Run exactly the user-selected fifth-fold screen, export results; never auto-resume folds."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

out=Path('outputs/pure_dl_paired_independent_q80_sequence')
out.mkdir(exist_ok=True)
usage=shutil.disk_usage(out.resolve())
(out/'launch_storage.json').write_text(json.dumps({'path':str(out.resolve()),'free_bytes':usage.free,'total_bytes':usage.total,'expected_cdf_scratch_bytes':578891*3*256*4*3,'note':'additional compressed evidence shards consume space; no migration or unrelated cleanup'},indent=2),encoding='utf-8')
print('LAUNCH_FREE_DISK_GIB',usage.free/1024**3,flush=True)
with (out/'train.log').open('w',encoding='utf-8') as log, (out/'train.err').open('w',encoding='utf-8') as error:
    process=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_paired_independent_q80_sequence.py','--out',str(out),'--learning-rate','0.0006','--schedule','cosine'],stdout=log,stderr=error)
    print('TRAINING_PID',process.pid,flush=True)
    result=process.wait()
assert result==0,'Training did not complete; no restart'
assert (out/'screening_report.json').exists()
print('FIFTH_FOLD_TRAINING_COMPLETE; inspect score before any continuation',flush=True)
