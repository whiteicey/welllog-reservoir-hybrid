"""Run exactly the user-selected fifth-fold screen, then verify; never auto-resume folds."""
import subprocess
import sys
from pathlib import Path

out=Path('outputs/pure_dl_annealed_aux_sequence')
out.mkdir(exist_ok=True)
with (out/'train.log').open('w',encoding='utf-8') as log, (out/'train.err').open('w',encoding='utf-8') as error:
    process=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_annealed_aux_sequence.py','--out',str(out),'--learning-rate','0.001','--schedule','cosine'],stdout=log,stderr=error)
    print('TRAINING_PID',process.pid,flush=True)
    result=process.wait()
assert result==0,'Training did not complete; no restart'
assert (out/'screening_report.json').exists()
subprocess.run([sys.executable,'experiments/pure_dl/verify_annealed_aux_sequence.py','--out',str(out),'--folds','1'],check=True)
print('FIFTH_FOLD_VERIFIED; no automatic remaining folds',flush=True)

subprocess.run([sys.executable,'experiments/pure_dl/verify_global_recovery2_table.py','--out',str(out)],check=True)
