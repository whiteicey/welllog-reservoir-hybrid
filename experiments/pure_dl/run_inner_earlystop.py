import subprocess,sys
from pathlib import Path
out=Path('outputs/pure_dl_inner_earlystop')
with (out/'train.log').open('w') as log,(out/'train.err').open('w') as err:
    proc=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_inner_earlystop.py'],stdout=log,stderr=err)
    print('TRAINING_PID',proc.pid,flush=True)
    assert proc.wait()==0,'pilot failed; inspect logs'
