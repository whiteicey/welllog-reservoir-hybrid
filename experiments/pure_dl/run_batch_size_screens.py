"""Sequential fixed-fold batch-size comparison; no automatic remaining folds."""
import subprocess
import sys
from pathlib import Path
for batch in (32,128):
    out=Path(f'outputs/pure_dl_batch_size_{batch}');out.mkdir(exist_ok=True)
    with (out/'train.log').open('w') as log,(out/'train.err').open('w') as err:
        proc=subprocess.Popen([sys.executable,'-u','experiments/pure_dl/train_batch_size_sequence.py','--out',str(out),'--batch-size',str(batch)],stdout=log,stderr=err)
        print('TRAINING',batch,'PID',proc.pid,flush=True)
        assert proc.wait()==0,'Training failed; no restart'
    subprocess.run([sys.executable,'experiments/pure_dl/verify_batch_size_sequence.py','--out',str(out),'--folds','1'],check=True)
    print('VERIFIED_BATCH',batch,flush=True)
