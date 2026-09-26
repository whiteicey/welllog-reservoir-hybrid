"""Verify a completed first-fold screen; never auto-launch remaining folds."""
import argparse
import subprocess
import sys
from pathlib import Path
import psutil

ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);args=ap.parse_args()
try:
    process=psutil.Process(args.pid)
    assert any(Path(a).name=='train_curve_token_sequence.py' for a in process.cmdline())
    while True:
        try:process.wait(timeout=30);break
        except psutil.TimeoutExpired:continue
except psutil.NoSuchProcess:pass
out=Path('outputs/pure_dl_curve_token_sequence')
assert (out/'screening_report.json').exists(),'No completed screen; no automatic restart'
subprocess.run([sys.executable,'experiments/pure_dl/verify_curve_token_sequence.py','--out',str(out),'--folds','1'],check=True)
print('FIRST_FOLD_VERIFIED; remaining folds require a separate threshold-gated invocation',flush=True)
