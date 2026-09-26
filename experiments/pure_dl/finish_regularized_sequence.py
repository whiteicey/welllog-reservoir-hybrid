"""Wait for the fixed direct-reward candidate, then independently verify it."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import psutil

ap = argparse.ArgumentParser(); ap.add_argument('--pid', type=int, required=True)
args = ap.parse_args()
root = Path(__file__).resolve().parents[2]
assert Path.cwd().resolve() == root
out = root/'outputs/pure_dl_regularized_sequence'
try:
    process = psutil.Process(args.pid)
    assert any(Path(arg).name == 'train_regularized_sequence.py' for arg in process.cmdline())
    print('WAITING_FOR', args.pid, flush=True)
    while True:
        try:
            process.wait(timeout=30); break
        except psutil.TimeoutExpired:
            continue
except psutil.NoSuchProcess:
    pass
assert (out/'report.json').is_file(), 'Training ended without a complete report; no automatic restart'
for script in ['verify.py', 'verify_regularized_sequence.py']:
    subprocess.run([sys.executable, 'experiments/pure_dl/'+script, '--out', str(out)], check=True)
subprocess.run([sys.executable, 'experiments/pure_dl/summarize.py'], check=True)
print('VERIFIED_COMPLETE', json.loads((out/'verification.json').read_text())['score'], flush=True)
