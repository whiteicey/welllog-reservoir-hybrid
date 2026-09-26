import json,subprocess,sys
from pathlib import Path
for name in ['spectral','heterogeneous_moe']:
    subprocess.run([sys.executable,'-u',f'experiments/pure_dl/run_{name}_screen.py'],check=True)
    proof=json.loads(Path(f'outputs/pure_dl_{name}_sequence/screening_verification.json').read_text())
    print('VERIFIED_CANDIDATE',name,json.dumps(proof),flush=True)
    if proof['qualifies_for_remaining_folds']:
        print('QUALIFIED: stop queue for five-fold continuation',flush=True);break
