"""Record the explicitly approved SW351 recovery, without changing training."""
from pathlib import Path
import hashlib
import json
import time
import psutil

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

auth = read(STUDY / 'OOM_SW351_FOLD4_AUTHORIZATION.json')
launch = read(STUDY / 'SW351_FOLD4_RECOVERY_LAUNCH.json')
state = read(STUDY / 'BATCH_STATE.json')
assert auth['explicit_user_approved'] and launch['authorization'] == auth
assert state['maximum_GPU_jobs'] == 1 and len(state['active_jobs']) == 1
job = state['active_jobs'][0]
assert (job['identity'], job['fold'], job['stage']) == ('seed_20261013', 4, 'SW351')
assert job['resumed'] and '--resume' in job['command']
worker = psutil.Process(job['pid'])
assert abs(worker.create_time() - job['create_time']) < .01
assert '--resume' in worker.cmdline()
controller = psutil.Process(state['controller_pid'])
assert 'resume_sw351_fold4_serial_v370r5.py' in ' '.join(controller.cmdline())
hashes = read(STUDY / 'PREPARATION.json')['code_sha256']
assert all(sha(ROOT / path) == expected for path, expected in hashes.items())
prepared = read(STUDY / 'OOM_SW351_FOLD4_PREPARED.json')
assert sha(STUDY / 'OOM_SW351_FOLD4_PREPARED.json') == auth['prepared_sha256']
assert sha(STUDY / 'resume_sw351_fold4_serial_v370r5.py') == auth['controller_sha256']
assert sha(Path(job['dest']) / 'config.json') == prepared['failed_jobs'][0]['config_sha256']
proof = dict(observed_epoch=time.time(), explicit_user_reply=auth['user_reply'],
             maximum_GPU_jobs=1, controller_pid=controller.pid,
             controller_create_time=controller.create_time(),
             controller_command=controller.cmdline(), worker_pid=worker.pid,
             worker_create_time=worker.create_time(), worker_command=worker.cmdline(),
             completed_epoch_to_restore=11, config_unchanged=True,
             frozen_seven_source_hashes_exact=True,
             checkpoint_sha_verified_by_controller_before_launch=True,
             scope='Approved launch/process verification; terminal recovery and full OOF remain pending.')
dest = STUDY / 'SW351_FOLD4_RECOVERY_ACTIVATION_QA.json'
assert not dest.exists()
dest.write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
note = ('\n\n2026-10-02 user explicitly approved "重试断点续训就好了". '
        'Final seed13/fold4 SW351 recovery launched from completed epoch11 with unchanged '
        'logical batch64/config/optimizer/scheduler/RNG, serial maximum1 GPU. '
        f'Controller PID{controller.pid}, worker PID{worker.pid} with --resume were OS-verified; '
        'seven frozen training/evaluation source hashes unchanged. '
        'Evidence SW351_FOLD4_RECOVERY_ACTIVATION_QA.json and OOM_SW351_FOLD4_AUTHORIZATION.json. '
        'This authorizes the single failed stage and dependent native scans/final OOF only; '
        'future failure still stops, no automatic retry. Terminal score, paired4 and fullOOF remain pending.\n')
for path in (STUDY/'PROJECTS.md', STUDY/'README.md', ROOT/'PROJECTS.md', ROOT.parent/'PROJECTS.md'):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(note)
print(json.dumps(dict(controller_pid=controller.pid, worker_pid=worker.pid, maximum_GPU_jobs=1)))
