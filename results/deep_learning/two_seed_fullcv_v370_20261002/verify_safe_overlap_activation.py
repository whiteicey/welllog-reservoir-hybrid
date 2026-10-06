"""Bounded CPU readback of authorized dispatcher handoff; no training restart."""
from pathlib import Path
import hashlib,json,time,psutil,subprocess

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=read(STUDY/'BATCH_STATE.json')
handoff=read(STUDY/'SAFE_OVERLAP_HANDOFF.json')
launch=read(STUDY/'SAFE_OVERLAP_ROOT_LAUNCH.json')
prep=read(STUDY/'PREPARATION.json')
assert state['status']=='running' and state['maximum_GPU_jobs']==2
controller=psutil.Process(state['controller_pid'])
assert controller.create_time()==launch['create_time']
assert controller.cmdline()[-1]==str(STUDY/'run_safe_overlap_pair_fullcv_v370r3.py')
assert not psutil.pid_exists(handoff['old_controller']['pid'])
assert launch['GPU_workers_stopped']==0 and handoff['GPU_child_preserved']
checks={name:sha(ROOT/name)==digest for name,digest in prep['code_sha256'].items()}
assert all(checks.values())
jobs=[]
for job in state['active_jobs']:
    p=psutil.Process(job['pid'])
    assert p.create_time()==job['create_time'] and p.cmdline()==job['command']
    jobs.append(dict(identity=job['identity'],fold=job['fold'],stage=job['stage'],pid=job['pid'],create_time=job['create_time'],process_identity_exact=True))
assert len(jobs)==2
query=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True)
result=dict(observed_epoch=time.time(),explicit_user_reply='\u53ef\u4ee5\u52a0\u5165\u53e6\u5916\u4e00\u4e2a\u5e76\u53d1\u7ebf\u7a0b',
    controller_pid=state['controller_pid'],active_jobs=jobs,maximum_GPU_jobs=2,old_CPU_only_stopped=True,
    exact_GPU_worker_preserved_at_handoff=launch['preserved_GPU_child'],GPU_workers_stopped=0,
    frozen_code_sha_checks=checks,training_recipe_unchanged=True,automatic_failure_retry=False,
    maximum_PERM_jobs=1,minimum_launch_driver_free_MiB=7168,driver_snapshot=query.stdout.strip(),
    authorization_text_encoding_note='Earlier launch receipts contain replacement characters; exact direct user instruction preserved in this UTF-8 readback. Numerical authorization and controller hash checks unchanged.',
    score_scope='Two active training jobs; full five-fold OOF remains incomplete')
(STUDY/'SAFE_OVERLAP_ACTIVATION_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
note='\n2026-10-02 latest explicit user authorization restores maximum2 GPU workers for the unchanged original+seed13 fullCV batch. CPU-only handoff from27968 to77612 preserved original/fold2 PERM154 PID74592 without training restart; it completed and advanced normally. Actual process-verified activation jobs: '+json.dumps(jobs)+'. At most one SOURCE/SW/scan/PERM each; launch requires7GiB driver free. All frozen training code hashes unchanged; any new failure stops future launches without automatic retry. Evidence SAFE_OVERLAP_ACTIVATION_QA.json. Earlier serial recovery remains historical, current cap2. Actual overlap and complete same-work batch wall remain to be measured; memory wait is allowed. Full OOF not completed. Pending: finish both seeds folds2-4, uniform native scans, fixed-ratio OOF and diagnostic OOF; possible later stage simplification requires evidence.\n'
for target in [STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
    with target.open('a',encoding='utf-8') as stream:stream.write(note)
print(json.dumps(dict(controller_pid=result['controller_pid'],active_jobs=jobs,code_hashes_unchanged=True,driver=result['driver_snapshot'])))
