"""Verify CPU-only handoff and publish scoped stage scores."""
from pathlib import Path
import json,time,hashlib,psutil
S=Path(__file__).resolve().parent;R=S.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
state=read(S/'BATCH_STATE.json');launch=read(S/'SERIAL_DRAIN_ROOT_LAUNCH.json');auth=read(S/'SERIAL_DRAIN_AUTHORIZATION.json')
controller=psutil.Process(state['controller_pid'])
assert state['controller_pid']==launch['controller_pid'] and controller.create_time()==launch['create_time']
assert controller.cmdline()[-1]==str(S/'run_serial_drain_pair_fullcv_v370r4.py')
assert state['maximum_GPU_jobs']==1 and state['no_new_overlap']
assert launch['GPU_workers_stopped']==0 and not psutil.pid_exists(launch['old_CPU_controller_stopped'])
script=(S/'run_serial_drain_pair_fullcv_v370r4.py').read_text(encoding='utf-8')
assert 'if len(active)>=1:return False' in script and 'assert len(active)<1 and eligible(job)' in script
assert hashlib.sha256((S/'run_serial_drain_pair_fullcv_v370r4.py').read_bytes()).hexdigest()==auth['controller_sha256']
hash_checks={n:hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in read(S/'PREPARATION.json')['code_sha256'].items()};assert all(hash_checks.values())
preserved={j['pid']:j for j in launch['preserved_jobs']};jobs=[]
for j in state['active_jobs']:
    p=psutil.Process(j['pid']);assert p.create_time()==j['create_time'] and p.cmdline()==j['command']
    assert j['pid'] in preserved
    d=Path(j['dest']);runtime={}
    if (d/'runtime_status.json').exists():runtime=read(d/'runtime_status.json')
    if j['stage']=='SOURCE':runtime=read(d/f"fold_{j['fold']}_validation_history.json")[-1]
    jobs.append(dict(identity=j['identity'],fold=j['fold'],stage=j['stage'],pid=j['pid'],create_time=j['create_time'],preserved_exactly=True,runtime=runtime,score_scope='training snapshot, not terminal model or OOF'))
prefix=read(S/'fold_2/original_PREFIX_SNAPSHOT.json')
result=dict(observed_epoch=time.time(),controller_pid=controller.pid,maximum_future_GPU_jobs=1,existing_overlapping_jobs_allowed_to_finish=True,no_new_overlap_gate_verified=True,GPU_workers_stopped=0,frozen_code_hash_checks=hash_checks,active_jobs=jobs,fold2_original_completed_prefix=dict(stages=prefix['completed_prefix'],scores=prefix['prefix_scores'],scope=prefix['score_scope']),complete_five_fold_OOF=False)
(S/'SERIAL_DRAIN_ACTIVATION_QA.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
note='\n2026-10-02 latest direct user instruction: existing two workers may finish, thereafter strictly serial. CPU-only handoff77612->'+str(controller.pid)+' preserved exact SOURCE75188 and POR34156476; no GPU worker stopped/restarted. New dispatcher cannot launch while ANY worker is active, maximum future GPU jobs1. All frozen training/source hashes unchanged. Any failure blocks new launches without retry. Current completed fold2 original prefix '+str(len(prefix['completed_prefix']))+'/8 TOTAL '+str(prefix['prefix_scores']['TOTAL'])+' is not complete model/fullOOF. Evidence SERIAL_DRAIN_ACTIVATION_QA.json; pending finish original+seed13 folds2-4/native scans/full OOF. Subsequent speed decisions respect this explicit serial instruction.\n'
for p in [S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md',R/'docs/deep_learning/atlas/PROJECTS.md']:
    with p.open('a',encoding='utf-8') as stream:stream.write(note)
with (S/'AGENTS.md').open('a',encoding='utf-8') as stream:
    stream.write('\n## Latest user instruction: drain current overlap, then serial\nExisting SOURCE75188 and POR34156476 may finish without restart. No new GPU worker may launch until all existing workers are terminal; thereafter maximum1. Preserve model recipe, frozen hashes, explicit failure-recovery boundary and native fixed/diagnostic OOF separation. Maintain PROJECTS after batches. Evidence SERIAL_DRAIN_AUTHORIZATION.json and SERIAL_DRAIN_ACTIVATION_QA.json.\n')
print(json.dumps(dict(controller_pid=controller.pid,active_jobs=[dict(pid=j['pid'],stage=j['stage'],epoch=j['runtime'].get('epoch'),best=j['runtime'].get('best_score',j['runtime'].get('best'))) for j in jobs],completed_prefix=result['fold2_original_completed_prefix'])))
