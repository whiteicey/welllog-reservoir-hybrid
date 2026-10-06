"""Exact CPU dispatcher handoff after user's explicit second-lane request."""
import json,os,time,subprocess,sys
from pathlib import Path
import psutil

study=Path(__file__).resolve().parent;root=study.parents[2]
sys.path.insert(0,str(root/'experiments/pure_dl'))
from fullcv_pair_context_v370 import read,write,sha
from run_overlap_pair_fullcv_v370r1 import actual_gpu_pids

controller=study/'run_safe_overlap_pair_fullcv_v370r3.py'
assert not (study/'SAFE_OVERLAP_HANDOFF.json').exists() and not (study/'SAFE_OVERLAP_LAUNCH.json').exists()
assert not (root/'outputs/pure_dl_two_seed_fullcv_v370/STOP_BATCH.json').exists()
prep=read(study/'PREPARATION.json')
for name,digest in prep['code_sha256'].items():assert sha(root/name)==digest,name
write(study/'SAFE_OVERLAP_AUTHORIZATION.json',dict(explicit_user_approved=True,maximum_GPU_jobs=2,
    user_reply='可以加入另外一个并发线程',authorized_epoch=time.time(),controller_sha256=sha(controller),
    scope='Unchanged finite original+seed13 folds1-4/native scans, preserve current GPU worker; no new experiments or failure retries'))
old=psutil.Process(27968)
assert 'resume_pair_serial_v370r2.py' in ' '.join(old.cmdline())
old_info=dict(pid=old.pid,create_time=old.create_time(),cmdline=old.cmdline())
captured=None
for attempt in range(10):
    old.suspend()
    try:
        state=read(study/'BATCH_STATE.json')
        assert state['controller_pid']==old.pid and state['status']=='running' and state['maximum_GPU_jobs']==1
        assert len(state['active_jobs'])==1
        job=state['active_jobs'][0];worker=psutil.Process(job['pid'])
        assert abs(worker.create_time()-job['create_time'])<.01
        assert worker.cmdline()==job['command'] and sorted(actual_gpu_pids())==[worker.pid]
        captured=dict(pid=worker.pid,create_time=worker.create_time(),cmdline=worker.cmdline())
        break
    except (AssertionError,psutil.Error):
        old.resume();time.sleep(1)
else:raise RuntimeError('Could not capture a stable existing GPU worker; serial dispatcher left running')
try:
    handoff=dict(created_epoch=time.time(),old_controller=old_info,batch=state,attached_child=captured,
        GPU_child_preserved=True,no_model_training_changes=True,
        overlap_controller_sha256=sha(controller),latest_user_authorization='可以加入另外一个并发线程',
        maximum_GPU_jobs=2,maximum_PERM_jobs=1,minimum_driver_free_MiB=7*1024)
    write(study/'SAFE_OVERLAP_HANDOFF.json',handoff)
    old.kill();old.wait(timeout=10)
    worker=psutil.Process(captured['pid'])
    assert worker.create_time()==captured['create_time'] and worker.cmdline()==captured['cmdline']
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    with (study/'safe_overlap_controller_stdout.log').open('w',encoding='utf-8') as out,(study/'safe_overlap_controller_stderr.log').open('w',encoding='utf-8') as err:
        new=subprocess.Popen([sys.executable,'-u',str(controller)],cwd=root,env=env,
            stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    write(study/'SAFE_OVERLAP_ROOT_LAUNCH.json',dict(controller_pid=new.pid,create_time=psutil.Process(new.pid).create_time(),
        preserved_GPU_child=captured,old_CPU_controller_stopped=True,started_epoch=time.time(),GPU_workers_stopped=0))
    print(dict(new_controller_pid=new.pid,preserved_GPU_pid=worker.pid,maximum_GPU_jobs=2),flush=True)
except BaseException:
    if old.is_running():old.resume()
    raise
