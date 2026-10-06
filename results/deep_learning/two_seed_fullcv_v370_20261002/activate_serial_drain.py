"""Switch only the suspended CPU dispatcher; never stop GPU children."""
import os,sys,json,time,subprocess,hashlib,psutil
from pathlib import Path
S=Path(__file__).resolve().parent;R=S.parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
handoff=read(S/'DRAIN_TO_SERIAL_PREPARATION.json');approval=read(S/'SERIAL_DRAIN_AUTHORIZATION.json')
assert not (S/'SERIAL_DRAIN_ROOT_LAUNCH.json').exists()
p=S/'run_serial_drain_pair_fullcv_v370r4.py';assert sha(p)==approval['controller_sha256']
for name,digest in read(S/'PREPARATION.json')['code_sha256'].items():assert sha(R/name)==digest
old_info=handoff['old_CPU_controller'];old=psutil.Process(old_info['pid'])
assert old.create_time()==old_info['create_time'] and old.cmdline()==old_info['cmdline']
for saved in handoff['active_jobs']:
    try:
        worker=psutil.Process(saved['pid']);assert worker.create_time()==saved['create_time'] and worker.cmdline()==saved['command']
    except psutil.NoSuchProcess:
        report=read(Path(saved['expected']))
        assert report.get('predictions_reproduced_exactly') or report.get('best_checkpoint_predictions_reproduced')
old.kill();old.wait(timeout=10)
environment=os.environ.copy();environment.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
with (S/'serial_drain_controller_stdout.log').open('w',encoding='utf-8') as out,(S/'serial_drain_controller_stderr.log').open('w',encoding='utf-8') as err:
    child=subprocess.Popen([sys.executable,'-u',str(p)],cwd=R,env=environment,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
result=dict(controller_pid=child.pid,create_time=psutil.Process(child.pid).create_time(),started_epoch=time.time(),old_CPU_controller_stopped=old_info['pid'],GPU_workers_stopped=0,preserved_jobs=handoff['active_jobs'],maximum_future_GPU_jobs=1)
(S/'SERIAL_DRAIN_ROOT_LAUNCH.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(new_controller_pid=child.pid,preserved_GPU_pids=[j['pid'] for j in handoff['active_jobs']],maximum_future_GPU_jobs=1)))
