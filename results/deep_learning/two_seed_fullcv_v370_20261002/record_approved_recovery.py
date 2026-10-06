"""Record live activation of the explicit recovery, without training writes."""
import json,time
from pathlib import Path
import psutil

study=Path(__file__).resolve().parent
root=study.parents[2]
state=json.loads((study/'BATCH_STATE.json').read_text(encoding='utf-8'))
launch=json.loads((study/'SERIAL_RECOVERY_LAUNCH.json').read_text(encoding='utf-8'))
controller=psutil.Process(state['controller_pid'])
assert 'resume_pair_serial_v370r2.py' in ' '.join(controller.cmdline())
assert state['status']=='running' and state['maximum_GPU_jobs']==1
jobs=[]
for job in state['active_jobs']:
    p=psutil.Process(job['pid'])
    assert abs(p.create_time()-job['create_time'])<.01
    assert 'pair_fullcv_' in ' '.join(p.cmdline())
    runtime=json.loads((Path(job['dest'])/'runtime_status.json').read_text())
    jobs.append(dict(pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline(),runtime=runtime))
assert len(jobs)==1
phase=root/'outputs/pure_dl_two_seed_fullcv_v370/fold_1/original/PERM336'
report=json.loads((phase/'report.json').read_text())
timing=json.loads((phase/'PERM336_TIMING.json').read_text())
log=(phase/'PERM336_serial_resume_stdout.log').read_text()
assert timing['resumed'] and timing['exit_code']==0 and not psutil.pid_exists(timing['pid'])
assert '"epoch": 10' in log and report['stopped_epoch']==14
assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
result=dict(observed_epoch=time.time(),explicit_user_reply='从断点串行恢复（推荐）',
    controller_pid=controller.pid,controller_create_time=controller.create_time(),
    controller_cmdline=controller.cmdline(),maximum_GPU_jobs=1,active_jobs=jobs,
    original_PERM336_resume_complete=True,resumed_next_epoch=10,last_complete_epoch=9,
    terminal_report=report,timing=timing,unchanged_training_recipe=True,
    complete_five_fold_OOF=False,authorization_source='Direct user response; UTF-8 readback of original answer')
(study/'RECOVERY_ACTIVATION_VERIFIED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
note='\n2026-10-02 explicit recovery activated: user confirmed checkpoint serial recovery. Controller27968, maximum1 GPU. Original fold1 PERM336 resumed from complete epoch9, first new epoch10, stopped14/best0, target90.989045182657; disk predictions and frozen weights exact. This stage adds no optimizer gain, and is -0.000007525098 versus prior PERM154 due to initial projection. SW351 now active. Seed13 PERM154 epoch7 recovery remains queued. Same8-stage private models, fixed and foldwise-tuned scans unchanged; no fullOOF yet. Old OOM evidence retained; another failure stops further launches. RECOVERY_ACTIVATION_VERIFIED.json preserves actual Chinese user reply (older launch files encoded that display field as question marks).\n'
for path in (study/'PROJECTS.md',study/'README.md',root/'PROJECTS.md',root.parent/'PROJECTS.md',root/'docs/deep_learning/atlas/PROJECTS.md'):
    with path.open('a',encoding='utf-8') as f:f.write(note)
print(json.dumps(dict(controller_pid=controller.pid,jobs=jobs,original_PERM336_score=report['target_score'],maximum_GPU_jobs=1),ensure_ascii=False))
