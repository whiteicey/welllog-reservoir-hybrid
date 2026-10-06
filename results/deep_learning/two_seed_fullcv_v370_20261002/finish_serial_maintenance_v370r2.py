"""Continue six held CPU-only source compactions after explicit batch recovery."""
import os,sys,time
from pathlib import Path
import psutil

study=Path(__file__).resolve().parent
root=study.parents[2]
sys.path.insert(0,str(root/'experiments/pure_dl'))
from compact_source_diagnostics_v370 import compact
from fullcv_pair_context_v370 import read,write

receipt=study/'SERIAL_MAINTENANCE_LAUNCH.json'
assert not receipt.exists(),'No duplicate maintenance launch'
prior=read(study/'MAINTENANCE_LAUNCH.json')
assert not psutil.pid_exists(prior['pid'])
held=read(study/'MAINTENANCE_STATE.json')
assert held['status']=='held_after_batch_terminal_failure_or_stop'
assert read(study/'SERIAL_RECOVERY_LAUNCH.json')['authorization']['explicit_user_approved']
pending={tuple(item) for item in held['pending']}
assert pending=={(i,f) for i in ('original','seed_20261013') for f in (2,3,4)}
write(receipt,dict(pid=os.getpid(),started_epoch=time.time(),pending=sorted(pending),
    CPU_only=True,launch_training=False,checkpoint_deletions=0,
    authorization='Existing user cleanup scope, resumed finite training batch explicitly authorized'))
while pending:
    state=read(study/'BATCH_STATE.json')
    if state['status'] in ('failed_or_stopped','stopped'):
        write(study/'MAINTENANCE_STATE.json',dict(status='held_after_batch_terminal_failure_or_stop',pending=sorted(pending)));break
    for identity,fold in sorted(pending.copy()):
        if compact(identity,fold):pending.remove((identity,fold))
    write(study/'MAINTENANCE_STATE.json',dict(status='waiting_for_completed_sources' if pending else 'complete',
        pid=os.getpid(),pending=sorted(pending),observed_epoch=time.time(),GPU_work=False))
    if pending:time.sleep(30)
