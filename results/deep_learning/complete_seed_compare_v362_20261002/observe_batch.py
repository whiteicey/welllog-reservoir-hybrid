"""Bounded CPU observer. Never start, stop, signal, resume, or modify trainers."""
from pathlib import Path
import argparse, json, os, time
import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ap = argparse.ArgumentParser()
ap.add_argument('--seconds', type=int, default=180)
ap.add_argument('--interval', type=int, default=40)
args = ap.parse_args()
assert 0 < args.seconds <= 600 and 10 <= args.interval <= 50
initial = json.loads((HERE/'BATCH_STATE.json').read_text(encoding='utf-8'))
expected = (initial.get('seed'), initial.get('stage'), initial.get('active_pid'))
deadline = time.monotonic()+args.seconds
while True:
    try:
        batch = json.loads((HERE/'BATCH_STATE.json').read_text(encoding='utf-8'))
        actual = {}
        for key in ('controller_pid', 'active_pid'):
            proc = psutil.Process(batch[key])
            actual[key] = dict(pid=proc.pid, create_time=proc.create_time(),
                               status=proc.status(), cmdline=proc.cmdline())
        if (batch.get('seed'),batch.get('stage'),batch.get('active_pid')) != expected:
            print('HANDOFF', json.dumps(dict(batch=batch,actual=actual)),flush=True)
            break
        own = ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'/f"seed_{batch['seed']}"
        path = own/('SW351_fixed' if batch['stage']=='SW351' else batch['stage'])
        if batch['stage']=='SOURCE':
            history = json.loads((path/'fold_5_validation_history.json').read_text(encoding='utf-8'))
            state = history[-1]
            compact = dict(epoch=state['epoch'],best=state['best_score'],
                           best_epoch=state['best_epoch'],bad=state['bad_epochs'])
        else:
            state = json.loads((path/'runtime_status.json').read_text(encoding='utf-8'))
            compact = {k:state.get(k) for k in ('epoch','best','best_epoch','bad','effective_patience','SOTA_broken')}
        snapshot = dict(observed_epoch=time.time(),classification='verified_wait',
                        batch_snapshot=batch,confirmed_actual_processes=actual,
                        current_stage_runtime=state,scope='Stage snapshot only, not complete model or OOF',
                        no_restart_or_new_training=True)
        tmp = HERE/f'LIVE_OBSERVATION.{os.getpid()}.tmp'
        tmp.write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
        tmp.replace(HERE/'LIVE_OBSERVATION.json')
        print('LIVE_STAGE',json.dumps(dict(seed=batch['seed'],stage=batch['stage'],
                                          pid=batch['active_pid'],**compact)),flush=True)
    except (OSError,json.JSONDecodeError) as exc:
        print('TRANSIENT_READ',type(exc).__name__,flush=True)
    except psutil.NoSuchProcess as exc:
        print('PROCESS_MISSING_REQUIRES_HANDOFF_CHECK',exc.pid,flush=True)
        break
    remaining=deadline-time.monotonic()
    if remaining<=0:
        break
    time.sleep(min(args.interval,remaining))
print('BOUNDED_OBSERVER_DONE',flush=True)
