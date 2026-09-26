"""Bounded serial/parallel GPU comparison; temporarily suspend and restore owned jobs.

Measures identical synthetic training work, excluding startup/warmup/teardown.
Retains suspended jobs' GPU allocations in both modes. No training outputs edited.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def worker(args):
    import torch
    from adaptive_scale_sequence import Network
    from private_head_sequence import Network as SAMNetwork
    from multiscale_sequence import objective
    from sam_sequence_step import sam_step
    torch.set_num_threads(3); torch.manual_seed(20261006)
    start = time.perf_counter()
    model = (Network() if args.kind == 'adaptive' else SAMNetwork()).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x = torch.randn(64,256,206,device='cuda',dtype=torch.float16)
    depth = torch.linspace(-1,1,256,device='cuda')[None].expand(64,-1)
    valid = torch.ones(64,256,device='cuda',dtype=torch.bool)
    supervised = valid.clone(); supervised[:,:64]=False; supervised[:,192:]=False
    y = torch.randn(64,256,3,device='cuda'); defaults=torch.rand(64,256,3,device='cuda')>.3
    def closure():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=model(x,depth,valid,auxiliary=True)
        return objective(outputs,y,defaults,supervised,True)
    def step():
        if args.kind=='sam':
            loss=sam_step(model,optimizer,closure)
        else:
            optimizer.zero_grad(set_to_none=True);loss=closure()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2);optimizer.step()
        return loss
    for _ in range(6): step()
    torch.cuda.synchronize()
    write(args.out/'ready.json',{'startup_and_warmup_seconds':time.perf_counter()-start})
    deadline=time.monotonic()+240
    while not (args.out/'go').exists():
        if time.monotonic()>deadline: raise TimeoutError('start barrier')
        time.sleep(.01)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    for _ in range(args.steps): loss=step()
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    write(args.out/'done.json',{'kind':args.kind,'steps':args.steps,'seconds':elapsed,
        'ms_per_update':elapsed*1000/args.steps,'last_loss':float(loss),
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    # Keep both contexts resident throughout the measured pair in both modes.
    while not (args.out/'release').exists():
        if time.monotonic()>deadline: raise TimeoutError('release barrier')
        time.sleep(.01)


def controller(args):
    import psutil
    args.out.mkdir(parents=True,exist_ok=True)
    paused=[];children=[];handles=[];trials=[]
    expected={8980:'train_sam_sequence.py',40648:'train_adaptive_scale_sequence.py'}
    begin=time.time()
    try:
        for pid, script in expected.items():
            try: p=psutil.Process(pid)
            except psutil.NoSuchProcess: continue
            if not any(Path(a).name==script for a in p.cmdline()):
                raise RuntimeError(f'PID identity mismatch: {pid}')
            p.suspend();paused.append(p)
        write(args.out/'suspended.json',{'pids':[p.pid for p in paused],'started_unix':begin})
        time.sleep(2)
        def wait_file(path, process, timeout=180):
            deadline=time.monotonic()+timeout
            while not path.exists():
                if process.poll() is not None: raise RuntimeError(f'worker failed: {path.parent}')
                if time.monotonic()>deadline: raise TimeoutError(str(path))
                time.sleep(.01)
        for repetition in range(3):
            for mode in (('serial','parallel') if repetition%2==0 else ('parallel','serial')):
                jobs=[]
                for kind in ('sam','adaptive'):
                    dest=args.out/f'{repetition+1}_{mode}_{kind}';dest.mkdir()
                    log=(dest/'worker.log').open('w',encoding='utf-8');handles.append(log)
                    process=subprocess.Popen([sys.executable,__file__,'--worker','--kind',kind,
                        '--out',str(dest),'--steps',str(args.steps)],stdout=log,stderr=subprocess.STDOUT)
                    children.append(process);jobs.append((dest,process))
                for dest,process in jobs:wait_file(dest/'ready.json',process)
                start=time.perf_counter()
                if mode=='serial':
                    for dest,process in jobs:
                        (dest/'go').touch();wait_file(dest/'done.json',process)
                else:
                    for dest,_ in jobs:(dest/'go').touch()
                    for dest,process in jobs:wait_file(dest/'done.json',process)
                pair_seconds=time.perf_counter()-start
                trial={'repetition':repetition+1,'mode':mode,'pair_seconds':pair_seconds,
                       'workers':[json.loads((dest/'done.json').read_text()) for dest,_ in jobs]}
                trials.append(trial);write(args.out/'trials.json',trials)
                print(json.dumps(trial),flush=True)
                for dest,_ in jobs:(dest/'release').touch()
                for _,process in jobs:process.wait(timeout=30)
        means={mode:sum(t['pair_seconds'] for t in trials if t['mode']==mode)/3 for mode in ('serial','parallel')}
        write(args.out/'report.json',{'trials':trials,'mean_pair_seconds':means,
            'parallel_speedup':means['serial']/means['parallel'],
            'parallel_time_reduction_fraction':1-means['parallel']/means['serial'],
            'steps_per_worker':args.steps,'scope':'synthetic SAM + adaptive pair; excludes startup, warmup, preprocessing, inference, teardown',
            'same_resident_context_count':True,'suspended_existing_pids':[p.pid for p in paused]})
    finally:
        for process in children:
            if process.poll() is None:process.terminate()
        for process in children:
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for log in handles:log.close()
        restored=[]
        for p in paused:
            try:p.resume();restored.append(p.pid)
            except psutil.NoSuchProcess:pass
        write(args.out/'restored.json',{'pids':restored,'measurement_total_seconds':time.time()-begin})
        print('RESTORED',restored,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true')
    ap.add_argument('--kind',choices=['sam','adaptive']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=96);args=ap.parse_args()
    worker(args) if args.worker else controller(args)
