"""Bounded CPU observation of actual live process; timeout never means stopped."""
from pathlib import Path
import argparse,json,sys,time
import psutil

ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,default=180);ap.add_argument('--interval',type=int,default=40)
a=ap.parse_args();start=time.time();initial=None
while True:
    state=json.loads((STUDY/'BATCH_STATE.json').read_text(encoding='utf-8-sig'))
    actual={};pid=state.get('active_pid')
    if pid:
        try:
            p=psutil.Process(pid);cmd=p.cmdline();created=p.create_time()
            assert 'pair_fullcv_' in ' '.join(cmd) and any(k in ' '.join(cmd) for k in ('train_','evaluate_'))
            actual=dict(pid=pid,create_time=created,cmdline=cmd,alive=True)
        except psutil.Error:actual=dict(pid=pid,alive=False)
    controller=state.get('controller_pid')
    try:
        p=psutil.Process(controller);controller_actual=dict(pid=controller,alive=True,cmdline=p.cmdline(),create_time=p.create_time())
    except psutil.Error:controller_actual=dict(pid=controller,alive=False)
    stage=state.get('stage');fold=state.get('fold');identity=state.get('identity')
    phase=ROOT/'outputs/pure_dl_two_seed_fullcv_v370'/f'fold_{fold}'/str(identity)/str(stage)
    runtime={}
    path=phase/'runtime_status.json'
    if path.exists():runtime=json.loads(path.read_text(encoding='utf-8-sig'))
    path=phase/f'fold_{fold}_validation_history.json'
    if stage=='SOURCE' and path.exists():
        history=json.loads(path.read_text(encoding='utf-8-sig'))
        if history:runtime=history[-1]
    jobs=[]
    for job in state.get('active_jobs',[]):
        checked=dict(pid=job['pid'],identity=job.get('identity'),fold=job.get('fold'),stage=job.get('stage'))
        try:
            p=psutil.Process(job['pid']);cmd=p.cmdline();created=p.create_time()
            assert abs(created-job['create_time'])<.01
            assert 'pair_fullcv_' in ' '.join(cmd)
            checked.update(alive=True,create_time=created,cmdline=cmd)
        except psutil.Error:checked.update(alive=False)
        phase=Path(job['dest']);jr={}
        rp=phase/'runtime_status.json'
        if rp.exists():jr=json.loads(rp.read_text(encoding='utf-8-sig'))
        hp=phase/f'fold_{job.get("fold")}_validation_history.json'
        if job.get('stage')=='SOURCE' and hp.exists():
            history=json.loads(hp.read_text(encoding='utf-8-sig'))
            if history:jr=history[-1]
        checked['runtime']=jr;jobs.append(checked)
    value=dict(observed_epoch=time.time(),batch=state,actual=actual,controller=controller_actual,runtime=runtime,jobs=jobs)
    (STUDY/'LIVE_OBSERVATION.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    with (STUDY/'OBSERVATIONS.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')
    compact=dict(status=state['status'],identity=identity,fold=fold,stage=stage,pid=pid,actual_alive=actual.get('alive'),
                 epoch=runtime.get('epoch'),best=runtime.get('best_score',runtime.get('best')),bad=runtime.get('bad_epochs',runtime.get('bad')),
                 jobs=[dict(pid=j['pid'],identity=j['identity'],fold=j['fold'],stage=j['stage'],alive=j['alive'],
                       epoch=j['runtime'].get('epoch'),best=j['runtime'].get('best_score',j['runtime'].get('best')),
                       bad=j['runtime'].get('bad_epochs',j['runtime'].get('bad'))) for j in jobs])
    print(json.dumps(compact),flush=True)
    signature=(pid,actual.get('create_time'))
    if initial is None:initial=signature
    if state['status']!='running':print('AUTHORITATIVE_TERMINAL',flush=True);break
    if signature!=initial:print('HANDOFF_VERIFY_NEXT_CHILD',flush=True);break
    if not actual.get('alive'):
        print('HANDLE_MISSING_RECHECK_CONTROLLER_STATE_BEFORE_ANY_ACTION',flush=True);break
    if time.time()-start>=a.seconds:print('BOUNDED_LIVE_OBSERVATION_COMPLETE_NOT_A_STOP',flush=True);break
    time.sleep(min(a.interval,a.seconds-(time.time()-start)))
