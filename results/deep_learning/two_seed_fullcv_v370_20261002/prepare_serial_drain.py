"""Prepare CPU dispatcher which drains existing workers then runs serially."""
from pathlib import Path
import py_compile,hashlib,json,time

S=Path(__file__).resolve().parent
text=(S/'run_safe_overlap_pair_fullcv_v370r3.py').read_text(encoding='utf-8')
start=text.index('    require_authorization();handoff=')
end=text.index('    from record_pair_fullcv_v370 import',start)
bootstrap='''    require_authorization();handoff=read(STUDY/'DRAIN_TO_SERIAL_PREPARATION.json')
    if (STUDY/'SERIAL_DRAIN_LAUNCH.json').exists():raise RuntimeError('Already launched: no retry or implicit resume')
    approval=read(STUDY/'SERIAL_DRAIN_AUTHORIZATION.json')
    assert approval['explicit_user_approved'] and approval['maximum_new_GPU_jobs']==1
    assert approval['controller_sha256']==sha(Path(__file__))
    old=handoff['old_CPU_controller'];assert not psutil.pid_exists(old['pid'])
    preparation=read(STUDY/'PREPARATION.json')
    def verify_code():
        for path,digest in preparation['code_sha256'].items():assert sha(ROOT/path)==digest,path
        assert sha(Path(__file__))==approval['controller_sha256']
    verify_code();began=time.time();whole_started=read(STUDY/'LAUNCH.json')['started_epoch']
    active={};handles={};counter=0;events=[];failure=None
    captured={j['pid'] for j in handoff['active_jobs']}
    assert set(actual_gpu_pids()).issubset(captured)
    for saved in handoff['active_jobs']:
        job=dict(saved);job['attached']=True
        try:
            process=psutil.Process(job['pid'])
            assert process.create_time()==job['create_time'] and process.cmdline()==job['command']
            active[process.pid]=job;handles[process.pid]=process
        except psutil.NoSuchProcess:
            validate_terminal(job,None)
            write(Path(job['dest'])/(job['stage']+'_TIMING.json'),dict(identity=job['identity'],fold=job['fold'],stage=job['stage'],pid=job['pid'],terminal=True,exit_code=None,process_wall_seconds=time.time()-job['started_epoch'],concurrency='existing safe overlap then serial drain',includes='load/preprocess/warmup/training or scan/restore/write',attached_without_training_restart=True))
            events.append(dict(event='attached_worker_already_terminal',pid=job['pid'],stage=job['stage'],fold=job['fold'],identity=job['identity']))
    write(STUDY/'SERIAL_DRAIN_LAUNCH.json',dict(controller_pid=os.getpid(),started_epoch=began,whole_batch_started_epoch=whole_started,preserved_jobs=handoff['active_jobs'],maximum_new_GPU_jobs=1,maximum_existing_GPU_jobs=2,no_model_training_changes=True,no_retry=True,concurrency='Existing workers finish, thereafter serial; no new overlap'))
'''
text=text[:start]+bootstrap+text[end:]
text=text.replace('"""User-approved second GPU lane; existing worker preserved, no two PERM jobs.','"""User-requested serial drain; all existing workers preserved.')
text=text.replace('Maximum two GPU children, one','Existing two workers may finish, future tasks strictly serial; one')
text=text.replace("concurrency='safe overlap up to2; at most one SOURCE/SW/scan/PERM each; driver headroom7GiB',maximum_GPU_jobs=2,","concurrency='existing workers drain, thereafter serial; no new overlap',maximum_GPU_jobs=1,maximum_existing_GPU_jobs=2,draining_existing_overlap=len(active)>1,no_new_overlap=True,")
text=text.replace('if len(active)>=2:return False','if len(active)>=1:return False')
text=text.replace('assert len(active)<2 and eligible(job)','assert len(active)<1 and eligible(job)')
text=text.replace('_safe_parallel_stdout.log','_serial_drain_stdout.log').replace('_safe_parallel_stderr.log','_serial_drain_stderr.log')
text=text.replace("print('OVERLAP_CHILD'","print('SERIAL_DRAIN_CHILD'")
text=text.replace("('solo then safe overlap' if job['attached'] else 'safe overlap') if job['overlap_observed'] else 'solo'","'existing safe overlap then serial drain' if job['attached'] else 'solo after user requested serial'")
text=text.replace('if len(active)>=2:break','if len(active)>=1:break')
text=text.replace('overlap_controller_seconds=time.time()-began','serial_drain_controller_seconds=time.time()-began')
text=text.replace('# Preserve contiguous model chains; fill idle slot with the next SOURCE.','# Preserve contiguous model chains; do not fill any slot while an existing worker runs.')
p=S/'run_serial_drain_pair_fullcv_v370r4.py'
assert not p.exists()
p.write_text(text,encoding='utf-8');py_compile.compile(str(p),doraise=True)
approval=dict(explicit_user_approved=True,user_reply='\u73b0\u6709\u7684\u53cc\u5e76\u53d1\u53ef\u4ee5\u7ee7\u7eed\uff0c\u4e4b\u540e\u4e0d\u518d\u53cc\u5e76\u53d1\uff0c\u770b\u4e00\u4e0b\u9636\u6bb5\u6027\u5f97\u5206',authorized_epoch=time.time(),maximum_new_GPU_jobs=1,maximum_preserved_GPU_jobs=2,controller_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),no_training_restart=True,no_model_changes=True,no_failure_retry=True)
(S/'SERIAL_DRAIN_AUTHORIZATION.json').write_text(json.dumps(approval,ensure_ascii=False,indent=2),encoding='utf-8')
print('Compiled serial-drain dispatcher; existing GPU jobs unchanged.')
