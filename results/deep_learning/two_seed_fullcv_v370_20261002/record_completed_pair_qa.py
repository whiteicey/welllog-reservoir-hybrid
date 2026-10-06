"""CPU-only completed-fold readback; no extra training or rescan."""
from pathlib import Path
import argparse,json,time,hashlib,psutil,numpy as np

S=Path(__file__).resolve().parent;R=S.parents[2];O=R/'outputs/pure_dl_two_seed_fullcv_v370'
ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[2,3,4],required=True);f=ap.parse_args().fold
def read(p):return json.loads(p.read_text(encoding='utf-8'))
dest=S/f'FOLD{f}_PAIRED_NATIVE_QA.json';assert not dest.exists(),'Unchanged completed fold already recorded'
d=S/f'fold_{f}';report=read(d/'FOLD_REPORT.json');scan={};wall=0.
for target in ['POR','PERM','SW']:
    evidence=read(d/(target+'_GPU_REPORT.json'));timing=read(d/('SCAN_'+target+'_TIMING.json'))
    assert all(evidence[k] for k in ['native_disk_replay_exact','coverage_once','endpoints_exact'])
    assert evidence['fold']==f and evidence['rows']==report['rows'] and evidence['extra_training_stages']==0
    assert evidence['admission']['other_jobs']==[]
    assert timing['terminal'] and timing['exit_code']==0 and not psutil.pid_exists(timing['pid'])
    wall+=timing['process_wall_seconds'];scan[target]=dict(replay_exact=True,coverage_once=True,endpoints_exact=True,exit_code=0,peak_MiB=evidence['peak_MiB'],process_wall_seconds=timing['process_wall_seconds'])
for identity,label in [('original','original'),('seed_20261013','seed13')]:
    own=O/f'fold_{f}'/identity;model=read(own/'COMPLETE_MODEL.json');prefix=read(d/(identity+'_PREFIX_SNAPSHOT.json'))
    assert model['training_stages']==8 and prefix['complete8stage_single_fold'] and len(prefix['completed_prefix'])==8
    assert model['rows']==report['rows'] and all(abs(model['scores'][k]-report['standalone_scores'][label][k])<1e-9 for k in model['scores'])
    a=np.load(own/'complete_model_pred.npy');b=np.load(d/(label+'_prediction.npy'))
    assert a.shape==(report['rows'],3) and np.isfinite(a).all() and np.array_equal(a,b)
for label in ['fixed','diagnostic']:
    a=np.load(d/(label+'_prediction.npy'));assert a.shape==(report['rows'],3) and np.isfinite(a).all()
assert report['fixed_weights']=={'POR':.56,'PERM':.44,'SW':.68}
strata=read(d/'FUSION_STRATA.json')
state=read(S/'BATCH_STATE.json');jobs=[]
for j in state['active_jobs']:
    p=psutil.Process(j['pid']);assert p.create_time()==j['create_time'] and p.cmdline()==j['command']
    jobs.append(dict(identity=j['identity'],fold=j['fold'],stage=j['stage'],pid=j['pid'],alive=True))
assert state['maximum_GPU_jobs']==1 and len(jobs)<=1
if state['status']=='complete':
    # The final approved recovery controller expresses serialization with
    # maximum_GPU_jobs, not the earlier drain-controller no_new_overlap key.
    assert not jobs
    launch=read(S/'SW351_FOLD4_RECOVERY_LAUNCH.json')
    assert launch['maximum_GPU_jobs']==1 and launch['authorization']['explicit_user_approved']
    controller=S/'resume_sw351_fold4_serial_v370r5.py'
    assert hashlib.sha256(controller.read_bytes()).hexdigest()==launch['authorization']['controller_sha256']
    assert not psutil.pid_exists(state['controller_pid'])
else:
    assert state['no_new_overlap']
checks={name:hashlib.sha256((R/name).read_bytes()).hexdigest()==digest for name,digest in read(S/'PREPARATION.json')['code_sha256'].items()};assert all(checks.values())
remaining=[n for n in range(1,6) if not (S/f'fold_{n}/FOLD_REPORT.json').exists()]
result=dict(observed_epoch=time.time(),fold=f,rows=report['rows'],native_scans=scan,scan_serial_wall_seconds=wall,extra_training_stages=0,two_complete8stage_models=True,native_standalone_arrays_exact_to_complete_models=True,fixed_weights_unchanged=True,scores=dict(fixed=report['fixed_scores'],diagnostic=report['diagnostic_scores'],standalone=report['standalone_scores']),per_head_best_weights=report['per_head_best_weights'],frozen_code_hash_checks=checks,actual_current_jobs=jobs,maximum_future_GPU_jobs=1,full_OOF_report_present=(S/'FINAL.json').exists(),remaining_paired_folds=remaining,overall_scope='This is one completed paired fold; final OOF requires a separate canonical coverage and score audit',diagnostic_scope='Fold labels select per-head ratios; tuning diagnostic only',fixed_scope='Fifth-selected pair/ratios applied unchanged; disclose fifth selection',strata_analysis_source=str(d/'FUSION_STRATA.json'),CPU_only=True)
dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
note='\nV370 completed paired fold'+str(f)+': '+str(report['rows'])+' rows; original '+str(report['standalone_scores']['original']['TOTAL'])+', seed13 '+str(report['standalone_scores']['seed13']['TOTAL'])+', fifth-fixed ratios '+str(report['fixed_scores']['TOTAL'])+', fold-tuned diagnostic '+str(report['diagnostic_scores']['TOTAL'])+'. Both8-stage models actually complete; each native head disk replay/coverage/endpoints exact, all3 scans exit0, native endpoint arrays exact to saved full-model arrays. Serial scan wall '+str(wall)+'s includes load/cache/warmup/native forwards/101-grid/write, no added training stage. Ratios selected on this fold are diagnostic only; fixed weights stay.56/.44/.68. Frozen code unchanged; current OS-process-verified jobs '+json.dumps(jobs)+', future cap1. This verifies one fold, not full OOF; fifth selection boundary disclosed. Evidence '+dest.name+', FOLD_RESULTS.md and FUSION_STRATA.json. Remaining paired folds '+str(remaining)+'; final723556-row OOF audit remains separate; existing deliveries unchanged.\n'
for p in [S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md',R/'docs/deep_learning/atlas/PROJECTS.md']:
    with p.open('a',encoding='utf-8') as out:out.write(note)
print(json.dumps(dict(fold=f,rows=report['rows'],fixed=report['fixed_scores']['TOTAL'],diagnostic=report['diagnostic_scores']['TOTAL'],serial_scan_wall=wall,actual_jobs=jobs)))
