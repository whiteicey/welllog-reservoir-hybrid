"""One bounded matched POR joint-utility loss study; independent best/recovery states."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,importlib,json,random,signal,sys,time,traceback
import numpy as np
import torch
from train_dual8222_heads80 import atomic_json,atomic_array,atomic_torch,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
from fold5_screening_805 import qualifies

STOP=False
OWNED=None

def stop_request(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish current epoch, persist; explicit resume only',flush=True)

def brief(states):
    return {m:{k:s[k] for k in ('best','best_epoch','bad','done','last_epoch','stop_reason')} for m,s in states.items()}

def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=('POR',),required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    target=a.target;module_name='por_joint_utility_v99'
    module=importlib.import_module(module_name);MODES=module.MODES;c=module.TARGETS.index(target)
    if not (module.ROOT/'outputs/pure_dl_fullcv_v97/fold_4/root/fold_4_report.json').exists():
        raise RuntimeError('Wait for the authorized fourth-fold source before using the extra GPU slot')
    verified=module.ROOT/'results/deep_learning/fullcv_v97_20260930/OOF_VERIFICATION.json'
    if verified.exists():
        final=module.ROOT/'results/deep_learning/fullcv_v97_20260930/FINAL_REPORT.json'
        if final.exists() and json.loads(final.read_text(encoding='utf-8'))['strict_oof82_passed']:
            raise RuntimeError('Verified full OOF already exceeds82; no extra trial required')
    out=a.out.resolve()
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Existing run requires explicit --resume')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=4.),flush=True)
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    atomic_json(out/'runtime_status.json',dict(status='running',target=target,pid=os.getpid(),atlas_current_trial=True))
    start=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=module.load_data();reference=module.reference_prediction()
    refs=json.loads((module.PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    assert reference.shape==(144665,3) and len(data['train_raw'])==578891
    model=module.Network(target,master).cuda();evaluator=module.Evaluator(data,master,c,reference)
    criterion=module.DecisionObjective(evaluator,data).cuda()
    frozen=cpu_copy(model.frozen_state());initial={m:cpu_copy(model.heads[m].state_dict()) for m in MODES}
    names={m:{n for n,p in model.heads[m].named_parameters() if p.requires_grad} for m in MODES}
    ids=[{id(p) for p in model.heads[m].parameters() if p.requires_grad} for m in MODES]
    assert ids[0].isdisjoint(ids[1])
    common=[k for k in initial[MODES[0]] if k in initial[MODES[1]] and initial[MODES[0]][k].shape==initial[MODES[1]][k].shape]
    common_equal=all(torch.equal(initial[MODES[0]][k],initial[MODES[1]][k]) for k in common)
    assert common_equal
    files=[Path(__file__),Path(module.__file__)]+[Path(__file__).with_name(n) for n in
        ('por_effective_sam_v98.py','por_reparam_v31.py','por_student3_sequence.py','por_t3_patch_loss.py','fold5_screening_805.py','champion_coadapt_v30.py','champion_split_route_v29.py','champion_route_v28.py','champion_posterior_v27.py','champion_all_heads_v26.py','train_scaling_v21.py')]
    files.append(module.ROOT/'package/src/utils.py')
    cfg=dict(target=target,modes=list(MODES),module=module_name,screening_fold=5,
        parent_recipe=str((module.PREVIOUS/'CURRENT_RECIPE.json').relative_to(module.ROOT)),reference_scores=refs,
        per_candidate_trainable_parameters={m:sum(p.numel() for p in model.heads[m].parameters() if p.requires_grad) for m in MODES},
        trainable_names={m:sorted(names[m]) for m in MODES},same_shape_initial_tensors_equal=common_equal,
        feature_count=206,amplitude_side=63,train_points=578891,validation_points=144665,
        learning_rate=module.LEARNING_RATE,eta_min=1e-5,weight_decay=.01,clip_norm=2.,
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        quality=data['qinfo'],decoder='original singlecoverage256bins/800actions+default/batch32',
        mechanism=module.__doc__,metadata=getattr(model,'study_metadata',{}),
        optimizers={m:'AdamW' for m in MODES},new_capacity=0,
        forward_backward_evaluations_per_update={m:1 for m in MODES},only_own_target_labels=True,
        training_loss='own POR Student-t3 NLL+BCE/Q50 /3 plus alpha .0/.5 joint softaction regret/Q50 /3',
        utility_weight={m:module.UTILITY_WEIGHT if m=='joint' else 0. for m in MODES},
        softaction_temperature=module.TEMPERATURE,training_label_isolation=criterion.isolation,
        gradient_memory='1024 point utility chunks with nonreentrant activation checkpointing',
        utility_kernel_source='package/src/utils.py POR tolerance .08',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        torch_version=str(torch.__version__),deterministic_mode='warn_only; not a bitwise repeated-training guarantee',
        complete_five_folds=False,remaining_folds_started=False)
    if a.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    opts={m:torch.optim.AdamW(model.trainable_groups(m),lr=module.LEARNING_RATE,weight_decay=.01) for m in MODES}
    sched={m:torch.optim.lr_scheduler.CosineAnnealingLR(opts[m],80*64,eta_min=1e-5) for m in MODES}
    windows=torch.Generator(device='cuda').manual_seed(20261010);epoch=0;prior_seconds=0.;timing=[]
    if a.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.heads.load_state_dict(saved['heads'],strict=True)
        for m in MODES:opts[m].load_state_dict(saved['optimizers'][m]);sched[m].load_state_dict(saved['schedulers'][m])
        states=saved['states'];epoch=saved['epoch'];prior_seconds=saved['seconds'];timing=saved['timing']
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        windows.set_state(saved['rng']['windows']);np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python']);del saved
    else:
        values=evaluator.many(model,MODES);states={}
        for m,(pred,score,total) in values.items():
            assert np.array_equal(pred,reference[:,c]),(target,m,'initial current-function mismatch')
            assert abs(score-refs[target])<1e-9
            arm=out/m;arm.mkdir(exist_ok=True)
            proof=dict(scope='same-capacity loss ablation; epoch0 selection only; no separate preflight')
            atomic_json(arm/'initial_verification.json',dict(target=target,mode=m,score=score,TOTAL=total,all144665_exact=True,mechanism_probe=proof))
            states[m]=dict(best=score,best_epoch=0,best_state=cpu_copy(model.heads[m].state_dict()),best_pred=pred.copy(),
                best_updated=None,updated_state=None,updated_pred=None,bad=0,history=[],done=False,last_epoch=0,stop_reason=None,gradient=None)
        print('INITIAL',json.dumps(dict(target=target,TOTAL=refs['TOTAL'],modes=brief(states))),flush=True)
    def persist():
        for m,s in states.items():
            arm=out/m;arm.mkdir(exist_ok=True)
            common=dict(target=target,mode=m,config=cfg,parent_recipe=cfg['parent_recipe'],format='v99_candidate_state_plus_documented_frozen_dependencies')
            atomic_torch(arm/'model_best.pt',dict(delta_state=s['best_state'],best_epoch=s['best_epoch'],target_score=s['best'],**common))
            atomic_array(arm/'channel_pred.npy',s['best_pred']);atomic_json(arm/'validation_history.json',s['history'])
            if s['best_updated'] is not None:
                atomic_torch(arm/'model_best_updated.pt',dict(delta_state=s['updated_state'],**s['best_updated'],**common))
                atomic_array(arm/'best_updated_pred.npy',s['updated_pred'])
        atomic_torch(out/'last_state.pt',dict(config=cfg,heads=cpu_copy(model.heads.state_dict()),states=states,
            optimizers={m:cpu_copy(o.state_dict()) for m,o in opts.items()},schedulers={m:s.state_dict() for m,s in sched.items()},
            epoch=epoch,timing=timing,seconds=prior_seconds+time.perf_counter()-start,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        active=[m for m in MODES if not states[m]['done']]
        if STOP or not active:break
        tick=time.perf_counter();model.train();losses={m:0. for m in active}
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]
            h,o,context,_=model.extract(data['x'][ix],rel,valid,data['absolute'][ix])
            for m in active:
                opts[m].zero_grad(set_to_none=True)
                output=model.heads[m](h,o,context,data['x'][ix,...,206:],valid)
                loss,_,_=criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],criterion.physical_training_only[ix],m)
                if not bool(torch.isfinite(loss)):raise FloatingPointError((target,m,'nonfinite objective'))
                loss.backward()
                if states[m]['gradient'] is None and step==1:
                    group_norms={g['name']:float(torch.stack([p.grad.detach().float().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt()) for g in opts[m].param_groups}
                    assert all(v>0 for v in group_norms.values())
                    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                    states[m]['gradient']=dict(group_norms=group_norms,frozen_gradients_none=True)
                    atomic_json(out/m/'gradient_scope.json',states[m]['gradient'])
                for g in opts[m].param_groups:torch.nn.utils.clip_grad_norm_(g['params'],2.,error_if_nonfinite=True)
                opts[m].step();sched[m].step();losses[m]+=float(loss.detach())
            del h,o,context
        epoch=ep;values=evaluator.many(model,active)
        for m,(pred,score,total) in values.items():
            s=states[m];s['last_epoch']=ep
            if score>s['best']:
                s.update(best=score,best_epoch=ep,best_state=cpu_copy(model.heads[m].state_dict()),best_pred=pred.copy(),bad=0)
            else:s['bad']+=1
            if s['best_updated'] is None or score>s['best_updated']['target_score']:
                s.update(best_updated=dict(epoch=ep,target_score=score,reference_replacement_total=total),
                    updated_state=cpu_copy(model.heads[m].state_dict()),updated_pred=pred.copy())
            s['history'].append(dict(epoch=ep,target_score=score,reference_replacement_total=total,best_score=s['best'],best_epoch=s['best_epoch'],
                training_loss=losses[m]/64,bad_epochs=s['bad'],learning_rate=opts[m].param_groups[0]['lr']))
            if ep>=10 and s['bad']>=18:s.update(done=True,stop_reason='patience18')
            elif ep==80:s.update(done=True,stop_reason='budget80')
        timing.append(dict(epoch=ep,seconds=time.perf_counter()-tick,active_modes=active));persist()
        atomic_json(out/'runtime_status.json',dict(status='running',target=target,pid=os.getpid(),last_completed_epoch=epoch,modes=brief(states),atlas_current_trial=True))
        print('EPOCH',json.dumps(dict(epoch=epoch,target=target,modes=brief(states),seconds=timing[-1]['seconds'])),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',target=target,resume_available=True,auto_resume=False));return
    assert all(s['done'] for s in states.values())
    assert all(torch.equal(v.detach().cpu(),frozen[n]) for n,v in model.frozen_state().items())
    change={m:float(sum((v.detach().cpu().double()-initial[m][n].double()).square().sum()
        for n,v in model.heads[m].state_dict().items() if n in names[m]).sqrt()) for m in MODES}
    assert all(v>0 for v in change.values())
    for m,s in states.items():model.heads[m].load_state_dict(s['best_state'],strict=True)
    restored=evaluator.many(model,MODES);arms=[]
    for m,(pred,score,total) in restored.items():
        s=states[m];assert np.array_equal(pred,s['best_pred'])
        report=dict(target=target,mode=m,screening_fold=5,target_score=score,reference_replacement_total=total,
            best_epoch=s['best_epoch'],stopped_epoch=s['last_epoch'],stop_reason=s['stop_reason'],best_after_update=s['best_updated'],
            gain_vs_v96_total=total-refs['TOTAL'],strict_fifth805=qualifies(total),
            positive_gain_and_gate=qualifies(total) and total>refs['TOTAL']+1e-9,
            trainable_parameters=cfg['per_candidate_trainable_parameters'][m],all_frozen_state_unchanged=True,
            all144665_selected_predictions_restored_exactly=True,gradient_scope=s['gradient'],last_parameter_change_L2=change[m],
            mechanism_scope=module.factor_scope(model,data,m),checkpoint=str((out/m/'model_best.pt').relative_to(module.ROOT)),
            complete_five_folds=False,remaining_folds_started=False)
        atomic_json(out/m/'fold_5_report.json',report);arms.append(report)
    joint=next(r for r in arms if r['mode']=='joint');like=next(r for r in arms if r['mode']=='likelihood')
    report=dict(target=target,arms=arms,reference_scores=refs,
        predefined_ablation_delta_TOTAL=joint['reference_replacement_total']-like['reference_replacement_total'],
        predefined_ablation='same capacity and initialization: joint utility minus likelihood',seconds=prior_seconds+time.perf_counter()-start,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,epoch_timing=timing,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',target=target,pid=os.getpid(),modes=brief(states),atlas_current_trial=False))
    model.close();print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            failure=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',failure);atomic_json(OWNED/'runtime_status.json',failure)
        raise
