"""Train two matched v36 retrieval branches; parent fallback is not a checkpoint."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,sys,time,traceback
import numpy as np
import torch
import retrieval_smooth_v36 as module
from train_dual8222_heads80 import atomic_json,atomic_array,atomic_torch,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None

def stop_request(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED; finish epoch and save, no automatic resume',flush=True)

def brief(states):
    return {m:{k:s[k] for k in ('best','best_epoch','candidate_best','candidate_epoch','bad','done','last_epoch')} for m,s in states.items()}

def gradient_norms(head):
    norms={}
    for group in ('query','key','output'):
        values=[p.grad.detach().float().square().sum() for n,p in head.named_parameters() if n.startswith(group+'.') and p.grad is not None]
        norms[group]=float(torch.stack(values).sum().sqrt()) if values else 0.
    return norms

def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    out=args.out.resolve();OWNED=out;out.mkdir(parents=True,exist_ok=True)
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed run is immutable')
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Explicit --resume required')
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    atomic_json(out/'runtime_status.json',dict(status='running',target='SW',pid=os.getpid(),atlas_current_trial=True))
    start=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=module.load_data();reference=module.reference_prediction()
    refs=json.loads((module.PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    assert reference.shape==(144665,3) and len(data['train_raw'])==578891
    model=module.Network('SW',master).cuda();modes=module.MODES
    evaluator=module.Evaluator(data,master,2,reference)
    frozen=cpu_copy(model.frozen_state());initial={m:cpu_copy(model.heads[m].state_dict()) for m in modes}
    assert all(torch.equal(initial[modes[0]][k],initial[modes[1]][k]) for k in initial[modes[0]])
    assert {id(p) for p in model.heads[modes[0]].parameters()}.isdisjoint({id(p) for p in model.heads[modes[1]].parameters()})
    names=('retrieval_smooth_v36.py','retrieval_density_v35.py','prepare_retrieval_v35.py','crosscurve_v32.py','por_reparam_v31.py','champion_split_route_v29.py','champion_all_heads_v26.py','train_scaling_v21.py')
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in names]
    cfg=dict(module='retrieval_smooth_v36',target='SW',modes=list(modes),screening_fold=5,reference_scores=refs,
        parent_recipe=str((module.PREVIOUS/'CURRENT_RECIPE.json').relative_to(module.ROOT)),
        parameters_per_mode={m:sum(p.numel() for p in model.heads[m].parameters()) for m in modes},metadata=model.metadata,
        seed=20261010,epochs=80,patience=18,min_epochs=10,batch_size=64,steps_per_epoch=64,
        learning_rate=module.LEARNING_RATE,eta_min=1e-6,weight_decay=.01,clip_norm=2.,quality=data['qinfo'],
        validation_rows=144665,training_rows=578891,initial_alpha=module.INITIAL_ALPHA,
        candidate_initialization_is_parent=False,parent_fallback_epoch=-1,
        stopping='best of frozen parent and current candidate, patience18; separately record candidate best and post-update best',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__),complete_five_folds=False,remaining_folds_started=False)
    if args.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    opts={m:torch.optim.AdamW(model.trainable_groups(m),lr=module.LEARNING_RATE,weight_decay=.01) for m in modes}
    sched={m:torch.optim.lr_scheduler.CosineAnnealingLR(opts[m],80*64,eta_min=1e-6) for m in modes}
    windows=torch.Generator(device='cuda').manual_seed(20261010);epoch=0;timing=[];prior_seconds=0.
    if args.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.heads.load_state_dict(saved['heads'],strict=True);states=saved['states'];epoch=saved['epoch'];timing=saved['timing'];prior_seconds=saved['seconds']
        for m in modes:opts[m].load_state_dict(saved['optimizers'][m]);sched[m].load_state_dict(saved['schedulers'][m])
        torch.set_rng_state(saved['rng']['torch']);torch.cuda.set_rng_state_all(saved['rng']['cuda']);windows.set_state(saved['rng']['windows'])
        np.random.set_state(saved['rng']['numpy']);random.setstate(saved['rng']['python']);del saved
    else:
        # Explicitly disabled retrieval is the parent baseline. Smooth 1% models
        # are separately measured; their initial scores must not be fabricated.
        for head in model.heads.values():head.force_off=True
        disabled=evaluator.many(model,modes)
        for m,(p,s,t) in disabled.items():assert np.array_equal(p,reference[:,2]) and abs(s-refs['SW'])<1e-9
        for head in model.heads.values():head.force_off=False
        values=evaluator.many(model,modes);states={}
        for m,(pred,score,total) in values.items():
            arm=out/m;arm.mkdir(exist_ok=True)
            probe=module.factor_scope(model,data,m)
            assert abs(probe['alpha_mean']-module.INITIAL_ALPHA)<1e-6
            atomic_json(arm/'initial_verification.json',dict(parent_disabled_all144665_exact=True,candidate_SW=score,candidate_TOTAL=total,probe=probe))
            better=score>refs['SW']
            states[m]=dict(best=max(score,refs['SW']),best_epoch=0 if better else -1,
                candidate_best=score,candidate_epoch=0,candidate_state=cpu_copy(model.heads[m].state_dict()),candidate_pred=pred.copy(),
                initial_score=score,initial_TOTAL=total,best_updated=None,updated_state=None,updated_pred=None,
                bad=0,done=False,last_epoch=0,stop_reason=None,history=[],first_gradient=None,last_gradient=None)
        print('INITIAL',json.dumps(brief(states)),flush=True)
    def persist():
        for m,s in states.items():
            arm=out/m;arm.mkdir(exist_ok=True)
            common=dict(target='SW',mode=m,config=cfg,format='v36_smooth_retrieval_candidate; fallback stored separately')
            atomic_torch(arm/'model_best.pt',dict(delta_state=s['candidate_state'],epoch=s['candidate_epoch'],target_score=s['candidate_best'],**common))
            atomic_array(arm/'candidate_pred.npy',s['candidate_pred'])
            chosen=s['candidate_pred'] if s['best_epoch']>=0 else reference[:,2]
            atomic_array(arm/'channel_pred.npy',chosen);atomic_json(arm/'validation_history.json',s['history'])
            if s['best_updated'] is not None:
                atomic_torch(arm/'model_best_updated.pt',dict(delta_state=s['updated_state'],**s['best_updated'],**common))
                atomic_array(arm/'best_updated_pred.npy',s['updated_pred'])
        atomic_torch(out/'last_state.pt',dict(config=cfg,heads=cpu_copy(model.heads.state_dict()),states=states,epoch=epoch,timing=timing,
            seconds=prior_seconds+time.perf_counter()-start,optimizers={m:cpu_copy(o.state_dict()) for m,o in opts.items()},schedulers={m:s.state_dict() for m,s in sched.items()},
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        active=[m for m in modes if not states[m]['done']]
        if STOP or not active:break
        tick=time.perf_counter();model.train();losses={m:0. for m in active}
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper']);relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]
            h,parent,context,_=model.extract(data['x'][ix],relative,valid,data['absolute'][ix],rows=ix)
            for m in active:
                head=model.heads[m];opts[m].zero_grad(set_to_none=True)
                output=head(h,parent,context,data['x'][ix,...,206:],valid)
                loss,_,_=module.channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],2)
                if not bool(torch.isfinite(loss)):raise FloatingPointError((m,'nonfinite_loss'))
                loss.backward()
                if step in (1,63):
                    norms=gradient_norms(head);assert all(np.isfinite(v) for v in norms.values())
                    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                    if states[m]['first_gradient'] is None:
                        assert all(v>0 for v in norms.values()),('inactive_learning_path',m,norms)
                        states[m]['first_gradient']=norms;atomic_json(out/m/'gradient_scope.json',dict(first_nonzero=norms,frozen_gradients_none=True))
                    states[m]['last_gradient']=norms
                torch.nn.utils.clip_grad_norm_(head.parameters(),2.,error_if_nonfinite=True)
                opts[m].step();sched[m].step();losses[m]+=float(loss.detach())
            del h,parent,context
        epoch=ep;values=evaluator.many(model,active)
        for m,(pred,score,total) in values.items():
            s=states[m];s['last_epoch']=ep
            if score>s['best']:s.update(best=score,best_epoch=ep,bad=0)
            else:s['bad']+=1
            if score>s['candidate_best']:
                s.update(candidate_best=score,candidate_epoch=ep,candidate_state=cpu_copy(model.heads[m].state_dict()),candidate_pred=pred.copy())
            if s['best_updated'] is None or score>s['best_updated']['target_score']:
                s.update(best_updated=dict(epoch=ep,target_score=score,reference_replacement_total=total),updated_state=cpu_copy(model.heads[m].state_dict()),updated_pred=pred.copy())
            s['history'].append(dict(epoch=ep,target_score=score,reference_replacement_total=total,best_vs_parent=s['best'],best_epoch=s['best_epoch'],training_loss=losses[m]/64,bad=s['bad'],last_gradient=s['last_gradient']))
            if ep>=10 and s['bad']>=18:s.update(done=True,stop_reason='patience18')
            elif ep==80:s.update(done=True,stop_reason='budget80')
        timing.append(dict(epoch=ep,seconds=time.perf_counter()-tick,active_modes=active));persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),target='SW',last_completed_epoch=epoch,modes=brief(states),atlas_current_trial=True))
        print('EPOCH',ep,json.dumps(brief(states)),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),resume_available=True,automatic_resume=False));return
    assert all(s['done'] for s in states.values())
    assert all(torch.equal(v.detach().cpu(),frozen[n]) for n,v in model.frozen_state().items())
    last_scope={m:module.factor_scope(model,data,m) for m in modes}
    change={m:{prefix:float(sum((v.detach().cpu().double()-initial[m][n].double()).square().sum() for n,v in model.heads[m].state_dict().items() if n.startswith(prefix+'.')).sqrt()) for prefix in ('query','key','output')} for m in modes}
    for m,s in states.items():model.heads[m].load_state_dict(s['candidate_state'],strict=True)
    restored=evaluator.many(model,modes);arms=[]
    for m,(pred,score,total) in restored.items():
        s=states[m];assert np.array_equal(pred,s['candidate_pred']) and abs(score-s['candidate_best'])<1e-9
        report=dict(mode=m,target='SW',screening_fold=5,target_score=s['best'],selected_is_parent=s['best_epoch']<0,best_epoch=s['best_epoch'],
            initial_candidate_score=s['initial_score'],candidate_score=score,candidate_epoch=s['candidate_epoch'],best_after_update=s['best_updated'],
            stopped_epoch=s['last_epoch'],stop_reason=s['stop_reason'],trainable_parameters=cfg['parameters_per_mode'][m],
            candidate_predictions_restored_exactly=True,all_frozen_state_unchanged=True,first_gradient=s['first_gradient'],last_gradient=s['last_gradient'],
            parameter_changes=change[m],last_train_probe=last_scope[m],best_candidate_probe=module.factor_scope(model,data,m),
            checkpoint=str((out/m/'model_best.pt').relative_to(module.ROOT)),negative_control=m=='shuffled',remaining_folds_started=False)
        atomic_json(out/m/'fold_5_report.json',report);arms.append(report)
    report=dict(target='SW',arms=arms,reference_scores=refs,seconds=prior_seconds+time.perf_counter()-start,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,epoch_timing=timing,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),target='SW',modes=brief(states),atlas_current_trial=False))
    model.close();print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            fail=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',fail);atomic_json(OWNED/'runtime_status.json',fail)
        raise
