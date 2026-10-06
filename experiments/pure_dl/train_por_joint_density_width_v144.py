"""Explicit compact own-target stage; atomic resume and actual best restoration."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from por_joint_density_width_v144 import ROOT,STUDY,OUT,SOURCE,STAGES,load_data,make_model
from train_stage_fullcv_v97 import frozen_signature
from por_joint_density_width_v144 import Evaluator
from por_joint_density_width_v144 import channel_objective
from fullcv_resume_v97 import atomic_save,cpu_state,rng_state,restore_rng
from train_dual8222_heads80 import atomic_json,atomic_array
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None
def stop_request(*args):
    global STOP
    STOP=True;print('STOP_REQUESTED: save at epoch boundary; explicit resume only',flush=True)

def main():
    global OWNED
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=list(STAGES),required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    stage=a.stage;out=OUT/stage;out.mkdir(parents=True,exist_ok=True);OWNED=out
    if (out/'report.json').exists():raise RuntimeError('Finished stage immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Existing stage requires explicit resume')
    plan=json.loads((STUDY/'PLAN.json').read_text());assert plan['total_training_stages']==5 and plan['source_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert plan['parent_sha256']==hashlib.sha256((ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt').read_bytes()).hexdigest()
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    started=time.perf_counter();seed=20261010;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),stage=stage,epoch=0))
    data,master,metadata=load_data(stage);target,length,sam,alpha,lr=STAGES[stage];c=('POR','PERM','SW').index(target)
    model=make_model(stage,master,metadata).cuda()
    initialization=model.initialize_priors(data);initialization.update(training_NLL_and_predictive_CDF_synchronized=True)
    evaluator=Evaluator(data,master,c,length,stage)
    if c==2:
        from sw_utility_strength_v82 import DecisionObjective
        criterion=DecisionObjective(evaluator,metadata,'a100').cuda();criterion.alpha=alpha
        loss_fn=lambda output,ix,mask:criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])[0]
    else:loss_fn=lambda output,ix,mask:channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],c,stage)[0]
    groups=model.groups() if c<2 else [dict(params=[p for p in model.parameters() if p.requires_grad],lr=lr,name='private_SW')]
    params=[p for g in groups for p in g['params']];fixed=frozen_signature(model)
    cfg=dict(stage=stage,target=target,original_fold=5,screening_threshold=80.8,total_recipe_training_stages=5,source_stage_counted=True,
        source_sha256=plan['source_sha256'],only_original_source_pretrained=False,no_old_18stage_downstream_weights=True,warm_parent_sha256=plan['parent_sha256'],
        parent_compact_joint_sha256=hashlib.sha256((OUT/'sw_joint/model_best.pt').read_bytes()).hexdigest() if stage=='sw_density' else None,
        seed=seed,epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps=64,cosine_T_max=80*64,eta_min=1e-5,
        lr_groups={g['name']:g['lr'] for g in groups},sam_rho=.1 if sam else 0.,utility_weight=alpha,dropkey=.4 if c==2 else 0.,
        window_length=length,trainable_parameters=sum(p.numel() for p in params),parameters=sum(p.numel() for p in model.parameters()),
        training_rows=len(data['train_raw']),validation_rows=len(data['validation']),quality=data['qinfo'],initialization=initialization,
        fixed_signature=fixed,own_target_labels_only=True,complete_five_folds=False,remaining_folds_started=False,
        source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('train_por_joint_density_width_v144.py','por_joint_density_width_v144.py','compact_por_mixed_family_v113.py','compact_five_stage_v100.py','fold5_screening_808.py','training_awake_guard.py')})
    if a.resume:assert json.loads((out/'config.json').read_text())==cfg
    else:atomic_json(out/'config.json',cfg)
    opt=torch.optim.AdamW(groups,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(seed);epoch=bad=0;history=[];elapsed=0.;peak=0.;gradient=None
    if a.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert saved['config']==cfg
        model.load_state_dict(saved['current']);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler'])
        epoch=saved['epoch'];bad=saved['bad'];best=saved['best'];best_epoch=saved['best_epoch'];best_state=saved['best_state'];best_pred=saved['best_pred'];history=saved['history'];elapsed=saved['seconds'];peak=saved['peak'];gradient=saved['gradient']
        restore_rng(saved['rng']);windows.set_state(saved['windows']);initial=saved['initial'];del saved
    else:
        best_pred,best=evaluator(model);initial=best;best_epoch=0;best_state=cpu_state(model)
    def persist():
        nonlocal peak
        peak=max(peak,torch.cuda.max_memory_allocated()/2**20)
        atomic_save(dict(config=cfg,current=cpu_state(model),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,bad=bad,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,initial=initial,history=history,gradient=gradient,
            rng=rng_state(),windows=windows.get_state(),seconds=elapsed+time.perf_counter()-started,peak=peak),out/'last_state.pt')
    persist();torch.cuda.reset_peak_memory_stats();lo=length//2-64;core=torch.zeros(1,length,device='cuda',dtype=torch.bool);core[:,lo:lo+128]=True
    print('INITIAL',json.dumps(dict(stage=stage,target=target,score=initial,parameters=cfg['trainable_parameters'])),flush=True)
    early_requested=False
    while epoch<80 and not (epoch>=10 and bad>=18) and not STOP:
        if (STUDY/(stage+'_early_stop_request.json')).exists():
            assert epoch>=10
            early_requested=True;break
        tick=time.perf_counter();model.train();total=0.
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'],length=length);rel=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix];kwargs=dict(absolute_depth=data['absolute'][ix],well_inputs=data['well_bank'][data['row_well'][anchors]])
            opt.zero_grad(set_to_none=True);before=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(data['x'][ix],rel,valid,**kwargs)
            loss=loss_fn(output,ix,mask);assert bool(torch.isfinite(loss));loss.backward()
            if gradient is None and step==1:
                gradient={g['name']:float(torch.stack([p.grad.detach().float().square().sum() for p in g['params'] if p.grad is not None]).sum().sqrt()) for g in groups}
                assert all(v>0 for v in gradient.values())
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            if sam:
                after=(torch.get_rng_state(),torch.cuda.get_rng_state_all());active=[p for p in params if p.grad is not None]
                norm=torch.stack([p.grad.detach().float().square().sum() for p in active]).sum().sqrt();assert bool(torch.isfinite(norm))
                backups=[(p,p.detach().clone()) for p in active]
                try:
                    with torch.no_grad():
                        for p,_ in backups:p.add_(p.grad*(.1/(norm+1e-12)))
                    opt.zero_grad(set_to_none=True);torch.set_rng_state(before[0]);torch.cuda.set_rng_state_all(before[1])
                    with torch.autocast('cuda',dtype=torch.bfloat16):output=model(data['x'][ix],rel,valid,**kwargs)
                    loss=loss_fn(output,ix,mask);assert bool(torch.isfinite(loss));loss.backward()
                finally:
                    with torch.no_grad():
                        for p,original in backups:p.copy_(original)
                    torch.set_rng_state(after[0]);torch.cuda.set_rng_state_all(after[1])
            for g in groups:torch.nn.utils.clip_grad_norm_(g['params'],2.,error_if_nonfinite=True)
            opt.step();sched.step();total+=float(loss.detach())
        epoch+=1;pred,score=evaluator(model)
        if score>best:best=score;best_epoch=epoch;best_state=cpu_state(model);best_pred=pred.copy();bad=0
        else:bad+=1
        history.append(dict(epoch=epoch,target_score=score,best=best,best_epoch=best_epoch,bad=bad,loss=total/64,seconds=time.perf_counter()-tick))
        assert frozen_signature(model)==fixed;persist();atomic_json(out/'validation_history.json',history)
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),stage=stage,epoch=epoch,best=best,best_epoch=best_epoch,bad=bad,peak_MiB=peak))
        print('EPOCH',json.dumps(history[-1]|dict(stage=stage,peak_MiB=peak)),flush=True)
    if STOP:
        persist();atomic_json(out/'runtime_status.json',dict(status='stopped',pid=os.getpid(),epoch=epoch,explicit_resume_only=True));return
    model.load_state_dict(best_state);pred,score=evaluator(model);assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9
    assert frozen_signature(model)==fixed
    atomic_save(dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=score),out/'model_best.pt');atomic_array(out/'channel_pred.npy',pred)
    reference=json.loads((ROOT/'results/deep_learning/fullcv_v97_20260930/promotion.json').read_text())['scores'][target]
    report=dict(stage=stage,target=target,target_score=score,initial_score=initial,against_initial=score-initial,against_v97_target=score-reference,
        best_epoch=best_epoch,stopped_epoch=epoch,validation_rows=len(pred),predictions_reproduced_exactly=True,frozen_parameters_exact=True,
        gradient_groups=gradient,seconds=elapsed+time.perf_counter()-started,peak_MiB=peak,complete_five_folds=False,early_terminated=early_requested)
    atomic_json(out/'report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete_early_terminated' if early_requested else 'complete',pid=os.getpid(),**report))
    print('COMPACT_STAGE_FINAL',json.dumps(report),flush=True)

if __name__=='__main__':
    try:
        from training_awake_guard import TrainingAwake
        with TrainingAwake() as awake:
            print('TEMPORARY_SYSTEM_REQUIRED',awake.active,flush=True);main()
    except BaseException as exc:
        if OWNED is not None:atomic_json(OWNED/'runtime_status.json',dict(status='failed',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc()))
        raise
