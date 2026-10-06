"""Bounded, matched expert-supervision study; private POR only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,sys,time,traceback
import numpy as np
import torch
from expert_supervision_v24 import Network,MODES,ROOT,STUDY,REFERENCE,load_data,metadata_from_training,training_losses
from train_scaling_v21 import Evaluator
from train_dual8222_heads80 import atomic_json,atomic_torch,atomic_array,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None


def stop_request(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: finish current epoch, save; explicit resume only',flush=True)


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed experiment immutable')
    if (out/'config.json').exists() and not args.resume:raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',mode=args.mode,pid=os.getpid(),atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data();metadata=metadata_from_training(data,master)
    model=Network(metadata,args.mode).cuda()
    reference=np.load(REFERENCE/'fold_5_pred.npy')
    ref=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))['scores']
    evaluator=Evaluator(data,master,0,reference)
    parameters={'expert':list(model.members.parameters()),'router':list(model.router.parameters())}
    ids=[{id(p) for p in group} for group in parameters.values()]
    assert ids[0].isdisjoint(ids[1]) and ids[0]|ids[1]=={id(p) for p in model.parameters()}
    assert all(p.requires_grad for p in model.parameters())
    names={n for n,p in model.named_parameters()}
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in names}
    original={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    source_names=('expert_supervision_v24.py','scratch_scaling_v23.py','independent_channels_v8.py',
        'train_scaling_v21.py','champion_scaling_v21.py','paired_detached_router_sequence.py')
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in source_names]
    cfg=dict(mode=args.mode,target='POR',model_module='expert_supervision_v24',screening_fold=5,
        trainable_parameters=sum(p.numel() for p in model.parameters()),model_metadata=metadata,
        common_initial_state_sha256=model.scaling_manifest['common_initial_state_sha256'],
        source_network_weights_loaded=False,architecture='unchanged v23 base1 private dual',
        expert_objective='mixture responsibility weighted' if args.mode=='mixture' else 'mean of independent expert likelihoods',
        router_objective='mixture likelihood with detached expert outputs',
        expert_gradient_does_not_update_router=True,router_gradient_does_not_update_experts=True,
        optimizers='two AdamW; experts together and router separate; same for both arms',
        gradient_clipping='expert group norm2 and router group norm2 independently',
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        learning_rate=.0006,eta_min=.00001,weight_decay=.01,clip_norm=2.,quality=data['qinfo'],
        cache='results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
        decoder='source singlecoverage256bins/800actions+default/batch32',
        reference_scores=ref,validation_rows=144665,training_rows=578891,
        supervision='POR only; no deep auxiliary loss; inherited train-only Q50',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__),deterministic_mode='warn_only; not bitwise training reproducibility',
        complete_five_folds=False,remaining_folds_started=False)
    if args.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    opts={k:torch.optim.AdamW(v,lr=.0006,weight_decay=.01) for k,v in parameters.items()}
    scheds={k:torch.optim.lr_scheduler.CosineAnnealingLR(v,80*64,eta_min=1e-5) for k,v in opts.items()}
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;prior_seconds=0.;proof=None
    if args.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['current'],strict=True)
        for k in opts:opts[k].load_state_dict(state['optimizers'][k]);scheds[k].load_state_dict(state['schedulers'][k])
        epoch=state['epoch'];prior_seconds=state['seconds'];proof=state['proof']
        best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred']
        history=state['history'];initial=state['initial'];bad=state['bad']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        windows.set_state(state['rng']['windows']);np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python']);del state
    else:
        pred,score,total=evaluator(model)
        initial_reference=np.load(ROOT/'outputs/pure_dl_scratch_scaling_v23_base1_fold5/initial_pred.npy')
        assert np.array_equal(pred,initial_reference),'Changed initial network function'
        initial=dict(POR=score,TOTAL=total,all144665_equal_to_v23_initial=True,
            common_initial_state_sha256=cfg['common_initial_state_sha256'])
        atomic_json(out/'initial_verification.json',initial);atomic_array(out/'initial_pred.npy',pred)
        best,best_epoch,best_state,best_pred=score,0,cpu_copy(model.state_dict()),pred.copy()
        bad=0;history=[];print('INITIAL',json.dumps(initial),flush=True)

    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,metadata=metadata,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),
            optimizers={k:cpu_copy(v.state_dict()) for k,v in opts.items()},schedulers={k:v.state_dict() for k,v in scheds.items()},
            epoch=epoch,seconds=prior_seconds+time.perf_counter()-start,proof=proof,best=best,best_epoch=best_epoch,
            best_state=best_state,best_pred=best_pred,history=history,initial=initial,bad=bad,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),
                numpy=np.random.get_state(),python=random.getstate())))

    persist();torch.cuda.reset_peak_memory_stats()
    core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    for ep in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18):break
        tick=time.perf_counter();model.train();stats=np.zeros(3)
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]
            for opt in opts.values():opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs,logits=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix],return_parts=True)
            loss,expert_loss,router_loss=training_losses(outputs,logits,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],args.mode)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
            check=proof is None and step==1
            if check:
                er=torch.autograd.grad(expert_loss,parameters['router'][0],allow_unused=True,retain_graph=True)[0]
                re=torch.autograd.grad(router_loss,parameters['expert'][0],allow_unused=True,retain_graph=True)[0]
                assert er is None and re is None,'Gradient isolation failed'
            loss.backward()
            if check:
                def norm(group):
                    values=[p.grad.detach().float().square().sum() for p in group if p.grad is not None]
                    return float(torch.stack(values).sum().sqrt()) if values else 0.
                member_norms=[norm(list(m.parameters())) for m in model.members]
                router_norm=norm(parameters['router'])
                assert all(v>0 and np.isfinite(v) for v in member_norms+[router_norm])
                proof=dict(member_gradient_norms=member_norms,router_gradient_norm=router_norm,
                    expert_loss_to_router_gradient_none=True,router_loss_to_expert_gradient_none=True,
                    second_actual_step=True,POR_only_supervision=True)
                atomic_json(out/'gradient_scope.json',proof)
            for k,group in parameters.items():
                torch.nn.utils.clip_grad_norm_(group,2.,error_if_nonfinite=True);opts[k].step();scheds[k].step()
            stats+=np.array([float(loss.detach()),float(expert_loss.detach()),float(router_loss.detach())])
        epoch=ep;pred,score,total=evaluator(model)
        if score>best:best,best_epoch,best_state,best_pred,bad=score,epoch,cpu_copy(model.state_dict()),pred.copy(),0
        else:bad+=1
        row=dict(epoch=epoch,target_score=score,reference_replacement_total=total,best_target_score=best,best_epoch=best_epoch,bad_epochs=bad,
            training_loss=stats[0]/64,training_expert_loss=stats[1]/64,training_router_loss=stats[2]/64,
            learning_rates={k:v.param_groups[0]['lr'] for k,v in opts.items()},seconds=time.perf_counter()-tick)
        history.append(row);persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=args.mode,last_completed_epoch=epoch,
            best_target_score=best,best_epoch=best_epoch,reference_replacement_total=ref['TOTAL']+.3*(best-ref['POR']),atlas_current_trial=True))
        print('EPOCH',json.dumps(row),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',mode=args.mode,pid=os.getpid(),auto_resume=False,resume_available=True));return
    last_change=float(sum((p.detach().cpu().double()-original[n].double()).square().sum() for n,p in model.named_parameters()).sqrt())
    assert last_change>0
    model.load_state_dict(best_state,strict=True);pred,score,total=evaluator(model)
    assert np.array_equal(pred,best_pred)
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    selected_change=float(sum((p.detach().cpu().double()-original[n].double()).square().sum() for n,p in model.named_parameters()).sqrt())
    report=dict(fold=5,screening_fold=5,target='POR',mode=args.mode,scores={'POR':score},target_score=score,
        reference_replacement_total=total,delta_reference_POR=score-ref['POR'],best_epoch=best_epoch,stopped_epoch=epoch,
        best_after_update=max(history,key=lambda r:r['target_score']) if history else None,initial=initial,
        trainable_parameters=cfg['trainable_parameters'],gradient_scope=proof,
        last_parameter_change_L2=last_change,selected_parameter_change_L2=selected_change,
        all144665selected_predictions_restored_exactly=True,unused_packed_buffers_unchanged=True,
        seconds=prior_seconds+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=args.mode,atlas_current_trial=False))
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            failure=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',failure);atomic_json(OWNED/'runtime_status.json',failure)
        raise
