"""Train two random-initialized private POR sizes under one fixed protocol."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from datetime import datetime
from pathlib import Path
import argparse, hashlib, json, random, signal, sys, time, traceback
import numpy as np
import torch
from scratch_scaling_v23 import Network, metadata_from_training, load_data, REFERENCE, STUDY, ROOT, MODES
from train_scaling_v21 import Evaluator, channel_objective
from train_dual8222_heads80 import atomic_json, atomic_torch, atomic_array, cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None


def stop_request(*_):
    global STOP
    STOP=True
    print('STOP_REQUESTED: complete current epoch and save; no automatic resume',flush=True)


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=tuple(MODES),required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.out.resolve()
    if (out/'fold_5_report.json').exists(): raise RuntimeError('Completed result immutable')
    if (out/'config.json').exists() and not args.resume: raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3)
    print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=args.mode,atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter()
    torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data()
    metadata=metadata_from_training(data,master)
    model=Network(metadata,args.mode).cuda()
    reference=np.load(REFERENCE/'fold_5_pred.npy')
    ref_report=json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))
    evaluator=Evaluator(data,master,0,reference)
    train_names={n for n,p in model.named_parameters() if p.requires_grad}
    assert train_names and all(p.requires_grad for p in model.parameters())
    fixed={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in train_names}
    original_body={n:p.detach().cpu().clone() for n,p in model.backbone_parameters()}
    source_names=('scratch_scaling_v23.py','train_scaling_v21.py','champion_scaling_v21.py',
        'independent_channels_v8.py','paired_local_context_router_sequence.py','paired_detached_router_sequence.py',
        'paired_calibrated_view_sequence.py','paired_well_view_sequence.py','global_cosine_sequence.py',
        'blur_ungated_sequence.py','dual8222_location_readout_v5.py')
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in source_names]
    cfg=dict(mode=args.mode,target='POR',architecture='dual',model_module='scratch_scaling_v23',screening_fold=5,
        scaling=model.scaling_manifest,trainable_parameters=sum(p.numel() for p in model.parameters()),
        cold_network_initialization=True,source_network_weights_loaded=False,model_metadata=metadata,
        training_protocol='private POR final-mixture likelihood; not historical multitarget/deep-supervised pretraining',
        cache='results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
        inherited_train_only_preprocessing_and_Q=True,quality=data['qinfo'],
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        learning_rate=.0006,eta_min=.00001,weight_decay=.01,clip_norm=2.,
        objective='v21 final predictive POR hurdle Student-t likelihood /3, fixedQ50, no other target loss',
        auxiliary_supervision=False,router_gradient='joint POR predictive likelihood; same for both capacities',
        decoder='unchanged original singlecoverage 256bins/800actions+default/batch32',
        validation_rows=144665,training_rows=578891,input_features=206,relative_depth=True,window=256,core=[64,192],
        reference_scores=ref_report['scores'],data_statistics_fitted_on_original64=True,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__),deterministic='warn_only; no bitwise repeat-training claim',
        actual_dropout_types=sorted({type(m).__name__ for n,m in model.named_modules() if n.endswith('dropout')}),
        complete_five_folds=False,remaining_folds_started=False)
    if args.resume: assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else: atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0006,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;prior_seconds=0.;proof=None
    if args.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['current'],strict=True);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];prior_seconds=state['seconds'];proof=state['proof']
        best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred']
        bad=state['bad'];history=state['history'];initial=state['initial']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        windows.set_state(state['rng']['windows']);np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python']);del state
    else:
        pred,score,total=evaluator(model)
        initial=dict(POR=score,TOTAL=total,network_pretrained=False,
            common_state_sha256=model.scaling_manifest['common_initial_state_sha256'])
        atomic_json(out/'initial_verification.json',initial);atomic_array(out/'initial_pred.npy',pred)
        best,best_epoch,best_state,best_pred=score,0,cpu_copy(model.state_dict()),pred.copy()
        bad=0;history=[]
        print('INITIAL',json.dumps(initial),flush=True)

    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,metadata=metadata,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(),epoch=epoch,seconds=prior_seconds+time.perf_counter()-start,proof=proof,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,history=history,initial=initial,
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
            mask=valid&core&data['labeled'][ix];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            loss,bce,nll=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],0)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if proof is None and step==1:
                def norm(predicate):
                    values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if predicate(n) and p.grad is not None]
                    return float(torch.stack(values).sum().sqrt()) if values else 0.
                body=norm(lambda n:'.main_head.' not in n)
                extra=norm(lambda n:'.extra_' in n)
                assert body>0 and np.isfinite(body+extra)
                if args.mode=='ffn2': assert extra>0
                proof=dict(backbone_gradient_norm=body,extra_gradient_norm=extra,
                    extra_input_projection_gradient_norm=norm(lambda n:'.extra_expand.' in n or '.extra_reduce.' in n),
                    POR_only_supervision=True,second_actual_training_step=True)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();stats+=np.array([float(loss.detach()),float(bce.detach()),float(nll.detach())])
        epoch=ep;pred,score,total=evaluator(model)
        if score>best:best,best_epoch,best_state,best_pred,bad=score,epoch,cpu_copy(model.state_dict()),pred.copy(),0
        else:bad+=1
        row=dict(epoch=epoch,target_score=score,reference_replacement_total=total,best_target_score=best,best_epoch=best_epoch,bad_epochs=bad,
            training_loss=stats[0]/64,training_bce=stats[1]/64,training_nll=stats[2]/64,
            learning_rate=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-tick)
        history.append(row);persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=args.mode,last_completed_epoch=epoch,
            best_target_score=best,best_epoch=best_epoch,reference_replacement_total=ref_report['scores']['TOTAL']+.3*(best-ref_report['scores']['POR']),atlas_current_trial=True))
        print('EPOCH',json.dumps(row),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),mode=args.mode,auto_resume=False,resume_available=True));return
    last_change=float(sum((p.detach().cpu().double()-original_body[n].double()).square().sum() for n,p in model.backbone_parameters()).sqrt())
    assert last_change>0
    model.load_state_dict(best_state,strict=True);pred,score,total=evaluator(model)
    assert np.array_equal(pred,best_pred)
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    selected_change=float(sum((p.detach().cpu().double()-original_body[n].double()).square().sum() for n,p in model.backbone_parameters()).sqrt())
    report=dict(fold=5,screening_fold=5,target='POR',mode=args.mode,scores={'POR':score},target_score=score,
        reference_replacement_total=total,delta_reference_POR=score-ref_report['scores']['POR'],
        best_epoch=best_epoch,stopped_epoch=epoch,best_after_update=max(history,key=lambda r:r['target_score']) if history else None,
        initial=initial,trainable_parameters=cfg['trainable_parameters'],scaling=model.scaling_manifest,
        last_backbone_change_L2=last_change,selected_backbone_change_L2=selected_change,
        all144665selected_predictions_restored_exactly=True,unused_packed_rows_unchanged=True,
        gradient_scope=proof,cold_network_training=True,inherited_Q50_and_preprocessing=True,
        seconds=prior_seconds+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=args.mode,atlas_current_trial=False))
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            error=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',error);atomic_json(OWNED/'runtime_status.json',error)
        raise
