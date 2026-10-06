"""Matched SW train-well nuisance loss: blocked versus reversed gradients."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from well_invariance_v40 import ROOT,PREVIOUS,Network,MODES,load_data,reference_prediction,Evaluator
from train_scaling_v21 import channel_objective
from well_invariance_v40 import prepare_domains,coefficient
from torch.nn import functional as F
from train_dual8222_heads80 import atomic_json,atomic_array,atomic_torch,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None
def stop_request(*_):
    global STOP
    STOP=True

def main():
    global OWNED
    ap=argparse.ArgumentParser();ap.add_argument('--mode',required=True,choices=MODES)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    out=a.out.resolve();mode=a.mode
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Explicit resume required')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();torch.manual_seed(20261010);np.random.seed(20261010);random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data();reference=reference_prediction();refs=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    domain_ids,domain_meta=prepare_domains(data)
    model=Network('SW',master,mode).cuda();evaluator=Evaluator(data,master,2,reference)
    frozen=cpu_copy(model.frozen_state());initial=model.trainable_state()
    params=[p for p in model.parameters() if p.requires_grad]
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('well_invariance_v40.py','well_adversarial_sequence.py','backbone_norm_v39.py','crosscurve_v32.py','champion_split_route_v29.py','champion_route_v28.py','champion_posterior_v27.py','champion_all_heads_v26.py','independent_channels_v8.py','train_scaling_v21.py')]
    cfg=dict(target='SW',mode=mode,model_module='well_invariance_v40',screening_fold=5,
        parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),parent_choice=model.choice,
        trainable_names=sorted(model.trainable_names),trainable_parameters=sum(p.numel() for p in params),
        total_parameters=sum(p.numel() for p in model.parameters()),added_parameters=sum(p.numel() for p in model.domain_head.parameters()),added_inference_parameters=0,
        comparison='same normalization scope, discriminator and initialization; only GRL coefficient differs',domain_metadata=domain_meta,domain_loss_weight=.1,grl_schedule='zero through epoch3, linear ramp4..8 to1; blocked always0',prediction_trainable_parameters=sum(p.numel() for n,p in model.named_parameters() if n in model.normalization_names),discriminator_parameters=sum(p.numel() for p in model.domain_head.parameters()),input_features=206,existing_side_features=63,
        old_output_heads_and_routes_frozen=True,backbone_forward_differentiable=True,dropout_eval=True,
        own_target_loss='original SW predictive likelihood /3 fixedQ50 + .1 train-well CE; GRL acts only on CE backbone gradient',
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        lr=1e-4,discriminator_lr=3e-4,eta_min=1e-5,weight_decay=0.,discriminator_weight_decay=.01,clip_norm_per_group=2.,optimizer_reset=True,
        quality=data['qinfo'],reference_scores=refs,validation_rows=144665,training_rows=578891,
        decoder='original singlecoverage/256bins/800actions+default/batch32',
        deterministic='warn_only; CUDA interpolation backward may be nondeterministic',
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        complete_five_folds=False,remaining_folds_started=False)
    if a.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(model.optimizer_groups())
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;history=[];timings=[];proof=None;best_updated=None
    if a.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_trainable_state(state['current']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];elapsed=state['seconds'];history=state['history'];timings=state['timings'];proof=state['proof']
        best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred'];bad=state['bad'];best_updated=state['best_updated'];updated_pred=state['updated_pred']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);windows.set_state(state['rng']['windows'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    else:
        pred,score,total=evaluator.many(model,(mode,))[mode]
        assert np.array_equal(pred,reference[:,2]) and abs(score-refs['SW'])<1e-9
        best=score;best_epoch=0;best_state=model.trainable_state();best_pred=pred.copy();updated_pred=None;bad=0
        atomic_json(out/'initial_verification.json',dict(SW=score,TOTAL=total,all144665_exact=True))
        print('INITIAL',json.dumps(dict(SW=score,TOTAL=total,trainable=cfg['trainable_parameters'])),flush=True)
    def persist():
        atomic_torch(out/'model_best.pt',dict(config=cfg,trainable_state=best_state,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        if best_updated is not None:atomic_array(out/'best_updated_pred.npy',updated_pred)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=model.trainable_state(),optimizer=cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            epoch=epoch,seconds=elapsed+time.perf_counter()-start,history=history,timings=timings,proof=proof,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,best_updated=best_updated,updated_pred=updated_pred,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats();core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    reason=None
    for ep in range(epoch+1,81):
        if STOP:break
        tick=time.perf_counter();model.train();losses=0.;target_losses=0.;domain_losses=0.;domain_correct=0;domain_seen=0;grl=coefficient(mode,ep)
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper']);relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]
            optimizer.zero_grad(set_to_none=True)
            output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])[mode]
            target_loss,_,_=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],2)
            labels=domain_ids[anchors];assert bool((labels>=0).all())
            logits=model.domain_logits(grl)
            domain_loss=F.cross_entropy(logits,labels)
            loss=target_loss+.1*domain_loss
            target_losses+=float(target_loss.detach());domain_losses+=float(domain_loss.detach())
            domain_correct+=int((logits.detach().argmax(-1)==labels).sum());domain_seen+=len(labels)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite SW likelihood')
            loss.backward()
            if proof is None:
                gradients={n:float(p.grad.float().norm()) if p.grad is not None else None for n,p in model.named_parameters() if p.requires_grad}
                assert all(v is not None and np.isfinite(v) and v>0 for v in gradients.values())
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                proof=dict(trainable_gradient_norms=gradients,frozen_gradients_none=True,scope='first actual training batch; gradients only normalization affine and training-only classifier')
                atomic_json(out/'gradient_scope.json',proof)
            for group in optimizer.param_groups:torch.nn.utils.clip_grad_norm_(group['params'],2.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();losses+=float(loss.detach())
        epoch=ep;pred,score,total=evaluator.many(model,(mode,))[mode]
        if score>best:best=score;best_epoch=ep;best_state=model.trainable_state();best_pred=pred.copy();bad=0
        else:bad+=1
        if best_updated is None or score>best_updated['target_score']:
            best_updated=dict(epoch=ep,target_score=score,reference_replacement_total=total);updated_pred=pred.copy()
        history.append(dict(epoch=ep,target_score=score,reference_replacement_total=total,best=best,best_epoch=best_epoch,bad=bad,loss=losses/64,target_loss=target_losses/64,domain_cross_entropy=domain_losses/64,domain_training_accuracy=domain_correct/domain_seen,grl_coefficient=grl,lr=optimizer.param_groups[0]['lr']))
        timings.append(dict(epoch=ep,seconds=time.perf_counter()-tick));persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,last_completed_epoch=ep,best=best,best_epoch=best_epoch,bad=bad,atlas_current_trial=True))
        print('EPOCH',json.dumps(history[-1]),flush=True)
        if ep>=10 and bad>=18:reason='patience18';break
        if ep==80:reason='budget80'
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),explicit_resume_required=True));return
    assert reason is not None
    assert all(torch.equal(v.detach().cpu(),frozen[n]) for n,v in model.frozen_state().items())
    current=model.trainable_state();change=float(sum((v.double()-initial[n].double()).square().sum() for n,v in current.items()).sqrt())
    norm_change=float(sum((v.double()-initial[n].double()).square().sum() for n,v in current.items() if n in model.normalization_names).sqrt())
    assert change>0 and norm_change>0
    model.load_trainable_state(best_state);pred,score,total=evaluator.many(model,(mode,))[mode]
    assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9
    report=dict(target='SW',mode=mode,screening_fold=5,target_score=score,reference_replacement_total=total,delta_reference_target=score-refs['SW'],
        best_epoch=best_epoch,stopped_epoch=epoch,stop_reason=reason,best_after_update=best_updated,
        trainable_parameters=cfg['trainable_parameters'],all144665_selected_predictions_restored_exactly=True,all_frozen_state_unchanged=True,
        gradient_scope=proof,last_parameter_change_L2=change,last_normalization_change_L2=norm_change,checkpoint=str((out/'model_best.pt').relative_to(ROOT)),reference_scores=refs,
        domain_metadata=domain_meta,last_domain_diagnostic={k:history[-1][k] for k in ('domain_cross_entropy','domain_training_accuracy','grl_coefficient')},seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,epoch_timing=timings,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=mode,best=best,best_epoch=best_epoch,atlas_current_trial=False))
    model.close();print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('epoch_timing','gradient_scope')}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            event=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),automatic_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',event);atomic_json(OWNED/'runtime_status.json',event)
        raise
