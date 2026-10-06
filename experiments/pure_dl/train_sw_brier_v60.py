"""Train matched original gate and density heads with a Brier auxiliary."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from sw_brier_v60 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_cache,reference_prediction,Objective
from train_dual8222_heads80 import atomic_json,atomic_torch,atomic_array,cpu_copy
from gpu_job_guard_v2 import check_slot

STOP=False
OWNED=None
def stop_request(*_):
    global STOP
    STOP=True

def digest(state):return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in state.values())).hexdigest()

def main():
    global OWNED
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    out=a.out.resolve();mode=a.mode
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed run is immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Explicit resume required')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=2.),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,target='SW',atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    cache=load_cache();reference=reference_prediction();refs=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    model=Network(mode,cache['metadata']).cuda();core_hash=model.core_hash()
    assert core_hash==cache['manifest']['core_sha256']
    train={k:cache['train'][k].cuda() for k in ('h','gate','mask','y','default','quality')}
    evaluator=Evaluator(cache,reference)
    names=('sw_brier_v60.py','sw_rqs_staging_v58.py','quantile_density_v48.py','sw_interaction_v56.py','well_conditioned_v46.py','divergent_models_v45.py','train_scaling_v21.py','independent_sw_scratch_v10.py','por_t3_patch_loss.py','multiscale_sequence.py')
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in names]
    params=list(model.output.parameters())+list(model.gate.parameters());objective=Objective(cache['edges'],mode).cuda()
    assert {id(p) for p in params}=={id(p) for p in model.parameters() if p.requires_grad}
    cfg=dict(mode=mode,target='SW',screening_fold=5,model_module='sw_brier_v60',
        total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in params),
        initialization=model.initialization_info,initial_state_sha256=digest(model.state_dict()),initial_head_sha256=digest(model.output.state_dict()),
        frozen_core_sha256=core_hash,cache_manifest=cache['manifest'],cache_manifest_sha256=hashlib.sha256((STUDY/'cache_manifest.json').read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},reference_scores=refs,
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,lr=1e-4,eta_min=1e-5,weight_decay=.01,clip_norm=2.,
        training_sampler='uniform canonical training blocks; historical random-anchor contexts changed equally for both arms',
        quality='original Q50, cached unchanged',only_own_target_labels=True,validation_labels_train=False,
        matched_parameter_count=True,matched_initial_function=True,comparison='same heads: likelihood versus likelihood plus .5*(gate Brier +16*conditional categorical Brier)',
        frozen='body/norm/FiLM',decoder='original v48 RQS CDF; physical256 support/800 actions plus default',
        complete_five_folds=False,remaining_folds_started=False)
    if a.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;history=[];timings=[];proof=None;bad=0
    if a.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['current']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];elapsed=state['seconds'];history=state['history'];timings=state['timings'];proof=state['proof']
        best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred'];bad=state['bad']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);windows.set_state(state['rng']['windows'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    else:
        pred,score,scores=evaluator(model)
        assert abs(score-cache['manifest']['initial_score'])<1e-9
        best=score;best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy()
        atomic_json(out/'initial.json',dict(target_score=score,reference_replacement_scores=scores,all144665_parent_exact=True,
            initial_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
        print('INITIAL',json.dumps(dict(score=score,parameters=cfg['trainable_parameters'])),flush=True)
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            epoch=epoch,seconds=elapsed+time.perf_counter()-start,history=history,timings=timings,proof=proof,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats();reason=None
    for ep in range(epoch+1,81):
        if STOP:break
        if epoch>=10 and (out/'stop_requested.json').exists():
            request=json.loads((out/'stop_requested.json').read_text(encoding='utf-8-sig'));reason='evidence_stop: '+request['reason'];break
        tick=time.perf_counter();model.train();losses=0.
        for step in range(64):
            ids=torch.randint(len(train['h']),(64,),device='cuda',generator=windows)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                raw=model.output(train['h'][ids]).float();gate=model.gate(train['h'][ids]).float()
            loss,loss_stats=objective((gate,raw),train['y'][ids],train['default'][ids],train['mask'][ids],train['quality'][ids])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite own SW loss')
            loss.backward()
            if proof is None and step==1:
                grads={}
                for group in ('output.','gate.'):
                    values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith(group) and p.grad is not None]
                    grads[group]=float(torch.stack(values).sum().sqrt()) if values else 0.
                assert all(np.isfinite(v) and v>0 for v in grads.values())
                assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith(('output.','gate.')))
                proof=dict(gradient_norms=grads,frozen_parameters_no_gradient=True,only_training_pack_sampled=True,own_target='SW')
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(params,2.,error_if_nonfinite=True);optimizer.step();scheduler.step();losses+=float(loss.detach())
        epoch=ep;pred,score,scores=evaluator(model)
        if score>best:best=score;best_epoch=ep;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else:bad+=1
        history.append(dict(epoch=ep,target_score=score,reference_replacement_total=scores['TOTAL'],best=best,best_epoch=best_epoch,bad=bad,loss=losses/64,last_loss_stats=loss_stats,lr=optimizer.param_groups[0]['lr']))
        timings.append(dict(epoch=ep,seconds=time.perf_counter()-tick));persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,target='SW',last_completed_epoch=ep,best=best,best_epoch=best_epoch,bad=bad,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,atlas_current_trial=True))
        print('EPOCH',json.dumps(history[-1]|dict(seconds=timings[-1]['seconds'])),flush=True)
        if ep>=10 and bad>=18:reason='patience18';break
        if ep==80:reason='budget80'
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),explicit_resume_required=True));return
    assert reason is not None and model.core_hash()==core_hash
    model.load_state_dict(best_state);pred,score,scores=evaluator(model)
    assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9 and model.core_hash()==core_hash
    report=dict(target='SW',mode=mode,screening_fold=5,target_score=score,reference_replacement_scores=scores,delta_reference_target=score-refs['SW'],
        best_epoch=best_epoch,stopped_epoch=epoch,stop_reason=reason,trainable_parameters=cfg['trainable_parameters'],all144665_selected_predictions_restored_exactly=True,
        frozen_core_unchanged=True,gradient_scope=proof,checkpoint=str((out/'model_best.pt').relative_to(ROOT)),reference_scores=refs,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,epoch_timing=timings,
        cache_inference_only=True,raw_input_replay_pending=True,complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=mode,best=best,best_epoch=best_epoch,atlas_current_trial=False))
    print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            event=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),automatic_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',event);atomic_json(OWNED/'runtime_status.json',event)
        raise
