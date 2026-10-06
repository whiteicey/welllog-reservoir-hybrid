"""Single predefined private SW Student-t4 family candidate."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from sw_student4_v89 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction,SOURCE,DecisionObjective
from train_scaling_v21 import channel_objective
from train_dual8222_heads80 import atomic_json,atomic_torch,atomic_array,cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
from fold5_screening_805 import qualifies

STOP=False
OWNED=None
def stop_request(*_):
    global STOP
    STOP=True

def main():
    global OWNED
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=MODES,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    out=a.out.resolve();mode=a.mode;spec=MODES[mode];c=spec['c'];target=spec['target'];length=spec['length']
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed run is immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Explicit resume required')
    out.mkdir(parents=True,exist_ok=True);OWNED=out
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.5),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,target=target,atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,metadata=load_data();reference=reference_prediction();refs=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    model=Network(mode,metadata).cuda();prior=model.initialize_priors(data)
    evaluator=Evaluator(data,metadata,c,reference)
    criterion=DecisionObjective(evaluator,metadata,mode).cuda()
    atomic_json(out/'input_context_manifest.json',metadata['bank_manifest'])
    initial_body=cpu_copy(model.body.state_dict())
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('sw_student4_v89.py','sw_utility_strength_v82.py','sw_full_utility_v81.py','sw_head_utility_v80r1.py','fold5_screening_805.py','train_dual8222_heads80.py','sw_dropout_extend_v77.py','sw_dropout_rate_v76.py','sw_key_dropout_v74.py','sw_interaction_v56.py','well_conditioned_v46.py','divergent_models_v45.py','train_scaling_v21.py','independent_sw_scratch_v10.py','por_t3_patch_loss.py','multiscale_sequence.py')]
    cfg=dict(mode=mode,target=target,screening_fold=5,model_module='sw_student4_v89',
        trainable_parameters=sum(p.numel() for p in model.parameters()),random_network_initialization=False,
        loaded_network_weights=True,old_predictions_as_input=False,only_own_target_labels=True,
        input_features=206,input_variant='original206 plus input-only whole-well128 bank; no well ID feature',
        window_length=length,central_supervised_points=128,validation_stride=128,validation_batch=16,
        quality=data['qinfo'],decoder='original physical256 support/800 actions+default; own Student-t4 SW density',
        density_components=8,initialization=prior,initial_state_sha256=initial_hash,initial_head_sha256=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.output.state_dict().values())).hexdigest(),reference_scores=refs,
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        lr=1e-4,eta_min=1e-5,weight_decay=.01,clip_norm=2.,
        utility_weight=spec['utility_weight'],decision_temperature=.05,differentiable_student4_interval_mass=True,density_family='Student-t4',std_to_scale='divide_sqrt2',infinite_edge_tails_constant=True,loss='SWBCE+StudentT4NLL+alpha_jointsoftactionregret;divide3',inference_argmax_unchanged=True,
        target_normalization=dict(median=metadata['sw_target_median'],scale=metadata['sw_target_scale'],checkpoint=metadata['preprocessing_checkpoint'],sha256=metadata['preprocessing_sha256']),
        scoring_kernel_sha256=hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        training_rows=len(data['train_raw']),validation_rows=len(data['validation']),preprocessing_fit='original64trainingwells',
        training_data_cache='results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
        base_preprocessing_metadata_only=True,matched_context_comparison=False,matched_backbone_initialization_comparison=False,matched_full_loss_comparison=False,matched_density_family_comparison=True,training_key_drop_probability=spec['rate'],matched_control_mode='historical_gaussian_a100',historical_control_reused=True,training_objective_mode=mode,parent_source_checkpoint=str(SOURCE.relative_to(ROOT)),historical_p40_reference='outputs/pure_dl_sw_dropout_extend_v77_SW_p40_fold5',historical_alpha05_reference='outputs/pure_dl_sw_full_utility_v81_SW_joint_fold5',evaluation_drop_probability=0.,same_random_draws_both_arms=True,reset_output_prior=True,unmatched_architecture_and_target_comparison=False,bank_manifest=metadata['bank_manifest'],conditioning_bound=.5,all_backbone_and_head_trainable=True,
        deterministic_mode='warn_only; no bitwise retraining claim',remaining_folds_started=False,complete_five_folds=False)
    if a.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    masks=torch.Generator(device='cuda').manual_seed(20261145)
    epoch=0;elapsed=0.;history=[];timings=[];proof=None;bad=0;updated=None
    if a.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['current']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];elapsed=state['seconds'];history=state['history'];timings=state['timings'];proof=state['proof']
        updated=state['updated'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred'];bad=state['bad']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);windows.set_state(state['rng']['windows']);masks.set_state(state['rng']['masks'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    else:
        pred,score,scores=evaluator(model)
        assert pred.shape==reference[:,c].shape and np.isfinite(pred).all()
        best=score;best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy()
        atomic_json(out/'initial.json',dict(target_score=score,reference_replacement_scores=scores,initialization=prior,all144665_parent_exact=False,initial_head_prior_only=True,initial_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
        print('INITIAL',json.dumps(dict(target=target,score=score,parameters=cfg['trainable_parameters'],length=length)),flush=True)
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            epoch=epoch,seconds=elapsed+time.perf_counter()-start,history=history,timings=timings,proof=proof,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,updated=updated,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),masks=masks.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats();core=torch.zeros((1,length),device='cuda',dtype=torch.bool);core[:,length//2-64:length//2+64]=True
    reason=None;maskinfo=None
    for ep in range(epoch+1,81):
        if STOP:break
        # Explicit evidence-based cancellation at an atomic completed-epoch boundary.
        if epoch>=10 and (out/'stop_requested.json').exists():
            request=json.loads((out/'stop_requested.json').read_text(encoding='utf-8-sig'));reason='evidence_stop: '+request['reason'];break
        tick=time.perf_counter();model.train();losses=nlls=auxs=0.
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'],length=length)
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10);mask=valid&core&data['labeled'][ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output,aux,maskinfo=model(data['x'][ix],relative,valid,return_aux=True,well_inputs=data['well_bank'][data['row_well'][anchors]])
            loss,nll,aux=criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])
            if proof is None and step==1:
                pathgrads=torch.autograd.grad(aux,output,retain_graph=True,allow_unused=False)
                decision_paths={name:float(grad.detach().float().norm()) for name,grad in zip(('gate','StudentT4_density'),pathgrads)}
                assert all(np.isfinite(v) and v>0 for v in decision_paths.values())
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite own-target loss')
            loss.backward()
            if proof is None and step==1:
                grads={}
                for group in ('body.','output.','condition.','body.global_blocks.0.ff.'):
                    values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith(group) and p.grad is not None]
                    grads[group]=float(torch.stack(values).sum().sqrt())
                assert all(np.isfinite(v) and v>0 for v in grads.values())
                proof=dict(gradient_norms=grads,decision_gradient_norms=decision_paths,training_labels_only=True,own_target=target,first_actual_second_step=True,maskinfo=maskinfo,train_raw_anchors_only=True)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step();scheduler.step()
            losses+=float(loss.detach());nlls+=float(nll.detach());auxs+=float(aux.detach())
        epoch=ep;pred,score,scores=evaluator(model)
        if updated is None or score>updated['target_score']:
            updated=dict(epoch=ep,target_score=score,reference_replacement_total=scores['TOTAL'])
            atomic_torch(out/'model_best_updated.pt',dict(state_dict=cpu_copy(model.state_dict()),config=cfg,**updated))
            atomic_array(out/'best_updated_pred.npy',pred)
        if score>best:best=score;best_epoch=ep;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else:bad+=1
        history.append(dict(epoch=ep,target_score=score,reference_replacement_total=scores['TOTAL'],best=best,best_epoch=best_epoch,bad=bad,
            loss=losses/64,own_nll=nlls/64,joint_decision_loss=auxs/64,lr=optimizer.param_groups[0]['lr'],last_maskinfo=maskinfo))
        timings.append(dict(epoch=ep,seconds=time.perf_counter()-tick));persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,target=target,last_completed_epoch=ep,best=best,best_epoch=best_epoch,bad=bad,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,atlas_current_trial=True))
        print('EPOCH',json.dumps(history[-1]|dict(seconds=timings[-1]['seconds'],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)),flush=True)
        if ep>=10 and bad>=18:reason='patience18';break
        if ep==80:reason='budget80'
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),explicit_resume_required=True));return
    assert reason is not None
    model.load_state_dict(best_state);pred,score,scores=evaluator(model)
    assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9
    change=float(sum((v.detach().cpu().double()-initial_body[n].double()).square().sum() for n,v in model.body.state_dict().items() if v.is_floating_point()).sqrt())
    report=dict(target=target,mode=mode,screening_fold=5,target_score=score,reference_replacement_scores=scores,delta_reference_target=score-refs[target],
        best_epoch=best_epoch,stopped_epoch=epoch,stop_reason=reason,best_after_update=updated,trainable_parameters=cfg['trainable_parameters'],all144665_selected_predictions_restored_exactly=True,
        strict805_qualified=qualifies(scores['TOTAL']),selected_body_parameter_change_L2=change,conditioning_output_norm=float(model.condition.output.weight.detach().float().norm()),gradient_scope=proof,checkpoint=str((out/'model_best.pt').relative_to(ROOT)),reference_scores=refs,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,epoch_timing=timings,
        random_network_initialization=False,complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=mode,best=best,best_epoch=best_epoch,atlas_current_trial=False))
    print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            event=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),automatic_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',event);atomic_json(OWNED/'runtime_status.json',event)
        raise
