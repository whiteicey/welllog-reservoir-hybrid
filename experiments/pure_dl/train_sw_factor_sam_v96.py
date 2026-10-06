"""Matched fixed-atom private-density SAM loss comparison."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from sw_factor_sam_v96 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction,GATE,DENSITY,DecisionObjective
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
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=4.),flush=True)
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
    sources=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('sw_factor_sam_v96.py','sw_factorization_v95.py','sw_sam_radius_v94.py','sw_sam_v93.py','sw_utility_strength_v82.py','sw_full_utility_v81.py','sw_head_utility_v80r1.py','fold5_screening_805.py','train_dual8222_heads80.py','sw_dropout_extend_v77.py','sw_dropout_rate_v76.py','sw_key_dropout_v74.py','sw_interaction_v56.py','well_conditioned_v46.py','divergent_models_v45.py','train_scaling_v21.py','independent_sw_scratch_v10.py','por_t3_patch_loss.py','multiscale_sequence.py')]
    cfg=dict(mode=mode,target=target,screening_fold=5,model_module='sw_factor_sam_v96',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()),
        random_network_initialization=False,loaded_network_weights=True,only_own_target_labels=True,
        input_features=206,window_length=length,central_supervised_points=128,validation_stride=128,validation_batch=16,
        quality=data['qinfo'],decoder='original physical256 support/800 actions+default; one conditional SW law',
        initialization=prior,initial_state_sha256=initial_hash,initial_fixed_hash=model.fixed_hash(),reference_scores=refs,
        density_components=8,density_family='Gaussian',epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        lr=1e-4,eta_min=1e-5,weight_decay=.01,clip_norm=2.,utility_weight=spec['utility_weight'],decision_temperature=.05,
        loss='own SW GaussianNLL+constant frozen BCE+alpha_jointsoftactionregret; divide3',
        sam_rho=spec['rho'],sam_normalization='global_unclipped_private_density_gradient_L2',optimizer='two-pass SAM around AdamW',
        reuse_dropkey_mask=True,restore_parameters_by_exact_copy=True,forward_backward_evaluations_per_update=2,
        target_normalization=dict(median=metadata['sw_target_median'],scale=metadata['sw_target_scale'],checkpoint=metadata['preprocessing_checkpoint'],sha256=metadata['preprocessing_sha256']),
        scoring_kernel_sha256=hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        training_rows=len(data['train_raw']),validation_rows=len(data['validation']),preprocessing_fit='original64trainingwells',bank_manifest=metadata['bank_manifest'],
        frozen_atom_eval_no_grad=True,private_density_backbone_and_head_trainable=True,unused_density_gate_parameters=129,
        initialization_preserves_complete_v95_function=True,reset_output_prior=False,training_key_drop_probability=.4,evaluation_drop_probability=0.,
        serial_execution_required=True,same_random_draws_both_arms=True,inference_argmax_unchanged=True,
        deterministic_mode='warn_only; no bitwise retraining claim',remaining_folds_started=False,complete_five_folds=False)
    assert cfg['trainable_parameters']==839481 and cfg['total_parameters']==1678962
    if a.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    masks=torch.Generator(device='cuda').manual_seed(20261145)
    epoch=0;elapsed=0.;history=[];timings=[];proof=None;bad=0;updated=None;peak_allocated=peak_reserved=0.
    if a.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['current']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];elapsed=state['seconds'];history=state['history'];timings=state['timings'];proof=state['proof']
        peak_allocated=state['memory_peaks']['allocated_MiB'];peak_reserved=state['memory_peaks']['reserved_MiB']
        updated=state['updated'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred'];bad=state['bad']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);windows.set_state(state['rng']['windows']);masks.set_state(state['rng']['masks'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
    else:
        pred,score,scores=evaluator(model)
        assert pred.shape==reference[:,c].shape and np.isfinite(pred).all()
        assert np.array_equal(pred,reference[:,2]) and abs(score-refs['SW'])<1e-9
        best=score;best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy()
        atomic_json(out/'initial.json',dict(target_score=score,reference_replacement_scores=scores,initialization=prior,all144665_parent_exact=True,initial_head_prior_only=False,initial_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
        print('INITIAL',json.dumps(dict(target=target,score=score,parameters=cfg['trainable_parameters'],length=length)),flush=True)
    def persist():
        nonlocal peak_allocated,peak_reserved
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated()/2**20);peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved()/2**20)
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            epoch=epoch,seconds=elapsed+time.perf_counter()-start,history=history,timings=timings,proof=proof,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,updated=updated,memory_peaks=dict(allocated_MiB=peak_allocated,reserved_MiB=peak_reserved),
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
            rng_before=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output,_,maskinfo=model(data['x'][ix],relative,valid,return_aux=True,well_inputs=data['well_bank'][data['row_well'][anchors]])
            base_loss,_,_=criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])
            if not bool(torch.isfinite(base_loss)):raise FloatingPointError('Nonfinite base loss')
            base_loss.backward();rng_after=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
            params=[p for p in model.parameters() if p.grad is not None]
            grad_norm=torch.stack([p.grad.detach().float().square().sum() for p in params]).sum().sqrt()
            if not bool(torch.isfinite(grad_norm)):raise FloatingPointError('Nonfinite base gradient')
            backups=[(p,p.detach().clone()) for p in params]
            perturbations=[]
            try:
                with torch.no_grad():
                    factor=spec['rho']/(grad_norm+1e-12)
                    for p,old in backups:
                        e=p.grad*factor;perturbations.append(e.detach().float().square().sum());p.add_(e)
                perturb_norm=float(torch.stack(perturbations).sum().sqrt())
                assert abs(perturb_norm-spec['rho'])<1e-6
                optimizer.zero_grad(set_to_none=True)
                torch.set_rng_state(rng_before[0]);torch.cuda.set_rng_state_all(rng_before[1])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output,_,second_mask=model(data['x'][ix],relative,valid,return_aux=True,well_inputs=data['well_bank'][data['row_well'][anchors]])
                loss,nll,aux=criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])
                assert second_mask==maskinfo
                if proof is None and step==1:
                    gs=torch.autograd.grad(aux,output[1],retain_graph=True,allow_unused=False)
                    decision_paths={'Gaussian_density':float(gs[0].detach().float().norm())}
                    assert all(np.isfinite(v) and v>0 for v in decision_paths.values())
                if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite perturbed loss')
                loss.backward()
                rng_second=(torch.get_rng_state(),torch.cuda.get_rng_state_all())
                assert torch.equal(rng_second[0],rng_after[0]) and all(torch.equal(a,b) for a,b in zip(rng_second[1],rng_after[1])), 'Second pass RNG consumption mismatch'
                if proof is None and step==1:
                    grads={}
                    for group in ('density.body.','density.output.','density.condition.','density.body.global_blocks.0.ff.'):
                        values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith(group) and p.grad is not None]
                        grads[group]=float(torch.stack(values).sum().sqrt())
                    assert all(np.isfinite(v) and v>0 for v in grads.values())
                    assert all(p.grad is None for p in model.atom.parameters())
                    proof=dict(frozen_atom_has_no_gradient=True,gradient_norms=grads,decision_gradient_norms=decision_paths,first_actual_second_step=True,own_target=target,training_labels_only=True,train_raw_anchors_only=True,
                        base_gradient_norm=float(grad_norm),requested_radius=spec['rho'],actual_perturbation_norm=perturb_norm,same_dropkey_masks=True,rng_replay=True,second_pass_rng_consumption_exact=True,exact_parameter_copy_restore=True,maskinfo=maskinfo)
                    atomic_json(out/'gradient_scope.json',proof)
            finally:
                with torch.no_grad():
                    for p,old in backups:p.copy_(old)
                torch.set_rng_state(rng_after[0]);torch.cuda.set_rng_state_all(rng_after[1])
            del backups,perturbations
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True);optimizer.step();scheduler.step()
            losses+=float(loss.detach());nlls+=float(nll.detach());auxs+=float(aux.detach())
        assert model.fixed_hash()==cfg['initial_fixed_hash']
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
            peak_allocated_MiB=peak_allocated,atlas_current_trial=True))
        print('EPOCH',json.dumps(history[-1]|dict(seconds=timings[-1]['seconds'],peak_allocated_MiB=peak_allocated)),flush=True)
        if ep>=10 and bad>=18:reason='patience18';break
        if ep==80:reason='budget80'
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),explicit_resume_required=True));return
    assert reason is not None
    model.load_state_dict(best_state);pred,score,scores=evaluator(model)
    assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9
    change=float(sum((v.detach().cpu().double()-initial_body[n].double()).square().sum() for n,v in model.body.state_dict().items() if v.is_floating_point()).sqrt())
    assert model.fixed_hash()==cfg['initial_fixed_hash']
    report=dict(fixed_current_and_selected_bitwise_unchanged=True,target=target,mode=mode,screening_fold=5,target_score=score,reference_replacement_scores=scores,delta_reference_target=score-refs[target],
        best_epoch=best_epoch,stopped_epoch=epoch,stop_reason=reason,best_after_update=updated,trainable_parameters=cfg['trainable_parameters'],all144665_selected_predictions_restored_exactly=True,
        strict805_qualified=qualifies(scores['TOTAL']),selected_body_parameter_change_L2=change,conditioning_output_norm=float(model.condition.output.weight.detach().float().norm()),gradient_scope=proof,checkpoint=str((out/'model_best.pt').relative_to(ROOT)),reference_scores=refs,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=peak_allocated,peak_reserved_MiB=peak_reserved,peak_scope='maximum across initialization and all explicitly resumed completed segments',epoch_timing=timings,
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
