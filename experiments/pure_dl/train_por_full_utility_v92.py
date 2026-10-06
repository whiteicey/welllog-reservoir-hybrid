"""Independent complete POR candidate with atomic full-state checkpoints."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import argparse,hashlib,json,random,signal,time,traceback
import numpy as np
import torch
from por_full_utility_v92 import ROOT,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction,SOURCE,DecisionObjective
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
    out=a.out.resolve();mode=a.mode
    if (out/'fold_5_report.json').exists():raise RuntimeError('Completed run is immutable')
    if (out/'config.json').exists() and not a.resume:raise RuntimeError('Explicit resume required')
    out.mkdir(parents=True,exist_ok=True);OWNED=out;torch.set_num_threads(3)
    print('ADMISSION',check_slot(min_free_gib=3.5),flush=True)
    atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,target='POR',atlas_current_trial=True))
    signal.signal(signal.SIGINT,stop_request);signal.signal(signal.SIGTERM,stop_request)
    start=time.perf_counter();random.seed(20261010);np.random.seed(20261010);torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data();reference=reference_prediction();refs=json.loads((PREVIOUS/'CURRENT_RECIPE.json').read_text(encoding='utf-8'))['scores']
    assert reference.shape==(144665,3) and len(data['train_raw'])==578891
    model=Network(mode,master).cuda();evaluator=Evaluator(data,master,0,reference);criterion=DecisionObjective(evaluator,master,mode).cuda()
    initial_body=cpu_copy(model.source.state_dict());buffers=cpu_copy(dict(model.named_buffers()));frozen_parameters=cpu_copy({n:p for n,p in model.named_parameters() if not p.requires_grad})
    sourcefiles=[Path(__file__)]+[Path(__file__).with_name(n) for n in ('por_full_utility_v92.py','por_joint_utility_v86.py','por_reparam_v31.py','champion_coadapt_v30.py','champion_shape_v25.py','champion_split_route_v29.py','champion_route_v28.py','champion_posterior_v27.py','champion_all_heads_v26.py','independent_channels_v8.py','train_scaling_v21.py','train_dual8222_heads80.py','fold5_screening_805.py')]
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    cfg=dict(mode=mode,target='POR',screening_fold=5,model_module='por_full_utility_v92',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),added_parameters=0,
        complete_private_model_trainable=True,old_capture_detach_removed=True,other_target_rows_frozen_buffers=True,unused_source_readouts_and_anchor_parameters_frozen=True,trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],
        initialization=model.metadata,initial_state_sha256=initial_hash,reference_scores=refs,
        original_features=206,existing_amplitude_side=63,window_length=256,central_supervised_points=128,
        train_points=len(data['train_raw']),validation_points=len(data['validation']),quality=data['qinfo'],
        epochs=80,patience=18,minimum_epochs=10,batch_size=64,steps_per_epoch=64,seed=20261010,
        body_lr=1e-5,head_lr=3e-5,eta_min=1e-5,weight_decay=.01,clip_norm=2.,
        utility_coefficient=MODES[mode],temperature=.05,utility_chunk=512,
        loss='ownPOR_Q50_BCE_StudentT3NLL_plus_alpha_softaction_regret_divide3',
        official_utility='max(0,1-abs(action-truth)/(.08*(abs(truth)+1e-5))); default0.1',
        decoder='unchanged original256support/800uniformactions+default/singlecoverage',
        preprocessing_checkpoint=str(SOURCE.relative_to(ROOT)),preprocessing_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcefiles},
        scoring_sha256=hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest(),
        target_median=float(master['regression_median'][0]),target_scale=float(master['regression_scale'][0]),
        only_own_target_training_labels=True,old_predictions_as_input=False,optimizer_reset=True,
        complete_five_folds=False,remaining_folds_started=False,torch_version=str(torch.__version__))
    if a.resume:assert json.loads((out/'config.json').read_text(encoding='utf-8'))==cfg
    else:atomic_json(out/'config.json',cfg)
    optimizer=torch.optim.AdamW(model.parameter_groups(),weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,80*64,eta_min=1e-5)
    windows=torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0;elapsed=0.;history=[];timings=[];proof=None;bad=0;updated=None
    if a.resume:
        state=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);assert state['config']==cfg
        model.load_state_dict(state['current']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        epoch=state['epoch'];elapsed=state['seconds'];history=state['history'];timings=state['timings'];proof=state['proof']
        updated=state['updated'];best=state['best'];best_epoch=state['best_epoch'];best_state=state['best_state'];best_pred=state['best_pred'];bad=state['bad']
        torch.set_rng_state(state['rng']['torch']);torch.cuda.set_rng_state_all(state['rng']['cuda']);windows.set_state(state['rng']['windows'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python']);del state
    else:
        pred,score,scores=evaluator(model)
        assert np.array_equal(pred,reference[:,0]),'complete current POR function mismatch'
        assert abs(score-refs['POR'])<1e-9
        best=score;best_epoch=0;best_state=cpu_copy(model.state_dict());best_pred=pred.copy()
        atomic_json(out/'initial.json',dict(target_score=score,reference_replacement_scores=scores,all144665_parent_exact=True,
            initial_prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
        print('INITIAL',json.dumps(dict(target='POR',score=score,parameters=cfg['trainable_parameters'])),flush=True)
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best))
        atomic_array(out/'channel_pred.npy',best_pred);atomic_json(out/'validation_history.json',history)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),scheduler=scheduler.state_dict(),
            epoch=epoch,seconds=elapsed+time.perf_counter()-start,history=history,timings=timings,proof=proof,
            best=best,best_epoch=best_epoch,best_state=best_state,best_pred=best_pred,bad=bad,updated=updated,
            rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
    persist();torch.cuda.reset_peak_memory_stats();core=torch.zeros((1,256),device='cuda',dtype=torch.bool);core[:,64:192]=True
    reason=None
    for ep in range(epoch+1,81):
        if STOP:break
        if epoch>=10 and (out/'stop_requested.json').exists():
            request=json.loads((out/'stop_requested.json').read_text(encoding='utf-8-sig'));reason='evidence_stop: '+request['reason'];break
        tick=time.perf_counter();model.train();losses=nlls=auxs=0.
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper']);relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix];optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            loss,nll,aux=criterion(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])
            aux_norms=None
            if proof is None and step==1:
                gs=torch.autograd.grad(aux,output,retain_graph=True,allow_unused=False)
                aux_norms={name:float(g.detach().float().norm()) for name,g in zip(('gate','density'),gs)}
                assert all(np.isfinite(v) and v>0 for v in aux_norms.values())
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite own POR loss')
            loss.backward()
            if proof is None and step==1:
                norms={}
                for prefix in ('source.members.0.','source.members.1.','source.router.','heads.'+mode+'.readouts.','heads.'+mode+'.route28.','heads.'+mode+'.route29.'):
                    vals=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    norms[prefix]=float(torch.stack(vals).sum().sqrt())
                assert all(np.isfinite(v) and v>0 for v in norms.values())
                proof=dict(gradient_norms=norms,decision_output_gradients=aux_norms,full_private_backbones_updated=True,
                    only_POR_labels=True,first_actual_second_step=True,other_target_rows_are_buffers=True,unused_source_readouts_and_anchors_frozen=True)
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                atomic_json(out/'gradient_scope.json',proof)
            for pg in optimizer.param_groups:torch.nn.utils.clip_grad_norm_(pg['params'],2.,error_if_nonfinite=True)
            optimizer.step();scheduler.step();losses+=float(loss.detach());nlls+=float(nll.detach());auxs+=float(aux.detach())
        epoch=ep;pred,score,scores=evaluator(model)
        if updated is None or score>updated['target_score']:
            updated=dict(epoch=ep,target_score=score,reference_replacement_total=scores['TOTAL'])
            atomic_torch(out/'model_best_updated.pt',dict(state_dict=cpu_copy(model.state_dict()),config=cfg,**updated))
            atomic_array(out/'best_updated_pred.npy',pred)
        if score>best:best=score;best_epoch=ep;best_state=cpu_copy(model.state_dict());best_pred=pred.copy();bad=0
        else:bad+=1
        history.append(dict(epoch=ep,target_score=score,reference_replacement_total=scores['TOTAL'],best=best,best_epoch=best_epoch,bad=bad,
            loss=losses/64,own_nll=nlls/64,joint_decision_loss=auxs/64,lr=[p['lr'] for p in optimizer.param_groups]))
        timings.append(dict(epoch=ep,seconds=time.perf_counter()-tick));persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=mode,target='POR',last_completed_epoch=ep,best=best,best_epoch=best_epoch,bad=bad,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,atlas_current_trial=True))
        print('EPOCH',json.dumps(history[-1]|dict(seconds=timings[-1]['seconds'],peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)),flush=True)
        if ep>=10 and bad>=18:reason='patience18';break
        if ep==80:reason='budget80'
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),explicit_resume_required=True));return
    assert reason is not None
    assert all(torch.equal(v.detach().cpu(),buffers[n]) for n,v in model.named_buffers())
    assert all(torch.equal(p.detach().cpu(),frozen_parameters[n]) for n,p in model.named_parameters() if n in frozen_parameters)
    latest_change=float(sum((v.detach().cpu().double()-initial_body[n].double()).square().sum() for n,v in model.source.state_dict().items() if v.is_floating_point()).sqrt())
    model.load_state_dict(best_state);pred,score,scores=evaluator(model)
    assert np.array_equal(pred,best_pred) and abs(score-best)<1e-9
    selected_change=float(sum((v.detach().cpu().double()-initial_body[n].double()).square().sum() for n,v in model.source.state_dict().items() if v.is_floating_point()).sqrt())
    report=dict(target='POR',mode=mode,screening_fold=5,target_score=score,reference_replacement_scores=scores,delta_reference_target=score-refs['POR'],
        best_epoch=best_epoch,stopped_epoch=epoch,stop_reason=reason,best_after_update=updated,trainable_parameters=cfg['trainable_parameters'],all144665_selected_predictions_restored_exactly=True,
        strict805_qualified=qualifies(scores['TOTAL']),selected_private_state_change_L2=selected_change,last_private_state_change_L2=latest_change,
        gradient_scope=proof,frozen_packed_rows_unchanged=True,frozen_source_readouts_and_anchor_parameters_unchanged=True,checkpoint=str((out/'model_best.pt').relative_to(ROOT)),reference_scores=refs,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,epoch_timing=timings,
        random_network_initialization=False,complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=mode,best=best,best_epoch=best_epoch,atlas_current_trial=False))
    model.close();print('FINAL',json.dumps({k:v for k,v in report.items() if k!='epoch_timing'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if OWNED is not None:
            event=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),automatic_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',event);atomic_json(OWNED/'runtime_status.json',event)
        raise
