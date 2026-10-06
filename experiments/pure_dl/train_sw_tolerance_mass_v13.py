"""Two explicitly launched private-SW loss experiments on the strong v8 dual.

Same architecture, source, random windows, AdamW, decoder, and fixed fifth fold.
No other-target model is trained. Interrupted work resumes only with --resume.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, random, signal, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from independent_channels_v8 import Network
from train_independent_channels_v8 import Evaluator, REFERENCE
from train_dual8222_heads80 import prepare_data, atomic_json, atomic_torch, atomic_array, cpu_copy, SOURCE
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
from por_t3_patch_loss import probability
from sw_tolerance_mass_v13 import ToleranceObjective, MODES, AUXILIARY_WEIGHT, verify_first_training_batch

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT/'outputs/pure_dl_independent_v8_SW_dual80_fold5/model_best.pt'
OWNED = None
STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    print('STOP_REQUESTED: save epoch-boundary state; no automatic restart', flush=True)


def main():
    global OWNED
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=MODES, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(); out = args.out.resolve()
    if (out/'fold_5_report.json').exists():
        raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not args.resume:
        raise RuntimeError('Existing configuration requires explicit --resume')
    out.mkdir(parents=True, exist_ok=True); OWNED = out
    torch.set_num_threads(3)
    print('GPU_ADMISSION', check_slot(min_free_gib=3.0), flush=True)
    atomic_json(out/'runtime_status.json', dict(status='running', pid=os.getpid(), mode=args.mode, atlas_current_trial=True))
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    begin = time.perf_counter()
    random.seed(20261010); np.random.seed(20261010); torch.manual_seed(20261010)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    master = torch.load(SOURCE, map_location='cpu', weights_only=False)
    previous = torch.load(REFERENCE/'readout_checkpoint.pt', map_location='cpu', weights_only=False)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False)
    data = prepare_data(master)
    reference = np.load(REFERENCE/'fold_5_pred.npy')
    model = Network(master, previous, 'SW', 'dual').cuda()
    model.load_state_dict(parent['state_dict'], strict=True)
    evaluator = Evaluator(data, master, 2, reference)
    objective = ToleranceObjective(master, args.mode).cuda()
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n not in names}
    initial_body = {n: p.detach().cpu().clone() for n, p in model.backbone_parameters()}
    cfg = dict(mode=args.mode, target='SW', architecture='dual', model_module='independent_channels_v8',
        source_checkpoint=str(PARENT.relative_to(ROOT)), source_best_epoch=parent['best_epoch'],
        source_target_score=parent['target_score'], source_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        private_backbone_branches=2, other_target_models_trainable=False, pretrained_network_used=True,
        screening_fold=5, seed=20261010, epochs=80, steps_per_epoch=64, batch_size=64, patience=18, minimum_epochs=10,
        backbone_lr=1e-5, readout_lr=3e-5, eta_min=1e-6, weight_decay=0.01, clip_norm=2.0,
        optimizer_reset=True, auxiliary_weight=objective.alpha, coverage_floor=1e-8,
        objective='proper hurdle Gaussian mixture NLL + alpha*(-log expected triangular tolerance coverage);both divided by3',
        coverage_support='same original SW256 physical centers and differentiable Gaussian bin masses plus default atom',
        proper_likelihood_retained=True, exact_argmax_loss_claimed=False, quality=data['qinfo'],
        training_rows=578891, validation_rows=144665, original_nine_curves=True, no_test_labels=True,
        decoder='unchanged v8 single coverage256bins800actions+default,inferencebatch32',
        remaining_folds_started=False, deterministic_mode='warn_only for unsupported CUDA interpolation backward',
        torch_version=str(torch.__version__), source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__), Path(__file__).with_name('sw_tolerance_mass_v13.py')]})
    if args.resume:
        assert json.loads((out/'config.json').read_text(encoding='utf-8')) == cfg
    else:
        atomic_json(out/'config.json', cfg)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80*64, eta_min=1e-6)
    windows = torch.Generator(device='cuda').manual_seed(20261010)
    epoch=0; elapsed=0.0; proof=None
    if args.resume:
        saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
        assert saved['config']==cfg
        model.load_state_dict(saved['current_state'],strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        epoch=saved['epoch']; best_epoch=saved['best_epoch']; best_score=saved['best_score']
        best_state=saved['best_state']; best_pred=saved['best_prediction']; bad=saved['bad']; history=saved['history']
        initial=saved['initial']; elapsed=saved['seconds']; proof=saved['gradient_proof']
        torch.set_rng_state(saved['rng']['torch']); torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        windows.set_state(saved['rng']['windows']); np.random.set_state(saved['rng']['numpy']); random.setstate(saved['rng']['python'])
        del saved
    else:
        pred, score, total=evaluator(model)
        assert np.array_equal(pred, reference[:,2])
        initial=dict(target_score=score, reference_replacement_total=total, all144665source_predictions_exact=True,
                     source='strong private SW dual; not the weaker v11 TCN')
        atomic_json(out/'initial_verification.json', initial)
        best_epoch=0; best_score=score; best_pred=pred.copy(); best_state=cpu_copy(model.state_dict()); bad=0; history=[]
        print('INITIAL', json.dumps(initial), flush=True)
    def persist():
        atomic_torch(out/'model_best.pt',dict(state_dict=best_state,config=cfg,best_epoch=best_epoch,target_score=best_score))
        atomic_array(out/'channel_pred.npy',best_pred)
        atomic_torch(out/'last_state.pt',dict(config=cfg,current_state=cpu_copy(model.state_dict()),optimizer=cpu_copy(optimizer.state_dict()),
          scheduler=scheduler.state_dict(),epoch=epoch,best_epoch=best_epoch,best_score=best_score,best_state=best_state,best_prediction=best_pred,
          bad=bad,history=history,initial=initial,gradient_proof=proof,seconds=elapsed+time.perf_counter()-begin,
          rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),windows=windows.get_state(),numpy=np.random.get_state(),python=random.getstate())))
        atomic_json(out/'validation_history.json',history)
    persist(); torch.cuda.reset_peak_memory_stats()
    core=torch.zeros(1,256,device='cuda',dtype=torch.bool); core[:,64:192]=True
    for current in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18): break
        model.train(); accum=np.zeros(5); started=time.perf_counter()
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            loss,stats,aux,p=objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix])
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite training loss')
            if proof is None:
                proof=verify_first_training_batch(aux,output,p,evaluator.edges)
                rr=output[1][mask][:,0].float().detach()
                expected=probability(2,rr[:,0].softmax(-1),rr[:,1],F.softplus(rr[:,2])+0.025,evaluator.edges)
                error=float((p.detach()-expected).abs().max()); proof['CDF_forward_max_error_vs_original']=error
                if error>1e-6: raise RuntimeError('Differentiable bin mass differs from evaluator')
            loss.backward()
            if current==1 and step==0:
                norm=float(torch.stack([p.grad.detach().float().square().sum() for _,p in model.backbone_parameters() if p.grad is not None]).sum().sqrt())
                if not np.isfinite(norm) or norm<=0: raise RuntimeError('Backbone gradient missing/nonfinite')
                proof['private_backbone_gradient_norm']=norm; atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.0,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); accum+=stats.detach().cpu().numpy()
        epoch=current; pred,score,total=evaluator(model)
        if score>best_score:
            best_score=score; best_pred=pred.copy(); best_state=cpu_copy(model.state_dict()); best_epoch=epoch; bad=0
        else: bad+=1
        row=dict(epoch=epoch,target_score=score,best_target_score=best_score,best_epoch=best_epoch,bad_epochs=bad,
          reference_replacement_total=total,proper_loss=accum[0]/64,training_bce=accum[1]/64,training_nll=accum[2]/64,
          log_coverage_loss=accum[3]/64,mean_training_coverage=accum[4]/64,seconds=time.perf_counter()-started,
          learning_rates=[g['lr'] for g in optimizer.param_groups])
        history.append(row); persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=args.mode,last_completed_epoch=epoch,
          best_epoch=best_epoch,best_target_score=best_score,atlas_current_trial=True))
        print('EPOCH',json.dumps(row),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),mode=args.mode,epoch=epoch,resume_available=True,auto_resume=False)); return
    last_change=float(sum((p.detach().cpu().double()-initial_body[n].double()).square().sum() for n,p in model.backbone_parameters()).sqrt())
    model.load_state_dict(best_state,strict=True); pred,score,total=evaluator(model)
    assert np.array_equal(pred,best_pred)
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in frozen.items())
    post=max(history,key=lambda row:row['target_score']) if history else None
    report=dict(fold=5,screening_fold=5,target='SW',architecture='dual',mode=args.mode,scores={'SW':score},target_score=score,
      initial_target_score=initial['target_score'],delta_initial=score-initial['target_score'],reference_replacement_total=total,
      TOTAL_scope='fixed previously verified POR/PERM plus current SW replacement diagnostic; no new complete OOF',
      best_epoch=best_epoch,stopped_epoch=epoch,best_after_update=post,trainable_parameters=cfg['trainable_parameters'],
      auxiliary_weight=objective.alpha,gradient_scope=proof,last_backbone_change_L2=last_change,
      all144665target_predictions_restored_exactly=True,frozen_packed_rows_unchanged=True,
      other_target_models_updated=False,complete_five_folds=False,remaining_folds_started=False,
      seconds=elapsed+time.perf_counter()-begin,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
      finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=args.mode,atlas_current_trial=False))
    print('FINAL_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if OWNED is not None:
            failure=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',failure); atomic_json(OWNED/'runtime_status.json',failure)
        raise
