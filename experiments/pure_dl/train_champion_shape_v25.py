"""Two small location-only candidates on the frozen, verified champion."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, random, signal, sys, time, traceback
import numpy as np
import torch
from champion_shape_v25 import ROOT, STUDY, FEATURES, MODES, REFERENCE, PRIVATE, HEADS
from champion_shape_v25 import load_data, make_model, state_signature
from train_scaling_v21 import Evaluator, channel_objective
from train_dual8222_heads80 import atomic_json, atomic_array, atomic_torch, cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

STOP = False
OWNED = None


def stop_request(*_):
    global STOP
    STOP = True
    print('STOP_REQUESTED: complete current epoch and save; explicit resume only', flush=True)


@torch.no_grad()
def factor_check(model, data):
    anchors = data['train_raw'][:2]
    ix, valid = window_indices(anchors, data['lower'], data['upper'])
    relative = torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
    model.eval()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        current = model(data['x'][ix], relative, valid, absolute_depth=data['absolute'][ix])
        for m in model.members: m.main_head.residual.enabled = False
        try:
            original = model(data['x'][ix], relative, valid, absolute_depth=data['absolute'][ix])
        finally:
            for m in model.members: m.main_head.residual.enabled = True
    assert torch.equal(current[0], original[0])
    assert torch.equal(current[1][...,0,:], original[1][...,0,:])
    assert torch.equal(current[1][...,2,:], original[1][...,2,:])
    delta = (current[1][...,1,:]-original[1][...,1,:]).abs()[valid]
    return dict(training_windows=2, gate_logits_exactly_unchanged=True,
        mixture_logits_exactly_unchanged=True, raw_scales_exactly_unchanged=True,
        selected_location_mean_absolute_change=float(delta.mean()),
        selected_location_max_absolute_change=float(delta.max()),
        scope='inference factor identity, no extra scoring candidate')


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if (out/'fold_5_report.json').exists(): raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not args.resume: raise RuntimeError('Existing run: explicit resume required')
    out.mkdir(parents=True, exist_ok=True); OWNED = out
    torch.set_num_threads(3)
    print('GPU_ADMISSION', check_slot(min_free_gib=3.), flush=True)
    atomic_json(out/'runtime_status.json', dict(status='running', mode=args.mode, pid=os.getpid(), atlas_current_trial=True))
    signal.signal(signal.SIGINT, stop_request); signal.signal(signal.SIGTERM, stop_request)
    start = time.perf_counter()
    torch.manual_seed(20261010); np.random.seed(20261010); random.seed(20261010)
    torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.benchmark = False
    data, master = load_data(args.mode)
    model = make_model(args.mode, master).cuda()
    ref = json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))['scores']
    reference = np.load(REFERENCE/'fold_5_pred.npy')
    evaluator = Evaluator(data, master, 0, reference)
    trained_names = {n for n,p in model.named_parameters() if p.requires_grad}
    parameters = [p for p in model.parameters() if p.requires_grad]
    fixed = {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in trained_names}
    original_new = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    paths = [Path(__file__)]+[Path(__file__).with_name(n) for n in
        ('champion_shape_v25.py','independent_channels_v8.py','champion_scaling_v21.py','train_scaling_v21.py','dual8222_location_readout_v5.py')]
    cfg = dict(mode=args.mode, target='POR', model_module='champion_shape_v25', screening_fold=5,
        private_checkpoint=str(PRIVATE.relative_to(ROOT)), head_initialization=str(HEADS.relative_to(ROOT)),
        reference_scores=ref, frozen_original_parameters=1290490, trainable_parameters=sum(p.numel() for p in parameters),
        total_parameters=sum(p.numel() for p in model.parameters()), trainable_names=sorted(trained_names),
        initial_state_sha256=state_signature(model.state_dict()), input_view=args.mode,
        main_features=206, side_features=63, head_input=191, head_hidden=32, outputs_per_expert=8,
        bound_fraction=.25, old_function_frozen=True, update_scope='new component location residual only',
        epochs=80, patience=18, minimum_epochs=10, steps_per_epoch=64, batch_size=64, seed=20261010,
        learning_rate=.0003, eta_min=.000001, weight_decay=.01, clip_norm=2., quality=data['qinfo'],
        objective='unchanged champion POR final-mixture likelihood /3, fixedQ50',
        decoder='source singlecoverage256bins/800actions+default/batch32',
        feature_manifest='results/deep_learning/feature_structure_v20_20260928/feature_manifest.json',
        feature_manifest_sha256=hashlib.sha256((FEATURES/'feature_manifest.json').read_bytes()).hexdigest(),
        preprocessing='champion exact cache; side global constants only64trainingwells',
        training_rows=578891, validation_rows=144665,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        torch_version=str(torch.__version__), deterministic_mode='warn_only; no bitwise repeat-training claim',
        complete_five_folds=False, remaining_folds_started=False)
    assert cfg['trainable_parameters']==12816 and cfg['total_parameters']==1303306
    if args.resume: assert json.loads((out/'config.json').read_text(encoding='utf-8')) == cfg
    else: atomic_json(out/'config.json', cfg)
    optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80*64, eta_min=1e-6)
    windows = torch.Generator(device='cuda').manual_seed(20261010)
    epoch, elapsed, proof = 0, 0., None
    updated_record, updated_state, updated_pred = None, None, None
    if args.resume:
        state = torch.load(out/'last_state.pt', map_location='cpu', weights_only=False)
        assert state['config'] == cfg
        model.load_state_dict(state['current'], strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        epoch, elapsed, proof = state['epoch'], state['seconds'], state['proof']
        best, best_epoch, best_state, best_pred = state['best'], state['best_epoch'], state['best_state'], state['best_pred']
        initial, history, bad = state['initial'], state['history'], state['bad']
        updated_record, updated_state, updated_pred = state['updated_record'], state['updated_state'], state['updated_pred']
        torch.set_rng_state(state['rng']['torch']); torch.cuda.set_rng_state_all(state['rng']['cuda'])
        windows.set_state(state['rng']['windows']); np.random.set_state(state['rng']['numpy']); random.setstate(state['rng']['python'])
        del state
    else:
        pred, score, total = evaluator(model)
        assert np.array_equal(pred, reference[:,0]), 'Initial predictor differs from champion'
        initial = dict(POR=score, TOTAL=total, all144665_initial_predictions_exact=True)
        atomic_json(out/'initial_verification.json', initial)
        best, best_epoch, best_state, best_pred = score, 0, cpu_copy(model.state_dict()), pred.copy()
        history, bad = [], 0
        print('INITIAL', json.dumps(initial), flush=True)

    def persist():
        atomic_torch(out/'model_best.pt', dict(state_dict=best_state, config=cfg, best_epoch=best_epoch, target_score=best))
        atomic_array(out/'channel_pred.npy', best_pred)
        atomic_json(out/'validation_history.json', history)
        if updated_record is not None:
            atomic_torch(out/'model_best_updated.pt', dict(state_dict=updated_state, config=cfg, **updated_record))
            atomic_array(out/'best_updated_pred.npy', updated_pred)
        atomic_torch(out/'last_state.pt', dict(config=cfg, current=cpu_copy(model.state_dict()),
            optimizer=cpu_copy(optimizer.state_dict()), scheduler=scheduler.state_dict(), epoch=epoch,
            seconds=elapsed+time.perf_counter()-start, proof=proof, best=best, best_epoch=best_epoch,
            best_state=best_state, best_pred=best_pred, initial=initial, history=history, bad=bad,
            updated_record=updated_record, updated_state=updated_state, updated_pred=updated_pred,
            rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), windows=windows.get_state(),
                numpy=np.random.get_state(), python=random.getstate())))

    persist(); torch.cuda.reset_peak_memory_stats()
    core = torch.zeros((1,256), device='cuda', dtype=torch.bool); core[:,64:192] = True
    for ep in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18): break
        tick=time.perf_counter(); model.train(); loss_sum=0.
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            loss,_,_=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],0)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite objective')
            loss.backward()
            if proof is None and step==1:
                assert all(p.grad is None for n,p in model.named_parameters() if n not in trained_names)
                def norm(part):
                    values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if part in n and p.grad is not None]
                    return float(torch.stack(values).sum().sqrt()) if values else 0.
                proof=dict(delta_input_gradient_norm=norm('.delta_in.'), delta_output_gradient_norm=norm('.delta_out.'),
                    all_old_parameter_gradients_none=True, POR_only_supervision=True, second_actual_step=True)
                assert proof['delta_input_gradient_norm']>0 and proof['delta_output_gradient_norm']>0
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(parameters,2.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); loss_sum+=float(loss.detach())
        epoch=ep; pred,score,total=evaluator(model)
        if score>best: best,best_epoch,best_state,best_pred,bad=score,epoch,cpu_copy(model.state_dict()),pred.copy(),0
        else: bad+=1
        if updated_record is None or score>updated_record['target_score']:
            updated_record=dict(epoch=epoch,target_score=score,reference_replacement_total=total)
            updated_state=cpu_copy(model.state_dict()); updated_pred=pred.copy()
        row=dict(epoch=epoch,target_score=score,reference_replacement_total=total,best_target_score=best,
            best_epoch=best_epoch,bad_epochs=bad,training_loss=loss_sum/64,learning_rate=optimizer.param_groups[0]['lr'],seconds=time.perf_counter()-tick)
        history.append(row); persist()
        atomic_json(out/'runtime_status.json',dict(status='running',mode=args.mode,pid=os.getpid(),last_completed_epoch=epoch,
            best_target_score=best,best_epoch=best_epoch,reference_replacement_total=ref['TOTAL']+.3*(best-ref['POR']),atlas_current_trial=True))
        print('EPOCH',json.dumps(row),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',mode=args.mode,pid=os.getpid(),auto_resume=False,resume_available=True)); return
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    last_change=float(sum((p.detach().cpu().double()-original_new[n].double()).square().sum()
        for n,p in model.named_parameters() if n in trained_names).sqrt())
    assert last_change>0
    model.load_state_dict(best_state,strict=True)
    pred,score,total=evaluator(model)
    assert np.array_equal(pred,best_pred)
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    factors=factor_check(model,data)
    report=dict(fold=5,screening_fold=5,target='POR',mode=args.mode,scores={'POR':score},target_score=score,
        reference_replacement_total=total,delta_reference_TOTAL=total-ref['TOTAL'],best_epoch=best_epoch,stopped_epoch=epoch,
        best_after_update=updated_record,initial=initial,trainable_parameters=12816,total_parameters=1303306,
        all_old_state_unchanged_before_and_after_restore=True,all144665_selected_predictions_restored_exactly=True,
        gradient_scope=proof,factor_identity=factors,last_new_parameter_change_L2=last_change,
        seconds=elapsed+time.perf_counter()-start,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',mode=args.mode,pid=os.getpid(),atlas_current_trial=False))
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if OWNED is not None:
            value=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',value); atomic_json(OWNED/'runtime_status.json',value)
        raise
