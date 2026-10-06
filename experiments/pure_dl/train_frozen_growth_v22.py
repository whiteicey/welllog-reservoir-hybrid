"""Fixed-original-parameter width scaling, two explicit POR trials only.

The v21 model, initialization, cache, objective, decoder and learning rates are
reused. Only the trainable parameter set differs from matching v21 runs.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from datetime import datetime
import argparse, hashlib, json, random, signal, time, traceback
import numpy as np
import torch
from frozen_growth_v22 import make_model, load_data, MODES
from champion_scaling_v21 import REFERENCE, PRIVATE, HEADS
from train_scaling_v21 import Evaluator, channel_objective
from train_dual8222_heads80 import SOURCE, atomic_json, atomic_torch, atomic_array, cpu_copy
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/frozen_growth_v22_20260928'
STOP = False
OWNED = None


def stop_request(*_):
    global STOP
    STOP = True
    print('STOP_REQUESTED: finish current epoch; explicit resume only', flush=True)


def state_signature(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    global OWNED
    sys_stdout = __import__('sys').stdout
    sys_stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=tuple(MODES), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    if (out/'fold_5_report.json').exists():
        raise RuntimeError('Completed trial immutable')
    if (out/'config.json').exists() and not args.resume:
        raise RuntimeError('Existing trial requires explicit resume')
    out.mkdir(parents=True, exist_ok=True)
    OWNED = out
    torch.set_num_threads(3)
    print('ADMISSION', check_slot(min_free_gib=3.), flush=True)
    atomic_json(out/'runtime_status.json', dict(status='running', pid=os.getpid(), mode=args.mode, atlas_current_trial=True))
    signal.signal(signal.SIGINT, stop_request)
    signal.signal(signal.SIGTERM, stop_request)
    started = time.perf_counter()
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    random.seed(20261010)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    data, master = load_data()
    reference = np.load(REFERENCE/'fold_5_pred.npy')
    ref_report = json.loads((REFERENCE/'fold_5_report.json').read_text(encoding='utf-8'))
    model = make_model(args.mode, master).cuda()
    evaluator = Evaluator(data, master, 0, reference)
    trainable_names = {n for n,p in model.named_parameters() if p.requires_grad}
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable_names and all('.extra_' in n for n in trainable_names)
    fixed = {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in trainable_names}
    original_extra = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    initial_signature = state_signature(model.state_dict())
    sources = [Path(__file__), Path(__file__).with_name('frozen_growth_v22.py'),
        Path(__file__).with_name('champion_scaling_v21.py'), Path(__file__).with_name('train_scaling_v21.py'),
        Path(__file__).with_name('independent_channels_v8.py')]
    cfg = dict(target='POR', architecture='dual', mode=args.mode, screening_fold=5, model_module='frozen_growth_v22',
        scaling=model.scaling_manifest, total_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in trainable), frozen_original_parameters=1290490,
        trainable_parameter_names=sorted(trainable_names), update_scope='only .extra_ parameters',
        initial_state_sha256=initial_signature, source_checkpoint=str(SOURCE.relative_to(ROOT)),
        private_checkpoint=str(PRIVATE.relative_to(ROOT)), head_initialization=str(HEADS.relative_to(ROOT)),
        reference_scores=ref_report['scores'], quality=data['qinfo'], data_cache='results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',
        epochs=80, patience=18, minimum_epochs=10, steps_per_epoch=64, batch_size=64, seed=20261010,
        backbone_lr=1e-5, readout_lr=3e-5, eta_min=1e-6, weight_decay=.01, clip_norm=2.,
        objective='v21 channel_objective, POR only, fixedQ50, mean divided by3',
        decoder='unchanged source singlecoverage 256bins/800actions+default/batch32',
        train_eval_behavior='same model.train()/model.eval() as v21, no dropout change',
        frozen_parameters_not_frozen_activations=True, validation_rows=144665, training_rows=578891,
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        torch_version=str(torch.__version__), deterministic_mode='warn_only; no bitwise training repeatability claim',
        matched_full_update_run='outputs/pure_dl_champion_scaling_v21_'+args.mode+'_fold5',
        complete_five_folds=False, remaining_folds_started=False)
    if args.resume:
        assert json.loads((out/'config.json').read_text(encoding='utf-8')) == cfg
    else:
        atomic_json(out/'config.json', cfg)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=.01)
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert optimizer_ids == {id(p) for p in trainable}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80*64, eta_min=1e-6)
    windows = torch.Generator(device='cuda').manual_seed(20261010)
    epoch, elapsed, proof = 0, 0., None
    best_updated, updated_state, updated_pred = None, None, None
    if args.resume:
        s = torch.load(out/'last_state.pt', map_location='cpu', weights_only=False)
        assert s['config'] == cfg
        model.load_state_dict(s['current'], strict=True)
        optimizer.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler'])
        epoch, elapsed, proof = s['epoch'], s['seconds'], s['proof']
        best, best_epoch, best_state, best_pred = s['best'], s['best_epoch'], s['best_state'], s['best_pred']
        best_updated, updated_state, updated_pred = s['best_updated'], s['updated_state'], s['updated_pred']
        initial, history, bad = s['initial'], s['history'], s['bad']
        torch.set_rng_state(s['rng']['torch']); torch.cuda.set_rng_state_all(s['rng']['cuda'])
        windows.set_state(s['rng']['windows']); np.random.set_state(s['rng']['numpy']); random.setstate(s['rng']['python'])
        del s
    else:
        pred, score, total = evaluator(model)
        assert np.array_equal(pred, reference[:,0]), 'Initial champion function changed'
        initial = dict(POR=score, TOTAL=total, all144665initial_predictions_exact=True, initial_state_sha256=initial_signature)
        atomic_json(out/'initial_verification.json', initial)
        best, best_epoch, best_state, best_pred = score, 0, cpu_copy(model.state_dict()), pred.copy()
        history, bad = [], 0
        print('INITIAL', json.dumps(initial), flush=True)

    def persist():
        atomic_torch(out/'model_best.pt', dict(state_dict=best_state, config=cfg, best_epoch=best_epoch, target_score=best))
        atomic_array(out/'channel_pred.npy', best_pred)
        atomic_json(out/'validation_history.json', history)
        if updated_state is not None:
            atomic_torch(out/'model_best_updated.pt', dict(state_dict=updated_state, config=cfg, **best_updated))
            atomic_array(out/'best_updated_pred.npy', updated_pred)
        atomic_torch(out/'last_state.pt', dict(config=cfg, current=cpu_copy(model.state_dict()), optimizer=cpu_copy(optimizer.state_dict()),
            scheduler=scheduler.state_dict(), epoch=epoch, seconds=elapsed+time.perf_counter()-started, proof=proof,
            best=best, best_epoch=best_epoch, best_state=best_state, best_pred=best_pred, best_updated=best_updated,
            updated_state=updated_state, updated_pred=updated_pred, initial=initial, history=history, bad=bad,
            rng=dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(), windows=windows.get_state(),
                numpy=np.random.get_state(), python=random.getstate())))

    persist(); torch.cuda.reset_peak_memory_stats()
    core = torch.zeros((1,256), device='cuda', dtype=torch.bool); core[:,64:192] = True
    for current in range(epoch+1,81):
        if STOP or (epoch>=10 and bad>=18): break
        tick=time.perf_counter(); model.train(); losses=np.zeros(3)
        for step in range(64):
            anchors=data['train_raw'][torch.randint(len(data['train_raw']),(64,),device='cuda',generator=windows)]
            ix,valid=window_indices(anchors,data['lower'],data['upper'])
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask=valid&core&data['labeled'][ix]; optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            loss,bce,nll=channel_objective(output,data['labels'][ix],data['default'][ix],mask,data['quality'][ix],0)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            if proof is None and step==1:
                assert all(p.grad is None for n,p in model.named_parameters() if n not in trainable_names)
                def grad_norm(part):
                    values=[p.grad.detach().float().square().sum() for n,p in model.named_parameters() if part(n) and p.grad is not None]
                    return float(torch.stack(values).sum().sqrt()) if values else 0.
                body=grad_norm(lambda n: '.extra_' in n and '.main_head.' not in n)
                head=grad_norm(lambda n: '.extra_' in n and '.main_head.' in n)
                assert body>0 and head>0 and np.isfinite(body+head)
                proof=dict(new_backbone_gradient_norm=body,new_head_gradient_norm=head,all_old_parameter_gradients_none=True,
                    new_input_projection_gradient_norm=grad_norm(lambda n: '.extra_expand.' in n or '.extra_reduce.' in n),
                    only_POR_labels=True,second_actual_training_step=True)
                atomic_json(out/'gradient_scope.json',proof)
            torch.nn.utils.clip_grad_norm_(trainable,2.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); losses+=np.array([float(loss.detach()),float(bce.detach()),float(nll.detach())])
        epoch=current; pred,score,total=evaluator(model)
        if score>best:
            best,best_epoch,best_state,best_pred,bad=score,epoch,cpu_copy(model.state_dict()),pred.copy(),0
        else: bad+=1
        if best_updated is None or score>best_updated['target_score']:
            best_updated=dict(epoch=epoch,target_score=score,reference_replacement_total=total)
            updated_state=cpu_copy(model.state_dict()); updated_pred=pred.copy()
        row=dict(epoch=epoch,target_score=score,reference_replacement_total=total,best_target_score=best,best_epoch=best_epoch,
            bad_epochs=bad,training_loss=losses[0]/64,training_bce=losses[1]/64,training_nll=losses[2]/64,
            learning_rates=[g['lr'] for g in optimizer.param_groups],seconds=time.perf_counter()-tick)
        history.append(row); persist()
        atomic_json(out/'runtime_status.json',dict(status='running',pid=os.getpid(),mode=args.mode,last_completed_epoch=epoch,
            best_target_score=best,best_epoch=best_epoch,reference_replacement_total=ref_report['scores']['TOTAL']+.3*(best-ref_report['scores']['POR']),atlas_current_trial=True))
        print('EPOCH',json.dumps(row),flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json',dict(status='stopped_by_user',pid=os.getpid(),mode=args.mode,resume_available=True,auto_resume=False)); return
    # Check frozen weights before restoring the selected state, including a possible epoch0 selection.
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    last_extra_change=float(sum((p.detach().cpu().double()-original_extra[n].double()).square().sum()
        for n,p in model.named_parameters() if n in original_extra).sqrt())
    assert last_extra_change>0
    model.load_state_dict(best_state,strict=True)
    pred,score,total=evaluator(model)
    assert np.array_equal(pred,best_pred)
    assert all(torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in fixed.items())
    selected_extra_change=float(sum((p.detach().cpu().double()-original_extra[n].double()).square().sum()
        for n,p in model.named_parameters() if n in original_extra).sqrt())
    report=dict(fold=5,screening_fold=5,target='POR',mode=args.mode,scores={'POR':score},target_score=score,
        reference_replacement_total=total,delta_reference_POR=score-ref_report['scores']['POR'],
        best_epoch=best_epoch,stopped_epoch=epoch,best_after_update=best_updated,initial=initial,
        total_parameters=cfg['total_parameters'],trainable_parameters=cfg['trainable_parameters'],frozen_original_parameters=1290490,
        all_old_state_unchanged_before_and_after_restore=True,gradient_scope=proof,
        last_extra_parameter_change_L2=last_extra_change,selected_extra_parameter_change_L2=selected_extra_change,
        all144665selected_predictions_restored_exactly=True,TOTAL_scope='fixed other columns; separate recorded final assembly',
        seconds=elapsed+time.perf_counter()-started,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(out/'fold_5_report.json',report)
    atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),mode=args.mode,atlas_current_trial=False))
    print('FINAL',json.dumps(report),flush=True)


if __name__=='__main__':
    try: main()
    except BaseException as exc:
        if OWNED is not None:
            error=dict(status='failed',pid=os.getpid(),error=repr(exc),traceback=traceback.format_exc(),auto_resume=False,atlas_current_trial=False)
            atomic_json(OWNED/'failure.json',error); atomic_json(OWNED/'runtime_status.json',error)
        raise
