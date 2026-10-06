"""Matched warm-start SW likelihood versus joint-decision objective;fixed architecture."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, random, signal, sys, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from sw_joint_decision_v12 import Network, MODES, DecisionObjective, verify_loss_paths
from train_dual8222_heads80 import prepare_data, SOURCE, atomic_json, atomic_torch, atomic_array, cpu_copy
from train_independent_sw_v9 import Evaluator, FIXED
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
import train_invariant as base

ROOT = Path(__file__).resolve().parents[2]
WARM = ROOT/'outputs/pure_dl_independent_sw_v11_shared80_fold5'
STOP = False
OWNED = None


def request_stop(_sig, _frame):
    global STOP
    STOP = True
    print('STOP_REQUESTED: save epoch-boundary state, no automatic restart', flush=True)


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=MODES, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    args.architecture = 'shared'
    out = args.out.resolve()
    if (out/'fold_5_report.json').exists():
        raise RuntimeError('Completed experiment is immutable')
    if (out/'config.json').exists() and not args.resume:
        raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(parents=True, exist_ok=True)
    OWNED = out
    torch.set_num_threads(3)
    print('GPU_ADMISSION', check_slot(min_free_gib=3.), flush=True)
    atomic_json(out/'runtime_status.json', {'status': 'running', 'pid': os.getpid(),
                'architecture': args.architecture, 'mode':args.mode, 'atlas_current_trial': True})
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    started = time.perf_counter()
    torch.manual_seed(20261010); np.random.seed(20261010); random.seed(20261010)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    # Only preprocessing, support grid and data contracts are reused from SOURCE.
    master = torch.load(SOURCE, map_location='cpu', weights_only=False)
    reference = np.load(FIXED)
    data = prepare_data(master)
    model = Network(args.architecture, master['centers'][2],
                    master['regression_median'][2], master['regression_scale'][2]).cuda()
    warm = torch.load(WARM/'model_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(warm['state_dict'], strict=True)
    initialization = {'kind':'warmstart_v11shared_best13', 'source':str((WARM/'model_best.pt').relative_to(ROOT)), 'source_target_score':float(warm['target_score']), 'source_sha256':hashlib.sha256((WARM/'model_best.pt').read_bytes()).hexdigest()}
    del warm
    evaluator = Evaluator(data, master, 2, reference)
    criterion = DecisionObjective(evaluator.actions, evaluator.utilities, model.target_median, model.target_scale, args.mode).cuda()
    reference_scores = base.metric(data['truth'][data['validation']], reference)
    initial_body = {n: p.detach().cpu().clone() for n,p in model.named_parameters()
                    if not n.startswith('output.')}
    cfg = {'target': 'SW', 'architecture': args.architecture, 'mode':args.mode,
           'model_module': 'independent_sw_separation_v11', 'screening_fold': 5,
           'pretrained_network_used': True, 'teacher_or_prior_prediction_used': False, 'warmstart_not_resume':True,
           'shared_learnable_parameters_across_targets': False,
           'input_features': 206, 'relative_depth_features': 1, 'hidden_width': 48,
           'sequence_window': 256, 'supervised_center_points': 128,
           'structure': 'two distinct48dimTCNs;four gated depthwise5 blocks each;dilations1/2/4/8',
           'within_SW_representation': 'both heads see mean(E0,E1)' if args.architecture=='shared' else 'gate seesE0; density seesE1',
           'parameter_count_matched': 92273,
           'readout': 'one learned default logit plus256 categorical continuous-density logits',
           'trainable_parameters': sum(p.numel() for p in model.parameters()),
           'initialization': initialization, 'reference_scores': reference_scores,
           'epochs': 80, 'steps_per_epoch': 64, 'batch_size': 64, 'seed': 20261010,
           'patience': 18, 'minimum_epochs': 10, 'lr': 1e-4, 'eta_min': 1e-5, 'utility_weight':criterion.alpha, 'temperature':criterion.temperature,
           'weight_decay': .01, 'clip_norm': 2., 'dropout': 0.,
           'loss':'BCE+numericsoftCE + alpha*jointsoftactionregret;alltraininglabels;fixedQ;divide3', 'actual_action_count':len(evaluator.actions),
           'no_prior_KL': True, 'no_logit_correction_bound': True,
           'quality': data['qinfo'], 'validation_rows': 144665, 'training_rows': 578891,
           'decoder': 'original256support/800actions+default/singlecoverage',
           'preprocessing_source': str(SOURCE.relative_to(ROOT)),
           'training_bitwise_reproducibility_claimed': False, 'deterministic_mode': 'warn_only',
           'compute': 'botharms_bfloat16AMP', 'gradient_clipping': 'separatelyencoder0+gate_and_encoder1+density',
           'torch_version': str(torch.__version__), 'remaining_folds_started': False,
           'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__), Path(__file__).with_name('independent_sw_separation_v11.py'), Path(__file__).with_name('sw_joint_decision_v12.py')]}}
    if args.resume:
        assert json.loads((out/'config.json').read_text()) == cfg
    else:
        atomic_json(out/'config.json', cfg)
    assert sum(p.numel() for p in model.parameters()) == 92273
    clip_groups = model.clipping_groups()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80*64, eta_min=1e-5)
    windows = torch.Generator(device='cuda').manual_seed(20261010)
    epoch, elapsed, proof = 0, 0., None
    if args.resume:
        saved = torch.load(out/'last_state.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == cfg
        model.load_state_dict(saved['current_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        epoch, elapsed = saved['epoch'], saved['seconds']
        best_state, best_pred = saved['best_state'], saved['best_prediction']
        best_score, best_epoch, bad = saved['best_score'], saved['best_epoch'], saved['bad']
        history, initial, proof = saved['history'], saved['initial'], saved['gradient_proof']
        torch.set_rng_state(saved['rng']['torch']); torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        windows.set_state(saved['rng']['windows'])
        np.random.set_state(saved['rng']['numpy']); random.setstate(saved['rng']['python'])
        del saved
    else:
        pred, score, total = evaluator(model)
        initial = {'target_score': score, 'reference_replacement_total': total,
                   'different_target_elements_from_old_model': int(np.count_nonzero(pred != reference[:,2])),
                   'scope':'v11shared checkpoint;fresh matched optimizer;not coldstart', 'warm_predictions_exact':bool(np.array_equal(pred,np.load(WARM/'channel_pred.npy')))}
        atomic_json(out/'initial_verification.json', initial)
        assert initial['warm_predictions_exact'] and abs(score-initialization['source_target_score'])<1e-10
        best_state, best_pred = cpu_copy(model.state_dict()), pred.copy()
        best_score, best_epoch, bad, history = score, 0, 0, []
        print('INITIAL', json.dumps(initial), flush=True)
    torch.cuda.reset_peak_memory_stats()
    def persist():
        atomic_torch(out/'model_best.pt', {'state_dict': best_state, 'config': cfg,
                     'best_epoch': best_epoch, 'target_score': best_score,
                     'target_median': model.target_median, 'target_scale': model.target_scale})
        atomic_array(out/'channel_pred.npy', best_pred)
        atomic_json(out/'validation_history.json', history)
        atomic_torch(out/'last_state.pt', {'config': cfg, 'current_state': cpu_copy(model.state_dict()),
                     'optimizer': cpu_copy(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
                     'epoch': epoch, 'seconds': elapsed+time.perf_counter()-started,
                     'best_state': best_state, 'best_prediction': best_pred,
                     'best_score': best_score, 'best_epoch': best_epoch, 'bad': bad,
                     'initial': initial, 'history': history, 'gradient_proof': proof,
                     'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                             'windows': windows.get_state(), 'numpy': np.random.get_state(),
                             'python': random.getstate()}})
    persist()
    core = torch.zeros((1,256), dtype=torch.bool, device='cuda'); core[:,64:192] = True
    for current in range(epoch+1, 81):
        if STOP or (epoch >= 10 and bad >= 18):
            break
        tick = time.perf_counter(); model.train(); stats = np.zeros(6)
        for step in range(64):
            anchors = data['train_raw'][torch.randint(len(data['train_raw']), (64,),
                                      device='cuda', generator=windows)]
            ix, valid = window_indices(anchors, data['lower'], data['upper'])
            relative = torch.tanh((data['depth'][ix]-data['depth'][anchors,None])/10)
            mask = valid & core & data['labeled'][ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(data['x'][ix], relative, valid, absolute_depth=data['absolute'][ix])
            loss, terms, decision_loss = criterion(model, output, data['labels'][ix], data['default'][ix],
                                      mask, data['quality'][ix])
            if current == 1 and step == 0:
                loss_paths = verify_loss_paths(decision_loss, output)
                atomic_json(out/'decision_gradient_scope.json', loss_paths)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if current == 1 and step == 1:
                grads = [(n,p.grad) for n,p in model.named_parameters()
                         if n in initial_body and p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for _,g in grads)
                norm = float(torch.stack([g.float().square().sum() for _,g in grads]).sum().sqrt())
                assert norm > 0
                proof = {'new_private_backbone_gradient_norm': norm, 'backbone_tensors_with_gradient': len(grads),
                         'only_SW_loss_used': True, 'old_model_or_other_target_parameters_present': False,
                         'warm_start_head_is_nonzero': True}
                atomic_json(out/'gradient_scope.json', proof)
            for parameters in clip_groups:
                torch.nn.utils.clip_grad_norm_(parameters, 2.)
            optimizer.step(); scheduler.step()
            stats += np.r_[float(loss.detach()), terms.cpu().numpy().astype(np.float64)]
        epoch = current
        pred, score, total = evaluator(model)
        if score > best_score:
            best_state, best_pred, best_score = cpu_copy(model.state_dict()), pred.copy(), score
            best_epoch, bad = epoch, 0
        else:
            bad += 1
        row = {'epoch': epoch, 'target_score': score, 'reference_replacement_total': total,
               'best_target_score': best_score, 'best_epoch': best_epoch, 'bad_epochs': bad,
               'training_loss':stats[0]/64, 'likelihood_loss':stats[1]/64, 'training_bce':stats[2]/64,
               'training_numeric_ce':stats[3]/64, 'joint_decision_loss':stats[4]/64, 'soft_action_reward':stats[5]/64, 'lr':optimizer.param_groups[0]['lr'],
               'seconds': time.perf_counter()-tick}
        history.append(row); persist()
        atomic_json(out/'runtime_status.json', {'status': 'running', 'pid': os.getpid(),
                    'architecture': args.architecture, 'mode':args.mode, 'last_completed_epoch': epoch,
                    'best_target_score': best_score, 'best_epoch': best_epoch, 'atlas_current_trial': True})
        print('EPOCH', json.dumps(row), flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json', {'status': 'stopped_by_user', 'pid': os.getpid(),
                    'last_completed_epoch': epoch, 'resume_available': True,
                    'auto_resume': False, 'atlas_current_trial': False})
        return
    model.load_state_dict(best_state, strict=True)
    pred, score, total = evaluator(model)
    assert np.array_equal(pred, best_pred)
    change = float(sum((p.detach().cpu().double()-initial_body[n].double()).square().sum()
                   for n,p in model.named_parameters() if n in initial_body).sqrt())
    diagnostic = json.loads((out/'decision_gradient_scope.json').read_text(encoding='utf-8'))
    report = {'fold':5, 'screening_fold':5, 'target':'SW', 'architecture':args.architecture, 'mode':args.mode, 'source_model':initialization,
              'scores': {'SW': score}, 'target_score': score, 'initial_target_score': initial['target_score'],
              'delta_initial': score-initial['target_score'], 'delta_old_SW': score-reference_scores['SW'],
              'reference_replacement_total': total,
              'TOTAL_scope': 'diagnostic with fixed previously verified independent POR/PERM',
              'best_epoch': best_epoch, 'stopped_epoch': epoch,
              'trainable_parameters':cfg['trainable_parameters'], 'pretrained_network_used':True, 'joint_utility_weight':criterion.alpha, 'temperature':criterion.temperature,
              'gradient_scope_verified': proof is not None, 'selected_backbone_change_L2': change,
              'gradient_diagnostic': diagnostic,
              'all144665target_predictions_restored_exactly': True,
              'strict805_passed_diagnostic': total > 80.5, 'complete_five_folds': False,
              'remaining_folds_started': False, 'seconds': elapsed+time.perf_counter()-started,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'finished_at': datetime.now().astimezone().isoformat()}
    atomic_json(out/'fold_5_report.json', report)
    atomic_json(out/'runtime_status.json', {'status': 'complete', 'pid': os.getpid(),
                'architecture': args.architecture, 'mode':args.mode, 'atlas_current_trial': False})
    print('FINAL', json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OWNED is not None:
            failure = {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc),
                       'traceback': traceback.format_exc(), 'auto_resume': False,
                       'atlas_current_trial': False}
            atomic_json(OWNED/'failure.json', failure)
            atomic_json(OWNED/'runtime_status.json', failure)
        raise
