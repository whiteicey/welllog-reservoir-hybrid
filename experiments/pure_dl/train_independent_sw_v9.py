"""Two predefined SW-private TCN objectives; original grouped fifth-fold score."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, random, signal, sys, time, traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from independent_sw_distribution_v9 import Network, objective
from train_dual8222_heads80 import prepare_data, SOURCE, atomic_json, atomic_torch, atomic_array, cpu_copy
from train_independent_channels_v8 import Evaluator as PriorEvaluator, REFERENCE
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
import train_invariant as base

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT/'outputs/pure_dl_independent_v8_SW_dual80_fold5/model_best.pt'
FIXED = ROOT/'outputs/pure_dl_independent_v8_selected_fold5/fold_5_pred.npy'
STOP = False
OWNED = None


def request_stop(_sig, _frame):
    global STOP
    STOP = True
    print('STOP_REQUESTED: checkpoint at epoch boundary; no automatic restart', flush=True)


class Evaluator(PriorEvaluator):
    @torch.no_grad()
    def __call__(self, model):
        d = self.data
        model.eval()
        result = np.full(len(d['validation']), np.nan, np.float64)
        coverage = np.zeros(len(result), np.int8)
        for begin in range(0, len(self.centers), 32):
            centers = torch.tensor(self.centers[begin:begin+32], device='cuda')
            anchors = torch.tensor(self.anchors[begin:begin+32], device='cuda')
            ix, valid = window_indices(centers, d['lower'], d['upper'], anchors=anchors)
            relative = torch.tanh((d['depth'][ix]-d['depth'][anchors, None])/10)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(d['x'][ix], relative, valid, absolute_depth=d['absolute'][ix])
            rows = d['raw_to_valid'][ix[:, 64:192].cpu().numpy()]
            use = valid[:, 64:192].cpu().numpy() & (rows >= 0)
            loc = self.local[rows[use]]
            assert (loc >= 0).all() and (coverage[loc] == 0).all()
            assert len(np.unique(loc)) == len(loc)
            coverage[loc] += 1
            mask = torch.zeros_like(valid)
            mask[:, 64:192] = torch.as_tensor(use, device='cuda')
            g, p, _, _ = model.distributions(output, mask)
            assert torch.isfinite(p).all() and (p >= 0).all()
            joint = torch.cat([g[:, None], (1-g[:, None])*p], -1)
            result[loc] = self.actions[(joint@self.utilities).argmax(-1).cpu().numpy()]
        assert (coverage == 1).all() and np.isfinite(result).all()
        combined = self.reference.copy()
        combined[:, 2] = result
        scores = base.metric(d['truth'][d['validation']], combined)
        return result, float(scores['SW']), float(scores['TOTAL'])


def construct(master=None, previous=None):
    if master is None:
        master = torch.load(SOURCE, map_location='cpu', weights_only=False)
    if previous is None:
        previous = torch.load(REFERENCE/'readout_checkpoint.pt', map_location='cpu', weights_only=False)
    prior = torch.load(PRIOR, map_location='cpu', weights_only=False)
    return Network(master, previous, prior['state_dict'])


def main():
    global OWNED
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('ce', 'ce_utility'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    if (out/'fold_5_report.json').exists():
        raise RuntimeError('Completed run is immutable')
    if (out/'config.json').exists() and not args.resume:
        raise RuntimeError('Existing run requires explicit resume')
    out.mkdir(exist_ok=True, parents=True)
    OWNED = out
    torch.set_num_threads(3)
    print('GPU_ADMISSION', check_slot(min_free_gib=3.), flush=True)
    atomic_json(out/'runtime_status.json', {'status': 'running', 'pid': os.getpid(), 'mode': args.mode, 'atlas_current_trial': True})
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    start = time.perf_counter()
    torch.manual_seed(20261010)
    np.random.seed(20261010)
    random.seed(20261010)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    master = torch.load(SOURCE, map_location='cpu', weights_only=False)
    previous = torch.load(REFERENCE/'readout_checkpoint.pt', map_location='cpu', weights_only=False)
    reference = np.load(FIXED)
    data = prepare_data(master)
    model = construct(master, previous).cuda()
    evaluator = Evaluator(data, master, 2, reference)
    source_score = base.metric(data['truth'][data['validation']], reference)
    frozen_prior = cpu_copy(model.prior.state_dict())
    body_initial = {n: p.detach().cpu().clone() for n, p in model.correction.named_parameters() if not n.startswith('output.')}
    alpha = 0. if args.mode == 'ce' else .1
    trainable = list(model.correction.parameters())
    cfg = {'target': 'SW', 'mode': args.mode, 'model_module': 'independent_sw_distribution_v9',
           'screening_fold': 5, 'private_new_backbone': 'TCN207to64;four depthwise5 gated blocks;dilations1,2,4,8;256densitybins',
           'new_branch_receptive_field_points': 61, 'shared_backbone_across_targets': False,
           'prior': str(PRIOR.relative_to(ROOT)), 'reference_prediction': str(FIXED.relative_to(ROOT)),
           'reference_scores': source_score, 'trainable_parameters': sum(p.numel() for p in trainable),
           'total_parameters': sum(p.numel() for p in model.parameters()),
           'epochs': 80, 'steps_per_epoch': 64, 'batch_size': 64, 'patience': 18, 'minimum_epochs': 10,
           'seed': 20261010, 'lr': 3e-4, 'eta_min': 1e-5, 'weight_decay': .01, 'gradient_clip': 2.,
           'conditional_utility_weight': alpha, 'prior_kl_weight': .01, 'log_density_adjustment_bound': 2.,
           'loss': 'soft2bin CE on true nondefault SW;optional negative log triangular conditional utility;priorKL on all labeled points;fixedQ;divide3',
           'frozen_default_probability': True, 'density_changes_can_change_final_default_actions': True,
           'quality': data['qinfo'], 'validation_rows': 144665, 'training_rows': 578891,
           'decoder': 'unchanged original source256support/800actions/single coverage',
           'no_other_target_supervision': True, 'deterministic_mode': 'warn_only;trainingbitwiseequivalence not claimed',
           'remaining_folds_started': False, 'torch_version': str(torch.__version__),
           'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [Path(__file__), Path(__file__).with_name('independent_sw_distribution_v9.py')]}}
    if args.resume:
        assert json.loads((out/'config.json').read_text()) == cfg
    else:
        atomic_json(out/'config.json', cfg)
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 80*64, eta_min=1e-5)
    windows = torch.Generator(device='cuda').manual_seed(20261010)
    epoch = 0
    prior_seconds = 0.
    proof = None
    if args.resume:
        saved = torch.load(out/'last_state.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == cfg
        model.load_state_dict(saved['current_state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        epoch = saved['epoch']
        best_epoch, best_score = saved['best_epoch'], saved['best_score']
        best_state, best_pred = saved['best_state'], saved['best_prediction']
        initial, history, bad = saved['initial'], saved['history'], saved['bad']
        prior_seconds, proof = saved['seconds'], saved['gradient_proof']
        torch.set_rng_state(saved['rng']['torch'])
        torch.cuda.set_rng_state_all(saved['rng']['cuda'])
        windows.set_state(saved['rng']['windows'])
        np.random.set_state(saved['rng']['numpy'])
        random.setstate(saved['rng']['python'])
        del saved
    else:
        pred, score, total = evaluator(model)
        initial = {'target_score': score, 'reference_replacement_total': total,
                   'different_elements_vs_prior': int(np.count_nonzero(pred != reference[:, 2])),
                   'score_delta_vs_prior': score-source_score['SW'],
                   'probability_floor_and_renormalization': 1e-12}
        atomic_json(out/'initial_verification.json', initial)
        # The protection only detects a large construction mismatch. Tiny
        # floating point discrepancies are explicitly recorded, not hidden.
        assert abs(score-source_score['SW']) < .01, initial
        best_score, best_pred = score, pred.copy()
        best_state = cpu_copy(model.state_dict())
        best_epoch, bad, history = 0, 0, []
        print('INITIAL', json.dumps(initial), flush=True)
    torch.cuda.reset_peak_memory_stats()
    def persist():
        atomic_torch(out/'model_best.pt', {'state_dict': best_state, 'config': cfg,
                     'best_epoch': best_epoch, 'target_score': best_score})
        atomic_array(out/'channel_pred.npy', best_pred)
        atomic_json(out/'validation_history.json', history)
        atomic_torch(out/'last_state.pt', {'config': cfg, 'current_state': cpu_copy(model.state_dict()),
              'optimizer': cpu_copy(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
              'epoch': epoch, 'best_epoch': best_epoch, 'best_score': best_score, 'best_state': best_state,
              'best_prediction': best_pred, 'initial': initial, 'history': history, 'bad': bad,
              'gradient_proof': proof, 'seconds': prior_seconds+time.perf_counter()-start,
              'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                      'windows': windows.get_state(), 'numpy': np.random.get_state(), 'python': random.getstate()}})
    persist()
    core = torch.zeros(1, 256, dtype=torch.bool, device='cuda')
    core[:, 64:192] = True
    for current in range(epoch+1, 81):
        if STOP or (epoch >= 10 and bad >= 18):
            break
        begin = time.perf_counter()
        model.train()
        stats = np.zeros(4)
        for step in range(64):
            anchors = data['train_raw'][torch.randint(len(data['train_raw']), (64,), device='cuda', generator=windows)]
            ix, valid = window_indices(anchors, data['lower'], data['upper'])
            relative = torch.tanh((data['depth'][ix]-data['depth'][anchors, None])/10)
            mask = valid & core & data['labeled'][ix]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(data['x'][ix], relative, valid, absolute_depth=data['absolute'][ix])
            loss, metrics = objective(model, output, data['labels'][ix], data['default'][ix],
                                      mask, data['quality'][ix], alpha)
            assert torch.isfinite(loss)
            loss.backward()
            if current == 1 and step == 1:
                body = [p.grad for n, p in model.correction.named_parameters()
                        if not n.startswith('output.') and p.grad is not None]
                norm = float(torch.stack([g.float().square().sum() for g in body]).sum().sqrt())
                assert norm > 0 and np.isfinite(norm)
                assert all(p.grad is None for p in model.prior.parameters())
                proof = {'new_private_TCN_backbone_gradient_norm': norm,
                         'frozen_prior_no_grad': True, 'only_SW_labels_used': True,
                         'checked_step': 2, 'first_step_zero_output_blocks_upstream_grad_expected': True}
                atomic_json(out/'gradient_scope.json', proof)
            torch.nn.utils.clip_grad_norm_(trainable, 2.)
            optimizer.step()
            scheduler.step()
            stats += [float(loss.detach()), metrics['ce'], metrics['conditional_utility'], metrics['prior_kl']]
        epoch = current
        pred, score, total = evaluator(model)
        if score > best_score:
            best_score, best_pred = score, pred.copy()
            best_state, best_epoch, bad = cpu_copy(model.state_dict()), epoch, 0
        else:
            bad += 1
        row = {'epoch': epoch, 'target_score': score, 'reference_replacement_total': total,
               'best_target_score': best_score, 'best_epoch': best_epoch, 'bad_epochs': bad,
               'loss': stats[0]/64, 'numeric_ce': stats[1]/64,
               'numeric_conditional_utility': stats[2]/64, 'prior_kl': stats[3]/64,
               'lr': optimizer.param_groups[0]['lr'], 'seconds': time.perf_counter()-begin}
        history.append(row)
        persist()
        atomic_json(out/'runtime_status.json', {'status': 'running', 'pid': os.getpid(), 'mode': args.mode,
                    'last_completed_epoch': epoch, 'best_target_score': best_score,
                    'best_epoch': best_epoch, 'atlas_current_trial': True})
        print('EPOCH', json.dumps(row), flush=True)
    if STOP:
        atomic_json(out/'runtime_status.json', {'status': 'stopped_by_user', 'pid': os.getpid(),
                    'last_completed_epoch': epoch, 'resume_available': True, 'auto_resume': False,
                    'atlas_current_trial': False})
        return
    body_change = float(sum((p.detach().cpu().double()-body_initial[n].double()).square().sum()
                        for n, p in model.correction.named_parameters() if n in body_initial).sqrt())
    assert body_change > 0
    model.load_state_dict(best_state, strict=True)
    pred, score, total = evaluator(model)
    assert np.array_equal(pred, best_pred)
    assert all(torch.equal(model.prior.state_dict()[n].detach().cpu(), v) for n, v in frozen_prior.items())
    report = {'fold': 5, 'screening_fold': 5, 'target': 'SW', 'mode': args.mode,
              'scores': {'SW': score}, 'target_score': score, 'initial_target_score': initial['target_score'],
              'delta_initial': score-initial['target_score'], 'delta_prior_SW': score-source_score['SW'],
              'reference_replacement_total': total, 'TOTAL_scope': 'fixed validated v8 POR/PERM plus currentSW;final disk replay pending',
              'best_epoch': best_epoch, 'stopped_epoch': epoch, 'trainable_parameters': cfg['trainable_parameters'],
              'new_TCN_backbone_parameter_change_L2_before_best_restore': body_change,
              'gradient_scope_verified': proof is not None, 'frozen_prior_exactly_unchanged': True,
              'all144665target_predictions_restored_exactly': True, 'other_target_models_unchanged': True,
              'strict805_passed_diagnostic': total > 80.5, 'complete_five_folds': False,
              'remaining_folds_started': False, 'seconds': prior_seconds+time.perf_counter()-start,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'finished_at': datetime.now().astimezone().isoformat()}
    atomic_json(out/'fold_5_report.json', report)
    atomic_json(out/'runtime_status.json', {'status': 'complete', 'pid': os.getpid(),
                'mode': args.mode, 'atlas_current_trial': False})
    print('FINAL', json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OWNED is not None:
            failure = {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc),
                       'traceback': traceback.format_exc(), 'auto_resume': False, 'atlas_current_trial': False}
            atomic_json(OWNED/'failure.json', failure)
            atomic_json(OWNED/'runtime_status.json', failure)
        raise
