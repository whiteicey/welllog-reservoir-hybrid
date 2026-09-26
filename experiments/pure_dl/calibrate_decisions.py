"""Fixed inner-well official-utility gate-bias selection, without neural retraining.

The same 54/10/16 partitions as train_ple_calibration.py are reused. All 31
biases are evaluated only on the ten internal calibration wells with their
54-well network's own support. The selected bias transfers to the corresponding
64-well calibrated outer model. No outer score selects parameters or sources.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'package'))
from src.data import load_split
from src.utils import BOUNDS, DEFAULTS, kernel_matrix
from train import TS, dmask, metric, write
import train_ple as ple
from train_ple_calibration import apply_gate

BIASES = np.round(np.linspace(-1.5, 1.5, 31), 1)
PENALTY = 1e-4
BATCH = 1024


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def point_score(predicted, truth, target):
    if target == 'PERM':
        ratio = (predicted/truth.clamp_min(1e-5)).clamp_min(1e-5)
        return (1-ratio.log10().abs()).clamp_min(0)
    tolerance = .08 if target == 'POR' else .05
    return (1-(predicted-truth).abs()/(tolerance*(truth.abs()+1e-5))).clamp_min(0)


def grid_and_kernel(centers, target, device):
    action = np.linspace(centers[0], centers[-1], 800) if target == 'SW' else np.geomspace(max(centers[0]*.9, .001), centers[-1]*1.1, 800)
    action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
    kernel = torch.tensor(kernel_matrix(action, np.r_[DEFAULTS[target], centers], target).T,
                          dtype=torch.float32, device=device)
    return action, kernel


def pick_bias(scores):
    objective = scores-PENALTY*BIASES**2
    best = objective.max()
    tied = np.flatnonzero(np.abs(objective-best) <= 1e-12)
    selected = min(tied.tolist(), key=lambda i: (abs(BIASES[i]), BIASES[i]))
    return selected, objective


def cpu_smoke():
    torch.set_num_threads(2)
    rng = np.random.default_rng(725)
    for target, centers in [('POR', np.array([.101, 1., 5., 10., 20., 30.])),
                            ('PERM', np.array([.0101, .02, .1, 1., 10., 100.])),
                            ('SW', np.array([10., 40., 65., 85., 98., 99.899]))]:
        actions, kernel = grid_and_kernel(centers, target, 'cpu')
        kernel = kernel.double()
        cp = torch.tensor(rng.dirichlet(np.ones(len(centers)), 12))
        logits = torch.tensor(rng.normal(size=12))
        g = (logits[:, None]+torch.tensor(BIASES)[None]).sigmoid()
        conditional = cp@kernel[1:]
        fast = conditional[:, None, :]+g[:, :, None]*(kernel[0][None, None, :]-conditional[:, None, :])
        for b in range(len(BIASES)):
            full = torch.cat([g[:, b, None], (1-g[:, b, None])*cp], 1)@kernel
            assert torch.allclose(fast[:, b], full, rtol=0, atol=5e-15)
        assert 1 < len(actions) <= 801 and np.any(actions == DEFAULTS[target])
    chosen, _ = pick_bias(np.ones(len(BIASES)))
    assert BIASES[chosen] == 0
    print(json.dumps({'cpu_smoke': 'passed', 'all31_full_action_utilities_match_joint_distribution': True,
                      'default_and_near_default_actions_present': True, 'tie_prefers_zero': True}), flush=True)


@torch.no_grad()
def scan_inner_target(inner, target_index, target, truth, centers, edges, slope, intercept, stdscale):
    """Evaluate every action for every fixed bias; no binary-decision shortcut."""
    action, kernel = grid_and_kernel(centers, target, 'cuda')
    action_tensor = torch.tensor(action, device='cuda', dtype=torch.float64)
    edge_tensor = torch.tensor(edges, device='cuda', dtype=torch.float32)
    bias_tensor = torch.tensor(BIASES, device='cuda', dtype=torch.float32)
    total = torch.zeros(len(BIASES), device='cuda', dtype=torch.float64)
    weight = inner['mixture_weight'][:, target_index]
    mu = inner['mixture_mu'][:, target_index]
    sd = inner['mixture_std'][:, target_index]*stdscale
    base_logit = (inner['gate_logits'][:, target_index].astype(np.float64)*slope+intercept).astype(np.float32)
    for start in range(0, len(truth), BATCH):
        sl = slice(start, start+BATCH)
        cp = ple.conditional_probability(torch.tensor(weight[sl], device='cuda'),
                                         torch.tensor(mu[sl], device='cuda'), torch.tensor(sd[sl], device='cuda'), edge_tensor)
        conditional_utility = cp@kernel[1:]
        gates = (torch.tensor(base_logit[sl], device='cuda')[:, None]+bias_tensor[None]).sigmoid()
        # Both the default atom and the complete conditional distribution score
        # every action, including nearby nondefault actions and the default itself.
        utility = conditional_utility[:, None, :]+gates[:, :, None]*(kernel[0][None, None, :]-conditional_utility[:, None, :])
        predictions = action_tensor[utility.argmax(-1)]
        yy = torch.tensor(truth[sl], device='cuda', dtype=torch.float64)[:, None]
        total += point_score(predictions, yy, target).sum(0)
    scores = total.cpu().numpy()/len(truth)
    choice, objective = pick_bias(scores)
    return {'bias_candidates': BIASES.tolist(), 'inner_score_0_to1': scores.tolist(),
            'penalized_inner_objective': objective.tolist(), 'penalty_coefficient': PENALTY,
            'selected_index': choice, 'selected_bias': float(BIASES[choice]),
            'inner_score_at_zero': float(scores[np.flatnonzero(BIASES == 0)[0]]),
            'inner_score_at_selected': float(scores[choice]), 'inner_rows': len(truth),
            'action_count': len(action), 'actions_sha256': hashlib.sha256(action.tobytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=ROOT/'outputs/pure_dl_ple_calibrated')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_decision_calibrated')
    ap.add_argument('--cpu-smoke', action='store_true')
    args = ap.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    assert torch.cuda.is_available()
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(.95, 1.5*2**30/device.total_memory))
    started = time.perf_counter()
    source_config = json.loads((args.source/'config.json').read_text(encoding='utf-8'))
    assert json.loads((args.source/'verification.json').read_text())['passed']
    assert json.loads((args.source/'nested_verification.json').read_text())['passed']
    source_files = [args.source/'config.json', args.source/'oof.npz']
    for f in range(5):
        source_files.extend([args.source/f'fold_{f+1}_inner_calibration.npz',
                             args.source/f'fold_{f+1}_calibration_parameters.json',
                             args.source/f'fold_{f+1}_calibration_model.pt',
                             args.source/f'fold_{f+1}_distributions.npz',
                             Path(source_config['base'])/f'fold_{f+1}.pt'])
    hashes = {str(path): sha256(path) for path in source_files}
    rec = load_split(ROOT.parent/'data', 'train')
    y = np.concatenate([frame.loc[frame[TS].notna().all(1), TS].to_numpy() for _, frame in rec])
    well = np.concatenate([np.full(frame[TS].notna().all(1).sum(), i, np.int32) for i, (_, frame) in enumerate(rec)])
    assert y.shape == (723556, 3) and len(rec) == 80
    fold = np.full(len(y), -1, np.int8)
    for f, (_, rows) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[rows] = f
    source_oof = np.load(args.source/'oof.npz')
    for key, expected in [('truth', y), ('well', well), ('fold', fold)]:
        assert np.array_equal(source_oof[key], expected), key
    default = dmask(y)
    config = {'out': str(args.out), 'source': str(args.source), 'source_sha256': sha256(__file__),
              'locked_source_sha256': hashes, 'valid_rows': len(y), 'test_access': False,
              'neural_training_updates': 0, 'bias_candidates': BIASES.tolist(), 'penalty': PENALTY,
              'objective_units': 'mean original per-target official score in[0,1] minus1e-4*bias^2',
              'tie_rule': 'objective ties within1e-12 choose smallest absolute bias, then smaller signed bias',
              'selection': 'only10 inner calibration wells; existing54-well model and its own training-only support; outer16 labels never select a bias',
              'decoder': 'full default-plus-conditional expected official utility over all800 actions including near-default alternatives',
              'transfer': 'add selected bias to same-fold existing64-well calibrated model gate; conditional distributions unchanged',
              'budget_estimate_seconds': 120, 'budget_estimate_range_seconds': [90, 165],
              'allocator_limit_gib': 1.5, 'device': device.name, 'torch': torch.__version__,
              'outer_wells': source_config['outer_wells'], 'nested_well_splits': source_config['nested_well_splits']}
    config_path = args.out/'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        for key in ['source_sha256', 'locked_source_sha256', 'bias_candidates', 'penalty', 'nested_well_splits']:
            assert previous[key] == config[key], f'Incompatible resume {key}'
    write(config_path, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k not in ['locked_source_sha256', 'outer_wells', 'nested_well_splits']}), flush=True)
    pred = np.full_like(y, np.nan)
    reports = []
    torch.cuda.reset_peak_memory_stats()
    for f in range(5):
        fold_started = time.perf_counter()
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for name in [f'fold_{f+1}_choices.json', f'fold_{f+1}_report.json', f'fold_{f+1}_distributions.npz']:
                assert (args.out/name).exists(), name
            pred[fold == f] = np.load(cache)
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()))
            continue
        split = source_config['nested_well_splits'][f]
        inner = np.load(args.source/f'fold_{f+1}_inner_calibration.npz')
        prior = json.loads((args.source/f'fold_{f+1}_calibration_parameters.json').read_text(encoding='utf-8'))
        calibration_rows = np.flatnonzero(np.isin(well, split['inner_calibration_well_indices']))
        training_mask = np.isin(well, split['inner_training_well_indices'])
        assert np.array_equal(inner['indices'], calibration_rows)
        assert np.all(fold[calibration_rows] != f) and not training_mask[calibration_rows].any()
        choices = {}
        print(f'FOLD {f+1} scan31 biases on{len(calibration_rows)} inner-only rows', flush=True)
        for j, target in enumerate(TS):
            training = ple.TRANSFORM[j](y[training_mask & ~default[:, j], j])
            transformed_centers = np.linspace(training.min(), training.max(), ple.DENSITY_BINS)
            centers = ple.INVERSE[j](transformed_centers)
            edges = (np.r_[-np.inf, (transformed_centers[:-1]+transformed_centers[1:])/2, np.inf]-inner['regression_median'][j])/inner['regression_scale'][j]
            par = prior['targets'][target]
            choices[target] = scan_inner_target(inner, j, target, y[calibration_rows, j], centers, edges,
                                                par['gate']['slope'], par['gate']['intercept'], par['stdscale']['multiplier'])
            choices[target]['inner_centers'] = centers.tolist()
            choices[target]['training_support_min_transformed'] = float(training.min())
            choices[target]['training_support_max_transformed'] = float(training.max())
        choice_record = {'fold': f+1, 'nested_split': split, 'targets': choices,
                         'inner_indices_sha256': hashlib.sha256(calibration_rows.tobytes()).hexdigest(),
                         'inner_source_sha256': hashes[str(args.source/f'fold_{f+1}_inner_calibration.npz')],
                         'outer_source_sha256': hashes[str(args.source/f'fold_{f+1}_distributions.npz')],
                         'well_lists_sha256': hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
                         'outer_labels_used_to_select': False}
        write(args.out/f'fold_{f+1}_choices.json', choice_record)
        print('CHOICES', json.dumps({'fold': f+1, 'biases': {t: choices[t]['selected_bias'] for t in TS},
                                    'inner_scores': {t: choices[t]['inner_score_at_selected'] for t in TS}}), flush=True)
        # All choices are fixed and saved before touching this fold's outer score.
        base = np.load(args.source/f'fold_{f+1}_distributions.npz')
        checkpoint = torch.load(Path(source_config['base'])/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
        rows = np.flatnonzero(fold == f)
        assert np.array_equal(base['indices'], rows)
        bias = np.array([choices[t]['selected_bias'] for t in TS])
        original_gate = base['gate']
        gate = np.column_stack([apply_gate(original_gate[:, j], 1., bias[j]) for j in range(3)])
        weight, mu, sd = base['mixture_weight'], base['mixture_mu'], base['mixture_std']
        actions, kernels = [], []
        for target in TS:
            action, kernel = grid_and_kernel(base[target+'_centers'], target, 'cuda')
            actions.append(action)
            kernels.append(kernel)
        zero_bias_prediction = np.empty((len(rows), 3), np.float64)
        with torch.no_grad():
            for start in range(0, len(rows), BATCH):
                sl = slice(start, start+BATCH)
                ww = torch.tensor(weight[sl], device='cuda')
                mm = torch.tensor(mu[sl], device='cuda')
                ss = torch.tensor(sd[sl], device='cuda')
                gg = torch.tensor(gate[sl], device='cuda')
                original_gg = torch.tensor(original_gate[sl], device='cuda')
                for j in range(3):
                    edge = torch.tensor(checkpoint['density_edges'][j], device='cuda', dtype=torch.float32)
                    cp = ple.conditional_probability(ww[:, j], mm[:, j], ss[:, j], edge)
                    zero_prob = torch.cat([original_gg[:, j, None], (1-original_gg[:, j, None])*cp], 1)
                    zero_bias_prediction[sl, j] = actions[j][(zero_prob@kernels[j]).argmax(1).cpu().numpy()]
                    prob = torch.cat([gg[:, j, None], (1-gg[:, j, None])*cp], 1)
                    pred[rows[sl], j] = actions[j][(prob@kernels[j]).argmax(1).cpu().numpy()]
        assert np.array_equal(zero_bias_prediction, base['pred']), 'Zero-bias decoder changed the existing predictions'
        deployed = {key: base[key] for key in base.files}
        deployed['predecision_gate'] = original_gate
        deployed['gate'] = gate
        deployed['decision_gate_bias'] = bias
        deployed['pred'] = pred[rows]
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **deployed)
        fr = {'fold': f+1, 'scores': metric(y[rows], pred[rows]),
              'source_scores': metric(y[rows], source_oof['pred'][rows]),
              'seconds': time.perf_counter()-fold_started, 'selected_biases': dict(zip(TS, bias.tolist())),
              'inner_calibration_rows': len(calibration_rows), 'zero_bias_reproduces_source': True,
              'conditional_distribution_unchanged': True,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        reports.append(fr)
        write(args.out/f'fold_{f+1}_report.json', fr)
        np.save(cache, pred[rows])
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps(fr), flush=True)
        del base, deployed, inner, checkpoint, kernels
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert all(sha256(path) == value for path, value in hashes.items())
    report = {'config': config, 'oof_scores': metric(y, pred), 'source_scores': metric(y, source_oof['pred']),
              'folds': reports, 'seconds': time.perf_counter()-started, 'neural_training_updates': 0,
              'source_artifacts_unchanged': True, 'outer_labels_used_to_select': False,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(args.out/'report.json', report)
    print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'source_scores', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
