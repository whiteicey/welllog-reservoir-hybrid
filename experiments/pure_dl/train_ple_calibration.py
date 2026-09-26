"""Inner-well held-out calibration transferred to each existing outer PLE model.

For each outer fold, a new PLE is fitted on 54 of its 64 training wells. Its ten
disjoint calibration wells fit only bounded gate Platt maps and mixture scales.
Those maps are applied to the existing 64-well model's outer predictions. Outer
labels are used only for the final descriptive scores, never for calibration.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import random
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'package'))
from src.data import load_split
from src.utils import BOUNDS, DEFAULTS, kernel_matrix
from train import TS, dmask, metric, write
import train_ple as ple

EPOCHS = 40
STEPS = 64
BATCH_POINTS = 4096
TRAIN_SEED = 20260927
SPLIT_SEED = 20260930
CALIBRATION_WELLS = 10
PLATT_REGULARIZATION = .01
PLATT_SLOPE_BOUNDS = (.5, 2.)
PLATT_INTERCEPT_BOUNDS = (-1.5, 1.5)
STD_SCALE_BOUNDS = (.5, 2.)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_gate(logits, labels):
    """Two bounded parameters with a fixed mild identity-map penalty."""
    logits = np.asarray(logits, np.float64)
    labels = np.asarray(labels, np.float64)
    assert logits.shape == labels.shape and np.isfinite(logits).all()

    def objective(parameters):
        log_a, b = parameters
        a = np.exp(log_a)
        calibrated = a*logits+b
        residual = expit(calibrated)-labels
        nll = np.mean(np.logaddexp(0, calibrated)-labels*calibrated)
        penalty = PLATT_REGULARIZATION*(log_a*log_a+b*b)
        gradient = np.array([np.mean(residual*a*logits)+2*PLATT_REGULARIZATION*log_a,
                             np.mean(residual)+2*PLATT_REGULARIZATION*b])
        return float(nll+penalty), gradient

    result = minimize(objective, np.zeros(2), jac=True, method='L-BFGS-B',
                      bounds=[tuple(np.log(PLATT_SLOPE_BOUNDS)), PLATT_INTERCEPT_BOUNDS],
                      options={'maxiter': 100, 'ftol': 1e-11, 'gtol': 1e-7})
    assert result.success, result.message
    a = float(np.exp(result.x[0]))
    b = float(result.x[1])
    before = float(np.mean(np.logaddexp(0, logits)-labels*logits))
    mapped = a*logits+b
    after = float(np.mean(np.logaddexp(0, mapped)-labels*mapped))
    return {'slope': a, 'intercept': b, 'regularization': PLATT_REGULARIZATION,
            'rows': len(labels), 'default_rows': int(labels.sum()),
            'uncalibrated_bce': before, 'calibrated_bce': after,
            'penalized_objective': float(result.fun), 'iterations': int(result.nit),
            'optimizer_success': bool(result.success)}


def fit_standard_deviation(weight, mu, sd, targets):
    """A single bounded scalar fitted by conditional nondefault likelihood."""
    weight = np.asarray(weight, np.float64)
    mu = np.asarray(mu, np.float64)
    sd = np.asarray(sd, np.float64)
    targets = np.asarray(targets, np.float64)
    assert mu.shape == weight.shape == sd.shape == (len(targets), ple.MIX)
    assert len(targets) > 0 and np.isfinite(mu).all() and np.all(sd > 0)
    base = np.log(np.maximum(weight, 1e-300))-np.log(sd)-.5*np.log(2*np.pi)
    standardized_squared_error = ((targets[:, None]-mu)/sd)**2

    def nll(log_scale):
        lp = base-log_scale-.5*standardized_squared_error*np.exp(-2*log_scale)
        return float(-np.mean(logsumexp(lp, axis=1)))

    bounds = tuple(np.log(STD_SCALE_BOUNDS))
    result = minimize_scalar(nll, bounds=bounds, method='bounded',
                             options={'maxiter': 80, 'xatol': 1e-5})
    assert result.success, result.message
    # Endpoint optima are evaluated explicitly; all choices use only this NLL.
    candidates = [float(result.x), float(bounds[0]), float(bounds[1])]
    values = [nll(v) for v in candidates]
    chosen = int(np.argmin(values))
    scale = float(np.exp(candidates[chosen]))
    return {'multiplier': scale, 'nondefault_rows': len(targets),
            'uncalibrated_nll': nll(0.), 'calibrated_nll': values[chosen],
            'function_evaluations': int(result.nfev)+3, 'optimizer_success': bool(result.success)}


def apply_gate(probability, slope, intercept):
    p = np.asarray(probability, np.float64)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    # Positive slopes preserve exact zero/one limits without arbitrary clipping.
    with np.errstate(divide='ignore'):
        logits = np.log(p)-np.log1p(-p)
    return expit(slope*logits+intercept).astype(np.float32)


def cpu_smoke():
    rng = np.random.default_rng(613)
    logits = rng.normal(size=2048)*2
    labels = rng.binomial(1, expit(.7*logits+.25))
    gate = fit_gate(logits, labels)
    assert .5 <= gate['slope'] <= 2 and -1.5 <= gate['intercept'] <= 1.5
    assert gate['calibrated_bce'] <= gate['uncalibrated_bce']+1e-8
    weight = np.full((2048, ple.MIX), 1/ple.MIX)
    mu = np.zeros_like(weight)
    sd = np.ones_like(weight)
    density = fit_standard_deviation(weight, mu, sd, rng.normal(size=2048)*1.4)
    assert .5 <= density['multiplier'] <= 2
    assert density['calibrated_nll'] <= density['uncalibrated_nll']+1e-8
    endpoints = apply_gate(np.array([0., .5, 1.]), 1., 0.)
    assert np.array_equal(endpoints, np.array([0., .5, 1.], np.float32))
    print(json.dumps({'cpu_smoke': 'passed', 'synthetic_gate': gate,
                      'synthetic_stdscale': density, 'real_data_read': False}), flush=True)


def predict_inner(model, x, rows):
    model.eval()
    saved = {'gate_logits': [], 'mixture_weight': [], 'mixture_mu': [], 'mixture_std': []}
    with torch.no_grad():
        for start in range(0, len(rows), 1024):
            ix = torch.tensor(rows[start:start+1024], device='cuda', dtype=torch.long)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits, raw = model(x[ix])
            logw, mu, sd = ple.mixture_params(raw)
            saved['gate_logits'].append(logits.float().cpu().numpy())
            saved['mixture_weight'].append(logw.exp().cpu().numpy())
            saved['mixture_mu'].append(mu.cpu().numpy())
            saved['mixture_std'].append(sd.cpu().numpy())
    return {key: np.concatenate(values) for key, values in saved.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_ple_calibrated')
    ap.add_argument('--base', type=Path, default=ROOT/'outputs/pure_dl_ple')
    ap.add_argument('--cpu-smoke', action='store_true')
    args = ap.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(.95, 1.5*2**30/device.total_memory))
    started = time.perf_counter()
    base_config = json.loads((args.base/'config.json').read_text(encoding='utf-8'))
    assert (base_config['epochs'], base_config['steps'], base_config['batch_points']) == (EPOCHS, STEPS, BATCH_POINTS)
    assert sha256(ple.__file__) == base_config['source_sha256'], 'Existing PLE code changed'
    source_files = [args.base/'config.json', args.base/'oof.npz']
    for f in range(5):
        source_files.extend([args.base/f'fold_{f+1}.pt', args.base/f'fold_{f+1}_distributions.npz'])
    source_hashes = {str(path): sha256(path) for path in source_files}
    rec = load_split(ROOT.parent/'data', 'train')
    ys = [frame[TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    y = np.concatenate([v[mask] for v, mask in zip(ys, valid)])
    well = np.concatenate([np.full(mask.sum(), i, np.int32) for i, mask in enumerate(valid)])
    assert len(rec) == 80 and y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[ix] = f
    well_fold = {i: int(fold[np.flatnonzero(well == i)[0]]) for i in range(80)}
    original = np.load(args.base/'oof.npz')
    for key, values in [('truth', y), ('well', well), ('fold', fold)]:
        assert np.array_equal(original[key], values), key
    source_pred = original['pred'].copy()
    splits = []
    for f in range(5):
        outer_train = [i for i in range(80) if well_fold[i] != f]
        outer_validation = [i for i in range(80) if well_fold[i] == f]
        permuted = np.random.default_rng(SPLIT_SEED+f).permutation(outer_train)
        calibration = sorted(permuted[:CALIBRATION_WELLS].tolist())
        fit = sorted(permuted[CALIBRATION_WELLS:].tolist())
        assert len(fit) == 54 and len(calibration) == 10 and len(outer_validation) == 16
        assert not (set(fit) & set(calibration)) and not (set(outer_train) & set(outer_validation))
        splits.append({'fold': f+1, 'split_seed': SPLIT_SEED+f,
                       'inner_training_well_indices': fit, 'inner_calibration_well_indices': calibration,
                       'outer_validation_well_indices': outer_validation,
                       'inner_training_wells': [rec[i][0] for i in fit],
                       'inner_calibration_wells': [rec[i][0] for i in calibration],
                       'outer_validation_wells': [rec[i][0] for i in outer_validation]})
    config = {'out': str(args.out), 'base': str(args.base), 'source_sha256': sha256(__file__),
              'ple_source_sha256': sha256(ple.__file__), 'locked_source_sha256': source_hashes,
              'epochs': EPOCHS, 'steps': STEPS, 'batch_points': BATCH_POINTS, 'seed': TRAIN_SEED,
              'split_seed': SPLIT_SEED, 'valid_rows': len(y), 'test_access': False,
              'parameters': sum(p.numel() for p in ple.Network().parameters()),
              'device': device.name, 'torch': torch.__version__, 'allocator_limit_gib': 1.5,
              'architecture': base_config['architecture'],
              'protocol': 'For each outer fold: train a fresh PLE on54 wells, fit calibration only on10 disjoint wells, transfer maps to the locked existing64-well PLE outer predictions',
              'preprocessing': 'inner-network medians, scales, PLE knots and target transforms fit on54 inner training wells only',
              'platt': {'positive_slope_bounds': PLATT_SLOPE_BOUNDS, 'intercept_bounds': PLATT_INTERCEPT_BOUNDS,
                        'identity_map_penalty': PLATT_REGULARIZATION, 'objective': 'inner10 gate BCE + lambda*((log slope)^2+intercept^2)'},
              'stdscale': {'bounds': STD_SCALE_BOUNDS, 'objective': 'inner10 nondefault mixture NLL per target'},
              'decoder': 'same existing64-well support centers, density edges and800 action grid; official expected utility with default atom',
              'selection': 'one fixed configuration; no outer-label fitting, threshold scan, checkpoint selection or fold-dependent model choice',
              'budget_estimate_seconds': 480, 'budget_estimate_range_seconds': [390, 570],
              'outer_wells': {str(f): splits[f]['outer_validation_wells'] for f in range(5)},
              'nested_well_splits': splits}
    config_path = args.out/'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        for key in ['source_sha256', 'locked_source_sha256', 'epochs', 'steps', 'batch_points', 'seed', 'nested_well_splits']:
            assert old[key] == config[key], f'Incompatible resume {key}'
    write(config_path, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k not in ['outer_wells', 'nested_well_splits', 'locked_source_sha256']}), flush=True)
    pred = np.full_like(y, np.nan)
    reports = []
    for f in range(5):
        fold_started = time.perf_counter()
        seed = TRAIN_SEED+f
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for filename in [f'fold_{f+1}_report.json', f'fold_{f+1}_calibration_parameters.json',
                             f'fold_{f+1}_distributions.npz', f'fold_{f+1}_inner_calibration.npz',
                             f'fold_{f+1}_calibration_model.pt']:
                assert (args.out/filename).exists(), filename
            pred[fold == f] = np.load(cache)
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()))
            continue
        split = splits[f]
        outer_train = sorted(split['inner_training_well_indices']+split['inner_calibration_well_indices'])
        fit_wells = split['inner_training_well_indices']
        calibration_wells = split['inner_calibration_well_indices']
        inner_rec = [rec[i] for i in outer_train]
        inner_valid = [valid[i] for i in outer_train]
        outer_rows = np.flatnonzero(fold != f)
        inner_y = y[outer_rows]
        inner_well = well[outer_rows]
        local_row_ids = [np.flatnonzero(inner_well == i) for i in outer_train]
        fit_local_wells = [i for i, global_well in enumerate(outer_train) if global_well in fit_wells]
        training_mask = np.isin(inner_well, fit_wells)
        calibration_mask = np.isin(inner_well, calibration_wells)
        assert np.all(training_mask ^ calibration_mask)
        assert set(well[outer_rows[training_mask]]) == set(fit_wells)
        assert set(well[outer_rows[calibration_mask]]) == set(calibration_wells)
        print(f'FOLD {f+1} preparing54 training wells;10 calibration wells;16 outer wells excluded', flush=True)
        x, builder, fm, fs, ple_edges = ple.prepare_features(inner_rec, fit_local_wells, inner_valid, local_row_ids)
        default = dmask(inner_y)
        med, scale, init, target_columns = [], [], [], []
        for j in range(3):
            transformed = ple.TRANSFORM[j](inner_y[:, j])
            training = transformed[training_mask & ~default[:, j]]
            q25, q50, q75 = np.percentile(training, [25, 50, 75])
            sc = max(float(q75-q25), .1)
            med.append(q50)
            scale.append(sc)
            target_columns.append((transformed-q50)/sc)
            init.append(np.quantile((training-q50)/sc, np.linspace(.05, .95, ple.MIX)))
        med = np.asarray(med)
        scale = np.asarray(scale)
        normalized_targets = np.column_stack(target_columns)
        targets = torch.tensor(normalized_targets, device='cuda', dtype=torch.float32)
        gates = torch.tensor(default, device='cuda', dtype=torch.float32)
        training_indices = torch.tensor(np.flatnonzero(training_mask), device='cuda', dtype=torch.long)
        model = ple.Network(ple_edges, np.stack(init)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS*STEPS, eta_min=.00001)
        history = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(EPOCHS):
            model.train()
            losses = np.zeros(3)
            for step in range(STEPS):
                losses += ple.training_step(model, opt, x, gates, targets, training_indices, BATCH_POINTS)
                scheduler.step()
            history.append((losses/STEPS).tolist())
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/{EPOCHS} loss={history[-1][0]:.4f} seconds={time.perf_counter()-fold_started:.1f}', flush=True)
        local_calibration_rows = np.flatnonzero(calibration_mask)
        global_calibration_rows = outer_rows[local_calibration_rows]
        assert np.all(fold[global_calibration_rows] != f)
        inner_predictions = predict_inner(model, x, local_calibration_rows)
        calibration_labels = default[local_calibration_rows]
        calibration_targets = normalized_targets[local_calibration_rows]
        np.savez_compressed(args.out/f'fold_{f+1}_inner_calibration.npz',
                            indices=global_calibration_rows, well=well[global_calibration_rows],
                            default=calibration_labels, normalized_targets=calibration_targets,
                            regression_median=med, regression_scale=scale, **inner_predictions)
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config,
                    'feature_median': fm.tolist(), 'feature_scale': fs.tolist(),
                    'ple_edges_normalized': ple_edges.tolist(), 'medians': builder.medians_,
                    'enhanced_feature_names': builder.enhanced_feature_names_,
                    'regression_median': med.tolist(), 'regression_scale': scale.tolist(),
                    'nested_split': split}, args.out/f'fold_{f+1}_calibration_model.pt')
        training_seconds = time.perf_counter()-fold_started
        del model, opt, scheduler, x, targets, gates, training_indices
        gc.collect()
        torch.cuda.empty_cache()
        parameters = {}
        for j, target in enumerate(TS):
            gp = fit_gate(inner_predictions['gate_logits'][:, j], calibration_labels[:, j])
            active = ~calibration_labels[:, j]
            sp = fit_standard_deviation(inner_predictions['mixture_weight'][active, j],
                                        inner_predictions['mixture_mu'][active, j],
                                        inner_predictions['mixture_std'][active, j],
                                        calibration_targets[active, j])
            parameters[target] = {'gate': gp, 'stdscale': sp}
        param_record = {'fold': f+1, 'nested_split': split, 'targets': parameters,
                        'inner_training_rows': int(training_mask.sum()), 'inner_calibration_rows': len(global_calibration_rows),
                        'outer_validation_rows': int((fold == f).sum()), 'outer_labels_used_to_fit': False,
                        'calibration_fit_targets': 'Only the10 inner calibration wells; gate BCE and nondefault NLL, no official-score search',
                        'source_checkpoint_sha256': source_hashes[str(args.base/f'fold_{f+1}.pt')],
                        'source_distributions_sha256': source_hashes[str(args.base/f'fold_{f+1}_distributions.npz')]}
        write(args.out/f'fold_{f+1}_calibration_parameters.json', param_record)
        print('CALIBRATION', json.dumps({'fold': f+1, 'parameters': parameters}), flush=True)
        base_pack = np.load(args.base/f'fold_{f+1}_distributions.npz')
        checkpoint = torch.load(args.base/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
        validation_rows = np.flatnonzero(fold == f)
        assert np.array_equal(base_pack['indices'], validation_rows)
        assert np.array_equal(base_pack['pred'], source_pred[validation_rows])
        assert np.all(fold[global_calibration_rows] != fold[validation_rows[0]])
        raw_gate = base_pack['gate']
        calibrated_gate = np.column_stack([apply_gate(raw_gate[:, j], parameters[t]['gate']['slope'],
                                                      parameters[t]['gate']['intercept']) for j, t in enumerate(TS)])
        weight = base_pack['mixture_weight']
        mu = base_pack['mixture_mu']
        std = base_pack['mixture_std'].copy()
        std_multiplier = np.array([parameters[t]['stdscale']['multiplier'] for t in TS])
        std *= std_multiplier[None, :, None]
        centers = [base_pack[t+'_centers'] for t in TS]
        for j in range(3):
            assert np.array_equal(centers[j], np.asarray(checkpoint['centers'][j]))
        actions, kernels = [], []
        for j, target in enumerate(TS):
            support = centers[j]
            action = np.linspace(support[0], support[-1], 800) if j == 2 else np.geomspace(max(support[0]*.9, .001), support[-1]*1.1, 800)
            action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
            actions.append(action)
            kernels.append(torch.tensor(kernel_matrix(action, np.r_[DEFAULTS[target], support], target).T,
                                         device='cuda', dtype=torch.float32))
        saved_prob = [[], [], []]
        with torch.no_grad():
            for start in range(0, len(validation_rows), 1024):
                sl = slice(start, start+1024)
                ww = torch.tensor(weight[sl], device='cuda')
                mm = torch.tensor(mu[sl], device='cuda')
                ss = torch.tensor(std[sl], device='cuda')
                gg = torch.tensor(calibrated_gate[sl], device='cuda')
                for j in range(3):
                    edge = torch.tensor(checkpoint['density_edges'][j], device='cuda', dtype=torch.float32)
                    cp = ple.conditional_probability(ww[:, j], mm[:, j], ss[:, j], edge)
                    probability = torch.cat([gg[:, j, None], (1-gg[:, j, None])*cp], 1)
                    pred[validation_rows[sl], j] = actions[j][(probability@kernels[j]).argmax(1).cpu().numpy()]
                    saved_prob[j].append(cp.cpu().numpy().astype(np.float16))
        deployed = {'indices': validation_rows, 'gate': calibrated_gate, 'uncalibrated_gate': raw_gate,
                    'mixture_weight': weight, 'mixture_mu': mu, 'mixture_std': std,
                    'regression_median': base_pack['regression_median'], 'regression_scale': base_pack['regression_scale'],
                    'regression_transforms': np.array(['log1p', 'log10', 'log']), 'pred': pred[validation_rows],
                    'gate_calibration_slope': np.array([parameters[t]['gate']['slope'] for t in TS]),
                    'gate_calibration_intercept': np.array([parameters[t]['gate']['intercept'] for t in TS]),
                    'std_calibration_multiplier': std_multiplier}
        for j, target in enumerate(TS):
            deployed[target+'_prob'] = np.concatenate(saved_prob[j])
            deployed[target+'_centers'] = centers[j]
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **deployed)
        fr = {'fold': f+1, 'scores': metric(y[validation_rows], pred[validation_rows]),
              'uncalibrated_source_scores': metric(y[validation_rows], source_pred[validation_rows]),
              'seconds': time.perf_counter()-fold_started, 'inner_training_and_prediction_seconds': training_seconds,
              'loss': history, 'calibration': parameters, 'inner_training_wells': 54, 'inner_calibration_wells': 10,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        reports.append(fr)
        write(args.out/f'fold_{f+1}_report.json', fr)
        np.save(cache, pred[validation_rows])
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k not in ['loss', 'calibration']}), flush=True)
        del base_pack, checkpoint, deployed, saved_prob, kernels, inner_predictions
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert all(sha256(path) == digest for path, digest in source_hashes.items())
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': reports,
              'uncalibrated_source_scores': metric(y, source_pred), 'seconds': time.perf_counter()-started,
              'source_artifacts_unchanged': True, 'outer_labels_used_to_fit_calibration': False,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(args.out/'report.json', report)
    print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'uncalibrated_source_scores', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
