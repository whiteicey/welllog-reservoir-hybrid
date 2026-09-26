"""Fixed curve-token Transformer over fold-local invariant numeric features.

The audited 206 input features and 16-segment PLE are reused unchanged. Features
are partitioned once by exact curve name or curve-name-plus-underscore prefix;
all remaining names form OTHER. Only interactions between input curves are new.
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
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'package'))
from src.data import load_split
from src.utils import BOUNDS, DEFAULTS, kernel_matrix
from train import TS, dmask, metric, write
import train_invariant as inv

GROUP_NAMES = ['GR', 'AC', 'DEN', 'CNL', 'RT', 'RXO', 'PE', 'SP', 'CAL', 'OTHER']
EPOCHS = 40
STEPS = 64
BATCH_POINTS = 2048
SEED = 20260930


def grouping_contract(feature_names):
    assert len(feature_names) == inv.FEATURES == 206 and len(set(feature_names)) == 206
    groups = {name: [] for name in GROUP_NAMES}
    for index, feature in enumerate(feature_names):
        matched = [name for name in GROUP_NAMES[:-1] if feature == name or feature.startswith(name+'_')]
        assert len(matched) <= 1, (feature, matched)
        groups[matched[0] if matched else 'OTHER'].append(index)
    flat = [index for name in GROUP_NAMES for index in groups[name]]
    assert sorted(flat) == list(range(206)) and len(set(flat)) == 206
    assert all(groups[name] for name in GROUP_NAMES)
    return {'rule': 'feature == curve or feature starts with curve + underscore; otherwise OTHER',
            'group_order': GROUP_NAMES, 'all_features_covered_once': True,
            'groups': {name: {'indices': groups[name], 'features': [feature_names[i] for i in groups[name]],
                              'feature_count': len(groups[name]), 'encoded_input_dimensions': len(groups[name])*17}
                       for name in GROUP_NAMES},
            'feature_count': 206, 'token_count': 10, 'token_dimensions': 64}


class CurveEncoding(nn.Module):
    def __init__(self, edges, group_indices):
        super().__init__()
        edges = torch.as_tensor(edges, dtype=torch.float32)
        assert edges.shape == (206, 17)
        widths = edges[:, 1:]-edges[:, :-1]
        self.register_buffer('left', edges[:, :-1])
        self.register_buffer('inverse_width', widths.clamp_min(1e-6).reciprocal())
        self.register_buffer('valid_segment', (widths > 1e-6).float())
        self.projections = nn.ModuleList([nn.Linear(len(indices)*17, 64) for indices in group_indices])
        for i, indices in enumerate(group_indices):
            self.register_buffer(f'group_{i}', torch.tensor(indices, dtype=torch.long))

    def forward(self, x):
        raw = x.float()
        ple = ((raw[:, :, None]-self.left)*self.inverse_width).clamp(0, 1)*self.valid_segment
        features = torch.cat([raw[:, :, None], ple], -1)
        tokens = []
        for i, projection in enumerate(self.projections):
            selected = features.index_select(1, getattr(self, f'group_{i}')).flatten(1)
            tokens.append(projection(selected))
        return torch.stack(tokens, 1)


class Network(nn.Module):
    def __init__(self, group_indices, edges=None, initial_means=None):
        super().__init__()
        if edges is None:
            edges = np.tile(np.linspace(-3, 3, 17), (206, 1))
        self.encoding = CurveEncoding(edges, group_indices)
        self.curve_identity = nn.Parameter(torch.empty(10, 64))
        nn.init.normal_(self.curve_identity, std=.02)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=192,
                                            dropout=.1, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(64), enable_nested_tensor=False)
        self.fusion = nn.Sequential(nn.Linear(640, 256), nn.GELU())
        self.gate = nn.Linear(256, 3)
        self.density = nn.Linear(256, 3*3*inv.MIX)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        with torch.no_grad():
            bias = self.density.bias.view(3, 3, inv.MIX)
            if initial_means is not None:
                bias[:, 1].copy_(torch.as_tensor(initial_means, dtype=torch.float32))
            bias[:, 2].fill_(float(np.log(np.expm1(.5-inv.SIGMA_FLOOR))))

    def forward(self, x):
        tokens = self.encoding(x)
        tokens = tokens+self.curve_identity.to(tokens.dtype)[None]
        encoded = self.encoder(tokens)
        h = self.fusion(encoded.flatten(1))
        return self.gate(h), self.density(h).reshape(-1, 3, 3, inv.MIX)


def contracts():
    feature = inv.feature_contract()
    grouped = grouping_contract(feature['whitelist'])
    return feature, grouped, [grouped['groups'][name]['indices'] for name in GROUP_NAMES]


def cpu_smoke():
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    _, grouped, indices = contracts()
    model = Network(indices, initial_means=np.tile(np.linspace(-2, 2, inv.MIX), (3, 1)))
    x = torch.randn(24, 206)
    gate, raw = model(x)
    assert gate.shape == (24, 3) and raw.shape == (24, 3, 3, inv.MIX)
    logw, mu, sd = inv.mixture_params(raw)
    target_gate = torch.randint(0, 2, (24, 3)).float()
    target_gate[0] = 0
    targets = torch.randn(24, 3)
    lp = logw-.5*((targets[:, :, None]-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi)
    nll = -torch.logsumexp(lp, -1)
    loss = F.binary_cross_entropy_with_logits(gate, target_gate)+sum(nll[:, j][target_gate[:, j] < .5].mean() for j in range(3))/3
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        eval_gate, eval_raw = model(x)
    assert torch.isfinite(eval_gate).all() and torch.isfinite(eval_raw).all()
    print(json.dumps({'cpu_smoke': 'passed', 'real_data_read': False,
                      'groups': {k: v['feature_count'] for k, v in grouped['groups'].items()},
                      'covers206_features_once': True, 'parameters': sum(p.numel() for p in model.parameters()),
                      'all_parameter_gradients_finite': True, 'eval_outputs_finite': True}), flush=True)


def benchmark(out):
    feature, grouped, group_indices = contracts()
    out.mkdir(parents=True, exist_ok=True)
    write(out/'feature_contract.json', feature)
    write(out/'group_contract.json', grouped)
    torch.manual_seed(SEED)
    model = Network(group_indices, initial_means=np.tile(np.linspace(-2, 2, inv.MIX), (3, 1))).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x = torch.randn((723556, 206), device='cuda', dtype=torch.float16).clamp(-8, 8)
    gates = (torch.rand((723556, 3), device='cuda') < .7).float()
    targets = torch.randn((723556, 3), device='cuda')
    rows = torch.arange(578000, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(8):
        inv.training_step(model, opt, x, gates, targets, rows, BATCH_POINTS)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(40):
        inv.training_step(model, opt, x, gates, targets, rows, BATCH_POINTS)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    step_seconds = seconds/40
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        go, raw = model(x[:1024])
    assert torch.isfinite(go).all() and torch.isfinite(raw).all()
    training_projection = step_seconds*EPOCHS*STEPS*5
    result = {'benchmark_only': True, 'real_data_read': False, 'synthetic_updates': 48,
              'timed_updates': 40, 'mean_step_ms': step_seconds*1000,
              'fixed_epochs': EPOCHS, 'fixed_steps': STEPS, 'fixed_batch_points': BATCH_POINTS,
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'parameters': sum(p.numel() for p in model.parameters()),
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'projected_five_fold_update_seconds': training_projection,
              'projected_total_seconds_range': [training_projection+145, training_projection+200],
              'budget_gate_passed': training_projection+200 <= 600 and torch.cuda.max_memory_reserved() < 1.5*2**30,
              'projection_note': 'Actual synthetic updates at the current GPU contention plus145-200 seconds for five-fold preprocessing, complete inference and archive I/O; not a completed CV runtime',
              'cuda_eval_finite': True}
    write(out/'benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)


def train(out):
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    benchmark_path = out/'benchmark.json'
    assert benchmark_path.exists(), 'Run the bounded synthetic benchmark first'
    bench = json.loads(benchmark_path.read_text())
    assert bench['budget_gate_passed'], 'Projected total exceeds10 minutes or GPU memory budget'
    feature, grouped, group_indices = contracts()
    for filename, value in [('feature_contract.json', feature), ('group_contract.json', grouped)]:
        path = out/filename
        if path.exists():
            assert json.loads(path.read_text()) == value
        else:
            write(path, value)
    rec = load_split(ROOT.parent/'data', 'train')
    assert len(rec) == 80
    ys = [frame[TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    y = np.concatenate([v[keep] for v, keep in zip(ys, valid)])
    well = np.concatenate([np.full(keep.sum(), i, np.int32) for i, keep in enumerate(valid)])
    assert y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, rows) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[rows] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    well_fold = {i: int(fold[row_ids[i][0]]) for i in range(80)}
    hash_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    config = {'out': str(out), 'epochs': EPOCHS, 'steps': STEPS, 'batch_points': BATCH_POINTS, 'seed': SEED,
              'source_sha256': hash_file(__file__), 'invariant_helper_sha256': hash_file(inv.__file__),
              'feature_builder_source_sha256': hash_file(ROOT/'package/src/feature.py'),
              'feature_contract_sha256': hash_file(out/'feature_contract.json'),
              'group_contract_sha256': hash_file(out/'group_contract.json'),
              'parameters': sum(p.numel() for p in Network(group_indices).parameters()),
              'architecture': '206 invariant features with16-segment fold-local PLE;10 curve groups each projected to64D plus curve identity embedding;2-layer preLN TransformerEncoder,4 heads,FF192,dropout0.1;flatten640->256;3 gates and3x8 Gaussian mixtures',
              'hypothesis': 'Learn between-curve interactions with explicit curve tokens instead of a flat PLE MLP; no claimed or guaranteed improvement',
              'feature_whitelist': feature['whitelist'], 'excluded_features': feature['excluded'],
              'group_order': GROUP_NAMES, 'feature_group_indices': group_indices,
              'preprocessing': 'reuse audited invariant206 selection; medians, robust scaling and17 PLE knots fit only64 outer-training input wells',
              'targets': 'log1p(POR),log10(PERM),log(SW); medians,scales,support and initialization fit outer-training nondefault labels only',
              'loss': 'original independent gate BCE plus average nondefault Gaussian-mixture NLL',
              'decoder': 'unchanged256 transformed support bins and800 action grid with original official expected utility including default atom',
              'selection': 'single fixed40x64 configuration; no outer-fold early stopping, calibration, hyperparameter or model choice',
              'valid_rows': len(y), 'test_access': False, 'device': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'allocator_limit_gib': 1.5,
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if well_fold[i] == f] for f in range(5)}}
    config_path = out/'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ['source_sha256', 'invariant_helper_sha256', 'epochs', 'steps', 'batch_points', 'seed', 'group_contract_sha256']:
            assert previous[key] == config[key], f'Incompatible cached configuration {key}'
    write(config_path, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k not in ['outer_wells', 'feature_whitelist', 'excluded_features', 'feature_group_indices']}), flush=True)
    pred = np.full_like(y, np.nan)
    reports = []
    default = dmask(y)
    for f in range(5):
        fold_started = time.perf_counter()
        seed = SEED+f
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cache = out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for filename in [f'fold_{f+1}_report.json', f'fold_{f+1}_distributions.npz', f'fold_{f+1}.pt', f'fold_{f+1}_preprocessing.json']:
                assert (out/filename).exists(), filename
            pred[fold == f] = np.load(cache)
            reports.append(json.loads((out/f'fold_{f+1}_report.json').read_text()))
            continue
        training_wells = [i for i in range(80) if well_fold[i] != f]
        assert len(training_wells) == 64 and not any(well_fold[i] == f for i in training_wells)
        print(f'FOLD {f+1} fitting invariant206 PLE on64 training wells', flush=True)
        x, builder, fm, fs, edges = inv.prepare_features(rec, training_wells, valid, row_ids)
        chosen_names = [builder.enhanced_feature_names_[i] for i in inv.select_features(builder.enhanced_feature_names_)]
        assert chosen_names == feature['whitelist'] and grouping_contract(chosen_names) == grouped
        median, scale, initial_means, centers, density_edges, columns = [], [], [], [], [], []
        for j in range(3):
            transformed = inv.TRANSFORM[j](y[:, j])
            training = transformed[(fold != f) & ~default[:, j]]
            q25, q50, q75 = np.percentile(training, [25, 50, 75])
            sc = max(float(q75-q25), .1)
            median.append(q50)
            scale.append(sc)
            columns.append((transformed-q50)/sc)
            initial_means.append(np.quantile((training-q50)/sc, np.linspace(.05, .95, inv.MIX)))
            ct = np.linspace(training.min(), training.max(), inv.DENSITY_BINS)
            centers.append(inv.INVERSE[j](ct))
            density_edges.append((np.r_[-np.inf, (ct[:-1]+ct[1:])/2, np.inf]-q50)/sc)
        median, scale = np.asarray(median), np.asarray(scale)
        targets = torch.tensor(np.column_stack(columns), dtype=torch.float32, device='cuda')
        gates = torch.tensor(default, dtype=torch.float32, device='cuda')
        training_indices = torch.tensor(np.flatnonzero(fold != f), dtype=torch.long, device='cuda')
        preproc = {'fold': f+1, 'training_wells': [rec[i][0] for i in training_wells],
                   'validation_wells': config['outer_wells'][str(f)], 'training_rows': int((fold != f).sum()),
                   'input_statistics_rows_including_unlabeled': sum(len(rec[i][1]) for i in training_wells),
                   'target_statistics_nondefault_rows': [int(((fold != f) & ~default[:, j]).sum()) for j in range(3)],
                   'feature_whitelist': chosen_names, 'feature_group_indices': group_indices,
                   'input_and_target_fit_excludes_outer_wells': True}
        assert not set(preproc['training_wells']) & set(preproc['validation_wells'])
        write(out/f'fold_{f+1}_preprocessing.json', preproc)
        model = Network(group_indices, edges, np.stack(initial_means)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS*STEPS, eta_min=.00001)
        history = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(EPOCHS):
            model.train()
            losses = np.zeros(3)
            for step in range(STEPS):
                losses += inv.training_step(model, opt, x, gates, targets, training_indices, BATCH_POINTS)
                scheduler.step()
            history.append((losses/STEPS).tolist())
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/{EPOCHS} loss={history[-1][0]:.4f} gate={history[-1][1]:.4f} nll={history[-1][2]:.4f} seconds={time.perf_counter()-fold_started:.1f}', flush=True)
        model.eval()
        actions, kernels = [], []
        for j, target in enumerate(TS):
            support = centers[j]
            action = np.linspace(support[0], support[-1], 800) if j == 2 else np.geomspace(max(support[0]*.9, .001), support[-1]*1.1, 800)
            action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
            actions.append(action)
            kernels.append(torch.tensor(kernel_matrix(action, np.r_[DEFAULTS[target], support], target).T, device='cuda', dtype=torch.float32))
        validation_rows = np.flatnonzero(fold == f)
        saved_gate, saved_weight, saved_mu, saved_sd = [], [], [], []
        saved_prob = [[], [], []]
        with torch.no_grad():
            for start in range(0, len(validation_rows), 1024):
                rows = validation_rows[start:start+1024]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    go, raw = model(x[torch.tensor(rows, device='cuda')])
                gate = go.float().sigmoid()
                logw, mu, sd = inv.mixture_params(raw)
                weight = logw.exp()
                for j in range(3):
                    ed = torch.tensor(density_edges[j], device='cuda', dtype=torch.float32)
                    cp = inv.conditional_probability(weight[:, j], mu[:, j], sd[:, j], ed)
                    probability = torch.cat([gate[:, j, None], (1-gate[:, j, None])*cp], 1)
                    pred[rows, j] = actions[j][(probability@kernels[j]).argmax(1).cpu().numpy()]
                    saved_prob[j].append(cp.cpu().numpy().astype(np.float16))
                saved_gate.append(gate.cpu().numpy())
                saved_weight.append(weight.cpu().numpy())
                saved_mu.append(mu.cpu().numpy())
                saved_sd.append(sd.cpu().numpy())
        pack = {'indices': validation_rows, 'gate': np.concatenate(saved_gate), 'mixture_weight': np.concatenate(saved_weight),
                'mixture_mu': np.concatenate(saved_mu), 'mixture_std': np.concatenate(saved_sd),
                'regression_median': median, 'regression_scale': scale,
                'regression_transforms': np.array(['log1p', 'log10', 'log']), 'pred': pred[validation_rows]}
        for j, target in enumerate(TS):
            pack[target+'_prob'] = np.concatenate(saved_prob[j])
            pack[target+'_centers'] = centers[j]
        np.savez_compressed(out/f'fold_{f+1}_distributions.npz', **pack)
        fr = {'fold': f+1, 'scores': metric(y[validation_rows], pred[validation_rows]),
              'seconds': time.perf_counter()-fold_started, 'loss': history,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        reports.append(fr)
        write(out/f'fold_{f+1}_report.json', fr)
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config,
                    'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'ple_edges_normalized': edges.tolist(),
                    'medians': builder.medians_, 'enhanced_feature_names': builder.enhanced_feature_names_,
                    'selected_feature_names': chosen_names, 'selected_feature_indices': feature['selected_indices'],
                    'group_contract': grouped, 'regression_median': median.tolist(), 'regression_scale': scale.tolist(),
                    'centers': [c.tolist() for c in centers], 'density_edges': [e.tolist() for e in density_edges]}, out/f'fold_{f+1}.pt')
        np.save(cache, pred[validation_rows])
        np.savez_compressed(out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, scheduler, x, gates, targets, training_indices, kernels, pack
        del saved_prob, saved_weight, saved_mu, saved_sd, saved_gate
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert hash_file(__file__) == config['source_sha256']
    assert hash_file(inv.__file__) == config['invariant_helper_sha256']
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': reports,
              'seconds': time.perf_counter()-started,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(out/'report.json', report)
    print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'seconds']}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_curve_transformer')
    ap.add_argument('--cpu-smoke', action='store_true')
    ap.add_argument('--benchmark', action='store_true')
    args = ap.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(.95, 1.5*2**30/device.total_memory))
    if args.benchmark:
        benchmark(args.out)
    else:
        train(args.out)


if __name__ == '__main__':
    main()
