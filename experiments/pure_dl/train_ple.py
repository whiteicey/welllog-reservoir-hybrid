"""Fold-local piecewise-linear numeric encoding + residual neural MDN.

--benchmark runs synthetic GPU updates only and never reads real data. Full
training is a separate explicit invocation. PLE knots use training inputs only;
labels never enter the numeric encoding. No tree/test/v3 artifacts are used.
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
from src.feature import FeatureBuilder
from src.utils import BOUNDS, DEFAULTS, kernel_matrix
from train import metric, write, dmask, TS

FEATURES = 237
SEGMENTS = 16
MIX = 8
DENSITY_BINS = 256
TRANSFORM = [np.log1p, np.log10, np.log]
INVERSE = [np.expm1, lambda z: 10**z, np.exp]
SIGMA_FLOOR = .025
LOGICAL_ENCODED = FEATURES*(SEGMENTS+1)
PADDED_ENCODED = ((LOGICAL_ENCODED+7)//8)*8


class NumericEncoding(nn.Module):
    def __init__(self, edges):
        super().__init__()
        edges = torch.as_tensor(edges, dtype=torch.float32)
        assert edges.shape == (FEATURES, SEGMENTS+1)
        width = edges[:, 1:] - edges[:, :-1]
        self.register_buffer('left', edges[:, :-1])
        self.register_buffer('inverse_width', width.clamp_min(1e-6).reciprocal())
        self.register_buffer('valid_segment', (width > 1e-6).float())

    def forward(self, x):
        raw = x.float()
        encoded = ((raw[:, :, None]-self.left)*self.inverse_width).clamp(0, 1)
        encoded = encoded*self.valid_segment
        combined = torch.cat([raw, encoded.flatten(1)], 1)
        return F.pad(combined, (0, PADDED_ENCODED-LOGICAL_ENCODED))


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.net = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(.05), nn.Linear(256, 256))
        self.scale = nn.Parameter(torch.full((256,), .1))

    def forward(self, x):
        return x + self.scale*self.net(self.norm(x))


class Network(nn.Module):
    def __init__(self, edges=None, initial_means=None):
        super().__init__()
        if edges is None:
            edges = np.tile(np.linspace(-3, 3, SEGMENTS+1), (FEATURES, 1))
        self.encoding = NumericEncoding(edges)
        self.stem = nn.Sequential(nn.Linear(PADDED_ENCODED, 256), nn.GELU())
        self.blocks = nn.Sequential(*(ResidualBlock() for _ in range(3)))
        self.norm = nn.LayerNorm(256)
        self.gate = nn.Linear(256, 3)
        self.density = nn.Linear(256, 3*3*MIX)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        with torch.no_grad():
            bias = self.density.bias.view(3, 3, MIX)
            if initial_means is not None:
                bias[:, 1].copy_(torch.as_tensor(initial_means, dtype=torch.float32))
            bias[:, 2].fill_(float(np.log(np.expm1(.5-SIGMA_FLOOR))))

    def forward(self, x):
        h = self.norm(self.blocks(self.stem(self.encoding(x))))
        return self.gate(h), self.density(h).reshape(-1, 3, 3, MIX)


def mixture_params(raw):
    raw = raw.float()
    return raw[:, :, 0].log_softmax(-1), raw[:, :, 1], F.softplus(raw[:, :, 2])+SIGMA_FLOOR


def training_step(model, opt, x, gates, targets, training_indices, batch_points):
    rows = training_indices[torch.randint(len(training_indices), (batch_points,), device='cuda')]
    target_gate = gates[rows]
    target_z = targets[rows]
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        gate_logits, raw = model(x[rows])
    logw, mu, sd = mixture_params(raw)
    logp = logw-.5*((target_z[:, :, None]-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi)
    nll = -torch.logsumexp(logp, -1)
    gate_loss = F.binary_cross_entropy_with_logits(gate_logits.float(), target_gate)
    density_loss = sum(nll[:, j][target_gate[:, j] < .5].mean() for j in range(3))/3
    loss = gate_loss+density_loss
    assert torch.isfinite(loss)
    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2)
    opt.step()
    return float(loss.detach()), float(gate_loss.detach()), float(density_loss.detach())


def benchmark(args):
    """Real forward/backward/optimizer updates with synthetic data, no train/test reads."""
    torch.manual_seed(args.seed)
    model = Network(initial_means=np.tile(np.linspace(-2, 2, MIX), (3, 1))).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x = torch.randn((723556, FEATURES), device='cuda', dtype=torch.float16).clamp(-8, 8)
    gates = (torch.rand((723556, 3), device='cuda') < .7).float()
    targets = torch.randn((723556, 3), device='cuda')
    rows = torch.arange(578000, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(8):
        training_step(model, opt, x, gates, targets, rows, args.batch_points)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(40):
        training_step(model, opt, x, gates, targets, rows, args.batch_points)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    step_seconds = seconds/40
    gpu_projection = step_seconds*args.epochs*args.steps*5
    result = {'benchmark_only': True, 'real_data_read': False, 'synthetic_updates': 48,
              'timed_updates': 40, 'batch_points': args.batch_points,
              'features': FEATURES, 'segments': SEGMENTS,
              'encoded_dimensions_with_raw': FEATURES*(SEGMENTS+1),
              'matrix_dimensions_including_zero_padding': PADDED_ENCODED,
              'parameters': sum(p.numel() for p in model.parameters()),
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'mean_step_ms': step_seconds*1000, 'timed_seconds': seconds,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'planned_epochs': args.epochs, 'planned_steps': args.steps,
              'projected_five_fold_update_seconds': gpu_projection,
              'projected_total_seconds_range': [gpu_projection+145, gpu_projection+220],
              'projection_note': 'Measured updates plus estimated 20-35 seconds/fold for input preprocessing and 45 seconds total inference/artifact I/O. Other processes may share GPU/CPU; this is not a completed training runtime.'}
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out/'benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)


def prepare_features(rec, train_wells, valid, row_ids):
    builder = FeatureBuilder().fit([rec[i][1] for i in train_wells])
    features = [builder.transform(frame).enhanced for _, frame in rec]
    fm = np.empty(FEATURES, np.float32); fs = np.empty(FEATURES, np.float32)
    edges = np.empty((FEATURES, SEGMENTS+1), np.float32)
    for c in range(FEATURES):
        values = np.concatenate([features[i][:, c] for i in train_wells])
        quantiles = np.quantile(values, np.linspace(0, 1, SEGMENTS+1))
        fm[c] = quantiles[SEGMENTS//2]
        fs[c] = max(quantiles[3*SEGMENTS//4]-quantiles[SEGMENTS//4], 1e-3)
        edges[c] = np.clip((quantiles-fm[c])/fs[c], -8, 8)
    n = sum(len(rows) for rows in row_ids)
    x = torch.empty((n, FEATURES), dtype=torch.float16, device='cuda')
    for i, feature in enumerate(features):
        xx = np.clip((feature[valid[i]]-fm)/fs, -8, 8).astype(np.float16)
        assert xx.shape == (len(row_ids[i]), FEATURES) and np.isfinite(xx).all()
        lo, hi = int(row_ids[i][0]), int(row_ids[i][-1])+1
        assert hi-lo == len(row_ids[i])
        x[lo:hi] = torch.from_numpy(xx).cuda()
    del features, values, feature, xx
    gc.collect()
    return x, builder, fm, fs, edges


@torch.no_grad()
def conditional_probability(weight, mean, std, edges):
    z = (edges[None, :, None]-mean[:, None, :])/std[:, None, :]
    cdf = .5*(1+torch.erf(z/np.sqrt(2)))
    prob = ((cdf[:, 1:]-cdf[:, :-1]).clamp_min(0)*weight[:, None, :]).sum(-1)
    return prob/prob.sum(1, keepdim=True).clamp_min(1e-12)


def train(args):
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    rec = load_split(ROOT.parent/'data', 'train')
    assert len(rec) == 80
    ys = [frame[TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    y = np.concatenate([v[mask] for v, mask in zip(ys, valid)])
    well = np.concatenate([np.full(mask.sum(), i, np.int32) for i, mask in enumerate(valid)])
    assert y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[ix] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    well_fold = {i: int(fold[row_ids[i][0]]) for i in range(80)}
    config = {**vars(args), 'out': str(args.out),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'parameters': sum(p.numel() for p in Network().parameters()),
              'architecture': '237 enhanced inputs; 16-segment input-only quantile PLE + raw robust channels (4029); width256 3-block residual MLP; independent gates and 8-Gaussian target heads',
              'encoding_fit': 'FeatureBuilder fallback medians, robust scales and all 17 PLE knots use outer-training inputs only; no labels in encoding',
              'target_transforms': ['log1p', 'log10', 'log'],
              'target_statistics_fit': 'outer-training nondefault labels only',
              'loss': 'mean independent gate BCE + mean target nondefault Gaussian-mixture NLL',
              'decode': 'fixed 256 transformed support centers and 800 actions/target, expected official utility with default atom',
              'selection': 'one predeclared configuration; fixed final epoch; no outer-validation hyperparameter, threshold, decoder or checkpoint selection',
              'test_access': False, 'valid_rows': len(y),
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if well_fold[i] == f] for f in range(5)}}
    config_path = args.out/'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ['source_sha256', 'epochs', 'steps', 'batch_points', 'seed']:
            assert old[key] == config[key], f'Incompatible cached configuration: {key}'
    write(config_path, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k != 'outer_wells'}), flush=True)
    pred = np.full_like(y, np.nan); reports = []; default = dmask(y)
    for f in range(5):
        fold_started = time.perf_counter()
        seed = args.seed+f; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for required in [f'fold_{f+1}_report.json', f'fold_{f+1}_distributions.npz', f'fold_{f+1}.pt']:
                assert (args.out/required).exists(), required
            pred[fold == f] = np.load(cache)
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()))
            continue
        train_wells = [i for i in range(80) if well_fold[i] != f]
        print(f'FOLD {f+1} fitting input-only PLE and train-only target statistics', flush=True)
        x, builder, fm, fs, ple_edges = prepare_features(rec, train_wells, valid, row_ids)
        med = []; scale = []; init = []; centers = []; density_edges = []; z = []
        for j in range(3):
            transformed = TRANSFORM[j](y[:, j])
            training = transformed[(fold != f) & ~default[:, j]]
            q25, q50, q75 = np.percentile(training, [25, 50, 75])
            sc = max(float(q75-q25), .1)
            med.append(q50); scale.append(sc); z.append((transformed-q50)/sc)
            init.append(np.quantile((training-q50)/sc, np.linspace(.05, .95, MIX)))
            ct = np.linspace(training.min(), training.max(), DENSITY_BINS)
            centers.append(INVERSE[j](ct))
            density_edges.append((np.r_[-np.inf, (ct[:-1]+ct[1:])/2, np.inf]-q50)/sc)
        med = np.array(med); scale = np.array(scale)
        targets = torch.tensor(np.column_stack(z), dtype=torch.float32, device='cuda')
        gates = torch.tensor(default, dtype=torch.float32, device='cuda')
        training_indices = torch.tensor(np.flatnonzero(fold != f), device='cuda', dtype=torch.long)
        model = Network(ple_edges, np.stack(init)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs*args.steps, eta_min=.00001)
        history = []; torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            model.train(); losses = np.zeros(3)
            for step in range(args.steps):
                losses += training_step(model, opt, x, gates, targets, training_indices, args.batch_points)
                scheduler.step()
            history.append((losses/args.steps).tolist())
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/{args.epochs} loss={history[-1][0]:.4f} gate={history[-1][1]:.4f} nll={history[-1][2]:.4f} seconds={time.perf_counter()-fold_started:.1f}', flush=True)
        model.eval(); actions = []; kernels = []
        for j, target in enumerate(TS):
            support = centers[j]
            action = np.linspace(support[0], support[-1], 800) if j == 2 else np.geomspace(max(support[0]*.9, .001), support[-1]*1.1, 800)
            action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
            actions.append(action)
            kernels.append(torch.tensor(kernel_matrix(action, np.r_[DEFAULTS[target], support], target).T, dtype=torch.float32, device='cuda'))
        validation_rows = np.flatnonzero(fold == f)
        saved_gate = []; saved_prob = [[], [], []]; saved_weight = []; saved_mu = []; saved_sd = []
        with torch.no_grad():
            for start in range(0, len(validation_rows), 1024):
                rows = validation_rows[start:start+1024]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    go, raw = model(x[torch.tensor(rows, device='cuda')])
                gate = go.float().sigmoid(); logw, mu, sd = mixture_params(raw); weight = logw.exp()
                for j in range(3):
                    edge = torch.tensor(density_edges[j], device='cuda', dtype=torch.float32)
                    cp = conditional_probability(weight[:, j], mu[:, j], sd[:, j], edge)
                    prob = torch.cat([gate[:, j, None], (1-gate[:, j, None])*cp], 1)
                    pred[rows, j] = actions[j][(prob@kernels[j]).argmax(1).cpu().numpy()]
                    saved_prob[j].append(cp.cpu().numpy().astype(np.float16))
                saved_gate.append(gate.cpu().numpy()); saved_weight.append(weight.cpu().numpy())
                saved_mu.append(mu.cpu().numpy()); saved_sd.append(sd.cpu().numpy())
        pack = {'indices': validation_rows, 'gate': np.concatenate(saved_gate),
                'mixture_weight': np.concatenate(saved_weight), 'mixture_mu': np.concatenate(saved_mu),
                'mixture_std': np.concatenate(saved_sd), 'regression_median': med, 'regression_scale': scale,
                'regression_transforms': np.array(['log1p', 'log10', 'log']), 'pred': pred[validation_rows]}
        for j, target in enumerate(TS):
            pack[target+'_prob'] = np.concatenate(saved_prob[j]); pack[target+'_centers'] = centers[j]
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **pack)
        fr = {'fold': f+1, 'scores': metric(y[validation_rows], pred[validation_rows]),
              'seconds': time.perf_counter()-fold_started, 'loss': history,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20}
        reports.append(fr); write(args.out/f'fold_{f+1}_report.json', fr); np.save(cache, pred[validation_rows])
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config, 'feature_median': fm.tolist(),
                    'feature_scale': fs.tolist(), 'ple_edges_normalized': ple_edges.tolist(),
                    'medians': builder.medians_, 'enhanced_feature_names': builder.enhanced_feature_names_,
                    'regression_median': med.tolist(), 'regression_scale': scale.tolist(),
                    'centers': [ct.tolist() for ct in centers], 'density_edges': [ed.tolist() for ed in density_edges]}, args.out/f'fold_{f+1}.pt')
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, scheduler, x, gates, targets, training_indices, kernels, pack, saved_prob, saved_weight, saved_mu, saved_sd
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': reports,
              'seconds': time.perf_counter()-started,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(args.out/'report.json', report)
    print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_ple')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--batch-points', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(3); torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    if args.benchmark:
        benchmark(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
