"""Fixed five-fold POR specialist: input-anchored residual mixture-density MLP.

Only original train data is fitted. PERM/SW are copied from one predeclared,
already completed pure-neural analytic MDN OOF file, with exact row/fold checks
and a locked source hash. No truth-conditioned prediction combination is used.
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
sys.path.insert(0, str(ROOT / 'package'))
from src.data import load_split
from src.feature import FeatureBuilder, well_extra_features
from src.model import CURVES, prep_well
from src.utils import kernel_matrix
from train import metric, write, TS

MIX = 8
INPUTS = 450
DENSITY_BINS = 256
SIGMA_FLOOR = .02


class PointBlock(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                 nn.Dropout(.05), nn.Linear(width, width))
        self.scale = nn.Parameter(torch.full((width,), .1))

    def forward(self, x):
        return x + self.scale * self.net(self.norm(x))


class PORNetwork(nn.Module):
    def __init__(self, initial_means=None):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(INPUTS, 256), nn.GELU())
        self.blocks = nn.Sequential(*(PointBlock() for _ in range(4)))
        self.norm = nn.LayerNorm(256)
        self.gate = nn.Linear(256, 1)
        self.density = nn.Linear(256, 3 * MIX)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        if initial_means is not None:
            with torch.no_grad():
                self.density.bias[MIX:2*MIX].copy_(torch.as_tensor(initial_means, dtype=torch.float32))
                self.density.bias[2*MIX:].fill_(float(np.log(np.expm1(.4 - SIGMA_FLOOR))))

    def forward(self, x):
        h = self.norm(self.blocks(self.stem(x)))
        return self.gate(h).squeeze(-1), self.density(h).reshape(-1, 3, MIX)


def mixture_params(raw):
    raw = raw.float()
    return raw[:, 0].log_softmax(-1), raw[:, 1], F.softplus(raw[:, 2]) + SIGMA_FLOOR


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def mixture_to_bins(weight, mean_log1p, std_log1p, edges_log1p):
    z = (edges_log1p[None, :, None] - mean_log1p[:, None, :]) / std_log1p[:, None, :]
    cdf = .5 * (1 + torch.erf(z / np.sqrt(2)))
    masses = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0)
    prob = (masses * weight[:, None, :]).sum(-1)
    return prob / prob.sum(-1, keepdim=True).clamp_min(1e-12)


def prepare_features(rec, train_wells, valid, row_ids):
    """Only training wells contribute fitted population statistics."""
    builder = FeatureBuilder().fit([rec[i][1] for i in train_wells])
    features = []
    for _, frame in rec:
        mats = builder.transform(frame)
        extra = well_extra_features(frame, builder.medians_).to_numpy(np.float32)
        assert mats.enhanced.shape[1] == 237 and extra.shape[1] == 181
        features.append(np.column_stack([mats.enhanced, extra]))
    fm = np.empty(418, np.float32)
    fs = np.empty(418, np.float32)
    for c in range(418):
        values = np.concatenate([features[i][:, c] for i in train_wells])
        q = np.percentile(values, [25, 50, 75])
        fm[c] = q[1]
        fs[c] = max(q[2] - q[0], 1e-3)
    gs = {}
    for c in CURVES:
        values = np.concatenate([rec[i][1][c].dropna().to_numpy() for i in train_wells])
        if c in ['RT', 'RXO']:
            values = np.log10(np.maximum(values, .01))
        q1, q3 = np.percentile(values, [25, 75])
        gs[c] = {'median': float(np.median(values)), 'iqr': max(float(q3-q1), 1e-3)}
    n = sum(len(r) for r in row_ids)
    x = torch.empty((n, INPUTS), device='cuda', dtype=torch.float16)
    anchor = np.empty(n, np.float32)
    for i, (_, frame) in enumerate(rec):
        raw = np.clip(prep_well(frame, gs), -8, 8).T
        static = np.clip((features[i] - fm) / fs, -8, 8)
        xx = np.column_stack([raw, static])[valid[i]].astype(np.float16)
        assert xx.shape == (len(row_ids[i]), INPUTS) and np.isfinite(xx).all()
        lo, hi = int(row_ids[i][0]), int(row_ids[i][-1]) + 1
        assert hi-lo == len(row_ids[i])
        x[lo:hi] = torch.from_numpy(xx).cuda()
        den = frame['DEN'].interpolate(limit_direction='both').fillna(builder.medians_['DEN']).to_numpy()
        density_porosity = np.clip((2.65-den)/1.65*100, .1, 50)
        anchor[lo:hi] = np.log1p(density_porosity[valid[i]])
    del features, values, mats, extra, raw, static, xx
    gc.collect()
    return x, anchor, builder, fm, fs, gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_por_specialist')
    ap.add_argument('--baseline', type=Path, default=ROOT/'outputs/pure_dl_residual/analytic_800/oof.npz')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--batch-points', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=20260926)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    started = time.perf_counter()
    rec = load_split(ROOT.parent/'data', 'train')
    assert len(rec) == 80
    ys = [frame[TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(values).all(1) for values in ys]
    y = np.concatenate([values[mask] for values, mask in zip(ys, valid)])
    well = np.concatenate([np.full(mask.sum(), i, np.int32) for i, mask in enumerate(valid)])
    assert y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, rows) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[rows] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    well_fold = {i: int(fold[row_ids[i][0]]) for i in range(80)}

    baseline_hash = source_hash(args.baseline)
    baseline = np.load(args.baseline)
    for name, expected in [('truth', y), ('fold', fold), ('well', well)]:
        assert np.array_equal(baseline[name], expected), name
    baseline_pred = baseline['pred'].copy()
    assert baseline_pred.shape == y.shape and np.isfinite(baseline_pred).all()
    pred = baseline_pred.copy()
    pred[:, 0] = np.nan
    source_sha = source_hash(__file__)
    config = {**vars(args), 'out': str(args.out), 'baseline': str(args.baseline),
              'baseline_oof_sha256': baseline_hash,
              'source_sha256': source_sha,
              'parameters': sum(p.numel() for p in PORNetwork().parameters()),
              'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'architecture': '450-input point MLP width256 with 4 pre-LayerNorm residual blocks; independent POR default gate and 8-component anchored residual Gaussian mixture',
              'anchor': 'log1p(clip((2.65-DEN)/1.65*100,0.1,50)); input-only density proxy, not a label assumption',
              'target': '(log1p(POR)-anchor-train_residual_median)/train_residual_IQR on training nondefault rows',
              'loss': 'BCE POR default gate on all training valid rows + MDN NLL on training nondefault rows; no auxiliary loss',
              'optimizer': 'AdamW lr0.001 weight_decay0.01, cosine to0.00001, gradient_clip2',
              'sampling': '8192 independent uniformly sampled outer-training valid rows/update; point network uses precomputed within-well context features',
              'decode': '256 fixed log1p support points spanning POR0.1..50, infinite CDF tails at endpoints; 800 geometric POR actions including0.1; exact official kernel and learned default atom',
              'reported_method': 'POR-specialist replacement: PERM and SW are unchanged predictions from the one locked pure-neural analytic MDN OOF source',
              'normalization': 'all fitted feature statistics and residual statistics use outer-training wells only',
              'selection': 'one predeclared configuration, fixed final epoch, no outer-fold early stopping/calibration/selection',
              'test_access': False, 'valid_rows': len(y),
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if well_fold[i] == f] for f in range(5)}}
    existing_config = args.out/'config.json'
    if existing_config.exists():
        old = json.loads(existing_config.read_text(encoding='utf-8'))
        for key in ['source_sha256', 'baseline_oof_sha256', 'epochs', 'steps', 'batch_points', 'seed']:
            assert old[key] == config[key], f'Refusing incompatible resume: {key}'
    write(existing_config, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k != 'outer_wells'}), flush=True)

    centers_z = np.linspace(np.log1p(.1), np.log1p(50), DENSITY_BINS)
    centers = np.expm1(centers_z)
    edges_z = np.r_[-np.inf, (centers_z[:-1]+centers_z[1:])/2, np.inf]
    actions = np.geomspace(.1, 50, 800)
    folds_report = []
    for f in range(5):
        fold_started = time.perf_counter()
        seed = args.seed + f
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for required in [f'fold_{f+1}_report.json', f'fold_{f+1}_distributions.npz', f'fold_{f+1}.pt']:
                assert (args.out/required).exists(), f'Incomplete fold cache: {required}'
            pred[fold == f] = np.load(cache)
            assert np.array_equal(pred[fold == f, 1:], baseline_pred[fold == f, 1:])
            folds_report.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()))
            continue
        train_wells = [i for i in range(80) if well_fold[i] != f]
        print(f'FOLD {f+1} preparing train-only statistics', flush=True)
        x, anchor, builder, fm, fs, gs = prepare_features(rec, train_wells, valid, row_ids)
        train_idx_np = np.flatnonzero(fold != f)
        train_idx = torch.tensor(train_idx_np, dtype=torch.long, device='cuda')
        nondefault = y[:, 0] > .1
        residual = np.log1p(y[:, 0]) - anchor
        training_residual = residual[(fold != f) & nondefault]
        q25, rmedian, q75 = np.percentile(training_residual, [25, 50, 75])
        rscale = max(float(q75-q25), .1)
        z = torch.tensor((residual-rmedian)/rscale, device='cuda', dtype=torch.float32)
        gate_labels = torch.tensor(~nondefault, device='cuda', dtype=torch.float32)
        initial_means = np.quantile((training_residual-rmedian)/rscale, np.linspace(.05, .95, MIX))
        model = PORNetwork(initial_means).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs*args.steps, eta_min=.00001)
        history = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            model.train(); total = total_gate = total_nll = 0.
            for step in range(args.steps):
                batch_rows = train_idx[torch.randint(len(train_idx), (args.batch_points,), device='cuda')]
                xx = x[batch_rows]; truth_gate = gate_labels[batch_rows]; target_z = z[batch_rows]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gate_logits, raw = model(xx)
                logw, mean, sd = mixture_params(raw)
                active = truth_gate < .5
                logp = logw - .5*((target_z[:, None]-mean)/sd).square() - sd.log() - .5*np.log(2*np.pi)
                nll = -torch.logsumexp(logp, -1)
                loss_gate = F.binary_cross_entropy_with_logits(gate_logits.float(), truth_gate)
                loss_nll = nll[active].mean()
                loss = loss_gate + loss_nll
                assert torch.isfinite(loss)
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step(); sched.step()
                total += float(loss.detach()); total_gate += float(loss_gate.detach()); total_nll += float(loss_nll.detach())
            history.append({'total': total/args.steps, 'gate': total_gate/args.steps, 'nll': total_nll/args.steps})
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/{args.epochs} loss={history[-1]["total"]:.4f} gate={history[-1]["gate"]:.4f} nll={history[-1]["nll"]:.4f} seconds={time.perf_counter()-fold_started:.1f}', flush=True)
        model.eval()
        validation_rows = np.flatnonzero(fold == f)
        kernel = torch.tensor(kernel_matrix(actions, np.r_[.1, centers], 'POR').T, dtype=torch.float32, device='cuda')
        edge_tensor = torch.tensor(edges_z, device='cuda', dtype=torch.float32)
        saved_gate = []; saved_prob = []; saved_weight = []; saved_mu = []; saved_sd = []
        with torch.no_grad():
            for start in range(0, len(validation_rows), 1024):
                rows = validation_rows[start:start+1024]
                rows_t = torch.tensor(rows, device='cuda', dtype=torch.long)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gate_logits, raw = model(x[rows_t])
                logw, normalized_mu, normalized_sd = mixture_params(raw)
                weight = logw.exp()
                mu = normalized_mu*rscale + float(rmedian) + torch.tensor(anchor[rows, None], device='cuda')
                sd = normalized_sd*rscale
                gate = gate_logits.float().sigmoid()
                conditional = mixture_to_bins(weight, mu, sd, edge_tensor)
                probability = torch.cat([gate[:, None], (1-gate[:, None])*conditional], 1)
                pred[rows, 0] = actions[(probability@kernel).argmax(1).cpu().numpy()]
                saved_gate.append(gate.cpu().numpy())
                saved_prob.append(conditional.cpu().numpy().astype(np.float16))
                saved_weight.append(weight.cpu().numpy())
                saved_mu.append(mu.cpu().numpy())
                saved_sd.append(sd.cpu().numpy())
        assert np.array_equal(pred[validation_rows, 1:], baseline_pred[validation_rows, 1:])
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', indices=validation_rows,
                            POR_gate=np.concatenate(saved_gate), POR_prob=np.concatenate(saved_prob), POR_centers=centers,
                            POR_mixture_weight=np.concatenate(saved_weight), POR_mixture_mean_log1p=np.concatenate(saved_mu),
                            POR_mixture_std_log1p=np.concatenate(saved_sd), anchor_log1p=anchor[validation_rows],
                            residual_median=rmedian, residual_scale=rscale, actions=actions, pred=pred[validation_rows])
        fr = {'fold': f+1, 'scores': metric(y[validation_rows], pred[validation_rows]),
              'seconds': time.perf_counter()-fold_started, 'loss': history,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'baseline_PERM_SW_unchanged': True}
        folds_report.append(fr)
        write(args.out/f'fold_{f+1}_report.json', fr)
        np.save(cache, pred[validation_rows])
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config,
                    'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'gstats': gs,
                    'medians': builder.medians_, 'residual_median': float(rmedian), 'residual_scale': rscale,
                    'enhanced_feature_names': builder.enhanced_feature_names_,
                    'centers_log1p': centers_z.tolist(), 'actions': actions.tolist()}, args.out/f'fold_{f+1}.pt')
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, sched, x, z, gate_labels, train_idx, kernel, saved_prob, saved_weight, saved_mu, saved_sd
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert np.array_equal(pred[:, 1:], baseline_pred[:, 1:])
    assert source_hash(args.baseline) == baseline_hash, 'Baseline artifact changed during this experiment'
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': folds_report,
              'seconds': time.perf_counter()-started,
              'POR_specialist_score': metric(y, pred)['POR'],
              'baseline_oof_scores': metric(y, baseline_pred),
              'baseline_PERM_SW_bitwise_unchanged': True,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(args.out/'report.json', report)
    print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'POR_specialist_score', 'seconds', 'baseline_PERM_SW_bitwise_unchanged']}), flush=True)


if __name__ == '__main__':
    main()
