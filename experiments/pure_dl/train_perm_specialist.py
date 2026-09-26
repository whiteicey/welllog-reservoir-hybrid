"""Pure neural PERM specialist with an exact log-triangular utility decoder.

The only fitted target is PERM. POR/SW predictions come unchanged from a single
predeclared pure-neural OOF source, whose hash and row/fold identity are locked.
No external prediction is used as a feature, target, teacher or selection rule.
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
from src.feature import FeatureBuilder, well_extra_features
from src.model import CURVES, prep_well
from train import TS, metric, write

INPUTS = 450
MIX = 8
SIGMA_FLOOR = .02
DENSITY_BINS = 256


class PointBlock(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                 nn.Dropout(.05), nn.Linear(width, width))
        self.scale = nn.Parameter(torch.full((width,), .1))

    def forward(self, x):
        return x + self.scale*self.net(self.norm(x))


class PERMNetwork(nn.Module):
    def __init__(self, initial_means=None):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(INPUTS, 256), nn.GELU())
        self.blocks = nn.Sequential(*(PointBlock() for _ in range(4)))
        self.norm = nn.LayerNorm(256)
        self.gate = nn.Linear(256, 1)
        self.density = nn.Linear(256, 3*MIX)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        if initial_means is not None:
            with torch.no_grad():
                self.density.bias[MIX:2*MIX].copy_(torch.as_tensor(initial_means, dtype=torch.float32))
                self.density.bias[2*MIX:].fill_(float(np.log(np.expm1(.4-SIGMA_FLOOR))))

    def forward(self, x):
        h = self.norm(self.blocks(self.stem(x)))
        return self.gate(h).squeeze(-1), self.density(h).reshape(-1, 3, MIX)


def mixture_params(raw):
    raw = raw.float()
    return raw[:, 0].log_softmax(-1), raw[:, 1], F.softplus(raw[:, 2])+SIGMA_FLOOR


def source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangle_expectation(actions_log10, mu, sd):
    """Exact E[max(0, 1-|Z-action|)] for Gaussian Z in log10(PERM).

    Arguments are broadcast-compatible, including a component dimension. The
    two truncated first moments integrate the left/right halves of the triangle.
    """
    lo = (actions_log10-1-mu)/sd
    mid = (actions_log10-mu)/sd
    hi = (actions_log10+1-mu)/sd
    cdf_lo = .5*(1+torch.erf(lo/np.sqrt(2)))
    cdf_mid = .5*(1+torch.erf(mid/np.sqrt(2)))
    cdf_hi = .5*(1+torch.erf(hi/np.sqrt(2)))
    pdf_lo = torch.exp(-.5*lo.square())/np.sqrt(2*np.pi)
    pdf_mid = torch.exp(-.5*mid.square())/np.sqrt(2*np.pi)
    pdf_hi = torch.exp(-.5*hi.square())/np.sqrt(2*np.pi)
    left = cdf_mid-cdf_lo
    right = cdf_hi-cdf_mid
    utility = (1-actions_log10+mu)*left + (1+actions_log10-mu)*right
    utility = utility + sd*(pdf_lo-2*pdf_mid+pdf_hi)
    return utility.clamp(0, 1)


@torch.no_grad()
def mixture_to_bins(weight, mean_log10, std_log10, edges_log10):
    z = (edges_log10[None, :, None]-mean_log10[:, None, :])/std_log10[:, None, :]
    cdf = .5*(1+torch.erf(z/np.sqrt(2)))
    masses = (cdf[:, 1:]-cdf[:, :-1]).clamp_min(0)
    prob = (masses*weight[:, None, :]).sum(-1)
    return prob/prob.sum(-1, keepdim=True).clamp_min(1e-12)


def prepare_features(rec, train_wells, valid, row_ids):
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
        fs[c] = max(q[2]-q[0], 1e-3)
    gs = {}
    for c in CURVES:
        values = np.concatenate([rec[i][1][c].dropna().to_numpy() for i in train_wells])
        if c in ['RT', 'RXO']:
            values = np.log10(np.maximum(values, .01))
        q1, q3 = np.percentile(values, [25, 75])
        gs[c] = {'median': float(np.median(values)), 'iqr': max(float(q3-q1), 1e-3)}
    x = torch.empty((sum(len(r) for r in row_ids), INPUTS), device='cuda', dtype=torch.float16)
    for i, (_, frame) in enumerate(rec):
        raw = np.clip(prep_well(frame, gs), -8, 8).T
        static = np.clip((features[i]-fm)/fs, -8, 8)
        xx = np.column_stack([raw, static])[valid[i]].astype(np.float16)
        assert xx.shape == (len(row_ids[i]), INPUTS) and np.isfinite(xx).all()
        lo, hi = int(row_ids[i][0]), int(row_ids[i][-1])+1
        assert hi-lo == len(row_ids[i])
        x[lo:hi] = torch.from_numpy(xx).cuda()
    del features, values, mats, extra, raw, static, xx
    gc.collect()
    return x, builder, fm, fs, gs


def cpu_smoke():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    model = PERMNetwork(np.linspace(-1, 1, MIX))
    gate, raw = model(torch.randn(64, INPUTS))
    logw, mu, sd = mixture_params(raw)
    assert gate.shape == (64,) and raw.shape == (64, 3, MIX)
    targets = torch.randn(64, 1)
    nll = -torch.logsumexp(logw-.5*((targets-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi), -1)
    loss = nll.mean()+F.binary_cross_entropy_with_logits(gate, torch.randint(0, 2, (64,)).float())
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Independent numerical quadrature checks the analytic official-kernel integral.
    grid = torch.linspace(-12, 12, 200001, dtype=torch.float64)
    cases = [(-1.4, .3, -1.1), (.8, .7, -.2), (1.2, .08, 1.1), (-2., 1.5, 2.), (0., 4., -.5)]
    errors = []
    for mean, sigma, action in cases:
        density = torch.exp(-.5*((grid-mean)/sigma).square())/(sigma*np.sqrt(2*np.pi))
        numerical = torch.trapz((1-(grid-action).abs()).clamp_min(0)*density, grid)
        analytic = triangle_expectation(torch.tensor(action, dtype=torch.float64),
                                        torch.tensor(mean, dtype=torch.float64), torch.tensor(sigma, dtype=torch.float64))
        errors.append(float((numerical-analytic).abs()))
    assert max(errors) < 1e-6, errors
    print(json.dumps({'cpu_smoke': 'passed', 'parameters': sum(p.numel() for p in model.parameters()),
                      'max_analytic_integral_error': max(errors)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_perm_specialist')
    ap.add_argument('--baseline', type=Path, default=ROOT/'outputs/pure_dl_residual/analytic_800/oof.npz')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--batch-points', type=int, default=8192)
    ap.add_argument('--seed', type=int, default=20260927)
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
    torch.cuda.set_per_process_memory_fraction(min(.95, 2.5*2**30/device.total_memory))
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
    pred[:, 1] = np.nan
    config = {**vars(args), 'out': str(args.out), 'baseline': str(args.baseline),
              'source_sha256': source_hash(__file__), 'baseline_oof_sha256': baseline_hash,
              'parameters': sum(p.numel() for p in PERMNetwork().parameters()),
              'device': device.name, 'torch': torch.__version__, 'allocator_limit_gib': 2.5,
              'architecture': '450-input point MLP width256, 4 pre-LayerNorm residual blocks; independent PERM default gate and 8-Gaussian log10(PERM) mixture',
              'target': '(log10(PERM)-outer_training_nondefault_median)/outer_training_nondefault_IQR; no POR labels or predictions are model inputs',
              'loss': 'default BCE on all sampled training valid rows + Gaussian mixture NLL on sampled nondefault training rows',
              'optimizer': 'AdamW lr0.001 weight_decay0.01, cosine to0.00001, gradient_clip2',
              'sampling': '8192 independent uniform outer-training valid rows per step',
              'decode': 'exact analytic Gaussian convolution with the official log10 triangular kernel, plus learned default atom; 800 fixed actions from PERM0.01 to10000',
              'distribution_export': 'PERM-only arrays; 256 uniform log10 support points from0.01 to10000 with endpoint-folded CDF tails; exact mixture parameters also saved',
              'reported_method': 'PERM-specialist replacement; POR and SW unchanged from one SHA-locked pure-neural analytic MDN OOF artifact',
              'normalization': 'feature and target statistics fitted exclusively on 64 outer-training wells',
              'selection': 'one predeclared fixed-epoch configuration; no outer-fold calibration, checkpoint or row selection',
              'test_access': False, 'valid_rows': len(y),
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if well_fold[i] == f] for f in range(5)}}
    config_path = args.out/'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding='utf-8'))
        for key in ['source_sha256', 'baseline_oof_sha256', 'epochs', 'steps', 'batch_points', 'seed']:
            assert old[key] == config[key], f'Refusing incompatible resume: {key}'
    write(config_path, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k != 'outer_wells'}), flush=True)
    centers_log = np.linspace(-2, 4, DENSITY_BINS)
    centers = 10**centers_log
    edges_log = np.r_[-np.inf, (centers_log[:-1]+centers_log[1:])/2, np.inf]
    actions_log = np.linspace(-2, 4, 800)
    actions = 10**actions_log
    folds_report = []
    for f in range(5):
        fold_started = time.perf_counter()
        seed = args.seed+f
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for required in [f'fold_{f+1}_report.json', f'fold_{f+1}_distributions.npz', f'fold_{f+1}.pt']:
                assert (args.out/required).exists(), f'Incomplete fold cache: {required}'
            pred[fold == f] = np.load(cache)
            assert np.array_equal(pred[fold == f][:, [0, 2]], baseline_pred[fold == f][:, [0, 2]])
            folds_report.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()))
            continue
        train_wells = [i for i in range(80) if well_fold[i] != f]
        print(f'FOLD {f+1} preparing train-only feature and PERM target statistics', flush=True)
        x, builder, fm, fs, gs = prepare_features(rec, train_wells, valid, row_ids)
        train_idx = torch.tensor(np.flatnonzero(fold != f), dtype=torch.long, device='cuda')
        nondefault = y[:, 1] > .01
        log_target = np.log10(y[:, 1])
        train_target = log_target[(fold != f) & nondefault]
        q25, rmedian, q75 = np.percentile(train_target, [25, 50, 75])
        rscale = max(float(q75-q25), .1)
        normalized_target = (log_target-rmedian)/rscale
        z = torch.tensor(normalized_target, device='cuda', dtype=torch.float32)
        gate_labels = torch.tensor(~nondefault, device='cuda', dtype=torch.float32)
        initial_means = np.quantile((train_target-rmedian)/rscale, np.linspace(.05, .95, MIX))
        model = PERMNetwork(initial_means).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs*args.steps, eta_min=.00001)
        history = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            model.train()
            total = total_gate = total_nll = 0.
            for step in range(args.steps):
                rows = train_idx[torch.randint(len(train_idx), (args.batch_points,), device='cuda')]
                truth_gate = gate_labels[rows]
                target_z = z[rows]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gate_logits, raw = model(x[rows])
                logw, mean, sd = mixture_params(raw)
                logp = logw-.5*((target_z[:, None]-mean)/sd).square()-sd.log()-.5*np.log(2*np.pi)
                nll = -torch.logsumexp(logp, -1)
                active = truth_gate < .5
                loss_gate = F.binary_cross_entropy_with_logits(gate_logits.float(), truth_gate)
                loss_nll = (nll*active).sum()/active.sum().clamp_min(1)
                loss = loss_gate+loss_nll
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step()
                sched.step()
                total += float(loss.detach())
                total_gate += float(loss_gate.detach())
                total_nll += float(loss_nll.detach())
            history.append({'total': total/args.steps, 'gate': total_gate/args.steps, 'nll': total_nll/args.steps})
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/{args.epochs} loss={history[-1]["total"]:.4f} gate={history[-1]["gate"]:.4f} nll={history[-1]["nll"]:.4f} seconds={time.perf_counter()-fold_started:.1f}', flush=True)
        model.eval()
        validation_rows = np.flatnonzero(fold == f)
        action_tensor = torch.tensor(actions_log, device='cuda', dtype=torch.float32)[None, :, None]
        default_utility = (1-(action_tensor[:, :, 0]+2).abs()).clamp_min(0)
        edge_tensor = torch.tensor(edges_log, device='cuda', dtype=torch.float32)
        saved_gate, saved_prob, saved_weight, saved_mu, saved_sd = [], [], [], [], []
        with torch.no_grad():
            for start in range(0, len(validation_rows), 512):
                rows = validation_rows[start:start+512]
                rows_t = torch.tensor(rows, device='cuda', dtype=torch.long)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    gate_logits, raw = model(x[rows_t])
                logw, normalized_mu, normalized_sd = mixture_params(raw)
                weight = logw.exp()
                mu = normalized_mu*rscale+float(rmedian)
                sd = normalized_sd*rscale
                gate = gate_logits.float().sigmoid()
                expected = triangle_expectation(action_tensor, mu[:, None, :], sd[:, None, :])
                conditional_utility = (expected*weight[:, None, :]).sum(-1)
                utility = gate[:, None]*default_utility+(1-gate[:, None])*conditional_utility
                pred[rows, 1] = actions[utility.argmax(-1).cpu().numpy()]
                conditional = mixture_to_bins(weight, mu, sd, edge_tensor)
                saved_gate.append(gate.cpu().numpy())
                saved_prob.append(conditional.cpu().numpy().astype(np.float16))
                saved_weight.append(weight.cpu().numpy())
                saved_mu.append(mu.cpu().numpy())
                saved_sd.append(sd.cpu().numpy())
        assert np.array_equal(pred[validation_rows][:, [0, 2]], baseline_pred[validation_rows][:, [0, 2]])
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', indices=validation_rows,
                            PERM_gate=np.concatenate(saved_gate), PERM_prob=np.concatenate(saved_prob), PERM_centers=centers,
                            PERM_mixture_weight=np.concatenate(saved_weight), PERM_mixture_mean_log10=np.concatenate(saved_mu),
                            PERM_mixture_std_log10=np.concatenate(saved_sd), target_median=rmedian, target_scale=rscale,
                            actions=actions, pred=pred[validation_rows])
        fr = {'fold': f+1, 'scores': metric(y[validation_rows], pred[validation_rows]),
              'seconds': time.perf_counter()-fold_started, 'loss': history,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'baseline_POR_SW_unchanged': True}
        folds_report.append(fr)
        write(args.out/f'fold_{f+1}_report.json', fr)
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config,
                    'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'gstats': gs,
                    'medians': builder.medians_, 'target_median': float(rmedian), 'target_scale': rscale,
                    'enhanced_feature_names': builder.enhanced_feature_names_,
                    'centers_log10': centers_log.tolist(), 'actions': actions.tolist()}, args.out/f'fold_{f+1}.pt')
        np.save(cache, pred[validation_rows])
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, sched, x, z, gate_labels, train_idx, saved_prob, saved_weight, saved_mu, saved_sd
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert np.array_equal(pred[:, [0, 2]], baseline_pred[:, [0, 2]])
    assert source_hash(args.baseline) == baseline_hash, 'Baseline artifact changed during the run'
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': folds_report,
              'seconds': time.perf_counter()-started, 'PERM_specialist_score': metric(y, pred)['PERM'],
              'baseline_oof_scores': metric(y, baseline_pred), 'baseline_POR_SW_bitwise_unchanged': True,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(args.out/'report.json', report)
    print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'PERM_specialist_score', 'seconds', 'baseline_POR_SW_bitwise_unchanged']}), flush=True)


if __name__ == '__main__':
    main()
