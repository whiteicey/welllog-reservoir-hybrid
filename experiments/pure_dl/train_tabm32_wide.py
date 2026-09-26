"""TabM-inspired grouped learned PLE embeddings and32-member shared MLP.

Each member has trainable rank-one input/output modulation in every shared
linear layer and private gate/density heads. No OOF-based member selection.
"""
from pathlib import Path
import argparse, gc, hashlib, json, math, random, sys, time
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'package'))
from src.data import load_split
from src.utils import DEFAULTS, BOUNDS, kernel_matrix
from train import metric, write, dmask, TS
from train_invariant import (prepare_features, NumericEncoding, FEATURES, PADDED_ENCODED,
                             TRANSFORM, INVERSE, DENSITY_BINS, SIGMA_FLOOR, feature_contract)

MEMBERS = 32
WIDTH = 512
MIX = 8
DEPENDENCIES = ['experiments/pure_dl/train_tabm32_wide.py',
                'experiments/pure_dl/train_invariant.py', 'experiments/pure_dl/train.py',
                'package/src/data.py', 'package/src/feature.py',
                'package/src/model.py', 'package/src/utils.py']


def hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in DEPENDENCIES}


class BELinear(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(no, ni))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.input_gain = nn.Parameter(torch.ones(MEMBERS, ni))
        self.output_gain = nn.Parameter(torch.ones(MEMBERS, no))
        self.bias = nn.Parameter(torch.zeros(MEMBERS, no))

    def forward(self, x, member):
        return F.linear(x*self.input_gain[member], self.weight)*self.output_gain[member]+self.bias[member]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.first = BELinear(256, 256)
        self.second = BELinear(256, 256)
        self.dropout = nn.Dropout(.05)
        self.scale = nn.Parameter(torch.full((256,), .1))

    def forward(self, x, member):
        h = self.dropout(F.gelu(self.first(self.norm(x), member)))
        return x+self.scale*self.second(h, member)


class PrivateHeads(nn.Module):
    def __init__(self, initial_means=None):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty(MEMBERS, WIDTH, 3))
        self.gate_bias = nn.Parameter(torch.zeros(MEMBERS, 1, 3))
        nn.init.normal_(self.gate_weight, 0, .03)
        self.density_weight = nn.Parameter(torch.empty(MEMBERS, WIDTH, 72))
        nn.init.normal_(self.density_weight, 0, .005)
        self.density_bias = nn.Parameter(torch.zeros(MEMBERS, 1, 72))
        with torch.no_grad():
            bias = self.density_bias.view(MEMBERS, 3, 3, MIX)
            if initial_means is not None:
                bias[:, :, 1].copy_(torch.as_tensor(initial_means, dtype=torch.float32)[None])
            bias[:, :, 2].fill_(float(np.log(np.expm1(.5-SIGMA_FLOOR))))

    def forward(self, h):
        gate = torch.bmm(h, self.gate_weight)+self.gate_bias
        density = torch.bmm(h, self.density_weight)+self.density_bias
        return gate, density.reshape(MEMBERS, h.shape[1], 3, 3, MIX)


class LearnedEncoding(nn.Module):
    def __init__(self, edges):
        super().__init__()
        self.base = NumericEncoding(edges)
        self.weight = nn.Parameter(torch.randn(FEATURES, 17, 8)/np.sqrt(17))
        self.bias = nn.Parameter(torch.zeros(FEATURES, 8))

    def forward(self, x):
        ple = self.base(x)[:, :FEATURES*17].reshape(len(x), FEATURES, 17)
        return F.relu(torch.einsum('nfi,fij->nfj', ple, self.weight)+self.bias).flatten(1)


class Network(nn.Module):
    def __init__(self, edges=None, initial_means=None):
        super().__init__()
        if edges is None:
            edges = np.tile(np.linspace(-3, 3, 17), (FEATURES, 1))
        self.encoding = LearnedEncoding(edges)
        self.stem = BELinear(FEATURES*8, WIDTH)
        with torch.no_grad():
            grouped = torch.randn(MEMBERS, FEATURES).repeat_interleave(8, dim=1)
            self.stem.input_gain.copy_(grouped)
        self.hidden = nn.ModuleList([BELinear(WIDTH, WIDTH) for _ in range(2)])
        self.dropout = nn.Dropout(.1)
        self.heads = PrivateHeads(initial_means)

    def forward(self, x):
        assert x.ndim == 3 and x.shape[0] == MEMBERS
        batch = x.shape[1]
        member = torch.arange(MEMBERS, device=x.device).repeat_interleave(batch)
        h = self.dropout(F.relu(self.stem(self.encoding(x.reshape(-1, FEATURES)), member)))
        for layer in self.hidden:
            h = self.dropout(F.relu(layer(h, member)))
        return self.heads(h.reshape(MEMBERS, batch, WIDTH))


def mixture_params(raw):
    raw = raw.float()
    return raw[:, :, :, 0].log_softmax(-1), raw[:, :, :, 1], F.softplus(raw[:, :, :, 2])+SIGMA_FLOOR


def step(model, opt, x, gates, targets, training_indices, batch):
    rows = training_indices[torch.randint(len(training_indices), (MEMBERS, batch//MEMBERS), device='cuda')]
    target_gate = gates[rows]; target_z = targets[rows]
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        go, raw = model(x[rows])
    logw, mu, sd = mixture_params(raw)
    logp = logw-.5*((target_z[:, :, :, None]-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi)
    nll = -torch.logsumexp(logp, -1)
    active = target_gate < .5
    per_member_target_nll = (nll*active).sum(1)/active.sum(1).clamp_min(1)
    density_loss = per_member_target_nll.mean()
    gate_loss = F.binary_cross_entropy_with_logits(go.float(), target_gate)
    loss = gate_loss+density_loss
    assert torch.isfinite(loss)
    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2)
    opt.step()
    return float(loss.detach()), float(gate_loss.detach()), float(density_loss.detach())


@torch.no_grad()
def joint_average(gates, conditional):
    """(members, rows), (members, rows, bins) -> compatible exact joint mean."""
    atom = gates.mean(0)
    mass = ((1-gates[:, :, None])*conditional).mean(0)
    full = torch.cat([atom[:, None], mass], 1)
    full = full/full.sum(1, keepdim=True).clamp_min(1e-12)
    nondefault = full[:, 1:].sum(1, keepdim=True)
    fallback = conditional.mean(0)
    fallback = fallback/fallback.sum(1, keepdim=True).clamp_min(1e-12)
    cp = torch.where(nondefault > 0, full[:, 1:]/nondefault.clamp_min(1e-12), fallback)
    return full[:, 0], cp, full


def self_test():
    linear = BELinear(2, 3)
    values = torch.tensor([[1., 2.], [-.5, .3]])
    actual = linear(values, torch.tensor([0, 1]))
    expected_linear = torch.stack([F.linear(values[i], linear.weight*linear.output_gain[i, :, None]*linear.input_gain[i, None, :])+linear.bias[i] for i in range(2)])
    assert torch.allclose(actual, expected_linear, atol=1e-6)
    gates = torch.tensor([[.9], [.8], [.2], [.1]])
    cp = torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 1.]], [[0., 1.]]])
    g, p, full = joint_average(gates, cp)
    expected = torch.tensor([[.5, .075, .425]])
    assert torch.allclose(full, expected, atol=1e-7)
    assert not torch.allclose(full[:, 1:], (1-g[:, None])*cp.mean(0))
    assert torch.allclose(torch.cat([g[:, None], (1-g[:, None])*p], 1), full, atol=1e-7)
    g, p, full = joint_average(torch.ones_like(gates), cp)
    assert torch.isfinite(p).all() and torch.allclose(p.sum(1), torch.ones(1))
    model = Network(initial_means=np.tile(np.linspace(-2, 2, MIX), (3, 1)))
    go, raw = model(torch.randn(MEMBERS, 3, FEATURES))
    logw, mu, sd = mixture_params(raw)
    assert go.shape == (MEMBERS, 3, 3) and mu.shape == (MEMBERS, 3, 3, MIX)
    assert torch.isfinite(logw).all() and torch.all(sd > 0)
    return {'passed': True, 'joint_counterexample': expected.tolist(),
            'rank_one_member_weights_match_explicit_matrices': True,
            'separate_gate_and_conditional_averaging_rejected': True,
            'all_default_fallback_finite': True, 'member_shapes_checked': True}


def benchmark(args):
    checks = self_test(); contract = feature_contract(); torch.manual_seed(args.seed)
    model = Network(initial_means=np.tile(np.linspace(-2, 2, MIX), (3, 1))).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x = torch.randn((723556, FEATURES), device='cuda', dtype=torch.float16).clamp(-8, 8)
    gates = (torch.rand((723556, 3), device='cuda') < .7).float()
    targets = torch.randn((723556, 3), device='cuda')
    rows = torch.arange(578000, device='cuda')
    torch.cuda.reset_peak_memory_stats()
    for _ in range(8): step(model, opt, x, gates, targets, rows, args.batch_points)
    torch.cuda.synchronize(); start = time.perf_counter()
    for _ in range(40): step(model, opt, x, gates, targets, rows, args.batch_points)
    torch.cuda.synchronize(); sec = (time.perf_counter()-start)/40
    result = {'benchmark_only': True, 'real_data_read': False, 'self_test': checks,
              'features': FEATURES, 'whitelist_invariance_check': contract['synthetic_invariance_check'],
              'members': MEMBERS, 'batch_total': args.batch_points,
              'batch_per_member': args.batch_points//MEMBERS,
              'parameters': sum(p.numel() for p in model.parameters()),
              'mean_step_ms': sec*1000, 'five_fold_update_seconds': sec*args.epochs*args.steps*5,
              'estimated_total_seconds_range': [sec*args.epochs*args.steps*5+160, sec*args.epochs*args.steps*5+235],
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'dependency_sha256': hashes()}
    args.out.mkdir(parents=True, exist_ok=True); write(args.out/'feature_contract.json', contract); write(args.out/'benchmark.json', result)
    print(json.dumps(result, indent=2), flush=True)


@torch.no_grad()
def conditionals(weight, mu, sd, edges):
    # members x points x target-component parameters for ONE target.
    z = (edges[None, None, :, None]-mu[:, :, None, :])/sd[:, :, None, :]
    cdf = .5*(1+torch.erf(z/np.sqrt(2)))
    cp = ((cdf[:, :, 1:]-cdf[:, :, :-1]).clamp_min(0)*weight[:, :, None, :]).sum(-1)
    return cp/cp.sum(-1, keepdim=True).clamp_min(1e-12)


def train(args):
    run_started = time.perf_counter(); args.out.mkdir(parents=True, exist_ok=True)
    checks = self_test(); dependency_hashes = hashes(); contract = feature_contract()
    contract_path = args.out/'feature_contract.json'
    if contract_path.exists(): assert json.loads(contract_path.read_text()) == contract
    else: write(contract_path, contract)
    contract_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    rec = load_split(ROOT.parent/'data', 'train'); assert len(rec) == 80
    ys = [fr[TS].to_numpy(np.float64) for _, fr in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    y = np.concatenate([v[m] for v, m in zip(ys, valid)])
    well = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, rows) in enumerate(GroupKFold(5).split(y, groups=well)): fold[rows] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    wf = {i: int(fold[row_ids[i][0]]) for i in range(80)}
    config = {**vars(args), 'out': str(args.out), 'members': MEMBERS,
              'parameters': sum(p.numel() for p in Network().parameters()),
              'source_sha256': dependency_hashes[DEPENDENCIES[0]], 'dependency_sha256': dependency_hashes,
              'architecture': '206 inputs with input-only PLE16+raw, learned8-dim per-feature embeddings; width512 three-layer ReLU/dropout.1 MLP;32 members; only first scaling random with featurewise groups, all other scalings start at1; private gates and8-Gaussian heads',
              'fixed_composition': 'Increase shared backbone capacity relative to the32-member candidate: width512 and3 layers, same embeddings, initialization, seed,32 members, per-member batch256,40x64 updates and decoding',
              'feature_whitelist': contract['whitelist'], 'excluded_features': contract['excluded'],
              'feature_contract_sha256': contract_hash,
              'member_initialization': 'first-layer feature-grouped Normal(0,1) input scalings; all other scalings ones; each member independently samples256 rows, total8192 per step; no member selection',
              'loss': 'mean over members of BCE and mean per-target nondefault mixture NLL; each member has equal weight',
              'fusion': 'fixed arithmetic mean of complete [default, (1-default)*conditional] distributions; recover conditional only after joint averaging',
              'target_transforms': ['log1p', 'log10', 'log'],
              'preprocessing': 'reuse train_invariant.prepare_features with the exact previously audited 206-feature whitelist; fit all feature/PLE and target statistics only on the outer 64 training wells',
              'decode': 'fixed 256 training-fitted transformed supports, 800 action grid with default atom and official expected utility',
              'selection': 'single fixed 40x64 configuration; no member/checkpoint/threshold/weight selection using outer OOF labels',
              'test_access': False, 'valid_rows': len(y), 'self_test': checks,
              'torch': str(torch.__version__), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
              'device': torch.cuda.get_device_name(0),
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if wf[i] == f] for f in range(5)},
              'training_wells': {str(f): [rec[i][0] for i in range(80) if wf[i] != f] for f in range(5)}}
    config_path = args.out/'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ['dependency_sha256', 'epochs', 'steps', 'batch_points', 'seed']:
            assert old[key] == config[key], f'Incompatible resume: {key}'
    write(config_path, config); print('CONFIG', json.dumps({k: v for k, v in config.items() if k not in ['outer_wells', 'training_wells', 'feature_whitelist', 'excluded_features']}), flush=True)
    pred = np.full_like(y, np.nan); reports = []; defaults = dmask(y)
    for f in range(5):
        t0 = time.perf_counter(); seed = args.seed+f
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for needed in [f'fold_{f+1}_report.json', f'fold_{f+1}.pt', f'fold_{f+1}_distributions.npz']:
                assert (args.out/needed).exists(), needed
            pred[fold == f] = np.load(cache); reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text())); continue
        training_wells = [i for i in range(80) if wf[i] != f]
        assert len(training_wells) == 64
        print(f'FOLD {f+1} fitting train-only statistics', flush=True)
        x, builder, fm, fs, ple_edges = prepare_features(rec, training_wells, valid, row_ids)
        med = []; scale = []; initial = []; centers = []; density_edges = []; z = []
        for j in range(3):
            transformed = TRANSFORM[j](y[:, j]); selected = transformed[(fold != f) & ~defaults[:, j]]
            q25, q50, q75 = np.percentile(selected, [25, 50, 75]); sc = max(float(q75-q25), .1)
            med.append(q50); scale.append(sc); z.append((transformed-q50)/sc)
            initial.append(np.quantile((selected-q50)/sc, np.linspace(.05, .95, MIX)))
            ct = np.linspace(selected.min(), selected.max(), DENSITY_BINS)
            centers.append(INVERSE[j](ct)); density_edges.append((np.r_[-np.inf, (ct[:-1]+ct[1:])/2, np.inf]-q50)/sc)
        med = np.asarray(med); scale = np.asarray(scale)
        targets = torch.tensor(np.column_stack(z), dtype=torch.float32, device='cuda')
        gates = torch.tensor(defaults, dtype=torch.float32, device='cuda')
        training_indices = torch.tensor(np.flatnonzero(fold != f), device='cuda')
        model = Network(ple_edges, np.stack(initial)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs*args.steps, eta_min=.00001)
        history = []; torch.cuda.reset_peak_memory_stats()
        for epoch in range(args.epochs):
            model.train(); losses = np.zeros(3)
            for _ in range(args.steps):
                losses += step(model, opt, x, gates, targets, training_indices, args.batch_points)
                scheduler.step()
            history.append((losses/args.steps).tolist())
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/{args.epochs} loss={history[-1][0]:.4f} gate={history[-1][1]:.4f} nll={history[-1][2]:.4f} seconds={time.perf_counter()-t0:.1f}', flush=True)
        model.eval(); actions = []; kernels = []
        for j, target in enumerate(TS):
            sup = centers[j]
            action = np.linspace(sup[0], sup[-1], 800) if j == 2 else np.geomspace(max(sup[0]*.9, .001), sup[-1]*1.1, 800)
            action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
            actions.append(action); kernels.append(torch.tensor(kernel_matrix(action, np.r_[DEFAULTS[target], sup], target).T, dtype=torch.float32, device='cuda'))
        rows_val = np.flatnonzero(fold == f)
        saved_gate = []; saved_member_gate = []; saved_prob = [[], [], []]
        saved_weight = []; saved_mu = []; saved_sd = []; joint_error = 0.
        with torch.no_grad():
            for start in range(0, len(rows_val), 128):
                rows = rows_val[start:start+128]
                xx = x[torch.tensor(rows, device='cuda')][None].expand(MEMBERS, -1, -1)
                with torch.autocast('cuda', dtype=torch.bfloat16): go, raw = model(xx)
                gate_members = go.float().sigmoid(); logw, mu, sd = mixture_params(raw); weight = logw.exp()
                aggregate_gate = []; effective_weights = []
                for j in range(3):
                    edges = torch.tensor(density_edges[j], device='cuda', dtype=torch.float32)
                    cp_members = conditionals(weight[:, :, j], mu[:, :, j], sd[:, :, j], edges)
                    gate, cp, full = joint_average(gate_members[:, :, j], cp_members)
                    restored = torch.cat([gate[:, None], (1-gate[:, None])*cp], 1)
                    error = float((restored-full).abs().max()); joint_error = max(joint_error, error)
                    assert error < 3e-7
                    pred[rows, j] = actions[j][(full@kernels[j]).argmax(1).cpu().numpy()]
                    aggregate_gate.append(gate.cpu().numpy()); saved_prob[j].append(cp.cpu().numpy().astype(np.float16))
                    ew = ((1-gate_members[:, :, j, None])*weight[:, :, j]).permute(1, 0, 2).reshape(len(rows), MEMBERS*MIX)
                    ew_total = ew.sum(1, keepdim=True)
                    fallback_weight = weight[:, :, j].permute(1, 0, 2).reshape(len(rows), MEMBERS*MIX)/MEMBERS
                    ew = torch.where(ew_total > 0, ew/ew_total.clamp_min(1e-12), fallback_weight)
                    effective_weights.append(ew.cpu().numpy().astype(np.float16))
                saved_gate.append(np.column_stack(aggregate_gate))
                saved_member_gate.append(gate_members.permute(1, 0, 2).cpu().numpy().astype(np.float16))
                saved_weight.append(np.stack(effective_weights, 1))
                saved_mu.append(mu.permute(1, 2, 0, 3).reshape(len(rows), 3, MEMBERS*MIX).cpu().numpy().astype(np.float16))
                saved_sd.append(sd.permute(1, 2, 0, 3).reshape(len(rows), 3, MEMBERS*MIX).cpu().numpy().astype(np.float16))
        pack = {'indices': rows_val, 'gate': np.concatenate(saved_gate), 'member_gate': np.concatenate(saved_member_gate),
                'mixture_weight': np.concatenate(saved_weight), 'mixture_mu': np.concatenate(saved_mu), 'mixture_std': np.concatenate(saved_sd),
                'regression_median': med, 'regression_scale': scale, 'regression_transforms': np.array(['log1p', 'log10', 'log']), 'pred': pred[rows_val]}
        for j, target in enumerate(TS): pack[target+'_prob'] = np.concatenate(saved_prob[j]); pack[target+'_centers'] = centers[j]
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **pack)
        fr = {'fold': f+1, 'scores': metric(y[rows_val], pred[rows_val]), 'seconds': time.perf_counter()-t0, 'loss': history,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_gpu_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'joint_reconstruction_max_abs_error': joint_error}
        reports.append(fr); write(args.out/f'fold_{f+1}_report.json', fr); np.save(cache, pred[rows_val])
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config,
                    'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'ple_edges_normalized': ple_edges.tolist(),
                    'medians': builder.medians_, 'enhanced_feature_names': builder.enhanced_feature_names_,
                    'selected_feature_names': contract['whitelist'], 'selected_feature_indices': contract['selected_indices'],
                    'regression_median': med.tolist(), 'regression_scale': scale.tolist(),
                    'centers': [v.tolist() for v in centers], 'density_edges': [v.tolist() for v in density_edges]}, args.out/f'fold_{f+1}.pt')
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, scheduler, x, gates, targets, training_indices, kernels, pack, saved_prob, saved_weight, saved_mu, saved_sd
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred).all() and hashes() == dependency_hashes
    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == contract_hash
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': reports, 'seconds': time.perf_counter()-run_started,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(args.out/'report.json', report); print('FINAL', json.dumps({k: report[k] for k in ['oof_scores', 'seconds']}), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_tabm32_wide')
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--batch-points', type=int, default=8192); ap.add_argument('--seed', type=int, default=20261004)
    ap.add_argument('--benchmark', action='store_true'); args = ap.parse_args()
    assert args.batch_points % MEMBERS == 0
    torch.set_num_threads(3); torch.backends.cudnn.benchmark = False; assert torch.cuda.is_available()
    assert torch.cuda.mem_get_info()[0] > 4*1024**3, 'Need4GiB free GPU memory'
    benchmark(args) if args.benchmark else train(args)


if __name__ == '__main__': main()
