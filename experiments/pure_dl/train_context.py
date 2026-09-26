"""Residual tabular MLP with local and complete-well neural context.

The context input is a fixed, label-free summary of the complete current well.
It is supplied during both crop training and complete-well inference. All fitted
statistics, target transforms and bins use only the outer-fold training wells.
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
from src.feature import FeatureBuilder
from src.model import CURVES, prep_well
from src.utils import DEFAULTS, BOUNDS, kernel_matrix
from train import TS, BINS, OFF, FW, IV, dmask, metric, write

CONTEXT_TOKENS = 128
REG_FW = [np.log, np.log10, np.log]


class PointResidual(nn.Module):
    """Independent nonlinear regression capacity at each depth point."""
    def __init__(self, width=256):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                 nn.Dropout(.10), nn.Linear(width, width))
        self.scale = nn.Parameter(torch.full((width,), .2))

    def forward(self, x):
        return x + self.scale * self.mlp(self.norm(x))


class LocalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depth = nn.Conv1d(width, width, 5, padding=2*dilation,
                               dilation=dilation, groups=width)
        self.mlp = nn.Sequential(nn.Conv1d(width, 2*width, 1), nn.GELU(),
                                 nn.Conv1d(2*width, width, 1))
        self.scale = nn.Parameter(torch.full((1, width, 1), .1))

    def forward(self, x):
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x + self.scale*self.mlp(self.depth(z))


class Network(nn.Module):
    def __init__(self, n_in=269, width=256, local_width=64):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(n_in, width), nn.GELU(),
                                   *(PointResidual(width) for _ in range(3)))
        self.local = nn.Sequential(nn.Conv1d(32, local_width, 7, padding=3),
                                   *(LocalBlock(local_width, d) for d in [1, 2, 4, 8, 16, 32]))
        self.context_stem = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.GELU())
        self.context_rnn = nn.GRU(64, 48, batch_first=True, bidirectional=True)
        self.global_context = nn.Sequential(nn.Linear(96, 64), nn.GELU())
        self.fusion = nn.Sequential(nn.Conv1d(width+local_width+96+64, 160, 1),
                                    nn.GELU(), nn.Dropout(.05))
        self.state = nn.Sequential(nn.Conv1d(160, 96, 1), nn.GELU(), nn.Conv1d(96, 4, 1))
        self.gate = nn.Conv1d(160, 3, 1)
        self.value = nn.Sequential(nn.Conv1d(160, 160, 1), nn.GELU(), nn.Conv1d(160, 200, 1))
        self.regression = nn.Sequential(nn.Conv1d(160, 96, 1), nn.GELU(), nn.Conv1d(96, 3, 1))

    def forward(self, x, landmarks, positions):
        """positions is [B, L], the row's location in the 128 pooled tokens."""
        point = self.point(x.transpose(1, 2)).transpose(1, 2)
        local = self.local(x[:, :32])
        # The small GRU uses float32; this avoids mixed-precision RNN backend
        # limitations while the much larger point/CNN paths stay under autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            tokens = self.context_stem(landmarks.float()).transpose(1, 2)
            encoded, _ = self.context_rnn(tokens)
            pooled = self.global_context(encoded.mean(1))
        encoded = encoded.transpose(1, 2).to(point.dtype)
        pos = positions.clamp(0, CONTEXT_TOKENS-1)
        lo = pos.floor().long()
        hi = (lo+1).clamp_max(CONTEXT_TOKENS-1)
        alpha = (pos-lo).unsqueeze(1).to(point.dtype)
        left = encoded.gather(2, lo[:, None].expand(-1, encoded.shape[1], -1))
        right = encoded.gather(2, hi[:, None].expand(-1, encoded.shape[1], -1))
        contextual = left + alpha*(right-left)
        global_feature = pooled.to(point.dtype)[:, :, None].expand(-1, -1, x.shape[-1])
        h = self.fusion(torch.cat([point, local, contextual, global_feature], 1))
        return self.state(h), self.gate(h), self.value(h), self.regression(h)


def train_loss(outputs, state, gate, label, regression):
    so, go, vo, ro = [output.float() for output in outputs]
    valid = state >= 0
    loss = F.cross_entropy(so, state, ignore_index=-1, reduction='sum')/valid.sum().clamp_min(1)
    gl = F.binary_cross_entropy_with_logits(go, gate.clamp_min(0), reduction='none')
    loss = loss + .3*(gl*(gate >= 0)).sum()/(gate >= 0).sum().clamp_min(1)
    for j in range(3):
        active = label[:, j] >= 0
        ce = F.cross_entropy(vo[:, OFF[j]:OFF[j+1]], label[:, j],
                             ignore_index=-1, reduction='sum')
        # Auxiliary continuous predictions encourage nearby bins to share useful
        # representations. They do not change the fixed probability decoder.
        reg = F.smooth_l1_loss(ro[:, j], regression[:, j], reduction='none', beta=.15)
        loss = loss + (ce + .25*(reg*active).sum())/active.sum().clamp_min(1)/3
    return loss


def smoke_test():
    """Synthetic CPU-only shape/gradient check, with no dataset access."""
    torch.set_num_threads(2)
    torch.manual_seed(7)
    net = Network()
    x = torch.randn(2, 269, 96)
    tokens = torch.randn(2, 32, CONTEXT_TOKENS)
    positions = torch.linspace(0, CONTEXT_TOKENS-1, 96)[None].expand(2, -1)
    outputs = net(x, tokens, positions)
    assert [tuple(a.shape) for a in outputs] == [(2, 4, 96), (2, 3, 96), (2, 200, 96), (2, 3, 96)]
    state = torch.randint(0, 4, (2, 96))
    gate = torch.randint(0, 2, (2, 3, 96)).float()
    labels = torch.stack([torch.randint(0, b, (2, 96)) for b in BINS], 1)
    labels[:, :, :8] = -1
    loss = train_loss(outputs, state, gate, labels, torch.randn(2, 3, 96))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
    # Fully missing labels must produce a finite zero objective.
    absent = train_loss(outputs, torch.full_like(state, -1), torch.full_like(gate, -1),
                        torch.full_like(labels, -1), torch.zeros(2, 3, 96))
    assert absent.item() == 0
    print(json.dumps({'cpu_smoke': 'passed', 'parameters': sum(p.numel() for p in net.parameters()),
                      'loss': float(loss.detach()), 'output_shapes': [list(a.shape) for a in outputs]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_context')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--crop', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=20260926)
    ap.add_argument('--cpu-smoke', action='store_true')
    a = ap.parse_args()
    if a.cpu_smoke:
        smoke_test()
        return
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    # Cap the PyTorch allocator at 2.5 GiB; leave room for the CUDA context and
    # the other two authorized parallel training processes.
    device = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(.95, 2.5*2**30/device.total_memory))
    start = time.perf_counter()
    rec = load_split(ROOT.parent/'data', 'train')
    assert len(rec) == 80
    ys = [fr[TS].to_numpy(np.float64) for _, fr in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    y = np.concatenate([t[v] for t, v in zip(ys, valid)])
    well = np.concatenate([np.full(v.sum(), i, np.int32) for i, v in enumerate(valid)])
    assert y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, index) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[index] = f
    wf = {i: int(fold[np.flatnonzero(well == i)[0]]) for i in range(80)}
    config = {**vars(a), 'out': str(a.out), 'parameters': sum(p.numel() for p in Network().parameters()),
              'device': device.name, 'torch': torch.__version__,
              'architecture': '269-channel width256 3-block residual MLP + raw32 width64 local CNN + 128 full-well landmarks and bidirectional GRU48',
              'loss': '4-state CE + .3 independent BCE + quantile CE + .25 continuous log-space SmoothL1 per target',
              'decode': 'fixed 50:50 state and independent default probabilities; expected official utility including the default action',
              'selection': 'one precommitted configuration, fixed final epoch, no outer-fold checkpoint/threshold/blend search',
              'normalization': 'global features and target statistics fit only on outer-fold training wells',
              'context': 'complete current-well input curves, no labels, same access during train and inference',
              'feature_dimensions': {'sequence': 32, 'enhanced': 237, 'total': 269},
              'allocator_limit_gib': 2.5, 'resident_input_dtype': 'float16',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if wf[i] == f] for f in range(5)},
              'valid_rows': len(y), 'test_access': False}
    config_path = a.out/'config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        for key in ['epochs', 'steps', 'batch', 'crop', 'seed', 'source_sha256', 'outer_wells']:
            assert previous[key] == config[key], f'Existing output has incompatible {key}'
    write(config_path, config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k != 'outer_wells'}), flush=True)
    pred = np.full_like(y, np.nan)
    fold_reports = []
    for f in range(5):
        fstart = time.perf_counter()
        seed = a.seed+f
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        train = [i for i in range(80) if wf[i] != f]
        val = [i for i in range(80) if wf[i] == f]
        cache = a.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            pred[fold == f] = np.load(cache)
            fold_reports.append(json.loads((a.out/f'fold_{f+1}_report.json').read_text()))
            continue
        print(f'FOLD {f+1} preparing train-only feature and target statistics', flush=True)
        builder = FeatureBuilder().fit([rec[i][1] for i in train])
        enhanced = [builder.transform(fr).enhanced for _, fr in rec]
        assert all(x.shape[1] == 237 for x in enhanced)
        fm = np.empty(237, np.float32)
        fs = np.empty(237, np.float32)
        for c in range(237):
            values = np.concatenate([enhanced[i][:, c] for i in train])
            q = np.percentile(values, [25, 50, 75])
            fm[c] = q[1]
            fs[c] = max(q[2]-q[0], 1e-3)
        del values
        gs = {}
        for c in CURVES:
            v = np.concatenate([rec[i][1][c].dropna().to_numpy() for i in train])
            v = np.log10(np.maximum(v, .01)) if c in ['RT', 'RXO'] else v
            q1, q3 = np.percentile(v, [25, 75])
            gs[c] = {'median': float(np.median(v)), 'iqr': max(float(q3-q1), 1e-3)}
        yt = y[fold != f]
        dm = dmask(yt)
        edges, centers, rm, rs = [], [], [], []
        for j, k in enumerate(BINS):
            values = FW[j](yt[~dm[:, j], j])
            edge = np.unique(np.quantile(values, np.linspace(0, 1, k+1)))
            assert len(edge) == k+1
            cl = np.clip(np.searchsorted(edge, values, side='right')-1, 0, k-1)
            center = np.array([np.median(values[cl == i]) for i in range(k)])
            edges.append(edge)
            centers.append(IV[j](center))
            z = REG_FW[j](np.maximum(yt[~dm[:, j], j], 1e-6))
            rm.append(float(np.median(z)))
            rs.append(max(float(np.subtract(*np.percentile(z, [75, 25]))), .1))
        xs, landmarks, positions, states, gates, labels, regression = [], [], [], [], [], [], []
        for i, (_, fr) in enumerate(rec):
            raw = np.clip(prep_well(fr, gs), -8, 8)
            static = np.clip((enhanced[i]-fm)/fs, -8, 8).T.astype(np.float32)
            xx = np.concatenate([raw, static]).astype(np.float16)
            assert xx.shape == (269, len(fr)) and np.isfinite(xx).all()
            xs.append(torch.from_numpy(xx).cuda())
            tokens = F.adaptive_avg_pool1d(torch.from_numpy(raw)[None], CONTEXT_TOKENS)[0]
            landmarks.append(tokens.cuda())
            # Adaptive average-pooling cells are centered at (k+.5)/K.
            pos = (np.arange(len(fr), dtype=np.float32)+.5)/len(fr)*CONTEXT_TOKENS-.5
            positions.append(torch.from_numpy(pos).cuda())
            clean = np.where(valid[i][:, None], ys[i], np.array([.1, .01, 99.9]))
            default = dmask(clean)
            state = np.select([clean[:, 0] <= .1, clean[:, 2] >= 99.9, clean[:, 1] <= .01], [0, 2, 3], 1)
            state[~valid[i]] = -1
            states.append(torch.tensor(state, device='cuda', dtype=torch.long))
            gate = default.T.astype(np.float32)
            gate[:, ~valid[i]] = -1
            gates.append(torch.tensor(gate, device='cuda'))
            cl, rz = [], []
            for j, k in enumerate(BINS):
                values = FW[j](np.maximum(clean[:, j], .01))
                category = np.clip(np.searchsorted(edges[j], values, side='right')-1, 0, k-1)
                category[default[:, j] | ~valid[i]] = -1
                cl.append(category)
                rz.append((REG_FW[j](np.maximum(clean[:, j], 1e-6))-rm[j])/rs[j])
            labels.append(torch.tensor(np.stack(cl), device='cuda', dtype=torch.long))
            regression.append(torch.tensor(np.stack(rz), device='cuda', dtype=torch.float32))
        del enhanced, raw, static, xx
        gc.collect()
        model = Network().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.0012, weight_decay=.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs*a.steps, eta_min=.000012)
        lengths = np.array([len(ys[i]) for i in train])
        sample_prob = lengths/lengths.sum()
        history = []
        torch.cuda.reset_peak_memory_stats()
        for ep in range(a.epochs):
            model.train()
            total = 0.
            for step in range(a.steps):
                wells = rng.choice(train, a.batch, p=sample_prob)
                starts = [rng.integers(0, len(ys[i])-a.crop+1) for i in wells]
                x = torch.stack([xs[i][:, s:s+a.crop] for i, s in zip(wells, starts)])
                context = torch.stack([landmarks[i] for i in wells])
                pos = torch.stack([positions[i][s:s+a.crop] for i, s in zip(wells, starts)])
                st = torch.stack([states[i][s:s+a.crop] for i, s in zip(wells, starts)])
                ga = torch.stack([gates[i][:, s:s+a.crop] for i, s in zip(wells, starts)])
                cl = torch.stack([labels[i][:, s:s+a.crop] for i, s in zip(wells, starts)])
                z = torch.stack([regression[i][:, s:s+a.crop] for i, s in zip(wells, starts)])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(x, context, pos)
                loss = train_loss(outputs, st, ga, cl, z)
                assert torch.isfinite(loss)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step()
                sched.step()
                total += float(loss.detach())
            history.append(total/a.steps)
            if ep == 0 or (ep+1) % 5 == 0:
                print(f'fold={f+1} epoch={ep+1}/{a.epochs} loss={history[-1]:.4f} seconds={time.perf_counter()-fstart:.1f} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}', flush=True)
        model.eval()
        grids, kernels = [], []
        for j, t in enumerate(TS):
            sup = centers[j]
            grid = np.linspace(sup[0], sup[-1], 400) if j == 2 else np.geomspace(max(sup[0]*.9, .001), sup[-1]*1.1, 400)
            grid = np.unique(np.r_[DEFAULTS[t], np.clip(grid, *BOUNDS[t])])
            grids.append(grid)
            kernels.append(torch.tensor(kernel_matrix(grid, np.r_[DEFAULTS[t], sup], t).T, dtype=torch.float32, device='cuda'))
        saved_state, saved_independent, saved_gate, saved_regression = [], [], [], []
        saved_prob = [[], [], []]
        saved_rows = []
        for wi in val:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                so, go, vo, ro = model(xs[wi][None], landmarks[wi][None], positions[wi][None])
            sp = so.float().softmax(1)[0].T
            independent = go.float().sigmoid()[0].T
            marg = torch.stack([sp[:, 0], sp[:, 0]+sp[:, 3], sp[:, 0]+sp[:, 2]], 1)
            gate = .5*(marg+independent)
            vp = vo.float()[0].T
            pp = []
            saved_rows.append(np.flatnonzero(well == wi))
            saved_state.append(sp.cpu().numpy()[valid[wi]].astype(np.float16))
            saved_independent.append(independent.cpu().numpy()[valid[wi]].astype(np.float16))
            saved_gate.append(gate.cpu().numpy()[valid[wi]].astype(np.float16))
            saved_regression.append(ro.float()[0].T.cpu().numpy()[valid[wi]].astype(np.float16))
            for j, t in enumerate(TS):
                cond = vp[:, OFF[j]:OFF[j+1]].softmax(1)
                probability = torch.cat([gate[:, j:j+1], (1-gate[:, j:j+1])*cond], 1)
                with torch.no_grad():
                    best = (probability@kernels[j]).argmax(1).cpu().numpy()
                pp.append(grids[j][best])
                saved_prob[j].append(cond.cpu().numpy()[valid[wi]].astype(np.float16))
            pred[well == wi] = np.stack(pp, 1)[valid[wi]]
        indices = np.flatnonzero(fold == f)
        assert np.array_equal(np.concatenate(saved_rows), indices)
        pack = {'indices': indices, 'state': np.concatenate(saved_state),
                'gate': np.concatenate(saved_gate), 'independent_gate': np.concatenate(saved_independent),
                'regression': np.concatenate(saved_regression), 'regression_median': np.array(rm),
                'regression_scale': np.array(rs), 'pred': pred[fold == f]}
        for j, t in enumerate(TS):
            pack[f'{t}_prob'] = np.concatenate(saved_prob[j])
            pack[f'{t}_centers'] = centers[j]
        np.savez_compressed(a.out/f'fold_{f+1}_distributions.npz', **pack)
        fr = {'fold': f+1, 'scores': metric(y[fold == f], pred[fold == f]),
              'seconds': time.perf_counter()-fstart, 'loss': history,
              'peak_gpu_GiB': torch.cuda.max_memory_allocated()/2**30,
              'peak_gpu_reserved_GiB': torch.cuda.max_memory_reserved()/2**30}
        fold_reports.append(fr)
        write(a.out/f'fold_{f+1}_report.json', fr)
        torch.save({'state_dict': model.cpu().state_dict(), 'feature_median': fm.tolist(), 'feature_scale': fs.tolist(),
                    'gstats': gs, 'centers': [c.tolist() for c in centers], 'edges': [e.tolist() for e in edges],
                    'regression_median': rm, 'regression_scale': rs, 'medians': builder.medians_,
                    'enhanced_feature_names': builder.enhanced_feature_names_, 'config': config}, a.out/f'fold_{f+1}.pt')
        np.save(cache, pred[fold == f])
        np.savez_compressed(a.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, sched, xs, landmarks, positions, states, gates, labels, regression, kernels
        del pack, saved_state, saved_independent, saved_gate, saved_prob, saved_regression
        gc.collect()
        torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': fold_reports,
              'seconds': time.perf_counter()-start,
              'per_well': {rec[i][0]: metric(y[well == i], pred[well == i]) for i in range(80)}}
    write(a.out/'report.json', report)
    print('FINAL', json.dumps({k: v for k, v in report.items() if k in ['oof_scores', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()
