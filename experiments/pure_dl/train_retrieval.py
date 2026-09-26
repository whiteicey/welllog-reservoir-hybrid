"""Fold-local neural memory correction of a frozen, same-fold PLE network.

Only the 64 outer training wells enter the label-bearing memory. Training
queries cannot retrieve their own well. No released model or test data is used.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

import train_ple as ple
from train import ROOT, TS, dmask, metric, write
from src.data import load_split
from src.feature import FeatureBuilder
from src.utils import BOUNDS, DEFAULTS, kernel_matrix
from retrieval_network import CrossWellAttention, validate_memory_split


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_retrieval')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=20260928)
    args = parser.parse_args()
    import psutil
    active = [p for p in psutil.process_iter(['pid', 'name', 'cmdline'])
              if p.info['pid'] != os.getpid() and (p.info['name'] or '').lower() == 'python.exe'
              and any('pure_dl' in s and Path(s).name.startswith('train') for s in (p.info['cmdline'] or []))]
    if len(active) >= 3:
        raise RuntimeError('Three other training processes already active; wait for a slot')
    torch.set_num_threads(3)
    assert torch.cuda.is_available()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    rec = load_split(ROOT.parent/'data', 'train')
    ys = [frame[TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    y = np.concatenate([yy[mask] for yy, mask in zip(ys, valid)])
    well = np.concatenate([np.full(mask.sum(), i, np.int32) for i, mask in enumerate(valid)])
    assert len(rec) == 80 and y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, rows) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[rows] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    wf = {i: int(fold[rows[0]]) for i, rows in enumerate(row_ids)}
    sources = ROOT/'outputs/pure_dl_ple'
    config = {**vars(args), 'out': str(args.out), 'test_access': False,
              'method': 'frozen same-fold PLE hidden features + learned cross-well label attention + small residual density correction',
              'memory_rows_per_well': 64, 'correction_scale': .2,
              'training_query_own_well_masked': True,
              'selection': 'fixed 30x64 updates; no outer-fold model selection or threshold fitting',
              'parameters_trainable': sum(p.numel() for p in CrossWellAttention().parameters()),
              'source_sha256': {p.name: digest(p) for p in [Path(__file__), Path(ple.__file__), Path(__file__).with_name('retrieval_network.py')]},
              'base_checkpoint_sha256': {str(f+1): digest(sources/f'fold_{f+1}.pt') for f in range(5)},
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if wf[i] == f] for f in range(5)}}
    if (args.out/'config.json').exists():
        assert json.loads((args.out/'config.json').read_text()) == json.loads(json.dumps(config, default=str))
    write(args.out/'config.json', config)
    print('CONFIG', json.dumps({k: v for k, v in config.items() if k not in ['outer_wells', 'base_checkpoint_sha256']}, default=str), flush=True)
    pred = np.full_like(y, np.nan)
    reports = []
    default = dmask(y)
    for f in range(5):
        t0 = time.perf_counter()
        if (args.out/f'fold_{f+1}_pred.npy').exists():
            pred[fold == f] = np.load(args.out/f'fold_{f+1}_pred.npy')
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text()))
            continue
        seed = args.seed+f
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        tr = [i for i in range(80) if wf[i] != f]
        training_rows = np.flatnonzero(fold != f)
        validation_rows = np.flatnonzero(fold == f)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            ck = torch.load(sources/f'fold_{f+1}.pt', weights_only=True, map_location='cpu')
        assert ck['config']['outer_wells'][str(f)] == config['outer_wells'][str(f)]
        base = ple.Network(np.asarray(ck['ple_edges_normalized'])).cuda().eval()
        base.load_state_dict(ck['state_dict'])
        for p in base.parameters():
            p.requires_grad_(False)
        builder = FeatureBuilder().fit([rec[i][1] for i in tr])
        assert builder.medians_ == ck['medians']
        fm, fs = np.asarray(ck['feature_median']), np.asarray(ck['feature_scale'])
        # Store frozen features, never predictions fitted to the outer labels.
        frozen = torch.empty((len(y), 331), dtype=torch.float16, device='cuda')
        with torch.no_grad():
            for i, (_, frame) in enumerate(rec):
                x = np.clip((builder.transform(frame).enhanced[valid[i]]-fm)/fs, -8, 8).astype(np.float16)
                for start in range(0, len(x), 2048):
                    rows = row_ids[i][start:start+2048]
                    xx = torch.from_numpy(x[start:start+2048]).cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        h = base.norm(base.blocks(base.stem(base.encoding(xx))))
                        output = torch.cat([base.gate(h), base.density(h)], -1)
                    frozen[torch.as_tensor(rows, device='cuda')] = torch.cat([h, output], -1).half()
        del base, x, xx, h, output
        med, scale = np.asarray(ck['regression_median']), np.asarray(ck['regression_scale'])
        z = np.column_stack([(ple.TRANSFORM[j](y[:, j])-med[j])/scale[j] for j in range(3)])
        targets = torch.tensor(z, dtype=torch.float32, device='cuda')
        gates = torch.tensor(default, dtype=torch.float32, device='cuda')
        groups = torch.tensor(well, dtype=torch.long, device='cuda')
        train_ix = torch.tensor(training_rows, dtype=torch.long, device='cuda')
        memory_rows = np.concatenate([rng.choice(row_ids[i], 64, replace=False) for i in tr])
        validate_memory_split(memory_rows, training_rows, validation_rows)
        assert len(set(well[memory_rows])) == 64 and not set(well[memory_rows]) & set(well[validation_rows])
        memory_ix = torch.tensor(memory_rows, dtype=torch.long, device='cuda')
        mh = frozen[memory_ix, :256]
        mz = targets[memory_ix].clamp(-5, 5)*(1-gates[memory_ix])
        mv = torch.cat([gates[memory_ix], mz, mz.square()], -1)
        mw = groups[memory_ix]
        model = CrossWellAttention().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.02)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs*args.steps, eta_min=.000005)
        history = []
        torch.cuda.reset_peak_memory_stats()
        for ep in range(args.epochs):
            model.train(); total = 0.
            for _ in range(args.steps):
                rows = train_ix[torch.randint(len(train_ix), (args.batch,), device='cuda')]
                xx = frozen[rows]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    delta = model(xx[:, :256], mh, mv, mw, groups[rows])
                raw = xx[:, 256:].float()+.2*delta.float()
                logw, mu, sd = ple.mixture_params(raw[:, 3:].reshape(-1, 3, 3, 8))
                logp = logw-.5*((targets[rows, :, None]-mu)/sd).square()-sd.log()-.5*np.log(2*np.pi)
                nll = -torch.logsumexp(logp, -1)
                loss = F.binary_cross_entropy_with_logits(raw[:, :3], gates[rows])
                loss = loss+sum(nll[:, j][gates[rows, j] < .5].mean() for j in range(3))/3
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step(); sched.step(); total += float(loss.detach())
            history.append(total/args.steps)
            if ep == 0 or (ep+1) % 5 == 0:
                print(f'fold={f+1} epoch={ep+1}/{args.epochs} loss={history[-1]:.4f} seconds={time.perf_counter()-t0:.1f}', flush=True)
        model.eval(); actions = []; kernels = []
        for j, target in enumerate(TS):
            support = np.asarray(ck['centers'][j])
            action = np.linspace(support[0], support[-1], 800) if j == 2 else np.geomspace(max(support[0]*.9, .001), support[-1]*1.1, 800)
            action = np.unique(np.r_[DEFAULTS[target], np.clip(action, *BOUNDS[target])])
            actions.append(action)
            kernels.append(torch.tensor(kernel_matrix(action, np.r_[DEFAULTS[target], support], target).T, dtype=torch.float32, device='cuda'))
        saved_gate = []; saved_probs = [[], [], []]
        with torch.no_grad():
            for start in range(0, len(validation_rows), 1024):
                ix = validation_rows[start:start+1024]
                rows = torch.tensor(ix, device='cuda')
                xx = frozen[rows]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    delta = model(xx[:, :256], mh, mv, mw, groups[rows])
                raw = xx[:, 256:].float()+.2*delta.float()
                gate = raw[:, :3].sigmoid()
                logw, mu, sd = ple.mixture_params(raw[:, 3:].reshape(-1, 3, 3, 8))
                saved_gate.append(gate.cpu().numpy())
                for j in range(3):
                    edges = torch.tensor(ck['density_edges'][j], device='cuda', dtype=torch.float32)
                    cp = ple.conditional_probability(logw[:, j].exp(), mu[:, j], sd[:, j], edges)
                    joint = torch.cat([gate[:, j:j+1], (1-gate[:, j:j+1])*cp], -1)
                    pred[ix, j] = actions[j][(joint@kernels[j]).argmax(-1).cpu().numpy()]
                    saved_probs[j].append(cp.cpu().numpy().astype(np.float16))
        pack = {'indices': validation_rows, 'gate': np.concatenate(saved_gate)}
        for j, target in enumerate(TS):
            pack[target+'_prob'] = np.concatenate(saved_probs[j])
            pack[target+'_centers'] = np.asarray(ck['centers'][j])
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **pack)
        report = {'fold': f+1, 'scores': metric(y[validation_rows], pred[validation_rows]),
                  'seconds': time.perf_counter()-t0, 'loss': history,
                  'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
        reports.append(report)
        write(args.out/f'fold_{f+1}_report.json', report)
        np.save(args.out/f'fold_{f+1}_pred.npy', pred[validation_rows])
        torch.save({'state_dict': model.cpu().state_dict(), 'config': config,
                    'memory_rows': memory_rows.tolist(), 'memory_wells': well[memory_rows].tolist(),
                    'outer_training_rows': training_rows.tolist(), 'base_checkpoint': str(sources/f'fold_{f+1}.pt')}, args.out/f'fold_{f+1}.pt')
        np.savez_compressed(args.out/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in report.items() if k != 'loss'}), flush=True)
        del model, opt, sched, frozen, targets, gates, groups, train_ix, memory_ix, mh, mz, mv, mw, kernels, pack, saved_probs
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    report = {'config': config, 'oof_scores': metric(y, pred), 'folds': reports,
              'seconds': time.perf_counter()-started}
    write(args.out/'report.json', report)
    print('FINAL', json.dumps(report['oof_scores']), flush=True)


if __name__ == '__main__':
    train()
