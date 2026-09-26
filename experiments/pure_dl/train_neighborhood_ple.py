"""Fold-local PLE encoding for jointly learned cross-well kernel density."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import train_invariant as base
from neighborhood_ple import NeighborhoodDensity
from retrieval_network import validate_memory_split

ROOT = base.ROOT
OUT = ROOT/'outputs/pure_dl_neighborhood_ple'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(3)
    if torch.cuda.mem_get_info()[0] < 8*1024**3:
        raise RuntimeError('Requires 8 GiB free before starting')
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rec = base.load_split(ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    y = np.concatenate([v[m] for v, m in zip(ys, valid)])
    well = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and y.shape == (723556, 3)
    fold = np.full(len(y), -1, np.int8)
    for f, (_, rows) in enumerate(GroupKFold(5).split(y, groups=well)):
        fold[rows] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    contract = base.feature_contract()
    base.write(OUT/'feature_contract.json', contract)
    sources = [Path(__file__), Path(__file__).with_name('neighborhood_ple.py'), Path(__file__).with_name('neighborhood_density.py'),
               Path(base.__file__), ROOT/'package/src/feature.py',
               Path(__file__).with_name('retrieval_network.py')]
    config = {'seed': 20261005, 'epochs': 40, 'steps': 64, 'batch_points': 512,
              'memory_per_well_per_update': 128, 'inference_memory': 'all outer-training rows',
              'features': 206, 'numeric_encoding': 'train-input-only16-segment PLE plus raw; padded3504', 'hidden': 128, 'embedding_per_target': 32,
              'lr': .001, 'weight_decay': .01, 'test_access': False,
              'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
              'outer_wells': {str(f): [rec[i][0] for i in range(80) if fold[row_ids[i][0]] == f] for f in range(5)},
              'selection': 'fixed final epoch; no validation-based choices; no pretrained weights',
              'loss': 'mean gate log likelihood plus equal-target nondefault Gaussian kernel NLL',
              'memory_policy': 'resample128 rows per training well each update; exclude complete query well; full64-well memory at evaluation',
              'decode': '256 training-target support bins; fixed800 actions with default atom',
              'parameters': sum(p.numel() for p in NeighborhoodDensity().parameters())}
    if (OUT/'config.json').exists():
        assert json.loads((OUT/'config.json').read_text()) == config, 'Incompatible cached run'
    base.write(OUT/'config.json', config)
    default = base.dmask(y)
    pred = np.full_like(y, np.nan); reports = []
    for f in range(5):
        t0 = time.perf_counter()
        cache = OUT/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for suffix in ('report.json', 'distributions.npz'):
                assert (OUT/f'fold_{f+1}_{suffix}').exists()
            assert (OUT/f'fold_{f+1}.pt').exists()
            pred[fold == f] = np.load(cache)
            reports.append(json.loads((OUT/f'fold_{f+1}_report.json').read_text()))
            continue
        torch.manual_seed(config['seed']+f); np.random.seed(config['seed']+f)
        tr = [i for i in range(80) if fold[row_ids[i][0]] != f]
        training = np.flatnonzero(fold != f); validation = np.flatnonzero(fold == f)
        validate_memory_split(training, training, validation)
        print(f'FOLD {f+1} preparing64 training wells', flush=True)
        x, builder, fm, fs, ple_edges = base.prepare_features(rec, tr, valid, row_ids)
        med = []; scale = []; centers = []; edges = []; z = []
        for j in range(3):
            transformed = base.TRANSFORM[j](y[:, j])
            values = transformed[(fold != f) & ~default[:, j]]
            q25, q50, q75 = np.percentile(values, [25, 50, 75])
            sc = max(float(q75-q25), .1)
            med.append(q50); scale.append(sc); z.append((transformed-q50)/sc)
            ct = np.linspace(values.min(), values.max(), 256)
            centers.append(base.INVERSE[j](ct))
            edges.append((np.r_[-np.inf, (ct[:-1]+ct[1:])/2, np.inf]-q50)/sc)
        # Only training labels are allocated on GPU. Validation labels enter metrics after inference.
        tz = torch.tensor(np.column_stack(z)[training], device='cuda', dtype=torch.float32)
        td = torch.tensor(default[training], device='cuda', dtype=torch.bool)
        tw = torch.tensor(well[training], device='cuda', dtype=torch.long)
        ti = torch.tensor(training, device='cuda', dtype=torch.long)
        offsets = torch.tensor([np.searchsorted(training, row_ids[i][0]) for i in tr], device='cuda')
        lengths = torch.tensor([len(row_ids[i]) for i in tr], device='cuda')
        assert sum(lengths.tolist()) == len(training)
        model = NeighborhoodDensity(edges=ple_edges).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40*64, eta_min=.00001)
        torch.cuda.reset_peak_memory_stats(); history = []
        for epoch in range(40):
            model.train(); total = 0.
            for _ in range(64):
                q = torch.randint(len(training), (512,), device='cuda')
                m = (offsets[:, None]+(torch.rand(64, 128, device='cuda')*lengths[:, None]).long()).flatten()
                opt.zero_grad(set_to_none=True)
                loga = model.log_attention(x[ti[q]].float(), x[ti[m]].float(), tw[q], tw[m])
                default_logp = loga.masked_fill(~td[m].T[None], -torch.inf).logsumexp(-1)
                active_logp = loga.masked_fill(td[m].T[None], -torch.inf).logsumexp(-1)
                logweights = loga.masked_fill(td[m].T[None], -torch.inf)-active_logp[..., None]
                nll = model.conditional_nll(logweights, tz[q], tz[m])
                loss = -torch.where(td[q], default_logp, active_logp).mean()
                loss = loss+sum(nll[:, j][~td[q, j]].mean() for j in range(3))/3
                assert torch.isfinite(loss), 'Non-finite train-only loss'
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step(); sched.step(); total += float(loss.detach())
            history.append(total/64)
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/40 loss={history[-1]:.5f} seconds={time.perf_counter()-t0:.1f}', flush=True)
        model.eval()
        with torch.no_grad():
            memory = torch.cat([model.encode(x[ti[s:s+4096]].float()) for s in range(0, len(ti), 4096)])
            kernels = torch.empty(len(ti), 3, 256, device='cuda')
            sd = model.log_bandwidth.exp().clamp(.02, 1.)
            ed = torch.tensor(np.stack(edges), device='cuda', dtype=torch.float32)
            for start in range(0, len(ti), 4096):
                zz = (ed[None]-tz[start:start+4096, :, None])/sd[None, :, None]
                cdf = .5*(1+torch.erf(zz/np.sqrt(2)))
                cp = (cdf[..., 1:]-cdf[..., :-1]).clamp_min(0)
                kernels[start:start+4096] = cp/cp.sum(-1, keepdim=True).clamp_min(1e-12)
            actions = []; utilities = []
            for j, name in enumerate(base.TS):
                support = centers[j]
                action = np.linspace(support[0], support[-1], 800) if j == 2 else np.geomspace(max(support[0]*.9, .001), support[-1]*1.1, 800)
                action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
                actions.append(action)
                utilities.append(torch.tensor(base.kernel_matrix(action, np.r_[base.DEFAULTS[name], support], name).T, device='cuda', dtype=torch.float32))
            saved_gate = []; saved_probs = [[], [], []]
            for start in range(0, len(validation), 512):
                ix = validation[start:start+512]
                qr = torch.tensor(ix, device='cuda')
                qw = torch.tensor(well[ix], device='cuda')
                gate, cp = model.streaming_posterior(model.encode(x[qr].float()), memory, qw, tw, td, kernels)
                assert torch.isfinite(cp).all() and torch.isfinite(gate).all()
                saved_gate.append(gate.cpu().numpy())
                for j in range(3):
                    joint = torch.cat([gate[:, j:j+1], (1-gate[:, j:j+1])*cp[:, j]], -1)
                    pred[ix, j] = actions[j][(joint@utilities[j]).argmax(-1).cpu().numpy()]
                    saved_probs[j].append(cp[:, j].cpu().numpy().astype(np.float16))
        pack = {'indices': validation, 'gate': np.concatenate(saved_gate)}
        for j, name in enumerate(base.TS):
            pack[name+'_prob'] = np.concatenate(saved_probs[j]); pack[name+'_centers'] = centers[j]
        np.savez_compressed(OUT/f'fold_{f+1}_distributions.npz', **pack)
        checkpoint = {'state_dict': model.cpu().state_dict(), 'config': config,
            'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'ple_edges_normalized': ple_edges.tolist(),
            'medians': builder.medians_, 'selected_feature_names': contract['whitelist'],
            'selected_feature_indices': contract['selected_indices'], 'regression_median': med,
            'regression_scale': scale, 'centers': centers, 'density_edges': edges,
            'memory_rows': training, 'memory_wells': well[training], 'memory_embeddings': memory.cpu(),
            'memory_target_z': tz.cpu(), 'memory_default': td.cpu()}
        torch.save(checkpoint, OUT/f'fold_{f+1}.pt')
        fr = {'fold': f+1, 'scores': base.metric(y[validation], pred[validation]),
              'seconds': time.perf_counter()-t0, 'loss': history,
              'training_wells': [rec[i][0] for i in tr], 'validation_wells': config['outer_wells'][str(f)],
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        base.write(OUT/f'fold_{f+1}_report.json', fr); reports.append(fr)
        np.save(cache, pred[validation])
        np.savez_compressed(OUT/'oof.npz', truth=y, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, sched, x, tz, td, tw, ti, offsets, lengths, memory, kernels, utilities, checkpoint, pack, saved_gate, saved_probs
        del loga, logweights, default_logp, active_logp, nll, loss, cp, gate, joint, cdf, zz
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert all(digest(p) == config['source_sha256'][str(p.relative_to(ROOT))] for p in sources)
    report = {'config': config, 'oof_scores': base.metric(y, pred), 'folds': reports, 'seconds': time.perf_counter()-started}
    base.write(OUT/'report.json', report)
    print('FINAL', json.dumps(report['oof_scores']), flush=True)


if __name__ == '__main__':
    main()
