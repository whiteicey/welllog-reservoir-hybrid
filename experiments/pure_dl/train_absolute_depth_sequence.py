"""Fixed grouped CV for dense multiscale well-depth networks and loss ablation."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold
import train_invariant as base
from multiscale_sequence import objective, window_indices
from absolute_depth_sequence import Network
from fold_screening import qualifies, THRESHOLD
from gpu_job_guard import check_slot


def main():
    ap = argparse.ArgumentParser()
    ap.set_defaults(deep_supervision=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fold-count', type=int, choices=[1,5], default=1)
    args = ap.parse_args()
    if args.fold_count == 5:
        proof = json.loads((args.out/'screening_verification.json').read_text())
        assert proof['passed'] and qualifies(proof['first_fold_score']), 'First fold must be independently verified above82.5'
    torch.set_num_threads(3)
    print('GPU_ADMISSION', check_slot(), flush=True)
    start = time.perf_counter(); args.out.mkdir(parents=True, exist_ok=True)
    rec = base.load_split(base.ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(v).all(1) for v in ys]
    truth = np.concatenate([v[m] for v, m in zip(ys, valid)])
    well = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=well)):
        fold[ix] = f
    row_ids = [np.flatnonzero(well == i) for i in range(80)]
    starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
    nraw = int(starts[-1])
    valid_to_raw = np.concatenate([np.flatnonzero(m)+starts[i] for i, m in enumerate(valid)])
    raw_to_valid = np.full(nraw, -1, np.int64); raw_to_valid[valid_to_raw] = np.arange(len(truth))
    lower = torch.tensor(np.concatenate([np.full(len(frame), starts[i]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    upper = torch.tensor(np.concatenate([np.full(len(frame), starts[i+1]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    depth = torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _, frame in rec]), device='cuda')
    assert torch.isfinite(depth).all()
    sources = [Path(__file__), Path(__file__).with_name('multiscale_sequence.py'),
               Path(__file__).with_name('absolute_depth_sequence.py'), Path(__file__).with_name('fold_screening.py'), Path(__file__).with_name('gpu_job_guard.py'), Path(base.__file__),
               Path(__file__).with_name('train.py'), base.ROOT/'package/src/feature.py',
               base.ROOT/'package/src/data.py', base.ROOT/'package/src/utils.py']
    hashes = {str(p.relative_to(base.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    contract = base.feature_contract(); base.write(args.out/'feature_contract.json', contract)
    cfg = {'seed': 20261006, 'epochs': 40, 'steps': 64, 'windows_per_step': 64,
           'window_length': 256, 'supervised_core': [64, 192], 'deep_supervision': args.deep_supervision,
           'auxiliary_weights': [.2, .1] if args.deep_supervision else [],
           'architecture': 'baseline U-Net plus physical absolute depth channel',
           'absolute_depth': 'train labeled rows median/IQR; clipped [-8,8]; zero initialized extra stem channel', 'screening_threshold': THRESHOLD,
           'features': 206, 'input_encoding': 'robust scaling only; no PLE or learned per-feature embedding',
           'parameters': sum(p.numel() for p in Network().parameters()), 'dependency_sha256': hashes,
           'test_access': False, 'label_scope': 'only64 training wells allocated for loss; validation labels used after inference',
           'selection': 'fixed final epoch and fixed whole-well grouped5-fold; fixed absolute depth channel; default first-fold screening; resume remaining folds only after independently verified first fold >82.5',
           'outer_wells': {str(f): [rec[i][0] for i in range(80) if fold[row_ids[i][0]] == f] for f in range(5)}}
    if (args.out/'config.json').exists():
        assert json.loads((args.out/'config.json').read_text()) == cfg
    base.write(args.out/'config.json', cfg)
    defaults = base.dmask(truth); pred = np.full_like(truth, np.nan); reports = []
    core = torch.zeros(1, 256, device='cuda', dtype=torch.bool); core[:, 64:192] = True
    for f in range(args.fold_count):
        if f > 0:
            assert qualifies(base.metric(truth[fold == 0], pred[fold == 0])['TOTAL']), 'Cached first-fold score fails gate'
        t0 = time.perf_counter(); torch.manual_seed(cfg['seed']+f); np.random.seed(cfg['seed']+f)
        validation = np.flatnonzero(fold == f); training = np.flatnonzero(fold != f)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for suffix in ['_report.json', '_distributions.npz', '.pt']:
                assert (args.out/f'fold_{f+1}{suffix}').exists()
            cached_checkpoint = torch.load(args.out/f'fold_{f+1}.pt', map_location='cpu', weights_only=False)
            assert cached_checkpoint['config'] == cfg
            del cached_checkpoint
            print(f'REUSE_VERIFIED_CONFIG_FOLD {f+1}', flush=True)
            pred[validation] = np.load(cache)
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text())); continue
        tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != f]
        print(f'FOLD {f+1} train-only preprocessing; deep_supervision={args.deep_supervision}', flush=True)
        valid_x, builder, fm, fs, edges = base.prepare_features(rec, tr_wells, valid, row_ids)
        x = torch.empty(nraw, 206, dtype=torch.float16, device='cuda')
        for i, (_, frame) in enumerate(rec):
            value = builder.transform(frame).enhanced[:, contract['selected_indices']]
            value = np.clip((value-fm)/fs, -8, 8).astype(np.float16)
            x[starts[i]:starts[i+1]] = torch.from_numpy(value).cuda()
        assert torch.equal(x[torch.tensor(valid_to_raw, device='cuda')], valid_x)
        del valid_x, value
        med = []; scale = []; z = []; centers = []; density_edges = []; initial = []
        for j in range(3):
            transformed = base.TRANSFORM[j](truth[:, j])
            fit = transformed[(fold != f) & ~defaults[:, j]]
            q25, q50, q75 = np.percentile(fit, [25, 50, 75]); sc = max(float(q75-q25), .1)
            med.append(q50); scale.append(sc); z.append((transformed-q50)/sc)
            initial.append(np.quantile((fit-q50)/sc, np.linspace(.05, .95, 8)))
            ct = np.linspace(fit.min(), fit.max(), 256); centers.append(base.INVERSE[j](ct))
            density_edges.append((np.r_[-np.inf, (ct[:-1]+ct[1:])/2, np.inf]-q50)/sc)
        train_raw_np = valid_to_raw[training]
        train_raw = torch.tensor(train_raw_np, device='cuda')
        labels = torch.zeros(nraw, 3, device='cuda')
        state = torch.zeros(nraw, 3, dtype=torch.bool, device='cuda')
        labeled = torch.zeros(nraw, dtype=torch.bool, device='cuda')
        labels[train_raw] = torch.tensor(np.column_stack(z)[training], device='cuda', dtype=torch.float32)
        state[train_raw] = torch.tensor(defaults[training], device='cuda'); labeled[train_raw] = True
        assert not bool(labeled[torch.tensor(valid_to_raw[validation], device='cuda')].any())
        depth_fit = depth[train_raw].cpu().numpy().astype(np.float64)
        depth_q = np.percentile(depth_fit, [25,50,75])
        depth_median = float(depth_q[1]); depth_scale = max(float(depth_q[2]-depth_q[0]), .1)
        absolute = ((depth-depth_median)/depth_scale).clamp(-8,8)
        model = Network(initial_means=np.stack(initial)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40*64, eta_min=.00001)
        history = []; torch.cuda.reset_peak_memory_stats()
        for epoch in range(40):
            model.train(); total = 0.
            for _ in range(64):
                anchors = train_raw[torch.randint(len(train_raw), (64,), device='cuda')]
                ix, exists = window_indices(anchors, lower, upper)
                relative = torch.tanh((depth[ix]-depth[anchors, None])/10)
                supervised = exists & core & labeled[ix]
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(x[ix], relative, exists, auxiliary=args.deep_supervision, absolute_depth=absolute[ix])
                loss = objective(outputs, labels[ix], state[ix], supervised, args.deep_supervision)
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step(); sched.step(); total += float(loss.detach())
            history.append(total/64)
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/40 loss={history[-1]:.5f} seconds={time.perf_counter()-t0:.1f}', flush=True)
        actions = []; utility = []
        for j, name in enumerate(base.TS):
            ct = centers[j]
            action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800)
            action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
            actions.append(action)
            utility.append(torch.tensor(base.kernel_matrix(action, np.r_[base.DEFAULTS[name], ct], name).T, device='cuda', dtype=torch.float32))
        local = np.full(len(truth), -1, np.int64); local[validation] = np.arange(len(validation))
        covered = np.zeros(len(validation), np.int8)
        pack = {'indices': validation, 'gate': np.empty((len(validation), 3), np.float32),
                'mixture_weight': np.empty((len(validation), 3, 8), np.float16),
                'mixture_mu': np.empty((len(validation), 3, 8), np.float16),
                'mixture_std': np.empty((len(validation), 3, 8), np.float16)}
        for j, name in enumerate(base.TS):
            pack[name+'_prob'] = np.empty((len(validation), 256), np.float16); pack[name+'_centers'] = centers[j]
        eval_centers = []; eval_anchors = []
        for wi in range(80):
            if fold[row_ids[wi][0]] != f:
                continue
            cc = np.arange(starts[wi], starts[wi+1], 128)+64
            eval_centers.extend(cc); eval_anchors.extend(np.minimum(cc, starts[wi+1]-1))
        model.eval()
        with torch.no_grad():
            for begin in range(0, len(eval_centers), 32):
                cc = torch.tensor(eval_centers[begin:begin+32], device='cuda')
                anchor = torch.tensor(eval_anchors[begin:begin+32], device='cuda')
                ix, exists = window_indices(cc, lower, upper, anchors=anchor)
                relative = torch.tanh((depth[ix]-depth[anchor, None])/10)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    go, raw = model(x[ix], relative, exists, absolute_depth=absolute[ix])
                original = ix[:, 64:192].cpu().numpy(); global_rows = raw_to_valid[original]
                use = exists[:, 64:192].cpu().numpy() & (global_rows >= 0)
                ids = global_rows[use]; loc = local[ids]
                assert (loc >= 0).all() and (covered[loc] == 0).all()
                assert len(np.unique(loc)) == len(loc)
                covered[loc] += 1
                select = torch.from_numpy(use).cuda()
                gate = go[:, 64:192][select].float().sigmoid()
                rr = raw[:, 64:192][select].float()
                weight = rr[:, :, 0].softmax(-1); mu = rr[:, :, 1]; sd = F.softplus(rr[:, :, 2])+.025
                pack['gate'][loc] = gate.cpu().numpy()
                for key, value in [('mixture_weight', weight), ('mixture_mu', mu), ('mixture_std', sd)]:
                    pack[key][loc] = value.cpu().numpy().astype(np.float16)
                for j, name in enumerate(base.TS):
                    de = torch.tensor(density_edges[j], device='cuda', dtype=torch.float32)
                    cp = base.conditional_probability(weight[:, j], mu[:, j], sd[:, j], de)
                    joint = torch.cat([gate[:, j:j+1], (1-gate[:, j:j+1])*cp], -1)
                    pred[ids, j] = actions[j][(joint@utility[j]).argmax(-1).cpu().numpy()]
                    pack[name+'_prob'][loc] = cp.cpu().numpy().astype(np.float16)
        assert (covered == 1).all() and np.isfinite(pred[validation]).all()
        np.savez_compressed(args.out/f'fold_{f+1}_distributions.npz', **pack)
        checkpoint = {'state_dict': model.cpu().state_dict(), 'config': cfg, 'absolute_depth_median': depth_median, 'absolute_depth_scale': depth_scale,
            'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'ple_edges_normalized': edges.tolist(),
            'medians': builder.medians_, 'selected_feature_names': contract['whitelist'],
            'selected_feature_indices': contract['selected_indices'], 'regression_median': med,
            'regression_scale': scale, 'centers': centers, 'density_edges': density_edges,
            'training_raw_rows': train_raw_np, 'validation_raw_rows': valid_to_raw[validation]}
        torch.save(checkpoint, args.out/f'fold_{f+1}.pt')
        fr = {'fold': f+1, 'scores': base.metric(truth[validation], pred[validation]),
              'seconds': time.perf_counter()-t0, 'loss': history, 'all_validation_rows_covered_once': True,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        base.write(args.out/f'fold_{f+1}_report.json', fr); reports.append(fr)
        np.save(cache, pred[validation]); np.savez_compressed(args.out/'oof.npz', truth=truth, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, sched, x, labels, state, labeled, train_raw, outputs, loss, utility, pack, checkpoint, go, raw, rr
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred[fold < args.fold_count]).all()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(base.ROOT))] for p in sources)
    elapsed = time.perf_counter()-start
    first_scores = base.metric(truth[fold == 0], pred[fold == 0])
    screening = {'config': cfg, 'complete_five_folds': args.fold_count == 5,
                 'fold1_scores': first_scores, 'screening_passed': qualifies(first_scores['TOTAL']),
                 'threshold': THRESHOLD, 'folds': reports, 'seconds_this_invocation': elapsed}
    base.write(args.out/'screening_report.json', screening)
    if args.fold_count == 1:
        assert np.isnan(pred[fold != 0]).all()
        print('FIRST_FOLD_SCREEN_ONLY', json.dumps(first_scores), flush=True)
        return
    assert np.isfinite(pred).all()
    report = {'config': cfg, 'oof_scores': base.metric(truth, pred), 'folds': reports,
              'seconds': sum(r['seconds'] for r in reports), 'seconds_this_invocation': elapsed,
              'timing_scope': 'sum of fold work across screening and continuation; outside-fold startup excluded'}
    base.write(args.out/'report.json', report)
    print('FINAL', json.dumps(report['oof_scores']), flush=True)


if __name__ == '__main__':
    main()
