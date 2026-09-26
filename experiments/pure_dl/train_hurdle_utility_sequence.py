"""Fixed grouped CV for gated conditional expected-official-reward prediction."""
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
from multiscale_sequence import window_indices
from hurdle_utility_sequence import Network, objective, fit_priors, expected_utilities
from action_sequence_utils import make_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(3)
    assert torch.cuda.mem_get_info()[0] >= 6*1024**3
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
    sources = [Path(__file__), Path(__file__).with_name('multiscale_sequence.py'), Path(base.__file__),
               Path(__file__).with_name('hurdle_utility_sequence.py'), Path(__file__).with_name('score_aligned_loss.py'),
               Path(__file__).with_name('action_sequence_utils.py'),
               Path(__file__).with_name('train.py'), base.ROOT/'package/src/feature.py',
               base.ROOT/'package/src/data.py', base.ROOT/'package/src/utils.py']
    hashes = {str(p.relative_to(base.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    contract = base.feature_contract(); base.write(args.out/'feature_contract.json', contract)
    grid_manifest = {str(f): [a.tolist() for a in make_actions(truth[fold != f])] for f in range(5)}
    cfg = {'seed': 20261006, 'epochs': 40, 'steps': 64, 'windows_per_step': 64,
           'window_length': 256, 'supervised_core': [64, 192], 'action_grids': grid_manifest,
           'loss': 'default gate BCE plus nondefault conditional soft-BCE against official reward; fixed nondefault mean reward-mass scalar; conditional task weights .3/.35/.35',
           'decode': 'argmax of gate*default official reward plus (1-gate)*sigmoid conditional action logits; absolute expected utilities, not softmax densities',
           'architecture': 'dense depth U-Net; gated dilated depthwise convolutions; pointwise MLP skip; default gates and conditional action-utility logits per target',
           'features': 206, 'input_encoding': 'robust scaling only; no PLE or learned per-feature embedding',
           'parameters_by_fold': {f: sum(p.numel() for p in Network([len(a) for a in aa]).parameters()) for f, aa in grid_manifest.items()}, 'dependency_sha256': hashes,
           'test_access': False, 'label_scope': 'only64 training wells allocated for loss; validation labels used after inference',
           'selection': 'fixed final epoch and fixed whole-well grouped5-fold; one gated utility candidate declared before scores; backbone and update budget retained',
           'outer_wells': {str(f): [rec[i][0] for i in range(80) if fold[row_ids[i][0]] == f] for f in range(5)}}
    if (args.out/'config.json').exists():
        assert json.loads((args.out/'config.json').read_text()) == cfg
    base.write(args.out/'config.json', cfg)
    defaults = base.dmask(truth); pred = np.full_like(truth, np.nan); reports = []
    core = torch.zeros(1, 256, device='cuda', dtype=torch.bool); core[:, 64:192] = True
    for f in range(5):
        t0 = time.perf_counter(); torch.manual_seed(cfg['seed']+f); np.random.seed(cfg['seed']+f)
        validation = np.flatnonzero(fold == f); training = np.flatnonzero(fold != f)
        cache = args.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            for suffix in ['_report.json', '_actions.npz', '.pt']:
                assert (args.out/f'fold_{f+1}{suffix}').exists()
            pred[validation] = np.load(cache)
            reports.append(json.loads((args.out/f'fold_{f+1}_report.json').read_text())); continue
        tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != f]
        print(f'FOLD {f+1} train-only preprocessing; gated conditional official utility', flush=True)
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
        labeled = torch.zeros(nraw, dtype=torch.bool, device='cuda')
        labels[train_raw] = torch.tensor(truth[training], device='cuda', dtype=torch.float32)
        labeled[train_raw] = True
        states = torch.zeros(nraw, 3, dtype=torch.bool, device='cuda')
        states[train_raw] = torch.tensor(defaults[training], device='cuda')
        assert not bool(labeled[torch.tensor(valid_to_raw[validation], device='cuda')].any())
        actions = [np.asarray(a) for a in grid_manifest[str(f)]]
        action_tensors = [torch.tensor(a, device='cuda', dtype=torch.float32) for a in actions]
        gate_prior, conditional_priors, normalizers = fit_priors(labels[train_raw], states[train_raw], action_tensors)
        model = Network([len(a) for a in actions], conditional_priors, gate_prior).cuda()
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
                    outputs = model(x[ix], relative, exists)
                loss = objective(outputs, labels[ix], states[ix], supervised, action_tensors, normalizers)
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                opt.step(); sched.step(); total += float(loss.detach())
            history.append(total/64)
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(f'fold={f+1} epoch={epoch+1}/40 loss={history[-1]:.5f} seconds={time.perf_counter()-t0:.1f}', flush=True)
        local = np.full(len(truth), -1, np.int64); local[validation] = np.arange(len(validation))
        covered = np.zeros(len(validation), np.int8)
        witness_local = np.linspace(0, len(validation)-1, 64, dtype=int)
        witness_slot = np.full(len(validation), -1, np.int64); witness_slot[witness_local] = np.arange(64)
        witness_covered = np.zeros(64, dtype=bool)
        pack = {'indices': validation, 'action_index': np.empty((len(validation), 3), np.uint16),
                'selected_expected_utility': np.empty((len(validation), 3), np.float32),
                'action_margin': np.empty((len(validation), 3), np.float32),
                'witness_indices': validation[witness_local],
                'gate_probability': np.empty((len(validation), 3), np.float32),
                'witness_gate_logits': np.empty((64, 3), np.float32),
                'witness_features': np.empty((64, 256, 206), np.float16),
                'witness_relative_depth': np.empty((64, 256), np.float32),
                'witness_valid': np.empty((64, 256), bool),
                'witness_position': np.empty(64, np.int16),
                'witness_center': np.empty(64, np.int64), 'witness_anchor': np.empty(64, np.int64)}
        for j, name in enumerate(base.TS):
            pack[name+'_actions'] = actions[j]
            pack[name+'_witness_conditional_logits'] = np.empty((64, len(actions[j])), np.float32)
            pack[name+'_witness_utilities'] = np.empty((64, len(actions[j])), np.float32)
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
                batch_x = x[ix]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    decoded_output = model(batch_x, relative, exists)
                joint_scores = expected_utilities(decoded_output, action_tensors)
                original = ix[:, 64:192].cpu().numpy(); global_rows = raw_to_valid[original]
                use = exists[:, 64:192].cpu().numpy() & (global_rows >= 0)
                ids = global_rows[use]; loc = local[ids]
                assert (loc >= 0).all() and (covered[loc] == 0).all()
                assert len(np.unique(loc)) == len(loc); covered[loc] += 1
                select = torch.from_numpy(use).cuda()
                slots = witness_slot[loc]; hit = slots >= 0
                hit_gpu = torch.from_numpy(hit).cuda()
                selected_gate = decoded_output[0][:, 64:192][select].float()
                pack['gate_probability'][loc] = selected_gate.sigmoid().cpu().numpy()
                if hit.any():
                    coordinates = torch.nonzero(select)[hit_gpu]
                    rows = coordinates[:, 0]; ws = slots[hit]
                    pack['witness_features'][ws] = batch_x[rows].cpu().numpy()
                    pack['witness_relative_depth'][ws] = relative[rows].cpu().numpy()
                    pack['witness_valid'][ws] = exists[rows].cpu().numpy()
                    pack['witness_position'][ws] = (coordinates[:, 1]+64).cpu().numpy()
                    pack['witness_center'][ws] = cc[rows].cpu().numpy()
                    pack['witness_anchor'][ws] = anchor[rows].cpu().numpy()
                    pack['witness_gate_logits'][ws] = selected_gate[hit_gpu].cpu().numpy()
                    witness_covered[ws] = True
                for j, name in enumerate(base.TS):
                    score = joint_scores[j][:, 64:192][select].float()
                    assert torch.isfinite(score).all()
                    assert ((score >= 0) & (score <= 1)).all()
                    probability = score
                    winner = score.argmax(-1)
                    best_two = probability.topk(2, dim=-1).values
                    selected_probability = probability.gather(1, winner[:, None]).squeeze(1)
                    assert torch.allclose(selected_probability, best_two[:, 0])
                    winner_np = winner.cpu().numpy()
                    pred[ids, j] = actions[j][winner_np]
                    pack['action_index'][loc, j] = winner_np
                    pack['selected_expected_utility'][loc, j] = selected_probability.cpu().numpy()
                    pack['action_margin'][loc, j] = (best_two[:, 0]-best_two[:, 1]).cpu().numpy()
                    if hit.any():
                        pack[name+'_witness_utilities'][slots[hit]] = score[hit_gpu].cpu().numpy()
                        pack[name+'_witness_conditional_logits'][slots[hit]] = decoded_output[1][j][:, 64:192][select].float()[hit_gpu].cpu().numpy()
        assert (covered == 1).all() and witness_covered.all() and np.isfinite(pred[validation]).all()
        # Full logits are reproducible from weights; keep64 fixed input/output witnesses
        # and all action choices, avoiding several GB of unused density-like artifacts.
        np.savez_compressed(args.out/f'fold_{f+1}_actions.npz', **pack)
        checkpoint = {'state_dict': model.cpu().state_dict(), 'config': cfg,
            'feature_median': fm.tolist(), 'feature_scale': fs.tolist(), 'ple_edges_normalized': edges.tolist(),
            'medians': builder.medians_, 'selected_feature_names': contract['whitelist'],
            'selected_feature_indices': contract['selected_indices'], 'regression_median': med,
            'regression_scale': scale, 'actions': actions, 'loss_normalizers': normalizers,
            'default_prior': gate_prior, 'conditional_priors': conditional_priors,
            'training_raw_rows': train_raw_np, 'validation_raw_rows': valid_to_raw[validation]}
        torch.save(checkpoint, args.out/f'fold_{f+1}.pt')
        fr = {'fold': f+1, 'scores': base.metric(truth[validation], pred[validation]),
              'seconds': time.perf_counter()-t0, 'loss': history, 'all_validation_rows_covered_once': True,
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        base.write(args.out/f'fold_{f+1}_report.json', fr); reports.append(fr)
        np.save(cache, pred[validation]); np.savez_compressed(args.out/'oof.npz', truth=truth, pred=pred, fold=fold, well=well)
        print('FOLD_RESULT', json.dumps({k: v for k, v in fr.items() if k != 'loss'}), flush=True)
        del model, opt, sched, x, labels, labeled, states, train_raw, outputs, loss, action_tensors, pack, checkpoint, decoded_output, joint_scores, selected_gate, score, probability, batch_x
        gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(base.ROOT))] for p in sources)
    report = {'config': cfg, 'oof_scores': base.metric(truth, pred), 'folds': reports, 'seconds': time.perf_counter()-start}
    base.write(args.out/'report.json', report)
    print('FINAL', json.dumps(report['oof_scores']), flush=True)


if __name__ == '__main__':
    main()
