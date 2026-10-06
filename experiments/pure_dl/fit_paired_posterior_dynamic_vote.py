"""Fit a small posterior-conditioned vote router on fold-five training wells."""
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
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from por_t3_patch_loss import components
from posterior_vote_router import posterior_summary, VoteRouter


ROOT = base.ROOT
OUT = ROOT/'outputs/pure_dl_posterior_dynamic_vote_fold5'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261007, ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261007'),
    (20261008, ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261008'),
]


def main():
    torch.set_num_threads(3)
    started = time.perf_counter()
    checkpoints = []
    for seed, folder in SOURCES:
        path = folder/'fold_5.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['config']['seed'] == seed
        assert json.loads((folder/'fold_5_report.json').read_text(encoding='utf-8'))['best_checkpoint_predictions_reproduced']
        checkpoints.append((seed, checkpoint, hashlib.sha256(path.read_bytes()).hexdigest()))
    master = checkpoints[0][1]
    for _, checkpoint, _ in checkpoints[1:]:
        for field in ['selected_feature_names', 'selected_feature_indices',
                      'feature_median', 'feature_scale', 'regression_median',
                      'regression_scale', 'absolute_depth_median', 'absolute_depth_scale']:
            assert np.array_equal(checkpoint[field], master[field]), field
        assert np.array_equal(checkpoint['validation_raw_rows'], master['validation_raw_rows'])
    rec = base.load_split(ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    truth = np.concatenate([y[m] for y, m in zip(ys, valid)])
    wells = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    assert len(rec) == 80 and truth.shape == (723556, 3)
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=wells)):
        fold[ix] = f
    training = np.flatnonzero(fold != 4)
    validation = np.flatnonzero(fold == 4)
    assert len(validation) == 144665
    row_ids = [np.flatnonzero(wells == i) for i in range(80)]
    tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != 4]
    assert len(tr_wells) == 64
    starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
    nraw = int(starts[-1])
    valid_to_raw = np.concatenate([np.flatnonzero(m)+starts[i] for i, m in enumerate(valid)])
    assert np.array_equal(valid_to_raw[validation], master['validation_raw_rows'])
    raw_to_valid = np.full(nraw, -1, np.int64)
    raw_to_valid[valid_to_raw] = np.arange(len(truth))
    valid_x, builder, fm, fs, _ = base.prepare_features(rec, tr_wells, valid, row_ids)
    assert np.array_equal(fm, master['feature_median'])
    assert np.array_equal(fs, master['feature_scale'])
    contract = base.feature_contract()
    assert contract['whitelist'] == master['selected_feature_names']
    view_indices = retained_indices(contract['whitelist'])
    numeric_indices, conditioning_indices = calibration_indices(contract['whitelist'])
    x = torch.empty(nraw, 206, dtype=torch.float16, device='cuda')
    for i, (_, frame) in enumerate(rec):
        value = builder.transform(frame).enhanced[:, contract['selected_indices']]
        value = np.clip((value-fm)/fs, -8, 8).astype(np.float16)
        x[starts[i]:starts[i+1]] = torch.from_numpy(value).cuda()
    assert torch.equal(x[torch.as_tensor(valid_to_raw, device='cuda')], valid_x)
    del valid_x
    depth = torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32)
                                         for _, frame in rec]), device='cuda')
    lower = torch.tensor(np.concatenate([np.full(len(frame), starts[i])
                                         for i, (_, frame) in enumerate(rec)]), device='cuda')
    upper = torch.tensor(np.concatenate([np.full(len(frame), starts[i+1])
                                         for i, (_, frame) in enumerate(rec)]), device='cuda')
    absolute = ((depth-master['absolute_depth_median'])/
                master['absolute_depth_scale']).clamp(-8, 8)
    models = []
    for _, checkpoint, _ in checkpoints:
        model = Network(retain_indices=view_indices, numeric_indices=numeric_indices,
                        conditioning_indices=conditioning_indices).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False)
        models.append(model.eval())
    train_centers, train_anchors = [], []
    for wi in tr_wells:
        length = starts[wi+1]-starts[wi]
        assert length >= 512
        cc = np.rint(np.linspace(starts[wi]+64, starts[wi+1]-64, 9)[1::2]).astype(np.int64)
        assert len(cc) == 4
        train_centers.extend(cc)
        train_anchors.extend(cc)
    transformed = np.column_stack([base.TRANSFORM[j](truth[:, j]) for j in range(3)])
    z = (transformed-np.asarray(master['regression_median']))/np.asarray(master['regression_scale'])
    default = base.dmask(truth)
    logp_batches, summary_batches, well_batches, raw_batches = [], [], [], []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for begin in range(0, len(train_centers), 32):
            cc = torch.tensor(train_centers[begin:begin+32], device='cuda')
            anchor = torch.tensor(train_anchors[begin:begin+32], device='cuda')
            ix, exists = window_indices(cc, lower, upper, anchors=anchor)
            relative = torch.tanh((depth[ix]-depth[anchor, None])/10)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = [model(x[ix], relative, exists, absolute_depth=absolute[ix])
                           for model in models]
            raw_rows = ix[:, 64:192].cpu().numpy()
            valid_rows = raw_to_valid[raw_rows]
            use = exists[:, 64:192].cpu().numpy() & (valid_rows >= 0)
            ids = valid_rows[use]
            assert (fold[ids] != 4).all(), 'Meta-weight fitting must not see validation rows'
            selected_raw = raw_rows[use]
            target = torch.tensor(z[ids], device='cuda', dtype=torch.float32)
            is_default = torch.tensor(default[ids], device='cuda', dtype=torch.bool)
            select = torch.from_numpy(use).cuda()
            model_logp, model_summary = [], []
            for gate_logits, raw in outputs:
                gate = gate_logits[:, 64:192][select].float()
                r = raw[:, 64:192][select].float()
                conditional = (r[:, :, 0].log_softmax(-1)+components(r, target)).logsumexp(-1)
                logp = torch.where(is_default, F.logsigmoid(gate),
                                   F.logsigmoid(-gate)+conditional)
                model_logp.append(logp)
                model_summary.append(posterior_summary(gate, r))
            logp_batches.append(torch.stack(model_logp, dim=1).cpu())
            summary_batches.append(torch.cat(model_summary, dim=-1).cpu())
            well_batches.append(wells[ids])
            raw_batches.append(selected_raw)
    logp = torch.cat(logp_batches, dim=0)
    summaries = torch.cat(summary_batches, dim=0)
    sample_well = np.concatenate(well_batches)
    sampled_raw = np.concatenate(raw_batches)
    assert logp.shape[1:] == (3, 3) and torch.isfinite(logp).all()
    assert summaries.shape[1:] == (3, 12) and torch.isfinite(summaries).all()
    assert len(sampled_raw) == len(np.unique(sampled_raw))
    assert not np.isin(sampled_raw, valid_to_raw[validation]).any()
    counts = np.bincount(sample_well, minlength=80)
    assert np.count_nonzero(counts) == 64 and np.all(counts[tr_wells] > 0)
    logp = logp.permute(0, 2, 1).contiguous()
    inner_val_wells = tr_wells[::4]
    assert len(inner_val_wells) == 16
    fit_mask = torch.from_numpy(~np.isin(sample_well, inner_val_wells))
    inner_mask = ~fit_mask
    assert int(fit_mask.sum()) == 48*512 and int(inner_mask.sum()) == 16*512
    torch.manual_seed(20261101)
    router = VoteRouter()
    optimizer = torch.optim.Adam(router.parameters(), lr=.01, weight_decay=.001)
    def objective(mask):
        vote = router(summaries[mask])
        mixed = torch.logsumexp(logp[mask]+vote.clamp_min(1e-8).log(), dim=-1)
        data_loss = -mixed.mean()
        penalty = .01*(vote-1/3).square().mean()
        return data_loss+penalty, data_loss, penalty
    router.eval()
    with torch.no_grad():
        initial_inner_loss = float(objective(inner_mask)[0])
    best_inner_loss = float('inf')
    best_epoch = 0
    bad_epochs = 0
    best_state = None
    history = []
    for epoch in range(1, 81):
        router.train()
        optimizer.zero_grad(set_to_none=True)
        loss, train_data, train_penalty = objective(fit_mask)
        loss.backward()
        optimizer.step()
        router.eval()
        with torch.no_grad():
            inner_loss, inner_data, inner_penalty = objective(inner_mask)
        val = float(inner_loss)
        if val < best_inner_loss:
            best_inner_loss = val
            best_epoch = epoch
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
        else:
            bad_epochs += 1
        history.append({'epoch': epoch, 'fit_loss': float(loss.detach()),
                        'inner_loss': val, 'best_epoch': best_epoch,
                        'bad_epochs': bad_epochs})
        if epoch >= 10 and bad_epochs >= 18:
            break
    stopped_epoch = epoch
    assert best_state is not None
    router.load_state_dict(best_state)
    router.eval()
    with torch.no_grad():
        restored_inner_loss = float(objective(inner_mask)[0])
        average_votes = router(summaries).mean(0).numpy()
    assert abs(restored_inner_loss-best_inner_loss) < 1e-8
    assert np.isfinite(average_votes).all() and np.allclose(average_votes.sum(-1), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'screening_fold': 5, 'seed_bases': [seed for seed, _, _ in checkpoints],
              'mean_train_weights_by_target': {name: average_votes[j].tolist() for j, name in enumerate(base.TS)},
              'target_order': list(base.TS), 'member_order': [seed for seed, _, _ in checkpoints],
              'train_wells': len(tr_wells), 'sampled_train_points': len(sampled_raw),
              'sampled_points_per_well_min': int(counts[tr_wells].min()),
              'sampled_points_per_well_max': int(counts[tr_wells].max()),
              'router_fit_wells': 48, 'router_inner_validation_wells': 16,
              'router_inner_validation_well_ids': [int(i) for i in inner_val_wells],
              'validation_points_used_to_fit': 0,
              'objective': 'frozen-expert predictive-mixture NLL plus 0.01 mean squared distance from uniform',
              'optimizer': 'PyTorch Adam lr0.01, weight_decay0.001, max80 epochs, patience18, best inner-training-well checkpoint',
              'router_parameters': sum(p.numel() for p in router.parameters()),
              'initial_inner_loss': initial_inner_loss,
              'best_inner_loss': best_inner_loss, 'best_epoch': best_epoch,
              'stopped_epoch': stopped_epoch,
              'checkpoint_sha256': {str(seed): sha for seed, _, sha in checkpoints},
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'wall_seconds': time.perf_counter()-started}
    torch.save({'state_dict': best_state, 'report': result}, OUT/'router.pt')
    (OUT/'router_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (OUT/'router_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
