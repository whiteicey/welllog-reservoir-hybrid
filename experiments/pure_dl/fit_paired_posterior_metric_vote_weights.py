"""Fit nine vote logits to official-action reward on fold-five training wells."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

import train_invariant as base
from multiscale_sequence import window_indices
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from por_t3_patch_loss import probability


ROOT = base.ROOT
OUT = ROOT/'outputs/pure_dl_posterior_metric_vote_fold5'
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
    action_values, action_tensors, utilities, edges = [], [], [], []
    for j, name in enumerate(base.TS):
        ct = np.asarray(master['centers'][j])
        action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800)
        action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
        action_values.append(action)
        action_tensors.append(torch.tensor(action, device='cuda', dtype=torch.float32))
        utilities.append(torch.tensor(base.kernel_matrix(action,
                          np.r_[base.DEFAULTS[name], ct], name).T,
                          device='cuda', dtype=torch.float32))
        edges.append(torch.tensor(master['density_edges'][j], device='cuda', dtype=torch.float32))
    cross_batches, reward_batches, well_batches, raw_batches = [], [], [], []
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
            select = torch.from_numpy(use).cuda()
            target_cross, target_rewards = [], []
            for j, name in enumerate(base.TS):
                member_utilities = []
                for gate_logits, raw in outputs:
                    gate = gate_logits[:, 64:192][select][:, j].float().sigmoid()
                    r = raw[:, 64:192][select][:, j].float()
                    weight = r[:, 0].softmax(-1)
                    mu = r[:, 1]
                    sd = torch.nn.functional.softplus(r[:, 2])+.025
                    cp = probability(j, weight, mu, sd, edges[j])
                    joint = torch.cat([gate[:, None], (1-gate[:, None])*cp], -1)
                    member_utilities.append(joint@utilities[j])
                candidates = torch.stack([u.argmax(-1) for u in member_utilities], dim=-1)
                cross = torch.stack([u.gather(1, candidates) for u in member_utilities], dim=-1)
                action = action_tensors[j][candidates]
                y = torch.tensor(truth[ids, j], device='cuda', dtype=torch.float32)[:, None]
                if j == 1:
                    reward = (1-(action/y).clamp_min(1e-5).log10().abs()).clamp_min(0)
                else:
                    tolerance = .08 if j == 0 else .05
                    reward = (1-(action-y).abs()/(tolerance*(y.abs()+1e-5))).clamp_min(0)
                target_cross.append(cross)
                target_rewards.append(reward)
            cross_batches.append(torch.stack(target_cross, dim=1).cpu())
            reward_batches.append(torch.stack(target_rewards, dim=1).cpu())
            well_batches.append(wells[ids])
            raw_batches.append(selected_raw)
    cross_utility = torch.cat(cross_batches, dim=0)
    true_reward = torch.cat(reward_batches, dim=0)
    sample_well = np.concatenate(well_batches)
    sampled_raw = np.concatenate(raw_batches)
    assert cross_utility.shape[1:] == (3, 3, 3) and torch.isfinite(cross_utility).all()
    assert true_reward.shape[1:] == (3, 3) and torch.isfinite(true_reward).all()
    assert len(sampled_raw) == len(np.unique(sampled_raw))
    assert not np.isin(sampled_raw, valid_to_raw[validation]).any()
    counts = np.bincount(sample_well, minlength=80)
    assert np.count_nonzero(counts) == 64 and np.all(counts[tr_wells] > 0)
    point_weight = torch.tensor(1/(64*counts[sample_well]), dtype=torch.float32)
    assert abs(float(point_weight.sum())-1) < 1e-5
    vote_logits = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float32))
    optimizer = torch.optim.Adam([vote_logits], lr=.05)
    uniform = torch.full((3, 3), 1/3)
    def objective():
        vote = vote_logits.softmax(-1)
        expected_utility = (cross_utility*vote[None, :, None, :]).sum(-1)
        soft_action = (expected_utility/.02).softmax(-1)
        reward = (soft_action*true_reward).sum(-1)
        data_loss = -(point_weight[:, None]*reward).sum()/3
        penalty = .05*(vote-uniform).square().sum()/3
        return data_loss+penalty, data_loss, penalty
    initial_loss = float(objective()[0].detach())
    for step in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = objective()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss, data_loss, penalty = objective()
        vote = vote_logits.softmax(-1).numpy()
    assert np.isfinite(vote).all() and (vote > 0).all()
    assert np.allclose(vote.sum(-1), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'screening_fold': 5, 'seed_bases': [seed for seed, _, _ in checkpoints],
              'weights_by_target': {name: vote[j].tolist() for j, name in enumerate(base.TS)},
              'target_order': list(base.TS), 'member_order': [seed for seed, _, _ in checkpoints],
              'train_wells': len(tr_wells), 'sampled_train_points': len(sampled_raw),
              'sampled_points_per_well_min': int(counts[tr_wells].min()),
              'sampled_points_per_well_max': int(counts[tr_wells].max()),
              'validation_points_used_to_fit': 0,
              'objective': '64-well-balanced negative soft official-action reward over three member Bayes-action candidates, temperature0.02, plus 0.05 times squared distance from uniform',
              'optimizer': 'PyTorch Adam lr0.05, 300 fixed steps, only 3x3 vote logits',
              'initial_objective': initial_loss, 'final_objective': float(final_loss),
              'final_data_loss': float(data_loss), 'final_regularizer': float(penalty),
              'checkpoint_sha256': {str(seed): sha for seed, _, sha in checkpoints},
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'wall_seconds': time.perf_counter()-started}
    (OUT/'learned_weights.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
