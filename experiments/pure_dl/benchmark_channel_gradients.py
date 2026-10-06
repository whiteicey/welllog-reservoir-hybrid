"""Read-only training-set task-gradient diagnostic; never changes model weights."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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
from global_cosine_sequence import Network as SingleNetwork
from por_t3_patch_loss import components
from gpu_job_guard_v2 import check_slot

OUT = base.ROOT/'outputs/pure_dl_channel_gradient_diagnostic'
SOURCES = {
    'A_epoch6': 'pure_dl_abd_scratch_control80_fold5',
    'A_epoch52': 'pure_dl_paired_posterior_router80_sequence',
    'B_epoch43': 'pure_dl_posterior_three_seed_equal80/seed_20261008',
    'D_epoch52': 'pure_dl_global_cosine_sequence',
}


def loss_channels(output, target, default, mask):
    gate, raw = output
    if gate.shape[1] != target.shape[1]:
        repeat = target.shape[1]//gate.shape[1]
        gate, raw = gate.repeat_interleave(repeat, 1), raw.repeat_interleave(repeat, 1)
    g, r, d = gate[mask].float(), raw[mask].float(), default[mask]
    y = target[mask].masked_fill(d, 0)
    bce = F.binary_cross_entropy_with_logits(g, d.float(), reduction='none')
    nll = -(r[:, :, 0].log_softmax(-1)+components(r, y)).logsumexp(-1)
    masked = nll.masked_fill(d, 0)
    terms = (bce+masked).mean(0)
    stats = {'gate_bce': bce.mean(0).detach().cpu().tolist(),
             'continuous_nll_population': masked.mean(0).detach().cpu().tolist(),
             'continuous_nll_active': [float(nll[:, j][~d[:, j]].mean()) for j in range(3)],
             'nondefault_fraction': (~d).float().mean(0).detach().cpu().tolist()}
    return terms, stats


def gradient_summary(grads):
    dot = (grads @ grads.T).double()
    norms = dot.diag().clamp_min(0).sqrt()
    cosine = dot/(norms[:, None]*norms[None, :]).clamp_min(1e-30)
    summed = grads.mean(0)
    alignment = grads.double() @ summed.double()
    return {'norm': norms.cpu().tolist(), 'cosine': cosine.cpu().tolist(),
            'dot_with_equal_mean_gradient': alignment.cpu().tolist(),
            'norm_max_min_ratio': float(norms.max()/norms.min().clamp_min(1e-30))}


def main():
    started = time.perf_counter()
    torch.set_num_threads(3)
    admission = check_slot(min_free_gib=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    base.write(OUT/'runtime_status.json', {'status': 'running', 'pid': os.getpid()})
    torch.manual_seed(20260927)
    checkpoints = {role: torch.load(base.ROOT/'outputs'/folder/'fold_5.pt', map_location='cpu', weights_only=False)
                   for role, folder in SOURCES.items()}
    master = checkpoints['A_epoch6']
    rec = base.load_split(base.ROOT.parent/'data', 'train')
    ys = [frame[base.TS].to_numpy(np.float64) for _, frame in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    truth = np.concatenate([y[m] for y, m in zip(ys, valid)])
    wells = np.concatenate([np.full(m.sum(), i, np.int32) for i, m in enumerate(valid)])
    fold = np.full(len(truth), -1, np.int8)
    for f, (_, ix) in enumerate(GroupKFold(5).split(truth, groups=wells)):
        fold[ix] = f
    row_ids = [np.flatnonzero(wells == i) for i in range(80)]
    train_wells = [i for i in range(80) if fold[row_ids[i][0]] != 4]
    starts = np.r_[0, np.cumsum([len(frame) for _, frame in rec])]
    nraw = int(starts[-1])
    valid_to_raw = np.concatenate([np.flatnonzero(m)+starts[i] for i, m in enumerate(valid)])
    train_valid = np.flatnonzero(fold != 4)
    train_raw = valid_to_raw[train_valid]
    assert np.array_equal(master['validation_raw_rows'], valid_to_raw[fold == 4])
    valid_x, builder, fm, fs, _ = base.prepare_features(rec, train_wells, valid, row_ids)
    assert np.array_equal(fm, master['feature_median']) and np.array_equal(fs, master['feature_scale'])
    contract = base.feature_contract()
    x = torch.empty(nraw, 206, dtype=torch.float16, device='cuda')
    for i, (_, frame) in enumerate(rec):
        value = builder.transform(frame).enhanced[:, contract['selected_indices']]
        x[starts[i]:starts[i+1]] = torch.from_numpy(np.clip((value-fm)/fs, -8, 8).astype(np.float16)).cuda()
    del valid_x
    depth = torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _, frame in rec]), device='cuda')
    lower = torch.tensor(np.concatenate([np.full(len(frame), starts[i]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    upper = torch.tensor(np.concatenate([np.full(len(frame), starts[i+1]) for i, (_, frame) in enumerate(rec)]), device='cuda')
    absolute = ((depth-master['absolute_depth_median'])/master['absolute_depth_scale']).clamp(-8, 8)
    labels = torch.zeros(nraw, 3, device='cuda')
    default = torch.zeros(nraw, 3, dtype=torch.bool, device='cuda')
    labeled = torch.zeros(nraw, dtype=torch.bool, device='cuda')
    # Allocate labels exclusively for the 64 training wells.
    z = np.column_stack([(base.TRANSFORM[j](truth[train_valid, j])-master['regression_median'][j])/
                         master['regression_scale'][j] for j in range(3)])
    labels[train_raw] = torch.tensor(z, device='cuda', dtype=torch.float32)
    default[train_raw] = torch.tensor(base.dmask(truth[train_valid]), device='cuda')
    labeled[train_raw] = True
    assert not bool(labeled[valid_to_raw[fold == 4]].any())
    # Eight anchors per training well; shuffled into 16 fixed batches of32.
    rng = np.random.default_rng(20260927)
    anchors_np = np.concatenate([rng.choice(valid_to_raw[row_ids[i]], 8, replace=False) for i in train_wells])
    rng.shuffle(anchors_np)
    np.save(OUT/'sample_anchor_raw_rows.npy', anchors_np)
    core = torch.zeros(1, 256, dtype=torch.bool, device='cuda'); core[:, 64:192] = True
    numeric, conditioning = calibration_indices(contract['whitelist'])
    results = {}
    torch.cuda.reset_peak_memory_stats()
    for role, checkpoint in checkpoints.items():
        for key in ['feature_median', 'feature_scale', 'regression_median', 'regression_scale', 'validation_raw_rows']:
            assert np.array_equal(checkpoint[key], master[key]), (role, key)
        model = (SingleNetwork() if role.startswith('D') else Network(
            retain_indices=retained_indices(contract['whitelist']), numeric_indices=numeric,
            conditioning_indices=conditioning)).cuda().eval()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        member_count = 1 if role.startswith('D') else 2
        shared = [(n, p) for n, p in model.named_parameters() if not any(
            part in n for part in ('main_head.', 'aux_mid.', 'aux_coarse.', 'router.'))]
        params = [p for _, p in shared]
        sizes = [p.numel() for p in params]
        branch_ids = []
        for branch in range(member_count):
            chunks = []; offset = 0
            for (name, _), size in zip(shared, sizes):
                if member_count == 1 or name.startswith(f'members.{branch}.'):
                    chunks.append(torch.arange(offset, offset+size, device='cuda'))
                offset += size
            branch_ids.append(torch.cat(chunks))
        batches = []
        for batch_index, start in enumerate(range(0, len(anchors_np), 32)):
            anchors = torch.tensor(anchors_np[start:start+32], device='cuda')
            ix, exists = window_indices(anchors, lower, upper)
            relative = torch.tanh((depth[ix]-depth[anchors, None])/10)
            mask = exists & core & labeled[ix]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                if member_count == 1:
                    member_outputs = [model(x[ix], relative, exists, True, absolute[ix])]
                else:
                    member_outputs = [m(x[ix], relative, exists, True, absolute[ix]) for m in model.members]
            main_losses = []; all_losses = []; statistics = []
            factor = max(0., 1.-(checkpoint['best_epoch']-1)/40.)
            for outputs in member_outputs:
                main_loss, stats = loss_channels(outputs[0], labels[ix], default[ix], mask)
                mid_loss, _ = loss_channels(outputs[1], labels[ix], default[ix], mask)
                coarse_loss, _ = loss_channels(outputs[2], labels[ix], default[ix], mask)
                main_losses.append(main_loss); all_losses.append(main_loss+factor*(.2*mid_loss+.1*coarse_loss))
                statistics.append(stats)
            vectors = {'main_only': torch.stack(main_losses).mean(0),
                       'with_epoch_auxiliary': torch.stack(all_losses).mean(0)}
            entry = {'batch': batch_index, 'supervised_points': int(mask.sum()), 'member_stats': statistics, 'modes': {}}
            for mode, losses in vectors.items():
                channel_grads = []
                for j in range(3):
                    gradients = torch.autograd.grad(losses[j], params, retain_graph=True, allow_unused=True)
                    channel_grads.append(torch.cat([(g.detach().float().flatten() if g is not None else torch.zeros_like(p).flatten())
                                                    for g, p in zip(gradients, params)]))
                matrix = torch.stack(channel_grads)
                entry['modes'][mode] = {'loss': losses.detach().cpu().tolist(), 'shared_all': gradient_summary(matrix),
                                      'branches': [gradient_summary(matrix[:, indices]) for indices in branch_ids]}
                del channel_grads, matrix, gradients
            batches.append(entry)
            del vectors, main_losses, all_losses, member_outputs, outputs, main_loss, mid_loss, coarse_loss, losses
        # Diagnostic performs no optimizer updates and checks unchanged checkpoint tensors.
        assert all(torch.equal(value.cpu(), checkpoint['state_dict'][key]) for key, value in model.state_dict().items())
        summary = {}
        for mode in ('main_only', 'with_epoch_auxiliary'):
            norms = np.array([b['modes'][mode]['shared_all']['norm'] for b in batches])
            cosines = np.array([b['modes'][mode]['shared_all']['cosine'] for b in batches])
            alignment = np.array([b['modes'][mode]['shared_all']['dot_with_equal_mean_gradient'] for b in batches])
            summary[mode] = {'mean_loss': np.mean([b['modes'][mode]['loss'] for b in batches], 0).tolist(),
                'mean_norm': norms.mean(0).tolist(), 'median_norm': np.median(norms, 0).tolist(),
                'mean_cosine': cosines.mean(0).tolist(), 'negative_cosine_fraction': (cosines < 0).mean(0).tolist(),
                'equal_gradient_opposes_task_fraction': (alignment < 0).mean(0).tolist(),
                'median_batch_norm_max_min': float(np.median(norms.max(1)/norms.min(1)))}
        results[role] = {'checkpoint': SOURCES[role]+'/fold_5.pt', 'epoch': checkpoint['best_epoch'],
            'sha256': hashlib.sha256((base.ROOT/'outputs'/SOURCES[role]/'fold_5.pt').read_bytes()).hexdigest(),
            'parameters_shared': sum(sizes), 'parameter_names_shared': [n for n, _ in shared],
            'weights_unchanged': True, 'auxiliary_factor': factor, 'summary': summary, 'batches': batches}
        base.write(OUT/'partial_results.json', results)
        print('CHECKPOINT_RESULT', role, json.dumps(summary), flush=True)
        del model, params, shared, branch_ids
        torch.cuda.empty_cache()
    report = {'status': 'complete', 'scope': 'training-set gradient diagnostic, no optimizer/validation tuning',
        'channels': base.TS, 'sources': results, 'training_wells': train_wells,
        'sampling': '8 anchors each of64 train wells, fixed seed20260927,16 batches of32; not independent depth samples',
        'quality_weight': 'Q=1 for a common checkpoint comparison; not historical effective gradients',
        'precision': 'bf16 forward/float32 likelihood and autograd, same training precision',
        'training_nondefault_fraction': (~base.dmask(truth[train_valid])).mean(0).tolist(),
        'limitations': ['A6 and A52 are saved checkpoints from separate runs, not a matched recorded gradient trajectory',
                       'Epoch auxiliaries evaluated at saved epoch, not with next-epoch coefficient',
                       'No Adam moments, clipping or weight decay in gradient cosines; not the actual optimizer step',
                       'Pairwise negative cosine does not imply the combined direction harms a task',
                       'No causal attribution of validation ensemble gains from this diagnostic alone'],
        'gpu_admission': admission, 'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
        'seconds': time.perf_counter()-started}
    base.write(OUT/'report.json', report)
    base.write(OUT/'runtime_status.json', {'status': 'complete', 'pid': os.getpid()})
    print('DONE', report['seconds'], report['peak_gpu_allocated_mib'], flush=True)


if __name__ == '__main__':
    main()
