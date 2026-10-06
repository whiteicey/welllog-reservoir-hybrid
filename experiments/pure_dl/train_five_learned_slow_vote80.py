"""Fine-grained slow vote-router training including the equal-vote initial state."""
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
from paired_posterior_router_sequence import Network, retained_indices, calibration_indices
from por_t3_patch_loss import components, probability
from posterior_vote_router import posterior_summary
from paired_calibrated_view_sequence import Network as CalibratedNetwork
from global_cosine_sequence import Network as SingleNetwork

class VoteRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.zeros(3,5))
    def forward(self, summaries):
        return self.logits.softmax(-1)[None].expand(len(summaries),-1,-1)



ROOT = base.ROOT
OUT = ROOT/'outputs/pure_dl_five_learned_slow_vote_fold5'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261007, ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261007'),
    (20261008, ROOT/'outputs/pure_dl_posterior_three_seed_equal80/seed_20261008'),
    (20261006, ROOT/'outputs/pure_dl_paired_calibrated_view_sequence'),
    (20261006, ROOT/'outputs/pure_dl_global_cosine_sequence'),
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
    for member_index, (_, checkpoint, _) in enumerate(checkpoints):
        cls=CalibratedNetwork if member_index==3 else Network
        model=(SingleNetwork() if member_index==4 else cls(retain_indices=view_indices, numeric_indices=numeric_indices,
                        conditioning_indices=conditioning_indices)).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.requires_grad_(False)
        models.append(model.eval())
    train_centers, train_anchors = [], []
    for wi in tr_wells:
        cc = np.arange(starts[wi], starts[wi+1], 128)+64
        train_centers.extend(cc)
        train_anchors.extend(np.minimum(cc, starts[wi+1]-1))
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
    assert logp.shape[1:] == (5, 3) and torch.isfinite(logp).all()
    assert summaries.shape[1:] == (3, 20) and torch.isfinite(summaries).all()
    assert len(sampled_raw) == len(np.unique(sampled_raw))
    assert not np.isin(sampled_raw, valid_to_raw[validation]).any()
    assert np.array_equal(np.sort(sampled_raw), np.sort(valid_to_raw[training]))
    counts = np.bincount(sample_well, minlength=80)
    assert np.count_nonzero(counts) == 64 and np.all(counts[tr_wells] > 0)
    train_logp = logp.permute(0, 2, 1).contiguous().cuda()
    train_summaries = summaries.cuda()
    del logp, summaries, logp_batches, summary_batches
    actions, utilities, edges = [], [], []
    for j, name in enumerate(base.TS):
        ct = np.asarray(master['centers'][j])
        action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800)
        finer = np.linspace(ct[0], ct[-1], 3197) if j == 2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,3197)
        action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name]),np.clip(finer,*base.BOUNDS[name])])
        fine_support=np.linspace(base.TRANSFORM[j](ct[0]),base.TRANSFORM[j](ct[-1]),1024)
        ct=base.INVERSE[j](fine_support)
        fine_edges=(np.r_[-np.inf,(fine_support[:-1]+fine_support[1:])/2,np.inf]-master['regression_median'][j])/master['regression_scale'][j]
        actions.append(action)
        utilities.append(torch.tensor(base.kernel_matrix(action,
                          np.r_[base.DEFAULTS[name], ct], name).T,
                          device='cuda', dtype=torch.float32))
        edges.append(torch.tensor(fine_edges, device='cuda', dtype=torch.float32))
    val_joint = torch.zeros((3,len(validation),5,1025),device='cpu',dtype=torch.float32)
    core_weights=torch.sin(torch.pi*(torch.arange(128,device='cuda')+.5)/128).square()
    val_summaries = torch.empty((len(validation),3,20),device='cpu')
    val_local = np.full(len(truth), -1, np.int64)
    val_local[validation] = np.arange(len(validation))
    val_coverage = np.zeros(len(validation), np.int8)
    val_centers, val_anchors = [], []
    for wi in range(80):
        if fold[row_ids[wi][0]] != 4:
            continue
        cc = np.arange(starts[wi]-32, starts[wi+1]+64,32)
        val_centers.extend(cc)
        val_anchors.extend(np.clip(cc,starts[wi],starts[wi+1]-1))
    with torch.no_grad():
        for begin in range(0, len(val_centers), 32):
            cc = torch.tensor(val_centers[begin:begin+32], device='cuda')
            anchor = torch.tensor(val_anchors[begin:begin+32], device='cuda')
            ix, exists = window_indices(cc, lower, upper, anchors=anchor)
            relative = torch.tanh((depth[ix]-depth[anchor, None])/10)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = [model(x[ix], relative, exists, absolute_depth=absolute[ix])
                           for model in models]
            original = ix[:, 64:192].cpu().numpy()
            global_rows = raw_to_valid[original]
            use = exists[:, 64:192].cpu().numpy() & (global_rows >= 0)
            ids = global_rows[use]
            loc = val_local[ids]
            assert (loc >= 0).all()
            np.add.at(val_coverage,loc,1)
            loc_t = torch.as_tensor(loc, device='cuda')
            select = torch.from_numpy(use).cuda()
            selected = [(gate[:, 64:192][select].float(), raw[:, 64:192][select].float())
                        for gate, raw in outputs]
            position_weight=core_weights[None].expand(len(cc),-1)[select]
            loc_cpu=torch.as_tensor(loc)
            for member, (gate_logits, raw) in enumerate(selected):
                gate = gate_logits.sigmoid()
                for j in range(3):
                    r = raw[:, j]
                    weight = r[:, 0].softmax(-1)
                    mu = r[:, 1]
                    sd = F.softplus(r[:, 2])+.025
                    cp = probability(j, weight, mu, sd, edges[j])
                    joint=torch.cat([gate[:,j:j+1],(1-gate[:,j:j+1])*cp],dim=-1)
                    val_joint[j,:,member].index_add_(0,loc_cpu,(joint*position_weight[:,None]/2).cpu())
    assert (val_coverage == 4).all() and torch.isfinite(val_joint).all()
    del models, x, depth, lower, upper, absolute
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(20261101)
    router = VoteRouter().cuda()
    optimizer = torch.optim.AdamW(router.parameters(), lr=.0001, weight_decay=0.)
    meta_generator = torch.Generator(device='cuda')
    meta_generator.manual_seed(20261102)
    def evaluate_current():
        router.eval()
        prediction = np.empty((len(validation), 3), np.float64)
        with torch.no_grad():
            vote=router.logits.softmax(-1)
            for j in range(3):
                channel=val_joint[j].cuda()
                for begin in range(0,len(validation),4096):
                    end=min(begin+4096,len(validation))
                    mixed=(channel[begin:end]*vote[j,:,None]).sum(1)
                    chosen=(mixed@utilities[j]).argmax(-1).cpu().numpy()
                    prediction[begin:end,j]=actions[j][chosen]
                del channel
        assert np.isfinite(prediction).all()
        return base.metric(truth[validation], prediction), prediction
    initial_scores, initial_prediction = evaluate_current()
    best_score = initial_scores['TOTAL']
    best_epoch = 0.0
    best_update = 0
    bad_epochs = 0
    best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
    best_prediction = initial_prediction.copy()
    updates = 0
    history = [{'epoch': 0.0, 'update': 0, 'train_loss': None,
                'official_scores': initial_scores, 'best_epoch': 0.0,
                'best_score': best_score, 'bad_epochs': 0}]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'validation_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    print('INITIAL_EQUAL_VALIDATION', json.dumps(history[0]), flush=True)
    for epoch in range(1, 81):
        permutation = torch.randperm(len(train_logp), generator=meta_generator, device='cuda')
        total_loss = 0.0
        improved_this_epoch = False
        batch_size = 16384
        batches_per_epoch = (len(permutation)+batch_size-1)//batch_size
        for batch_index, begin in enumerate(range(0, len(permutation), batch_size)):
            router.train()
            index = permutation[begin:begin+batch_size]
            optimizer.zero_grad(set_to_none=True)
            vote = router(train_summaries[index])
            mixed = torch.logsumexp(train_logp[index]+vote.clamp_min(1e-8).log(), dim=-1)
            loss = -mixed.mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach())*len(index)
            if (epoch<=2 and (batch_index+1)%8==0) or batch_index+1==batches_per_epoch:
                scores, prediction = evaluate_current()
                val = scores['TOTAL']
                fraction = epoch-1+(batch_index+1)/batches_per_epoch
                if val > best_score:
                    best_score = val
                    best_epoch = fraction
                    best_update = updates
                    improved_this_epoch = True
                    best_state = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
                    best_prediction = prediction.copy()
                    print('NEW_BEST', json.dumps({'epoch': fraction, 'update': updates,
                                                  'scores': scores}), flush=True)
                history.append({'epoch': fraction, 'update': updates,
                                'train_loss': total_loss/(begin+len(index)),
                                'official_scores': scores, 'best_epoch': best_epoch,
                                'best_score': best_score, 'bad_epochs': bad_epochs})
                (OUT/'validation_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if improved_this_epoch:
            bad_epochs = 0
        else:
            bad_epochs += 1
        print('EPOCH_VALIDATION', json.dumps({'epoch': epoch,
                                             'train_loss': total_loss/len(train_logp),
                                             'official_scores': scores,
                                             'best_score': best_score,
                                             'best_epoch': best_epoch,
                                             'best_update': best_update,
                                             'bad_epochs': bad_epochs}), flush=True)
        if epoch >= 10 and bad_epochs >= 18:
            break
    stopped_epoch = epoch
    assert best_state is not None
    router.load_state_dict(best_state)
    restored_scores, restored_prediction = evaluate_current()
    assert np.array_equal(restored_prediction, best_prediction)
    assert abs(restored_scores['TOTAL']-best_score) < 1e-10
    with torch.no_grad():
        average_votes = router(val_summaries).mean(0).cpu().numpy()
    assert np.isfinite(average_votes).all() and np.allclose(average_votes.sum(-1), 1)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {'screening_fold': 5, 'seed_bases': [seed for seed, _, _ in checkpoints],
              'screening_scores': restored_scores,
              'screening_passed': bool(best_score > 80.4),
              'threshold_strict_gt': 80.4,
              'mean_validation_weights_by_target': {name: average_votes[j].tolist() for j, name in enumerate(base.TS)},
              'target_order': list(base.TS), 'member_order': [seed for seed, _, _ in checkpoints],
              'train_wells': len(tr_wells), 'sampled_train_points': len(sampled_raw),
              'sampled_points_per_well_min': int(counts[tr_wells].min()),
              'sampled_points_per_well_max': int(counts[tr_wells].max()),
              'validation_points_used_for_gradient': 0,
              'validation_points_used_for_checkpoint_selection': len(validation),
              'validation_rows_covered_four_times': True,
              'model_folders':[str(folder.relative_to(ROOT)) for _,folder in SOURCES],
              'objective': 'frozen5-expert training-only mixture NLL;15 global logits; official four-window fine-decoder validation TOTAL chooses checkpoint',
              'optimizer': 'PyTorch AdamW lr0.0001, weight_decay0, batch16384, grad_clip1, max80 epochs, patience18, min10, validation at initialization, every8 updates for first2 epochs, then epoch ends',
              'router_parameters': sum(p.numel() for p in router.parameters()),
              'initial_equal_scores': initial_scores,
              'best_epoch': best_epoch,
              'best_update': best_update,
              'validation_evaluations': len(history),
              'stopped_epoch': stopped_epoch,
              'best_checkpoint_predictions_reproduced': True,
              'checkpoint_persisted': True,
              'checkpoint_sha256': {str(SOURCES[i][1].relative_to(ROOT)):sha for i,(_,_,sha) in enumerate(checkpoints)},
              'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'wall_seconds': time.perf_counter()-started}
    torch.save({'state_dict': best_state, 'report': result}, OUT/'router.pt')
    np.save(OUT/'fold_5_pred.npy', restored_prediction)
    (OUT/'router_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
