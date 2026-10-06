"""Equal predictive-distribution vote of three independently trained neural models."""
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
from por_t3_patch_loss import probability


ROOT = base.ROOT
SOURCE_OUT = ROOT/'outputs/pure_dl_posterior_three_seed_equal80'
OUT = ROOT/'outputs/pure_dl_posterior_validation_grid_fold5'
SOURCES = [
    (20261006, ROOT/'outputs/pure_dl_paired_posterior_router80_sequence'),
    (20261007, SOURCE_OUT/'seed_20261007'),
    (20261008, SOURCE_OUT/'seed_20261008'),
]


def main():
    torch.set_num_threads(3)
    started = time.perf_counter()
    checkpoints = []
    for seed, folder in SOURCES:
        path = folder/'fold_5.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['config']['seed'] == seed
        report = json.loads((folder/'fold_5_report.json').read_text(encoding='utf-8'))
        assert report['best_checkpoint_predictions_reproduced']
        assert abs(report['scores']['TOTAL']-checkpoint['best_validation_score']) < 1e-10
        checkpoints.append((seed, folder, checkpoint, hashlib.sha256(path.read_bytes()).hexdigest()))
    master = checkpoints[0][2]
    for _, _, checkpoint, _ in checkpoints[1:]:
        for field in ['selected_feature_names', 'selected_feature_indices',
                      'feature_median', 'feature_scale', 'regression_median',
                      'regression_scale', 'absolute_depth_median', 'absolute_depth_scale']:
            assert np.array_equal(checkpoint[field], master[field]), field
        for field in ['centers', 'density_edges']:
            for left, right in zip(checkpoint[field], master[field]):
                assert np.array_equal(left, right), field
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
    validation = np.flatnonzero(fold == 4)
    assert len(validation) == 144665
    row_ids = [np.flatnonzero(wells == i) for i in range(80)]
    tr_wells = [i for i in range(80) if fold[row_ids[i][0]] != 4]
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
    for seed, folder, checkpoint, sha in checkpoints:
        model = Network(retain_indices=view_indices, numeric_indices=numeric_indices,
                        conditioning_indices=conditioning_indices).cuda()
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        models.append(model.eval())
    actions, utilities, edges = [], [], []
    for j, name in enumerate(base.TS):
        ct = np.asarray(master['centers'][j])
        action = np.linspace(ct[0], ct[-1], 800) if j == 2 else np.geomspace(max(ct[0]*.9, .001), ct[-1]*1.1, 800)
        action = np.unique(np.r_[base.DEFAULTS[name], np.clip(action, *base.BOUNDS[name])])
        actions.append(action)
        utilities.append(torch.tensor(base.kernel_matrix(action,
                          np.r_[base.DEFAULTS[name], ct], name).T,
                          device='cuda', dtype=torch.float32))
        edges.append(torch.tensor(master['density_edges'][j], device='cuda', dtype=torch.float32))
    local = np.full(len(truth), -1, np.int64)
    local[validation] = np.arange(len(validation))
    covered = np.zeros(len(validation), np.int8)
    single = np.empty((len(validation), 3, 3), np.float64)
    ensemble = np.empty((len(validation), 3), np.float64)
    cached = torch.empty((len(validation), 3, 3, 257), device="cuda", dtype=torch.float32)
    eval_centers, eval_anchors = [], []
    for wi in range(80):
        if fold[row_ids[wi][0]] != 4:
            continue
        cc = np.arange(starts[wi], starts[wi+1], 128)+64
        eval_centers.extend(cc)
        eval_anchors.extend(np.minimum(cc, starts[wi+1]-1))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for begin in range(0, len(eval_centers), 32):
            cc = torch.tensor(eval_centers[begin:begin+32], device='cuda')
            anchor = torch.tensor(eval_anchors[begin:begin+32], device='cuda')
            ix, exists = window_indices(cc, lower, upper, anchors=anchor)
            relative = torch.tanh((depth[ix]-depth[anchor, None])/10)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = [model(x[ix], relative, exists, absolute_depth=absolute[ix])
                           for model in models]
            original = ix[:, 64:192].cpu().numpy()
            global_rows = raw_to_valid[original]
            use = exists[:, 64:192].cpu().numpy() & (global_rows >= 0)
            ids = global_rows[use]
            loc = local[ids]
            assert (loc >= 0).all() and (covered[loc] == 0).all()
            covered[loc] += 1
            select = torch.from_numpy(use).cuda()
            for j in range(3):
                joints = []
                for member, (gate_logits, raw) in enumerate(outputs):
                    gate = gate_logits[:, 64:192][select][:, j].float().sigmoid()
                    r = raw[:, 64:192][select][:, j].float()
                    weight = r[:, 0].softmax(-1)
                    mu = r[:, 1]
                    sd = F.softplus(r[:, 2]) + .025
                    cp = probability(j, weight, mu, sd, edges[j])
                    joint = torch.cat([gate[:, None], (1-gate[:, None])*cp], -1)
                    joints.append(joint)
                    cached[torch.as_tensor(loc, device="cuda"), j, member] = joint
                    single[loc, member, j] = actions[j][(joint@utilities[j]).argmax(-1).cpu().numpy()]
                mixed = torch.stack(joints).mean(0)
                ensemble[loc, j] = actions[j][(mixed@utilities[j]).argmax(-1).cpu().numpy()]
    assert (covered == 1).all() and np.isfinite(single).all() and np.isfinite(ensemble).all()
    source_checks = []
    for member, (seed, folder, _, sha) in enumerate(checkpoints):
        saved = np.load(folder/'fold_5_pred.npy')
        assert saved.shape == single[:, member].shape
        agreement = float(np.mean(saved == single[:, member]))
        max_abs = float(np.max(np.abs(saved-single[:, member])))
        source_checks.append({'seed_base': seed, 'folder': str(folder.relative_to(ROOT)),
                              'checkpoint_sha256': sha, 'exact_action_agreement': agreement,
                              'max_absolute_action_difference': max_abs,
                              'saved_scores': base.metric(truth[validation], saved),
                              'recomputed_scores': base.metric(truth[validation], single[:, member])})
        assert agreement > .999 and abs(source_checks[-1]['saved_scores']['TOTAL']-
                                        source_checks[-1]['recomputed_scores']['TOTAL']) < .005
    OUT.mkdir(parents=True, exist_ok=True)
    del models, x
    torch.cuda.empty_cache()
    target_truth = truth[validation]
    history = []
    def decode(j, weights):
        pred = np.empty(len(validation), np.float64)
        w = torch.tensor(weights, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            for begin in range(0, len(validation), 8192):
                end = min(begin+8192, len(validation))
                mixed = (cached[begin:end, j]*w[None, :, None]).sum(1)
                pred[begin:end] = actions[j][(mixed@utilities[j]).argmax(-1).cpu().numpy()]
        return pred
    def score(j, pred):
        y = target_truth[:, j]
        if j == 1:
            reward = np.maximum(0, 1-np.abs(np.log10(np.maximum(pred/np.maximum(y,1e-5),1e-5))))
        else:
            reward = np.maximum(0, 1-np.abs(pred-y)/((.08 if j == 0 else .05)*(np.abs(y)+1e-5)))
        return float(reward.mean()*100)
    records = [{}, {}, {}]
    def evaluate(j, weights, stage):
        key = tuple(round(float(v), 8) for v in weights)
        if key not in records[j]:
            result = score(j, decode(j, key))
            records[j][key] = result
            history.append({'target': base.TS[j], 'weights': key, 'score': result, 'stage': stage})
        return records[j][key]
    grid = [(a/10,b/10,(10-a-b)/10) for a in range(11) for b in range(11-a)]
    grid.append((1/3,1/3,1/3))
    for i, w in enumerate(grid):
        for j in range(3):
            evaluate(j,w,'coarse')
        if i % 10 == 0:
            print('COARSE',i,len(grid),flush=True)
    coarse_best = [max(records[j], key=records[j].get) for j in range(3)]
    shared = max(records[0], key=lambda w: sum(f*records[j][w] for j,f in enumerate([.3,.35,.35])))
    def nearby(w):
        return sorted(set((round(w[0]+a*.02,8),round(w[1]+b*.02,8),round(1-w[0]-a*.02-w[1]-b*.02,8))
                          for a in range(-5,6) for b in range(-5,6)
                          if min(w[0]+a*.02,w[1]+b*.02,1-w[0]-a*.02-w[1]-b*.02)>=-1e-8))
    for i,w in enumerate(nearby(shared)):
        for j in range(3):
            evaluate(j,w,'shared_fine')
        if i % 20 == 0:
            print('SHARED_FINE',i,flush=True)
    common = set(records[0]) & set(records[1]) & set(records[2])
    shared = max(common, key=lambda w: sum(f*records[j][w] for j,f in enumerate([.3,.35,.35])))
    for j in range(3):
        for w in nearby(coarse_best[j]):
            evaluate(j,w,'channel_fine')
        print('CHANNEL_DONE',base.TS[j],max(records[j].values()),flush=True)
    chosen = [max(records[j], key=records[j].get) for j in range(3)]
    prediction = np.column_stack([decode(j, chosen[j]) for j in range(3)])
    shared_prediction = np.column_stack([decode(j,shared) for j in range(3)])
    scores = base.metric(target_truth,prediction)
    for j,name in enumerate(base.TS):
        assert abs(scores[name]-records[j][chosen[j]]) < 1e-8
    result = {'screening_fold':5,'label_scope':'validation labels explicitly used to select global/channel weights; optimistic tuned validation score, NOT independent OOF',
              'rows':len(validation),'all_rows_covered_once':bool((covered==1).all()),
              'member_order':[seed for seed,_,_,_ in checkpoints], 'source_checks':source_checks,
              'coarse_step':.1,'fine_step':.02,'fine_radius':.1,
              'channel_weights':dict(zip(base.TS,chosen)), 'tuned_validation_scores':scores,
              'shared_weights':shared,'shared_tuned_validation_scores':base.metric(target_truth,shared_prediction),
              'equal_scores':base.metric(target_truth,ensemble),
              'channel_candidate_counts':[len(r) for r in records],
              'threshold_exceeded':bool(scores['TOTAL']>80.4),
              'independent_screening_passed':False,
              'wall_seconds':time.perf_counter()-started,
              'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
    np.save(OUT/'fold_5_tuned_pred.npy',prediction)
    (OUT/'search_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (OUT/'grid_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    main()
