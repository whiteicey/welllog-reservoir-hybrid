import gc
import numpy as np
import torch
from train_invariant import FeatureBuilder, SEGMENTS, feature_contract as old_contract

def feature_contract():
    names=old_contract()['full_schema']
    assert len(names)==237 and not any(name in ['POR','PERM','SW','CASE'] for name in names)
    return {'full_feature_count':237,'selected_feature_count':237,'full_schema':names,'selected_indices':list(range(237)),'whitelist':names,'excluded':[],'scope':'all input-derived features; train-only fit'}

def prepare_features(rec, train_wells, valid, row_ids):
    assert len(train_wells) == 64 and len(set(train_wells)) == 64
    builder = FeatureBuilder().fit([rec[i][1] for i in train_wells])
    indices = np.arange(237,dtype=np.int64)
    features = [builder.transform(frame).enhanced[:,indices] for _, frame in rec]
    fm = np.empty(237, np.float32); fs = np.empty(237, np.float32)
    edges = np.empty((237, SEGMENTS+1), np.float32)
    for c in range(237):
        values = np.concatenate([features[i][:, c] for i in train_wells])
        assert len(values) == sum(len(rec[i][1]) for i in train_wells)
        quantiles = np.quantile(values, np.linspace(0, 1, SEGMENTS+1))
        fm[c] = quantiles[SEGMENTS//2]
        fs[c] = max(quantiles[3*SEGMENTS//4]-quantiles[SEGMENTS//4], 1e-3)
        edges[c] = np.clip((quantiles-fm[c])/fs[c], -8, 8)
    n = sum(len(rows) for rows in row_ids)
    x = torch.empty((n, 237), dtype=torch.float16, device='cuda')
    for i, feature in enumerate(features):
        xx = np.clip((feature[valid[i]]-fm)/fs, -8, 8).astype(np.float16)
        assert xx.shape == (len(row_ids[i]), 237) and np.isfinite(xx).all()
        lo, hi = int(row_ids[i][0]), int(row_ids[i][-1])+1
        assert hi-lo == len(row_ids[i])
        x[lo:hi] = torch.from_numpy(xx).cuda()
    del features, values, feature, xx
    gc.collect()
    return x, builder, fm, fs, edges
