"""Restore AZIM/DEVI features only; train-fold-only preprocessing without base mutation."""
import gc
import numpy as np
import torch
import train_invariant as base
FeatureBuilder=base.FeatureBuilder
SEGMENTS=base.SEGMENTS

def exclusion_reason(name):
    if name in ('DEPTH','DEPTH_MISSING'):return 'absolute depth excluded'
    if name.endswith(('_WELL_MEDIAN','_WELL_STD')):return 'well constant excluded'
    if 'BIT' in name.split('_'):return 'BIT excluded'
    return None

def select_features(names):return np.asarray([i for i,n in enumerate(names) if exclusion_reason(n) is None],np.int64)

def feature_contract():
    original=base.feature_contract();names=original['full_schema'];indices=select_features(names)
    return {'full_feature_count':len(names),'selected_feature_count':len(indices),'full_schema':names,'selected_indices':indices.tolist(),'whitelist':[names[i] for i in indices],'restored':[names[i] for i in indices if i not in original['selected_indices']],'excluded':[{'name':n,'reason':exclusion_reason(n)} for n in names if exclusion_reason(n)],'scope':'Restore trajectory AZIM/DEVI and derivatives; keep BIT, absolute DEPTH and explicit well constants excluded. CASE absent from FeatureBuilder schema. No validation-dependent feature choice.'}
FEATURES=feature_contract()['selected_feature_count']
def prepare_features(rec, train_wells, valid, row_ids):
    assert len(train_wells) == 64 and len(set(train_wells)) == 64
    builder = FeatureBuilder().fit([rec[i][1] for i in train_wells])
    indices = select_features(builder.enhanced_feature_names_)
    features = [builder.transform(frame).enhanced[:,indices] for _, frame in rec]
    fm = np.empty(FEATURES, np.float32); fs = np.empty(FEATURES, np.float32)
    edges = np.empty((FEATURES, SEGMENTS+1), np.float32)
    for c in range(FEATURES):
        values = np.concatenate([features[i][:, c] for i in train_wells])
        assert len(values) == sum(len(rec[i][1]) for i in train_wells)
        quantiles = np.quantile(values, np.linspace(0, 1, SEGMENTS+1))
        fm[c] = quantiles[SEGMENTS//2]
        fs[c] = max(quantiles[3*SEGMENTS//4]-quantiles[SEGMENTS//4], 1e-3)
        edges[c] = np.clip((quantiles-fm[c])/fs[c], -8, 8)
    n = sum(len(rows) for rows in row_ids)
    x = torch.empty((n, FEATURES), dtype=torch.float16, device='cuda')
    for i, feature in enumerate(features):
        xx = np.clip((feature[valid[i]]-fm)/fs, -8, 8).astype(np.float16)
        assert xx.shape == (len(row_ids[i]), FEATURES) and np.isfinite(xx).all()
        lo, hi = int(row_ids[i][0]), int(row_ids[i][-1])+1
        assert hi-lo == len(row_ids[i])
        x[lo:hi] = torch.from_numpy(xx).cuda()
    del features, values, feature, xx
    gc.collect()
    return x, builder, fm, fs, edges


