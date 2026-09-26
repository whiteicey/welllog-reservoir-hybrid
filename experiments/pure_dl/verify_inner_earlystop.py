import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import train_invariant as base
out=base.ROOT/'outputs/pure_dl_inner_earlystop'
r=json.loads((out/'pilot_report.json').read_text());cfg=r['config']
for name,sha in cfg['sha256'].items():assert hashlib.sha256((base.ROOT/name).read_bytes()).hexdigest()==sha
rec=base.load_split(base.ROOT.parent/'data','train')
valid=[np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1) for _,frame in rec]
truth=np.concatenate([frame[base.TS].to_numpy(np.float64)[m] for (_,frame),m in zip(rec,valid)])
well=np.concatenate([np.full(m.sum(),i) for i,m in enumerate(valid)])
outer_train,outer_val=next(GroupKFold(5).split(truth,groups=well))
train_ids=np.unique(well[outer_train]);inner_ids=np.sort(np.random.default_rng(20260925).permutation(train_ids)[:8])
fit_ids=sorted(set(train_ids.tolist())-set(inner_ids.tolist()))
assert r['training_wells']==fit_ids and r['inner_validation_wells']==inner_ids.tolist()
assert r['outer_validation_wells']==np.unique(well[outer_val]).tolist()
starts=np.r_[0,np.cumsum([len(frame) for _,frame in rec])]
mapping=np.concatenate([np.flatnonzero(m)+starts[i] for i,m in enumerate(valid)])
fit=np.isin(well,fit_ids);inner=np.isin(well,inner_ids)
with np.load(out/'scope.npz') as scope:
    assert np.array_equal(scope['training_raw_rows'],mapping[fit])
    assert np.array_equal(scope['inner_validation_raw_rows'],mapping[inner])
    assert np.array_equal(scope['outer_validation_raw_rows'],mapping[outer_val])
    assert not np.intersect1d(mapping[fit],mapping[inner]).size
    default=base.dmask(truth)
    for j in range(3):
        q=np.percentile(base.TRANSFORM[j](truth[fit&~default[:,j],j]),[25,50,75])
        assert scope['regression_median'][j]==q[1]
        assert scope['regression_scale'][j]==max(float(q[2]-q[0]),.1)
    depth=np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _,frame in rec]).astype(np.float64)
    q=np.percentile(depth[mapping[fit]],[25,50,75])
    assert scope['depth_median']==q[1] and scope['depth_scale']==max(float(q[2]-q[0]),.1)
assert cfg['patience']==5 and cfg['min_delta']==.002 and cfg['minimum_epochs']==10 and cfg['epochs']==40
best=float('inf');epoch=0;bad=0
for i,h in enumerate(r['history'],1):
    assert h['epoch']==i and np.isfinite(h['inner_loss'])
    if h['inner_loss']<best-.002:best=h['inner_loss'];epoch=i;bad=0
    else:bad+=1
    assert h['best_epoch']==epoch and h['bad_epochs']==bad
    if i<len(r['history']):assert not (i>=10 and bad>=5)
assert r['best_epoch']==epoch and r['best_inner_loss']==best
assert r['stopped_epoch']==len(r['history']) and (len(r['history'])==40 or (len(r['history'])>=10 and bad>=5))
assert r['official_score'] is None and not r['outer_predictions_produced']
proof={'passed':True,'fixed_56_8_16_well_scope':True,'target_and_depth_statistics_recomputed':True,'source_hashes_verified':True,'stopping_rule_replayed':True,'selected_epoch':epoch,'no_outer_score':True}
base.write(out/'pilot_verification.json',proof);print(json.dumps(proof,indent=2))
