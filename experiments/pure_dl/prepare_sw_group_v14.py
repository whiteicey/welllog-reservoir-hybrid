"""Prepare one strictly group-isolated development split for SW.

No network checkpoint, teacher, old Q, or old target grid is loaded. Original
fold5 remains excluded from the inner cache. This is a diagnostic split within
the original64 training wells, not a newly completed competition fold.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse,json,time
import numpy as np
from sklearn.model_selection import GroupKFold
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/sw_group_generalization_v14_20260928'

def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['inner','outer'],default='inner');a=ap.parse_args()
    started=time.perf_counter();STUDY.mkdir(parents=True,exist_ok=True)
    dest=STUDY/('data_'+a.phase+'.npz')
    if dest.exists():raise RuntimeError('Prepared data is immutable: '+str(dest))
    records=base.load_split(ROOT.parent/'data','train')
    valid=[np.isfinite(f[base.TS].to_numpy(np.float64)).all(1) for _,f in records]
    groups=np.concatenate([np.full(v.sum(),i,np.int32) for i,v in enumerate(valid)])
    assert len(records)==80 and len(groups)==723556
    folds=np.full(len(groups),-1,np.int8)
    for k,(_,idx) in enumerate(GroupKFold(5).split(np.zeros(len(groups)),groups=groups)):folds[idx]=k
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        assert np.array_equal(folds,z['fold']) and np.array_equal(groups,z['well'])
    original_train=np.unique(groups[folds!=4]);outer_holdout=np.unique(groups[folds==4])
    assert len(original_train)==64 and len(outer_holdout)==16
    if a.phase=='inner':
        inside=np.flatnonzero(folds!=4)
        tr,va=next(GroupKFold(4).split(np.zeros(len(inside)),groups=groups[inside]))
        fit_ids=np.unique(groups[inside[tr]]);monitor_ids=np.unique(groups[inside[va]])
        included=original_train
        assert len(fit_ids)==48 and len(monitor_ids)==16
        assert not np.intersect1d(included,outer_holdout).size
    else:
        fit_ids=original_train;monitor_ids=outer_holdout;included=np.arange(80)
    fit_set=set(fit_ids.tolist());monitor_set=set(monitor_ids.tolist())
    assert not fit_set&monitor_set
    # Feature construction is the historical fixed206 schema, fitted only here.
    builder=base.FeatureBuilder().fit([records[i][1] for i in fit_ids])
    selected=base.select_features(builder.enhanced_feature_names_)
    frames=[records[i][1] for i in included]
    features=[builder.transform(frame).enhanced[:,selected] for frame in frames]
    fit_feature=np.concatenate([features[j] for j,w in enumerate(included) if int(w) in fit_set])
    quant=np.quantile(fit_feature,[.25,.5,.75],axis=0)
    median=quant[1].astype(np.float32);scale=np.maximum(quant[2]-quant[0],1e-3).astype(np.float32)
    del fit_feature
    x=np.concatenate([np.clip((f-median)/scale,-8,8).astype(np.float16) for f in features]);del features
    starts=np.r_[0,np.cumsum([len(f) for f in frames])].astype(np.int64)
    wells=np.concatenate([np.full(len(f),int(w),np.int32) for w,f in zip(included,frames)])
    labeled=np.concatenate([valid[w] for w in included])
    truth=np.concatenate([f['SW'].to_numpy(np.float64) for f in frames])
    default=np.concatenate([base.dmask(f[base.TS].to_numpy(np.float64))[:,2] for f in frames])
    fitmask=labeled&np.isin(wells,fit_ids);monmask=labeled&np.isin(wells,monitor_ids)
    numeric=np.log(truth[fitmask&~default]);q25,q50,q75=np.quantile(numeric,[.25,.5,.75])
    target_median=float(q50);target_scale=max(float(q75-q25),.1)
    centers=np.exp(np.linspace(numeric.min(),numeric.max(),256))
    labels=np.zeros((len(x),3),np.float32)
    labels[fitmask,2]=(np.log(truth[fitmask])-target_median)/target_scale
    d=np.zeros((len(x),3),bool);d[fitmask,2]=default[fitmask]
    fit_raw=np.flatnonzero(fitmask);monitor_raw=np.flatnonzero(monmask)
    lower=np.repeat(starts[:-1],np.diff(starts));upper=np.repeat(starts[1:],np.diff(starts))
    depth=np.concatenate([f['DEPTH'].to_numpy(np.float32) for f in frames])
    count={int(w):int((fitmask&(wells==w)).sum()) for w in fit_ids}
    balanced=np.ones(len(x),np.float32)
    for w,n in count.items():balanced[(wells==w)&fitmask]=len(fit_raw)/(len(fit_ids)*n)
    assert np.isfinite(x).all() and abs(float(balanced[fitmask].mean())-1)<1e-6
    assert not np.any(fitmask&monmask)
    np.savez(dest,x=x,starts=starts,wells=wells,included_wells=included,depth=depth,lower=lower,upper=upper,
             fit_raw=fit_raw,monitor_raw=monitor_raw,fitmask=fitmask,labeled=labeled,truth=truth,default_truth=default,
             labels=labels,default=d,group_weight=balanced,centers=centers,target_median=target_median,target_scale=target_scale,
             feature_median=median,feature_scale=scale)
    rows=[]
    for w in included:
        use=labeled&(wells==w)
        rows.append(dict(well_id=int(w),name=str(records[w][0]),population='fit' if int(w) in fit_set else 'monitor',
                         valid_points=int(use.sum()),default_rate=float(default[use].mean()),
                         group_weight=float(balanced[use][0]) if int(w) in fit_set else None))
    manifest=dict(phase=a.phase,created_at=datetime.now().astimezone().isoformat(),
        outer_training_wells=original_train.tolist(),outer_fifth_wells=outer_holdout.tolist(),fit_wells=fit_ids.tolist(),monitor_wells=monitor_ids.tolist(),
        fit_valid_points=len(fit_raw),monitor_valid_points=len(monitor_raw),raw_context_points=len(x),
        outer_fifth_used_for_fit_or_selection=a.phase=='outer',
        inner_split='first deterministic GroupKFold(4) split of original outer64 only',
        old_pretrained_network_used=False,old_quality_or_target_statistics_used=False,quality='Q=1;groupweights are separate learning weights',
        fit_scope='imputation,feature median/IQR,target normalization/support,initial label priors use fit wells only',
        input_features=206,relative_depth=1,feature_names=[builder.enhanced_feature_names_[i] for i in selected],
        target_median=target_median,target_scale=target_scale,support_min=float(centers[0]),support_max=float(centers[-1]),
        point_counts_min=min(count.values()),point_counts_max=max(count.values()),
        group_weight_min=float(balanced[fitmask].min()),group_weight_max=float(balanced[fitmask].max()),
        well_statistics=rows,preparation_seconds=time.perf_counter()-started,cache_bytes=dest.stat().st_size)
    write(STUDY/('data_'+a.phase+'_manifest.json'),manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['feature_names','well_statistics']},ensure_ascii=False))

if __name__=='__main__':main()
