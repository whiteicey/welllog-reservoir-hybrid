"""Original grouped-fold context with explicit fold-local preprocessing and Q."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import train_invariant as base

ROOT=base.ROOT
STUDY=ROOT/'results/deep_learning/fullcv_v97_20260930'
RECIPE=ROOT/'results/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def original_population():
    records=base.load_split(ROOT.parent/'data','train')
    ys=[frame[base.TS].to_numpy(np.float64) for _,frame in records]
    valid=[np.isfinite(y).all(1) for y in ys]
    truth=np.concatenate([y[m] for y,m in zip(ys,valid)])
    wells=np.concatenate([np.full(m.sum(),i,np.int32) for i,m in enumerate(valid)])
    assert len(records)==80 and truth.shape==(723556,3)
    fold=np.full(len(truth),-1,np.int8)
    for f,(_,ix) in enumerate(GroupKFold(5).split(truth,groups=wells)):fold[ix]=f
    starts=np.r_[0,np.cumsum([len(frame) for _,frame in records])]
    raw=np.concatenate([np.flatnonzero(m)+starts[i] for i,m in enumerate(valid)])
    return dict(records=records,valid=valid,truth=truth,wells=wells,fold=fold,starts=starts,valid_to_raw=raw,
        row_ids=[np.flatnonzero(wells==i) for i in range(80)])

def fold_rows(population,fold_index):
    if fold_index not in range(5):raise ValueError('Original fold index must be 0..4')
    f=population['fold'];w=population['wells'];raw=population['valid_to_raw']
    validation=np.flatnonzero(f==fold_index);training=np.flatnonzero(f!=fold_index)
    train_wells=np.unique(w[training]);heldout_wells=np.unique(w[validation])
    assert len(train_wells)==64 and len(heldout_wells)==16 and not np.intersect1d(train_wells,heldout_wells).size
    return dict(training=training,validation=validation,train_wells=train_wells.tolist(),heldout_wells=heldout_wells.tolist(),
        training_raw=raw[training],validation_raw=raw[validation])

def validate_metadata_source(master,population,fold_index):
    rows=fold_rows(population,fold_index)
    if not np.array_equal(np.sort(master['training_raw_rows']),np.sort(rows['training_raw'])):
        raise ValueError('Checkpoint training population differs from the requested original fold')
    if not np.array_equal(master['validation_raw_rows'],rows['validation_raw']):
        raise ValueError('Checkpoint validation population/row order differs from the requested original fold')
    if master['selected_feature_names']!=base.feature_contract()['whitelist']:
        raise ValueError('Checkpoint does not use the original nine-curve206 feature contract')
    return rows

def prepare_fold_data(master,fold_index,population=None):
    """Fit raw inputs on this fold only; training stays disabled until Q is bound.

    The source may provide fold-local preprocessing metadata. No network weights
    are loaded here. This function does not select a network or change a score.
    """
    pop=population or original_population();rows=validate_metadata_source(master,pop,fold_index)
    rec=pop['records'];valid=pop['valid'];starts=pop['starts'];raw=pop['valid_to_raw'];nraw=int(starts[-1]);validation=rows['validation'];training=rows['training']
    valid_x,builder,fm,fs,_=base.prepare_features(rec,rows['train_wells'],valid,pop['row_ids'])
    if not np.array_equal(fm,master['feature_median']) or not np.array_equal(fs,master['feature_scale']):
        raise ValueError('Re-fitted feature normalization differs from fold-local metadata')
    contract=base.feature_contract();x=torch.empty(nraw,206,dtype=torch.float16,device='cuda')
    for wi,(_,frame) in enumerate(rec):
        feature=builder.transform(frame).enhanced[:,contract['selected_indices']]
        value=np.clip((feature-fm)/fs,-8,8).astype(np.float16)
        x[starts[wi]:starts[wi+1]]=torch.from_numpy(value).cuda()
    assert torch.equal(x[torch.as_tensor(raw,device='cuda')],valid_x);del valid_x
    depth=torch.tensor(np.concatenate([frame['DEPTH'].to_numpy(np.float32) for _,frame in rec]),device='cuda')
    lower=torch.tensor(np.repeat(starts[:-1],np.diff(starts)),device='cuda');upper=torch.tensor(np.repeat(starts[1:],np.diff(starts)),device='cuda')
    raw_to_valid=np.full(nraw,-1,np.int64);raw_to_valid[raw]=np.arange(len(raw))
    train_raw=torch.as_tensor(rows['training_raw'],device='cuda');labels=torch.zeros(nraw,3,device='cuda')
    default=torch.zeros(nraw,3,dtype=torch.bool,device='cuda');labeled=torch.zeros(nraw,dtype=torch.bool,device='cuda')
    yy=np.column_stack([(base.TRANSFORM[j](pop['truth'][training,j])-master['regression_median'][j])/master['regression_scale'][j] for j in range(3)])
    labels[train_raw]=torch.as_tensor(yy,dtype=torch.float32,device='cuda');default[train_raw]=torch.as_tensor(base.dmask(pop['truth'][training]),device='cuda');labeled[train_raw]=True
    val_raw=torch.as_tensor(rows['validation_raw'],device='cuda');assert not bool(labeled[val_raw].any()) and bool((labels[val_raw]==0).all())
    # Pending is explicit: no silent replacement of the retained conflict-Q rule.
    return dict(x=x,depth=depth,lower=lower,upper=upper,absolute=((depth-master['absolute_depth_median'])/master['absolute_depth_scale']).clamp(-8,8),
        labels=labels,default=default,labeled=labeled,train_raw=train_raw,truth=pop['truth'],fold=pop['fold'],wells=pop['wells'],starts=starts,
        row_ids=pop['row_ids'],raw_to_valid=raw_to_valid,validation=validation,contract=contract,fold_index=fold_index,
        quality=None,quality_ready=False,qinfo=dict(status='pending_fold_local_source_trace'),population_rows=rows)

def bind_fold_quality(data,master,trace_path):
    rows=data['population_rows'];path=Path(trace_path)
    with np.load(path) as trace:
        raw_ids=trace['raw_ids'];epochs=trace['epochs']
        if not np.array_equal(np.sort(raw_ids),np.sort(rows['training_raw'])):
            raise ValueError('Quality trace contains another fold or population')
        weights=trace['weights']
        if len(np.unique(raw_ids))!=len(raw_ids) or weights.shape!=(len(epochs),len(raw_ids),3):
            raise ValueError('Malformed quality trace')
        if not np.isfinite(weights).all() or (weights<.09999).any() or (weights>1.00001).any():raise ValueError('Invalid training quality weights')
        prior=np.flatnonzero(epochs<int(master['best_epoch']))
        # Before the first source observation, the original policy uses Q=1.
        if len(prior):index=int(prior[-1]);qepoch=int(epochs[index]);point_q=weights[index].min(1)
        else:qepoch=0;point_q=np.ones(len(raw_ids),np.float32)
    quality=torch.ones(len(data['x']),device='cuda');quality[torch.as_tensor(raw_ids,device='cuda')]=torch.as_tensor(point_q,device='cuda')
    assert bool((quality[torch.as_tensor(rows['validation_raw'],device='cuda')]==1).all())
    data.update(quality=quality,quality_ready=True,qinfo=dict(status='ready',epoch=qepoch,sha256=sha(path),dynamic_updates=False,
        rule='last original training-only source Q observation strictly before this fold best epoch; Q0=1 before first observation',
        minimum=float(point_q.min()),maximum=float(point_q.max()),downweighted_training_rows=int((point_q<1).sum())))
    return data

def require_training_context(data):
    if not data.get('quality_ready') or data.get('quality') is None:raise ValueError('Fold-local conflict-Q must be bound before training')
    rows=data['population_rows'];ix=torch.as_tensor(rows['validation_raw'],device='cuda')
    assert not bool(data['labeled'][ix].any())

def add_well_bank(data):
    starts=data['starts'];sample=np.stack([lo+np.minimum(((np.arange(128)+.5)*(hi-lo)/128).astype(np.int64),hi-lo-1) for lo,hi in zip(starts[:-1],starts[1:])])
    row_well=np.repeat(np.arange(80),np.diff(starts));train_wells=np.asarray(data['population_rows']['train_wells'])
    bank=data['x'][torch.as_tensor(sample,device='cuda')].detach();data['well_bank']=bank;data['row_well']=torch.as_tensor(row_well,device='cuda')
    return dict(reference_inputs=bank[torch.as_tensor(train_wells,device='cuda')].mean(0),bank_manifest=dict(points_per_well=128,features=206,
        train_wells=train_wells.tolist(),sample_raw_rows=sample.tolist(),sample_sha256=array_sha(sample),reference='equal-training-well input bank average',
        bank_shape=list(bank.shape),validation_fit=False,bank_labels_used=False))
