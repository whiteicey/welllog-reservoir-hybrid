"""NOT_USED: superseded old [1,1.5]-only priority prototype verifier.
CPU-only replay of saved training-probe priorities; no model inference or fitting.
Do not use for the new mild-downweighting and independent-quality recovery policy.
Usage: python verify_hierarchical_priority_trace.py --out OUTPUT --fold 5
Required log schema: PRIORITY_UPDATE {observed_epoch, applies_from_epoch,
increased_by_channel:[n0,n1,n2]}. This checks raw priority A before any batch loss
normalization. It does not establish sample corruption or label unreliability.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from hierarchical_priority import HierarchicalPriority
import hierarchical_priority

def verify(out,fold=5):
    out=Path(out);root=Path(__file__).resolve().parents[2]
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8-sig'))
    hashes=cfg.get('dependency_sha256',cfg.get('dependency_hash'))
    assert isinstance(hashes,dict),'Configuration must contain dependency hashes'
    source=Path(hierarchical_priority.__file__).resolve()
    entries=[(relative,value) for relative,value in hashes.items() if (root/relative).resolve()==source]
    assert len(entries)==1,'Frozen priority module missing or duplicated in dependencies'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==entries[0][1],'Priority source changed'
    checkpoint=torch.load(out/f'fold_{fold}.pt',map_location='cpu',weights_only=False)
    assert checkpoint['config']==cfg
    training=np.asarray(checkpoint['training_raw_rows']);validation=np.asarray(checkpoint['validation_raw_rows'])
    assert not np.intersect1d(training,validation).size
    with np.load(out/f'fold_{fold}_dynamics/training_dynamics_history.npz',allow_pickle=False) as archive:
        data={k:archive[k].copy() for k in ['epochs','raw_ids','well_ids','block_ids','default','nll']}
    with np.load(out/f'fold_{fold}_priority_history.npz',allow_pickle=False) as archive:
        trace={k:archive[k].copy() for k in ['epochs','raw_ids','weights']}
    epochs=data['epochs'];raw=data['raw_ids'];well=data['well_ids'];block=data['block_ids'];default=data['default'].astype(bool)
    assert epochs.ndim==1 and len(epochs)>0 and np.all(np.diff(epochs)>0)
    assert np.issubdtype(epochs.dtype,np.integer) and np.issubdtype(raw.dtype,np.integer)
    assert np.array_equal(epochs,trace['epochs']) and np.array_equal(raw,trace['raw_ids'])
    assert len(np.unique(raw))==len(raw) and len(raw)>0
    assert np.isin(raw,training).all() and not np.isin(raw,validation).any()
    assert well.shape==block.shape==raw.shape and default.shape==(len(raw),3)
    assert data['nll'].shape==trace['weights'].shape==(len(epochs),len(raw),3)
    assert trace['weights'].dtype==np.float32
    assert np.isfinite(trace['weights']).all() and ((trace['weights']>=1)&(trace['weights']<=1.5)).all()
    assert np.isnan(data['nll'][:,default]).all(),'Default channels must be absent from continuous NLL'
    assert np.isfinite(data['nll'][:,~default]).all(),'Nondefault probe NLL must be finite'
    assert np.all(trace['weights'][:,default]==1),'Default channel must remain neutral'
    # Probe cadence is epoch 1 followed by every even epoch, without record gaps.
    assert np.array_equal(epochs,np.asarray([1]+list(range(2,int(epochs[-1])+1,2))))
    payload=(out/'train.log').read_bytes()
    text=payload.decode('utf-16') if payload.startswith((b'\xff\xfe',b'\xfe\xff')) else payload.decode('utf-8-sig')
    events=[]
    for line in text.splitlines():
        if 'PRIORITY_UPDATE' in line:
            tail=line.split('PRIORITY_UPDATE',1)[1].strip()
            events.append(json.loads(tail[tail.index('{'):]))
    events=[e for e in events if int(e.get('fold',fold))==fold]
    assert len(events)==len(epochs),'Priority log count differs from trace'
    tracker=HierarchicalPriority();counts=[];first=None
    for index,epoch in enumerate(epochs):
        nll=data['nll'][index]
        # Stable softplus applied only to diagnostics, not the optimized loss.
        difficulty=np.logaddexp(0,nll)
        replay=tracker.update(int(epoch),raw,well,block,difficulty).astype(np.float32)
        assert np.array_equal(replay,trace['weights'][index]),f'Priority replay mismatch at epoch {epoch}'
        count=(replay>1).sum(0).astype(int).tolist();event=events[index]
        assert int(event['observed_epoch'])==int(epoch)
        assert int(event['applies_from_epoch'])==int(epoch)+1,'Priorities must not affect their own observed epoch'
        assert list(event['increased_by_channel'])==count,'Priority increased counts differ'
        if 'covered_points' in event:assert int(event['covered_points'])==len(raw)
        if any(count) and first is None:first=int(epoch)+1
        if int(epoch)<10:assert np.all(replay==1)
        counts.append({'observed_epoch':int(epoch),'applies_from_epoch':int(epoch)+1,'increased_by_channel':count,'any_channel_increased_points':int((replay>1).any(1).sum())})
    pairs=np.stack([well,block],1)
    result={'status':'NOT_USED_OLD_PROTOTYPE','passed':True,'fold':int(fold),'records':len(epochs),'epochs':epochs.astype(int).tolist(),'first_non_neutral_applies_epoch':first,'increased_counts':counts,'observed_priority_range':[float(trace['weights'].min()),float(trace['weights'].max())],'training_only_raw_ids':True,'validation_rows_intersect':0,'frozen_source_sha256':entries[0][1],'exact_float32_replay':True,'default_channels_neutral':True,'coverage':{'probe_unique_rows':len(raw),'training_labeled_rows':len(training),'fraction_training_labeled_rows':float(len(raw)/len(training)),'probe_wells':int(len(np.unique(well))),'probe_well_block_pairs':int(len(np.unique(pairs,axis=0))),'nondefault_points_by_channel':(~default).sum(0).astype(int).tolist(),'probe_points_per_well':{str(int(w)):int((well==w).sum()) for w in np.unique(well)}},'tracker_snapshot':tracker.snapshot(),'scope':'Sparse fixed training probes. Checks raw priority A and next-epoch application declarations; batch normalization is verified separately in loss/trainer audit. High loss is not evidence of anomalous or unreliable labels. No validation data used for priorities.'}
    (out/'priority_trace_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--fold',type=int,default=5)
    args=parser.parse_args();result=verify(args.out,args.fold)
    print(json.dumps({k:v for k,v in result.items() if k not in ['increased_counts','tracker_snapshot']},ensure_ascii=False,indent=2))
