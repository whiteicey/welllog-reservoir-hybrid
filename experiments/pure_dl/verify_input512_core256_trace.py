"""CPU replay of persistent-conflict probe weights; not an audit of label correctness.
Run only after training has completed: --out OUTPUT --fold 5.
Checks Q policy replay and next-epoch log declarations. Actual loss application
and general CV/checkpoint verification are handled by separate trainer audits.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from conflict_recovery2_policy import PersistentConflictPolicy

def verify(out,fold=5):
    out=Path(out);root=Path(__file__).resolve().parents[2]
    # Completed fold artifacts are required; this verifier does not poll partial NPZs.
    report=json.loads((out/f'fold_{fold}_report.json').read_text(encoding='utf-8-sig'))
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8-sig'))
    ck=torch.load(out/f'fold_{fold}.pt',map_location='cpu',weights_only=False)
    assert ck['config']==cfg and ck['stopped_epoch']==report['stopped_epoch']
    source_hashes={}
    for name in ['conflict_recovery2_policy.py','training_conflict_probe.py','training_full512_core256.py']:
        path=Path(__file__).with_name(name).resolve()
        matches=[v for k,v in cfg['dependency_sha256'].items() if (root/k).resolve()==path]
        assert len(matches)==1 and hashlib.sha256(path.read_bytes()).hexdigest()==matches[0],name
        source_hashes[name]=matches[0]
    train=np.asarray(ck['training_raw_rows']);val=np.asarray(ck['validation_raw_rows'])
    assert not np.intersect1d(train,val).size
    with np.load(out/f'fold_{fold}_dynamics/training_dynamics_history.npz',allow_pickle=False) as archive:
        names=['epochs','raw_ids','well_ids','block_ids','default','nll','conditional_entropy','target_tail','gate_probability','cdf_l1_change']
        probe={name:archive[name].copy() for name in names}
    with np.load(out/f'fold_{fold}_conflict_weights.npz',allow_pickle=False) as archive:
        epochs=archive['epochs'].copy();raw=archive['raw_ids'].copy();weights=archive['weights'].copy()
    assert np.array_equal(epochs,probe['epochs']) and np.array_equal(raw,probe['raw_ids'])
    assert epochs.ndim==1 and len(epochs)>0 and np.issubdtype(epochs.dtype,np.integer)
    assert np.all(np.diff(epochs)>0)
    expected=np.asarray([1]+list(range(2,int(ck['stopped_epoch'])+1,2)))
    assert np.array_equal(epochs,expected),'Unexpected missing or extra probe records'
    assert raw.ndim==1 and len(raw)>0 and np.issubdtype(raw.dtype,np.integer)
    assert len(np.unique(raw))==len(raw) and np.isin(raw,train).all() and not np.isin(raw,val).any()
    default=probe['default'].astype(bool);well=probe['well_ids'];block=probe['block_ids']
    assert default.shape==(len(raw),3) and well.shape==block.shape==raw.shape
    shape=(len(epochs),len(raw),3)
    assert weights.shape==shape and weights.dtype==np.float32
    assert np.isfinite(weights).all() and ((weights>=.1)&(weights<=1)).all()
    assert np.all(weights[:,default]==1),'Default continuous channels must remain neutral'
    for name in ['nll','conditional_entropy','target_tail','gate_probability','cdf_l1_change']:assert probe[name].shape==shape
    assert np.isnan(probe['nll'][:,default]).all() and np.isnan(probe['target_tail'][:,default]).all()
    assert np.isfinite(probe['nll'][:,~default]).all()
    data=(out/'train.log').read_bytes();text=data.decode('utf-16') if data.startswith((b'\xff\xfe',b'\xfe\xff')) else data.decode('utf-8-sig')
    events=[]
    for line in text.splitlines():
        if 'CONFLICT_UPDATE' in line:
            tail=line.split('CONFLICT_UPDATE',1)[1].strip();event=json.loads(tail[tail.index('{'):])
            if int(event.get('fold',fold))==fold:events.append(event)
    assert len(events)==len(epochs),'Log/trace record count mismatch'
    policy=PersistentConflictPolicy(len(raw));record_counts=[];first=None
    for i,epoch in enumerate(epochs):
        replay=policy.update(int(epoch),probe['nll'][i],probe['conditional_entropy'][i],probe['target_tail'][i],probe['gate_probability'][i],probe['cdf_l1_change'][i],~default).astype(np.float32)
        assert np.array_equal(replay,weights[i]),f'Policy replay mismatch at epoch {epoch}'
        event=events[i];counts=(replay<1).sum(0).astype(int).tolist()
        assert int(event['observed_epoch'])==int(epoch) and int(event['applies_from_epoch'])==int(epoch)+1
        assert list(event['downweighted_by_channel'])==counts
        if 'minimum' in event:assert abs(float(event['minimum'])-float(replay.min()))<1e-7
        if any(counts) and first is None:first=int(epoch)+1
        if epoch<10:assert np.all(replay==1)
        record_counts.append({'observed_epoch':int(epoch),'applies_from_epoch':int(epoch)+1,'downweighted_by_channel':counts})
    ever=(weights<1).any(0);final=weights[-1]<1;recover=weights[1:]>weights[:-1]
    groups={}
    for name,indicator in [('ever',ever),('final',final)]:
        by_channel=[]
        for c in range(3):
            selected=indicator[:,c];pairs=np.stack([well[selected],block[selected]],1)
            by_channel.append({'probe_points':int(selected.sum()),'wells_containing_flagged_probe_points':int(len(np.unique(well[selected]))),'well_block_pairs_containing_flagged_probe_points':int(len(np.unique(pairs,axis=0)))})
        groups[name]=by_channel
    result={'passed':True,'fold':int(fold),'records':len(epochs),'source_sha256':source_hashes,'exact_float32_policy_replay':True,'training_probe_rows_only':True,'default_channels_remain_one':True,'first_non_neutral_applies_epoch':first,'ever_downweighted_by_channel':ever.sum(0).astype(int).tolist(),'final_downweighted_by_channel':final.sum(0).astype(int).tolist(),'recovery_increase_transitions_by_channel':recover.sum((0,1)).astype(int).tolist(),'unique_recovered_probe_points_by_channel':recover.any(0).sum(0).astype(int).tolist(),'group_presence':groups,'record_counts':record_counts,'coverage':{'probe_rows':len(raw),'training_labeled_rows':len(train),'fraction':float(len(raw)/len(train)),'probe_wells':int(len(np.unique(well))),'probe_well_block_pairs':int(len(np.unique(np.stack([well,block],1),axis=0)))},'final_policy_snapshot':policy.snapshot(),'scope':'Replay of training-probe function rules and next-epoch log declarations only. Suspected conditional conflicts are not verified annotation errors. Counts of wells/blocks containing flagged probe points are not whole-well/block judgments or penalties. Sparse unobserved rows are not characterized; ordinary CV/loss integration audit is separate.'}
    (out/'conflict_trace_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--fold',type=int,default=5);args=parser.parse_args()
    result=verify(args.out,args.fold)
    print(json.dumps({k:v for k,v in result.items() if k not in ['record_counts','final_policy_snapshot']},ensure_ascii=False,indent=2))
