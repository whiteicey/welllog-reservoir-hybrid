"""Causal replay of baseline observations; NOT weighted training or a performance estimate."""
from pathlib import Path
import json,numpy as np,pandas as pd
import train_invariant as base
from hierarchical_priority import HierarchicalPriority
r=base.ROOT;out=r/'outputs/hierarchical_importance';a=np.load(r/'outputs/pure_dl_dynamics_probe/fold_5_dynamics/training_dynamics_history.npz')
t=HierarchicalPriority();log=[]
for e,loss in zip(a['epochs'],a['nll']):
 weights=t.update(int(e),a['raw_ids'],a['well_ids'],a['block_ids'],np.logaddexp(0,loss,out=np.full_like(loss,np.nan),where=np.isfinite(loss)))
 assert weights.shape==loss.shape and np.isfinite(weights).all() and (weights>=1).all() and (weights<=1.5).all()
 assert (weights[a['default']]==1).all()
 snap=t.snapshot();log.append(snap)
np.savez_compressed(out/'priority_replay_final.npz',raw_ids=a['raw_ids'],well_ids=a['well_ids'],block_ids=a['block_ids'],weights=weights)
(out/'priority_replay.json').write_text(json.dumps({'scope':'Past-only replay of unweighted baseline observations. No retraining; weights are not evidence of improved score. Actual weighted training would produce different later observations.','records':log,'last':log[-1]},indent=2),encoding='utf-8')
# Independent quality summaries at all three levels, training wells only.
validation=set(json.loads((r/'outputs/pure_dl_relative_only_sequence/config.json').read_text())['outer_wells']['4'])
cols=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT'];wellrows=[];blockrows=[];counts=[];wids=[];localids=[]
for wi,path in enumerate(sorted((r.parent/'data/train').glob('*.txt'))):
 if path.stem in validation:continue
 frame=pd.read_csv(path,skiprows=[1]);flags=[]
 for c in cols:
  v=pd.to_numeric(frame[c],errors='coerce').to_numpy(float);lo,hi=base.VALID_RANGES[c]
  flags.append(~np.isfinite(v)|(v==-99999)|(v<lo)|(v>hi))
 flags=np.stack(flags,1);k=flags.sum(1).astype(np.uint8)
 counts.append(k);wids.append(np.full(len(k),wi,np.int16));localids.append(np.arange(len(k),dtype=np.int32))
 wellrows.append({'well_index':wi,'well_id':path.stem,'rows':len(k),'flagged_rows':int((k>0).sum()),'flagged_fraction':float((k>0).mean()),'affected_channels':int(flags.any(0).sum()),'max_channels_at_point':int(k.max())})
 for lo in range(0,len(k),128):
  kk=k[lo:lo+128];blockrows.append({'well_index':wi,'block_id':lo//128,'rows':len(kk),'flagged_rows':int((kk>0).sum()),'flagged_fraction':float((kk>0).mean()),'max_channels_at_point':int(kk.max())})
assert len(wellrows)==64
pd.DataFrame(wellrows).to_csv(out/'quality_by_training_well.csv',index=False,encoding='utf-8-sig');pd.DataFrame(blockrows).to_csv(out/'quality_by_training_block.csv',index=False,encoding='utf-8-sig')
np.savez_compressed(out/'quality_by_training_point.npz',well_ids=np.concatenate(wids),local_row_ids=np.concatenate(localids),flagged_channel_count=np.concatenate(counts))
summary={'wells':len(wellrows),'blocks':len(blockrows),'raw_rows':int(sum(x['rows'] for x in wellrows)),'wells_with_any_input_flag':sum(x['flagged_rows']>0 for x in wellrows),'blocks_with_any_input_flag':sum(x['flagged_rows']>0 for x in blockrows),'raw_rows_with_any_input_flag':sum(x['flagged_rows'] for x in wellrows),'definition':'Existing input missing/sentinel/range flags, not confirmed label errors; no whole-well downweighting; includes raw rows with unavailable targets, unlike supervised training-row counts.'}
(out/'quality_hierarchy_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'last_priority':log[-1],'quality':summary},indent=2))
