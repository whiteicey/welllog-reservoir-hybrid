"""Training-only raw quality audit; no automatic anomaly or difficulty labels."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
import train_invariant as base
r=base.ROOT;out=r/'outputs/training_quality_audit';out.mkdir(exist_ok=True)
cfg=json.loads((r/'outputs/pure_dl_relative_only_sequence/config.json').read_text())
validation=set(cfg['outer_wells']['4']);files=sorted((r.parent/'data/train').glob('*.txt'))
a=np.load(r/'outputs/pure_dl_dynamics_probe/fold_5_dynamics/training_dynamics_history.npz')
used=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']
counts={c:{'sentinel':0,'nonfinite_other':0,'outside_existing_range':0} for c in used+base.TS}
wellrows=[];flags=np.zeros((len(a['raw_ids']),len(used)),bool);visited=np.zeros(len(flags),bool)
for wi,path in enumerate(files):
 if path.stem in validation:continue
 raw=pd.read_csv(path,skiprows=[1]);clean=base.load_split.__globals__['read_curve'](path)
 bads=[]
 for c in used+base.TS:
  v=pd.to_numeric(raw[c],errors='coerce').to_numpy(float);sent=v==-99999;lo,hi=base.VALID_RANGES[c]
  outside=np.isfinite(v)&~sent&((v<lo)|(v>hi));other=~np.isfinite(v)
  counts[c]['sentinel']+=int(sent.sum());counts[c]['nonfinite_other']+=int(other.sum());counts[c]['outside_existing_range']+=int(outside.sum())
  if c in used:bads.append(sent|outside|other)
 bad=np.stack(bads,1);depth=pd.to_numeric(raw['DEPTH'],errors='coerce')
 duplicate=depth.duplicated(keep=False)&depth.notna();conflicts=0
 if duplicate.any():
  for _,g in clean.loc[duplicate].groupby('DEPTH'):
   if any(g[c].dropna().nunique()>1 for c in base.TS):conflicts+=1
 valid=np.isfinite(clean[base.TS].to_numpy(float)).all(1)
 grid=np.arange(0,len(raw),128)+64;chosen=grid[np.unique(np.linspace(0,len(grid)-1,min(4,len(grid)),dtype=int))]
 local=np.concatenate([np.arange(max(0,int(center)-64),min(len(raw),int(center)+64)) for center in chosen]);local=local[valid[local]]
 pos=np.flatnonzero(a['well_ids']==wi);assert len(pos)==len(local)
 assert np.array_equal(a['block_ids'][pos],local//128)
 flags[pos]=bad[local];visited[pos]=True
 wellrows.append({'well_index':wi,'well_id':path.stem,'rows':len(raw),'complete_target_rows':int(valid.sum()),'rows_any_used_curve_masked':int(bad.any(1).sum()),'duplicate_depth_rows':int(duplicate.sum()),'duplicate_depth_target_conflict_groups':conflicts,'non_increasing_depth_steps':int((np.diff(depth.to_numpy(float))<=0).sum())})
assert len(wellrows)==64 and visited.all() and not(set(x['well_id'] for x in wellrows)&validation)
comparisons={}
for j,n in enumerate(base.TS):
 active=~a['default'][:,j];persistent=a['hard_p90'][:,:,j].mean(0)>=.5
 comparisons[n]={}
 for name,m in [('persistent_hard',active&persistent),('other_nondefault',active&~persistent)]:
  comparisons[n][name]={'points':int(m.sum()),'any_used_curve_masked_fraction':float(flags[m].any(1).mean()),'mean_missing_curve_count':float(flags[m].sum(1).mean())}
result={'scope':'Only fifth-fold64 training wells raw files parsed; validation file names used solely to exclude them. Existing loader ranges are implementation rules, not newly established universal physical limits.','wells':64,'rows':sum(x['rows'] for x in wellrows),'complete_target_rows':sum(x['complete_target_rows'] for x in wellrows),'existing_loader_masks':counts,'duplicate_depth_rows':sum(x['duplicate_depth_rows'] for x in wellrows),'conflicting_duplicate_depth_groups':sum(x['duplicate_depth_target_conflict_groups'] for x in wellrows),'probe_input_missingness_by_difficulty':comparisons,'automatic_deletion_or_reweighting':False}
(out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');pd.DataFrame(wellrows).to_csv(out/'training_wells.csv',index=False,encoding='utf-8-sig')
print(json.dumps(result,indent=2))
