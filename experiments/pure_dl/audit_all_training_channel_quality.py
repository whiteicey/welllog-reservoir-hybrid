"""All80 raw training wells: channel validity flags, missingness distinguished from finite range flags."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import train_invariant as base
r=base.ROOT;p=r/'outputs/training_channel_quality';p.mkdir(exist_ok=True)
inputs=base.INPUT_COLUMNS;targets=base.TS;used=['GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT']
files=sorted((r.parent/'data/train').glob('*.txt'));assert len(files)==80
chunks=[];missing_chunks=[];range_chunks=[];well_rows=[]
columns=inputs+targets
for path in files:
 f=pd.read_csv(path,skiprows=[1]);bad=[];missing=[];ranges=[]
 for c in columns:
  v=pd.to_numeric(f[c],errors='coerce').to_numpy(float) if c in f else np.full(len(f),np.nan);m=(v==-99999)|~np.isfinite(v);lo,hi=base.VALID_RANGES[c];o=~m&((v<lo)|(v>hi))
  missing.append(m);ranges.append(o);bad.append(m|o)
 b=np.stack(bad,1);chunks.append(b);missing_chunks.append(np.stack(missing,1));range_chunks.append(np.stack(ranges,1));well_rows.append({'well_id':path.stem,'rows':len(f),'any_input_flag':int(b[:,:len(inputs)].any(1).sum())})
b=np.concatenate(chunks);m=np.concatenate(missing_chunks);o=np.concatenate(range_chunks);n=len(b);assert n==730268 and np.array_equal(b,m|o) and not (m&o).any()
def scope(names):
 ix=[columns.index(c) for c in names];x=b[:,ix];counts=x.sum(1);h=np.bincount(counts,minlength=len(names)+1)
 return {'finite_range_histogram':[{'flagged_channels':i,'rows':int(v),'percent_all_rows':float(100*v/n)} for i,v in enumerate(np.bincount(o[:,ix].sum(1),minlength=len(names)+1))], 'finite_range_multiple_rows':int((o[:,ix].sum(1)>=2).sum()),'channels':names,'channel_count':len(names),'rows':n,'any_flag_rows':int((counts>=1).sum()),'any_flag_percent':float(100*(counts>=1).mean()),'multiple_flag_rows':int((counts>=2).sum()),'multiple_flag_percent_all_rows':float(100*(counts>=2).mean()),'multiple_percent_among_flagged_rows':float(100*(counts>=2).sum()/max(1,(counts>=1).sum())),'maximum_simultaneous_flagged_channels':int(counts.max()),'histogram':[{'flagged_channels':i,'rows':int(v),'percent_all_rows':float(100*v/n)} for i,v in enumerate(h)],'missing_any_rows':int(m[:,ix].any(1).sum()),'finite_outside_range_any_rows':int(o[:,ix].any(1).sum()),'finite_outside_range_any_percent':float(100*o[:,ix].any(1).mean())}
per=[{'channel':c,'total_flagged':int(b[:,j].sum()),'percent_rows':float(100*b[:,j].mean()),'missing_or_nonfinite_including_absent_column':int(m[:,j].sum()),'finite_outside_existing_range':int(o[:,j].sum()),'existing_range':base.VALID_RANGES[c]} for j,c in enumerate(columns)]
result={'scope':'80 raw train wells, all730268 depth rows; test excluded; default target values are not flagged merely for being defaults. Flags follow existing loader ranges, not confirmed geologic anomalies or label errors.','input14':scope(inputs),'common_input13':scope([c for c in inputs if c!='CASE']),'model_source_curves9':scope(used),'targets3':scope(targets),'all17':scope(columns),'per_channel':per}
(p/'statistics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');pd.DataFrame(per).to_csv(p/'per_channel.csv',index=False,encoding='utf-8-sig');pd.DataFrame(result['input14']['histogram']).to_csv(p/'input_channel_count_histogram.csv',index=False,encoding='utf-8-sig');pd.DataFrame(well_rows).to_csv(p/'per_well.csv',index=False,encoding='utf-8-sig')
print(json.dumps({'finite_range_histogram_13':result['common_input13']['finite_range_histogram'],'finite_range_multiple_rows':result['common_input13']['finite_range_multiple_rows']},indent=2))
