import json
from pathlib import Path
import numpy as np
import train_invariant as base
repo=base.ROOT;out=repo/'outputs/depth_change_diagnostic';out.mkdir(exist_ok=True)
source=repo/'outputs/pure_dl_lr55_cosine_1e3'
assert json.loads((source/'screening_verification.json').read_text())['passed']
oof=np.load(source/'oof.npz');truth=oof['truth'];pred=oof['pred'];fold=oof['fold'];well=oof['well']
rec=base.load_split(repo.parent/'data','train')
starts=np.r_[0,np.cumsum([len(frame) for _,frame in rec])]
valid=[np.isfinite(frame[base.TS].to_numpy(np.float64)).all(1) for _,frame in rec]
mapping=np.concatenate([np.flatnonzero(m)+starts[i] for i,m in enumerate(valid)])
default=base.dmask(truth);pdefault=base.dmask(pred)
pairs=(well[1:]==well[:-1])&(mapping[1:]==mapping[:-1]+1)
summary={}
for j,name in enumerate(base.TS):
    z=base.TRANSFORM[j](truth[:,j]);pz=base.TRANSFORM[j](np.where(fold==0,pred[:,j],truth[:,j]))
    active=pairs&(~default[1:,j])&(~default[:-1,j])
    train=active&(fold[1:]!=0);val=active&(fold[1:]==0)
    dy=np.diff(z);dp=np.diff(pz)
    threshold=float(np.quantile(abs(dy[train]),.75))
    sharp=val&(abs(dy)>threshold)
    energy=float(np.sqrt(np.mean(dp[val]**2))/np.sqrt(np.mean(dy[val]**2)))
    summary[name]={'train_consecutive_nondefault_pairs':int(train.sum()),'train_change_abs_q50_q75_q95':np.quantile(abs(dy[train]),[.5,.75,.95]).tolist(),'validation_pairs':int(val.sum()),'validation_truth_change_abs_q50_q95':np.quantile(abs(dy[val]),[.5,.95]).tolist(),'validation_prediction_change_abs_q50_q95':np.quantile(abs(dp[val]),[.5,.95]).tolist(),'prediction_to_truth_change_rms_ratio':energy,'change_correlation':float(np.corrcoef(dy[val],dp[val])[0,1]),'sharp_threshold_from_training':threshold,'sharp_pair_count':int(sharp.sum()),'sharp_direction_agreement':float(np.mean(np.sign(dp[sharp])==np.sign(dy[sharp]))),'validation_pairs_predicted_default_fraction':float(np.mean(pdefault[1:,j][val]|pdefault[:-1,j][val]))}
result={'source':str(source),'diagnostic_only':True,'no_new_predictions_or_oracle_selection':True,'transform':['log1p(POR)','log10(PERM)','log(SW)'],'targets':summary}
(out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
