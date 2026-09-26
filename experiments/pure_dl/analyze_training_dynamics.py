"""Audit observational training probes and summarize persistent difficulty without reweighting."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[2]
out=root/'outputs/pure_dl_dynamics_probe';p=out/'fold_5_dynamics'
a=np.load(p/'training_dynamics_history.npz');s=json.loads((p/'training_dynamics_summary.json').read_text())
base=root/'outputs/pure_dl_relative_only_sequence'
ha=json.loads((out/'fold_5_validation_history.json').read_text());hb=json.loads((base/'fold_5_validation_history.json').read_text())
assert ha==hb
assert np.array_equal(np.load(out/'fold_5_pred.npy'),np.load(base/'fold_5_pred.npy'))
c=torch.load(out/'fold_5.pt',map_location='cpu',weights_only=False);b=torch.load(base/'fold_5.pt',map_location='cpu',weights_only=False)
assert c['state_dict'].keys()==b['state_dict'].keys()
assert all(torch.equal(v,b['state_dict'][k]) for k,v in c['state_dict'].items())
ids=a['raw_ids'];assert len(np.unique(ids))==len(ids)
assert np.isin(ids,c['training_raw_rows']).all() and not np.isin(ids,c['validation_raw_rows']).any()
import train_invariant as invariant
rec=invariant.load_split(root.parent/'data','train');starts=np.r_[0,np.cumsum([len(frame) for _,frame in rec])]
actual_well=np.searchsorted(starts[1:],ids,side='right')
assert np.array_equal(actual_well,a['well_ids'])
assert np.array_equal((ids-starts[actual_well])//128,a['block_ids'])
assert len(np.unique(actual_well))==64
assert np.isnan(a['nll'][:,a['default']]).all()
for key in ['conditional_entropy','gate_entropy','gate_probability']:
 assert np.isfinite(a[key]).all() and a[key].min()>=-1e-5 and a[key].max()<=1.00001
correct=a['mean_probe_correct'];active=~a['default']
forget=((correct[:-1])&(~correct[1:])).sum(0)
result={'audit':{'passed':True,'train_only_points':len(ids),'train_wells':64,'records':len(a['epochs']),'history_exactly_matches_baseline':True,'final_predictions_exactly_match_baseline':True,'best_state_exactly_matches_baseline':True},'channels':{}}
fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
for j,name in enumerate(['POR','PERM','SW']):
 mask=active[:,j];nll=a['nll'][:,mask,j];hard=a['hard_p90'][:,mask,j].mean(0);persistent=hard>=.5
 ent=a['conditional_entropy'][:,mask,j];change=a['cdf_l1_change'][:,mask,j];utility=a['mean_probe_utility'][:,mask,j]
 groups={}
 for title,sel in [('persistent_hard',persistent),('other',~persistent)]:
  f=forget[mask,j][sel]
  groups[title]={'points':int(sel.sum()),'nll_first_mean':float(nll[0,sel].mean()),'nll_last_mean':float(nll[-1,sel].mean()),'nll_drop_mean':float((nll[0,sel]-nll[-1,sel]).mean()),'last_entropy_mean':float(ent[-1,sel].mean()),'late_cdf_change_mean':float(np.nanmean(change[-8:,sel])),'any_forgetting_fraction':float((f>0).mean()),'forgetting_per_point':float(f.mean()),'last_conditional_mean_utility':float(utility[-1,sel].mean())}
 result['channels'][name]={'nondefault_points':int(mask.sum()),'persistent_hard_fraction':float(persistent.mean()),'groups':groups}
 epochs=a['epochs']
 axes[j,0].plot(epochs,nll.mean(1),label='All nondefault probes');axes[j,0].plot(epochs,nll[:,persistent].mean(1),label='Persistently hard probes');axes[j,0].set_title(name+' conditional NLL');axes[j,0].legend(fontsize=8)
 axes[j,1].plot(epochs,ent[:,persistent].mean(1),label='Persistent');axes[j,1].plot(epochs,ent[:,~persistent].mean(1),label='Other');axes[j,1].set_title(name+' normalized conditional entropy');axes[j,1].legend(fontsize=8)
 axes[j,2].plot(epochs[1:],np.nanmean(change[1:,persistent],axis=1),label='Persistent');axes[j,2].plot(epochs[1:],np.nanmean(change[1:,~persistent],axis=1),label='Other');axes[j,2].set_title(name+' CDF change since previous probe');axes[j,2].legend(fontsize=8)
 for ax in axes[j]:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
fig.suptitle('Fixed training probes only: retrospective persistent-hard groups; no weighting',fontsize=12)
fig.savefig(p/'training_dynamics.png',dpi=140);plt.close(fig)
result['limitations']=['Persistent means top10% NLL in at least half of recorded epochs within target; retrospective definition, not a deployable online rule.','Four fixed windows per training well are not a census or representative random row sample.','Groups do not prove label errors; conditional-mean utility is not the model final expected-utility decode.','Cross-target NLL and discretized entropy are not directly comparable difficulty scales.','First-to-last decrease includes early fitting; no-improvement candidates require late-window analysis.']
(p/'independent_dynamics_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
