"""Summarize PERM bin-count effects, pairwise CIs and bootstrap rank stability."""
from pathlib import Path
import os
import json,numpy as np
MAIN=Path(os.environ.get("WELLLOG_ROOT", Path(__file__).resolve().parents[2]));OUT=MAIN/'paper_workspace/reviewer_response_20261117/bin_summary';OUT.mkdir(parents=True,exist_ok=True)
paths={30:MAIN/'outputs/review_perm_bins30_base/report.json',60:MAIN/'outputs/perm_bins60_base/report.json',120:MAIN/'outputs/review_perm_bins120_base/report.json'}
r={k:json.loads(p.read_text()) for k,p in paths.items()};wells=np.load(MAIN/'cache/tr_well.npy');ws=np.unique(wells);cnt=np.array([(wells==w).sum() for w in ws],float)
means={k:np.array([r[k]['full_v2gate']['utility']['per_well'][str(int(w))] for w in ws]) for k in [30,60,120]}
res={'metrics':{k:{'micro':r[k]['full_v2gate']['utility']['micro'],'nondefault':r[k]['nondefault_only']['utility'],'fold_scores':r[k]['full_v2gate']['utility']['fold_scores'],'args':r[k]['args']} for k in r},'best_bin_per_fold':{},'pairwise':{}}
folds=np.array([res['metrics'][k]['fold_scores'] for k in [30,60,120]])
for f in range(5):res['best_bin_per_fold'][str(f)]=[30,60,120][int(np.argmax(folds[:,f]))]
rng=np.random.default_rng(20261117);boot={k:[] for k in [30,60,120]};best_count={k:0 for k in [30,60,120]};n_boot=2000
for b in range(n_boot):
 ix=rng.integers(0,len(ws),len(ws));vals=[]
 for k in [30,60,120]:
  v=float((means[k][ix]*cnt[ix]).sum()/cnt[ix].sum());boot[k].append(v);vals.append(v)
 best_count[[30,60,120][int(np.argmax(vals))]]+=1
for a,b in [(30,60),(60,120),(30,120)]:
 vals=np.array(boot[a])-np.array(boot[b]);obs=float(((means[a]-means[b])*cnt).sum()/cnt.sum())
 res['pairwise'][f'{a}_minus_{b}']={'delta':obs,'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]}
res['bootstrap_best_frequency']={str(k):best_count[k]/n_boot for k in best_count}
res['absolute_score_ci95']={str(k):[float(np.quantile(boot[k],.025)),float(np.quantile(boot[k],.975))] for k in boot}
res['protocol']='Two-seed ensembles; identical five well folds, base features and 1200 iterations; paired well bootstrap over per-well micro scores.'
(OUT/'metrics.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps(res,indent=2),flush=True)
