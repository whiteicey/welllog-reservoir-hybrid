"""Compare fixed-budget CNN and CNN+MLP OOF runs on exactly the same wells."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_cv import scores

ROOT=Path(__file__).resolve().parents[2]
before=ROOT/'outputs/simple_cnn_cv';after=ROOT/'outputs/cnn_mlp_cv'
out=ROOT/'outputs/cnn_mlp_comparison';out.mkdir(parents=True,exist_ok=True)
a=np.load(before/'oof.npz');b=np.load(after/'oof.npz')
ra=json.loads((before/'report.json').read_text());rb=json.loads((after/'report.json').read_text())
for k in ['truth','fold','well']:assert np.array_equal(a[k],b[k]),k
for k in ['epochs','steps','batch','crop','seed','thresholds','optimizer','fold_wells']:
    assert ra['config'][k]==rb['config'][k],k
assert not rb['config']['anomaly_treatment']
rows=[{'fold':i+1,'cnn':ra['folds'][i]['scores']['TOTAL'],'cnn_mlp':rb['folds'][i]['scores']['TOTAL'],
       'delta':rb['folds'][i]['scores']['TOTAL']-ra['folds'][i]['scores']['TOTAL']} for i in range(5)]
well_delta={w:rb['per_well'][w]['TOTAL']-ra['per_well'][w]['TOTAL'] for w in ra['per_well']}
from collections import Counter
counts=Counter(a['well'].tolist())
ds=[];ns=[]
for wi in sorted(counts):
    ix=a['well']==wi
    ds.append(scores(a['truth'][ix],b['pred'][ix])['TOTAL']-scores(a['truth'][ix],a['pred'][ix])['TOTAL']);ns.append(counts[wi])
ds=np.array(ds);ns=np.array(ns);rng=np.random.default_rng(719);idx=rng.integers(0,80,(5000,80))
boot=(ds[idx]*ns[idx]).sum(1)/ns[idx].sum(1)
summary={'cnn':ra['oof_scores'],'cnn_mlp':rb['oof_scores'],
         'delta':{k:rb['oof_scores'][k]-ra['oof_scores'][k] for k in ra['oof_scores']},
         'folds':rows,'well_wins':sum(v>0 for v in well_delta.values()),'well_count':80,
         'paired_well_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
         'parameters':rb['config']['parameters'],'elapsed_seconds':rb['elapsed_seconds'],
         'limitations':'One seed per fold; same CV reused for model development. Bootstrap conditions on trained models, not retraining uncertainty. No held-out test score.'}
(out/'comparison.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(5)
axes[0].plot(x,[r['cnn'] for r in rows],'o-',label='CNN',color='#2463a6')
axes[0].plot(x,[r['cnn_mlp'] for r in rows],'o-',label='CNN + MLP',color='#dc7e2e')
axes[0].set_xticks(x,[f'Fold {i+1}' for i in x]);axes[0].set_ylabel('Official score / 100');axes[0].grid(alpha=.2);axes[0].legend()
keys=['POR','PERM','SW','TOTAL'];vals=[summary['delta'][k] for k in keys]
axes[1].bar(keys,vals,color=['#dc7e2e' if v>=0 else '#2463a6' for v in vals]);axes[1].axhline(0,color='black',lw=.8);axes[1].set_ylabel('Score change (CNN+MLP - CNN)');axes[1].margins(y=.3)
for i,v in enumerate(vals):axes[1].annotate(f'{v:+.3f}',(i,v),xytext=(0,5 if v>=0 else -15),textcoords='offset points',ha='center')
fig.suptitle(f"Fixed-budget 5-fold OOF: {summary['cnn']['TOTAL']:.3f} -> {summary['cnn_mlp']['TOTAL']:.3f}")
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps(summary,indent=2))
