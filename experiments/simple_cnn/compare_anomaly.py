"""Paired OOF comparison with fixed samples and well-level bootstrap."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
base=ROOT/'outputs/simple_cnn_cv';clean=ROOT/'outputs/simple_cnn_anomaly_cv'
out=ROOT/'outputs/anomaly_comparison';out.mkdir(parents=True,exist_ok=True)
a=np.load(base/'oof.npz');b=np.load(clean/'oof.npz')
ra=json.loads((base/'report.json').read_text());rb=json.loads((clean/'report.json').read_text())
for key in ['truth','well','fold']:assert np.array_equal(a[key],b[key]),key
for key in ['epochs','steps','batch','crop','seed','architecture','parameters','thresholds','optimizer','fold_wells']:
    assert ra['config'][key]==rb['config'][key],key
assert np.isfinite(b['pred']).all()

def utility(y,p):
    return np.column_stack([np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
        np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
        np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))])

delta=(utility(a['truth'],b['pred'])-utility(a['truth'],a['pred']))@np.array([.3,.35,.35])*100
well_ids=np.unique(a['well']);counts=np.array([(a['well']==w).sum() for w in well_ids])
sums=np.array([delta[a['well']==w].sum() for w in well_ids]);means=sums/counts
rng=np.random.default_rng(719);idx=rng.integers(0,len(well_ids),size=(5000,len(well_ids)))
boot=sums[idx].sum(1)/counts[idx].sum(1)
ci=np.quantile(boot,[.025,.975]);assert abs(delta.mean()-(rb['oof_scores']['TOTAL']-ra['oof_scores']['TOTAL']))<1e-8
rows=[]
for i in range(5):
    aa=ra['folds'][i]['scores'];bb=rb['folds'][i]['scores']
    rows.append({'fold':i+1,'before':aa['TOTAL'],'after':bb['TOTAL'],'delta':bb['TOTAL']-aa['TOTAL']})
pd.DataFrame(rows).to_csv(out/'fold_scores.csv',index=False)
audit={}
for i in range(5):
    spec=json.loads((clean/f'treatment_fold_{i+1}.json').read_text())
    assert set(spec['fit_wells']).isdisjoint(spec['validation_wells'])
    assert sorted(spec['validation_wells'])==sorted(ra['config']['fold_wells'][str(i)])
    for wid in spec['validation_wells']:
        for c,values in spec['audit'][wid].items():
            d=audit.setdefault(c,{'observed':0,'spikes':0,'clipped':0})
            for k,v in values.items():d[k]+=v
for d in audit.values():
    d['spike_pct']=100*d['spikes']/d['observed'];d['clip_pct']=100*d['clipped']/d['observed']
pd.DataFrame(audit).T.to_csv(out/'treatment_counts.csv',encoding='utf-8-sig')
summary={'before':ra['oof_scores'],'after':rb['oof_scores'],
         'delta':{k:rb['oof_scores'][k]-ra['oof_scores'][k] for k in ra['oof_scores']},
         'folds':rows,'paired_well_bootstrap_95ci':ci.tolist(),'well_wins':int((means>0).sum()),
         'well_losses':int((means<0).sum()),'validation_rows':len(delta),'audit':audit,
         'interpretation':'Basic cleaning vs additional fixed feature treatment; same labels, folds and seed. CI conditions on these trained models; not multi-seed evidence. No test data used.'}
(out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(13,5));colors=['#2463a6','#dc7e2e']
xx=np.arange(5)
axes[0].bar(xx-.18,[r['before'] for r in rows],.36,label='Basic cleaning',color=colors[0])
axes[0].bar(xx+.18,[r['after'] for r in rows],.36,label='Additional treatment',color=colors[1])
axes[0].set_xticks(xx,[f'Fold {i+1}' for i in xx]);axes[0].set_ylim(0,100);axes[0].set_ylabel('Official score / 100');axes[0].legend()
for i,r in enumerate(rows):axes[0].text(i,max(r['before'],r['after'])+2,f"{r['delta']:+.2f}",ha='center')
keys=['POR','PERM','SW','TOTAL'];xx=np.arange(4)
axes[1].bar(xx,[summary['delta'][k] for k in keys],color=[colors[1] if summary['delta'][k]>=0 else colors[0] for k in keys])
axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(xx,keys);axes[1].set_ylabel('Score change (after - before)')
for i,k in enumerate(keys):
    v=summary['delta'][k];axes[1].annotate(f'{v:+.3f}',(i,v),xytext=(0,6 if v>=0 else -14),textcoords='offset points',ha='center')
axes[1].margins(y=.3)
fig.suptitle(f"Same 5-fold CNN: {summary['before']['TOTAL']:.3f} -> {summary['after']['TOTAL']:.3f} | {summary['well_wins']}/80 wells improved")
fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4));order=np.argsort(means);ax.bar(np.arange(80),means[order],color=np.where(means[order]>=0,colors[1],colors[0]))
ax.axhline(0,color='black',lw=.8);ax.set_xlabel('Wells sorted by score change');ax.set_ylabel('Score change / 100');ax.set_title('Paired per-well changes (same validation samples)')
fig.tight_layout();fig.savefig(out/'well_changes.png',dpi=180);plt.close(fig)
text=['# 异常处理前后五折对照','',
      '对照组：已有基础清洗 CNN。实验组：在相同基础上加局部孤立尖峰处理与训练折 0.1%/99.9% 分位数裁剪。',
      '固定 11 点窗口、6 倍稳健局部偏差且超过 0.5 训练 IQR，只处理不超过 2 点的异常段。尖峰转缺失后沿用原插值逻辑；新增缺失也进入原缺失标记通道。',
      '9 条处理曲线为 GR/PE/SP/CAL/AC/DEN/CNL/RXO/RT。GR/RXO/RT 的处理空间为 log1p，其余为原始值。正常范围裁剪不能证明被修改的尾部值就是真异常。',
      '目标、有效行、五折划分、初始化种子、随机裁剪、网络、训练预算、固定 0.5 门控均相同。标准化仍使用原基础清洗训练折统计，避免同时改变归一化策略。',
      '', '| 指标 | 基础清洗 | 额外处理 | 差值 |','|---|---:|---:|---:|']
for k in keys:text.append(f"| {k} | {summary['before'][k]:.6f} | {summary['after'][k]:.6f} | {summary['delta'][k]:+.6f} |")
text+=['',f"改善井数：{summary['well_wins']}/80。按井配对 bootstrap 条件区间：[{ci[0]:+.3f}, {ci[1]:+.3f}] 分。",
       '该区间只反映当前已训练 OOF 结果的井间抽样变化，不包含重新训练或不同随机种子的波动。本次不根据结果修改规则继续挑选。',
       '', '![对照](comparison.png)','', '![逐井差异](well_changes.png)']
(out/'README.md').write_text('\n'.join(text),encoding='utf-8')
print(json.dumps(summary,indent=2))
