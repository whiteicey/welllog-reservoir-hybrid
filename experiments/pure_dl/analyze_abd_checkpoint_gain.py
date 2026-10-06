"""CPU-only attribution of the saved80.49231 gain, never a prediction selector."""
import json
from pathlib import Path
import numpy as np
import train_invariant as base

ROOT=base.ROOT
SOURCE=ROOT/'outputs/pure_dl_abd_scratch_control80_fold5'
OUT=ROOT/'results/deep_learning/abd_retrain_attribution'


def point_scores(y,p):
    return 100*np.c_[np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
        np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
        np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]


def main():
    with np.load(SOURCE/'oof.npz') as source:
        ids=np.flatnonzero(source['fold']==4)
        y=source['truth'][ids];p=source['pred'][ids];wells=source['well'][ids]
    with np.load(ROOT/'outputs/pure_dl_8040_pair_ablation/ablation_predictions.npz') as source:
        assert np.array_equal(source['validation_rows'],ids)
        before=source['four_fine__triple_drop_3']
    assert p.shape==before.shape==(144665,3)
    old=point_scores(y,before);new=point_scores(y,p);diff=new-old
    weights=np.array([.3,.35,.35])
    old_scores=base.metric(y,before);new_scores=base.metric(y,p)
    assert np.allclose(old.mean(0),[old_scores[k] for k in base.TS],atol=1e-10)
    assert np.allclose(new.mean(0),[new_scores[k] for k in base.TS],atol=1e-10)
    expected=new_scores['TOTAL']-old_scores['TOTAL']
    assert abs(diff.mean(0)@weights-expected)<1e-10
    default=base.dmask(y);old_default=base.dmask(before);new_default=base.dmask(p)
    channel_rows=[];state_rows=[]
    for j,name in enumerate(base.TS):
        channel_rows.append({'channel':name,'old':old_scores[name],'new':new_scores[name],
                             'delta':new_scores[name]-old_scores[name],
                             'total_contribution':float(diff[:,j].mean()*weights[j])})
        for title,mask in [('true_default',default[:,j]),('true_nondefault',~default[:,j])]:
            item={'channel':name,'population':title,'count':int(mask.sum()),'fraction':float(mask.mean()),
                  'old_score':float(old[mask,j].mean()),'new_score':float(new[mask,j].mean()),
                  'conditional_delta':float(diff[mask,j].mean()),
                  'total_contribution':float(np.where(mask,diff[:,j],0).mean()*weights[j]),
                  'pred_default_to_numeric':int((mask & old_default[:,j] & ~new_default[:,j]).sum()),
                  'pred_numeric_to_default':int((mask & ~old_default[:,j] & new_default[:,j]).sum())}
            state_rows.append(item)
    assert abs(sum(r['total_contribution'] for r in state_rows)-expected)<1e-10
    well_rows=[]
    for wi in np.unique(wells):
        mask=wells==wi
        well_rows.append({'well_index':int(wi),'count':int(mask.sum()),'fraction':float(mask.mean()),
                          'delta':float(diff[mask].mean(0)@weights),
                          'total_contribution':float((diff[mask]@weights).sum()/len(y)),
                          'channel_delta':dict(zip(base.TS,diff[mask].mean(0).tolist()))})
    assert abs(sum(r['total_contribution'] for r in well_rows)-expected)<1e-10
    well_rows.sort(key=lambda row:row['total_contribution'],reverse=True)
    top=well_rows[0]
    excluding_top=(expected-top['total_contribution'])/(1-top['fraction'])
    report={'rows':len(y),'old_scores':old_scores,'new_scores':new_scores,'total_delta':expected,
            'channel_deltas':channel_rows,'default_numeric_decomposition':state_rows,'well_deltas':well_rows,
            'positive_wells':sum(r['delta']>0 for r in well_rows),'negative_wells':sum(r['delta']<0 for r in well_rows),
            'delta_excluding_largest_positive_well':excluding_top,
            'scope':'post-evaluation CPU diagnostic only; every point retains the same fixed prediction, no per-row/well selection or new fitting'}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'gain_decomposition.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# 80.49231 的收益来自哪里','','## 探索重点','',
           '区分融合选模的提升来自哪个通道、默认/非默认值群体及哪些井；检验收益是否集中。只分析已有预测，不改变任何预测。','','## 已经探索的内容与验证','',
           f"固定第五折144665点，原ABD {old_scores['TOTAL']:.8f} → 新融合选模 {new_scores['TOTAL']:.8f}，差值 **{expected:+.8f}**。仅分析已有预测，不根据标签挑选任何点的输出。",'',
           '|目标|原分数|新分数|通道变化|对TOTAL的贡献|','|---|---:|---:|---:|---:|']
    for r in channel_rows:lines.append(f"|{r['channel']}|{r['old']:.6f}|{r['new']:.6f}|{r['delta']:+.6f}|{r['total_contribution']:+.6f}|")
    lines+=['','|目标|真实标签类型|点数|该类分数变化|对TOTAL贡献|','|---|---|---:|---:|---:|']
    for r in state_rows:
        label='默认值' if r['population']=='true_default' else '连续非默认值'
        lines.append(f"|{r['channel']}|{label}|{r['count']}|{r['conditional_delta']:+.6f}|{r['total_contribution']:+.6f}|")
    lines+=['','## 取得的收益','',f"井级：{report['positive_wells']}口改善、{report['negative_wells']}口下降。最大正贡献井为索引{top['well_index']}，对TOTAL贡献{top['total_contribution']:+.6f}；排除这口井后，其余井总体变化为{excluding_top:+.6f}。此排除仅用于集中度诊断，不改变正式评分或模型。",'',
            '## 不值得继续探索的点及证据','',
            '不能仅依据总体+0.08316声称跨井普遍改善；不按验证井或真实标签类型选择不同模型、输出或权重，也不把诊断中的排除操作带入正式评分。当前PERM连续值改善，而POR/SW连续值退步，应保留这种取舍的证据。',
            '本结果不能证明统计显著性或完整五折收益。默认/非默认分解采用原项目dmask，与训练口径一致；各通道按0.30/0.35/0.35加权，所有分解贡献严格加总回总差值。']
    (OUT/'GAIN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='well_deltas'},indent=2),flush=True)


if __name__=='__main__':main()
