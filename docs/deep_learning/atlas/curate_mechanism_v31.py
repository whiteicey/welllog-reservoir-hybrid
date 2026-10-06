"""Incremental v31 evidence; preserve previous curation and score scopes."""
from pathlib import Path
from datetime import datetime
import json


def main():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    study=root/'results/deep_learning/mechanism_checks_v31_20260929'
    result=json.loads((study/'assembled_report.json').read_text(encoding='utf-8'))
    assert result['all_four_complete'] and result['actual_combined_prediction_scored']
    path=here/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    for s in result['selections']:
        t=s['target'];key='pure_dl_mechanism_v31_'+t+'_fold5'
        rows=[r for r in result['candidates'] if r['target']==t]
        detail='；'.join(f"{r['mode']}更新后{r['post_update']['scores'][t]:.9f}，最终{r['report']['target_score']:.9f}@{r['report']['best_epoch']}" for r in rows)
        source='outputs/'+key+'/fold_5_report.json'
        cur.setdefault('nodes',{}).setdefault(key,{}).update(title='v31 '+t+('同范围路由重参数化' if t=='POR' else '单调条件密度'),stage='architecture',status='recorded',
            highlight=False,primary=dict(value=s['target_score'],scope='原第五折 · '+t+'单通道',kind='diagnostic',source=source),
            summary='当前80.336212上按诊断检验参数化与低自由度分布；共四项，原数据与解码保持。',
            conclusion=detail+f"。最终组合TOTAL={result['selected_scores']['TOTAL']:.12f}；微小POR增量不构成稳定泛化证据，不扩大本设置。",
            evidence=[source,'results/deep_learning/mechanism_checks_v31_20260929/FINAL.md','results/deep_learning/mechanism_checks_v31_20260929/METHOD.md'])
    key='dead:sw_lowdim_warp_v31'
    cur['deadends']=[v for v in cur.get('deadends',[]) if v.get('id')!=key]
    cur['deadends'].append(dict(id=key,title='v31 SW低自由度仿射/四段变换·本设置无收益',stage='architecture',
        parents=['pure_dl_mechanism_v31_SW_fold5'],summary='两项18轮早停并回退原SW73.663082797；训练后最高73.629397760/73.630069916。',
        reason='固定当前默认概率与骨干，本次两类条件变换无正收益；不继续分段数/边界网格，不否定所有单调密度模型。'))
    temp=path.with_suffix('.v31.tmp.json');temp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
    marker='## v31最终图谱'
    p=here/'PROJECTS.md';old=p.read_text(encoding='utf-8-sig')
    if marker not in old:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n'+datetime.now().astimezone().isoformat()+f"。四项完成，TOTAL={result['selected_scores']['TOTAL']:.12f}；保存POR极小增量和SW两项负结果，单通道分数仍为diagnostic，无其他折。\n")
    print(json.dumps(dict(scores=result['selected_scores'],delta=result['delta_reference_TOTAL'],curated=True),ensure_ascii=False))

if __name__=='__main__':main()
