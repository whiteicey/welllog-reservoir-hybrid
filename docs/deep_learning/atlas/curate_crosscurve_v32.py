"""Incremental, scope-labelled v32 feature evidence and finite negative branches."""
from pathlib import Path
from datetime import datetime
import json


def main():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    study=root/'results/deep_learning/crosscurve_features_v32_20260929'
    result=json.loads((study/'assembled_report.json').read_text(encoding='utf-8'))
    assert result['all_six_complete'] and result['actual_combined_prediction_scored']
    path=here/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    for s in result['selections']:
        t=s['target'];key='pure_dl_crosscurve_v32_'+t+'_fold5'
        rows=[r for r in result['candidates'] if r['target']==t]
        source='outputs/'+key+'/fold_5_report.json'
        description='；'.join(f"{r['mode']}更新后{r['post_update']['scores'][t]:.9f}，最终{r['report']['target_score']:.9f}@{r['report']['best_epoch']}" for r in rows)
        cur.setdefault('nodes',{}).setdefault(key,{}).update(title='v32 '+t+'跨曲线信息 · '+(s['mode'] or '保留v31'),stage='architecture',status='recorded',
            highlight=bool(s['adopted']),primary=dict(value=s['target_score'],kind='diagnostic',scope='原第五折 · '+t+'单通道',source=source),
            summary='固定当前模型，18维局部相关与同维冗余幅值对照；相同337→32头和初始状态。',
            conclusion=description+f"。实际组合TOTAL={result['selected_scores']['TOTAL']:.12f}，反复开发折结果，非独立泛化确认。",
            evidence=[source,'results/deep_learning/crosscurve_features_v32_20260929/FINAL.md','results/deep_learning/crosscurve_features_v32_20260929/feature_manifest.json'])
        dead_id='dead:crosscurve_v32_'+t
        cur['deadends']=[d for d in cur.get('deadends',[]) if d.get('id')!=dead_id]
        if not s['adopted']:
            cur['deadends'].append(dict(id=dead_id,title='v32 '+t+'当前特征接入未超越父模型',stage='architecture',parents=[key],
                summary=description,reason='本次六对三尺度相关/冗余统计及固定小头未获正增益；不追加配对窗口网格，结论限于该接入方式。'))
    tmp=path.with_suffix('.v32.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    marker='## v32最终图谱'
    project=here/'PROJECTS.md'
    if marker not in project.read_text(encoding='utf-8-sig'):
        with project.open('a',encoding='utf-8') as f:
            f.write('\n\n'+marker+'\n'+datetime.now().astimezone().isoformat()+f"。六项实际结果及父模型关系已记录，TOTAL={result['selected_scores']['TOTAL']:.12f}；未采用的设置标注有限负结论，不重复启动。\n")
    print(json.dumps(dict(scores=result['selected_scores'],delta=result['delta_reference_TOTAL'],choices=[dict(target=s['target'],mode=s['mode']) for s in result['selections']]),ensure_ascii=False))


if __name__=='__main__':main()
