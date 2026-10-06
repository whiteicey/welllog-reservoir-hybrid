"""Record completed v28 routing evidence without replacing historical annotations."""
from pathlib import Path
from datetime import datetime
import json


def main():
    here=Path(__file__).resolve().parent
    root=here.parents[2]
    study=root/'results/deep_learning/champion_route_v28_20260929'
    result=json.loads((study/'assembled_report.json').read_text(encoding='utf-8'))
    assert result['all_six_complete'] and result['actual_combined_prediction_scored']
    path=here/'curation.json'
    cur=json.loads(path.read_text(encoding='utf-8'))
    for selected in result['selections']:
        target=selected['target']
        key=f'pure_dl_champion_route_v28_{target}_fold5'
        candidates=[r for r in result['candidates'] if r['target']==target]
        source='outputs/'+key+'/fold_5_report.json'
        detail='；'.join(
            f"{r['mode']}={r['report']['target_score']:.9f}, 最佳{r['report']['best_epoch']}轮/停止{r['report']['stopped_epoch']}轮"
            for r in candidates)
        node=cur.setdefault('nodes',{}).setdefault(key,{})
        node.update(title=f'v28 {target}内部路由 · 采用{selected["mode"]}',
            stage='architecture',status='recorded',highlight=bool(selected['adopted']),
            primary=dict(value=selected['target_score'],scope='原第五折 · '+target+'单通道',kind='diagnostic',source=source),
            summary='冻结80.30基座与已采用的修正头；319→32→1有界路由残差，对照联合/仅条件密度路由。',
            conclusion=detail+f"。选中{selected['mode']}，完整144665点磁盘重载通过。组合TOTAL={result['selected_scores']['TOTAL']:.12f}，属于本次第五折开发结果。",
            evidence=[source,'results/deep_learning/champion_route_v28_20260929/assembled_report.json',
                'results/deep_learning/champion_route_v28_20260929/FINAL.md'])
    temporary=path.with_suffix('.v28.tmp.json')
    temporary.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)
    marker='## v28路由最终图谱'
    note=(marker+'\n'+datetime.now().astimezone().isoformat()+
          f"。三个目标节点保留六项匹配结果与最终选择；实际组合TOTAL={result['selected_scores']['TOTAL']:.12f}，相对v26{result['delta_reference_TOTAL']:+.12f}；"
          '14井提高、2井下降为该次组合开发诊断，非独立泛化确认。无新训练或其他折，旧交付保留。\n')
    project=here/'PROJECTS.md'
    previous=project.read_text(encoding='utf-8-sig') if project.exists() else ''
    if marker not in previous:
        with project.open('a',encoding='utf-8') as f:f.write('\n\n'+note)
    print(json.dumps(dict(completed=True,scores=result['selected_scores'],selections=[
        dict(target=s['target'],mode=s['mode'],best_epoch=s['best_epoch']) for s in result['selections']]),ensure_ascii=False))


if __name__=='__main__':main()
