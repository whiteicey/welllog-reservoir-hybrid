"""Preserve v35 inactive-gate evidence and the separately trained v36 repair."""
from pathlib import Path
from datetime import datetime
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'results/deep_learning/retrieval_density_v35_20260929'
STUDY=ROOT/'results/deep_learning/retrieval_smooth_gate_v36_20260929'

def main():
    path=HERE/'curation.json';cur=json.loads(path.read_text(encoding='utf-8'))
    nodes=cur.setdefault('nodes',{});edges=cur.setdefault('edges',[])
    old='pure_dl_retrieval_density_v35_SW_fold5';new='pure_dl_retrieval_smooth_v36_SW_fold5'
    diagnostic=json.loads((OLD/'gate_diagnostic.json').read_text(encoding='utf-8'))
    nodes.setdefault(old,{}).update(title='v35 跨井参考密度 · 门控失活',stage='architecture',status='recorded',highlight=False,
        primary=dict(value=73.6630827970656,kind='diagnostic',scope='原第五折 · SW单通道，检索分支失活',source='outputs/'+old+'/fold_5_report.json'),
        summary='4096训练参考/整井排除/Top16；硬截断参与门控落入负半轴。',
        conclusion='末态7973固定训练点的proposal全部负、alpha及全部检索梯度为0；18轮预测不变。不能当作有效训练后的检索负证据。v36单独修正门控，旧训练来源不改。',
        evidence=['outputs/'+old+'/fold_5_report.json','results/deep_learning/retrieval_density_v35_20260929/gate_diagnostic.json'])
    dead='dead:retrieval_gate_inactive_v35'
    cur['deadends']=[d for d in cur.get('deadends',[]) if d.get('id')!=dead]
    cur['deadends'].append(dict(id=dead,title='v35 硬截断门控失活设置',stage='architecture',parents=[old],
        summary='初始输出层有梯度，末态全关闭；权重变化不足以证明学习。',reason='修正门控后单独重试；只标记当前参数化失活，不否定跨井检索。'))
    status=ROOT/'outputs'/new/'runtime_status.json'
    final_path=STUDY/'assembled_report.json';final=json.loads(final_path.read_text(encoding='utf-8')) if final_path.exists() else None
    if status.exists():
        state=json.loads(status.read_text(encoding='utf-8'));done=state.get('status')=='complete'
        score=final['selected_scores']['SW'] if final else state.get('modes',{}).get('matched',{}).get('best',73.6630827970656)
        desc='同参考库、重排序与损失；两项11266参数，平滑1%起点，原模型单独回退。'
        if final:
            desc+='；'.join(f"{r['mode']}初始{r['report']['initial_candidate_score']:.9f}，训练后{r['post_update_scores']['SW']:.9f}" for r in final['candidates'])
            desc+=f"。实际TOTAL={final['selected_scores']['TOTAL']:.12f}；第五折开发比较，非独立确认。"
        else:desc+='运行快照：'+json.dumps(state.get('modes',{}),ensure_ascii=False)
        source='outputs/'+new+('/fold_5_report.json' if done else '/runtime_status.json')
        nodes.setdefault(new,{}).update(title='v36 检索平滑门控修正',stage='architecture',status='recorded' if done else 'running',
            highlight=bool(final and final['adopted']),primary=dict(value=score,kind='diagnostic',scope='原第五折 · SW单通道',source=source),
            summary='平滑门控恢复检索学习；真实匹配对比同井打乱标签。',conclusion=desc,
            evidence=[source,'results/deep_learning/retrieval_smooth_gate_v36_20260929/PROJECTS.md','results/deep_learning/retrieval_density_v35_20260929/memory_manifest.json'])
        for parent,note in [(old,'修正失活参数化；重新初始化分支，不继承v35已训练头'),('pure_dl_split_route_v29_SW_fold5','冻结当前SW原预测器')]:
            if not any(e.get('source')==parent and e.get('target')==new for e in edges):edges.append(dict(source=parent,target=new,type='evolution',note=note,evidence=source))
    tmp=path.with_suffix('.v36.tmp.json');tmp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
    project=HERE/'PROJECTS.md';marker='## v36检索图谱'+('完成' if final else '进行中')
    if marker not in project.read_text(encoding='utf-8-sig'):
        with project.open('a',encoding='utf-8') as f:f.write('\n\n'+marker+'\n'+datetime.now().astimezone().isoformat()+'。v35门控失活证据与v36有效训练分别记录，未覆盖旧来源。\n')
    print('V36_ATLAS',json.dumps(dict(final=bool(final),source=diagnostic['observed_at'])),flush=True)

if __name__=='__main__':main()
