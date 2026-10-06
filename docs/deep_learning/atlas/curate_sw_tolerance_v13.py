"""Idempotent v12/v13 overlay; preserve all unrelated current atlas curation."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def update():
    path=HERE/'curation.json'
    data=json.loads(path.read_text(encoding='utf-8-sig'))
    nodes=data.setdefault('nodes',{})
    edges=data.setdefault('edges',[])
    specs=[('v12','likelihood','v11 TCN·似然微调对照','pure_dl_independent_sw_v11_shared80_fold5'),
           ('v12','joint_utility','v11 TCN·联合动作效用','pure_dl_independent_sw_v11_shared80_fold5'),
           ('v13','likelihood','SW私有双专家·似然对照','pure_dl_independent_v8_SW_dual80_fold5'),
           ('v13','tolerance_mass','SW私有双专家·容差质量损失','pure_dl_independent_v8_SW_dual80_fold5')]
    recorded=[]
    for version,mode,title,parent in specs:
        ident='pure_dl_independent_sw_'+version+'_'+mode+'80_fold5'; out=ROOT/'outputs'/ident
        if not out.exists():continue
        read=lambda name: json.loads((out/name).read_text(encoding='utf-8-sig')) if (out/name).exists() else {}
        status=read('runtime_status.json'); report=read('fold_5_report.json'); history=read('validation_history.json')
        initial=read('initial_verification.json')
        best=float(report['target_score']) if report else status.get('best_target_score',initial.get('target_score'))
        source='outputs/'+ident+('/fold_5_report.json' if report else '/runtime_status.json')
        post=max(history,key=lambda row:row['target_score']) if isinstance(history,list) and history else None
        primary=None if best is None else dict(scope='第五折·SW单通道'+('' if report else '·训练快照'),value=best,
            scores={'SW':best},kind='target',fold=5,source=source)
        if version=='v12':
            summary='同一v11shared预训练TCN92273参数；只差0/0.5联合动作效用代理项，含默认动作竞争，温度0.05。'
        else:
            summary='较强私有SW双专家同初始化/参数/优化，NLL与NLL+0.5容差范围概率质量负对数对照；原评分/分折不改。'
        if report:
            conclusion=f"完成SW={best:.9f}，最佳第{report['best_epoch']}轮，停止{report['stopped_epoch']}轮；相对本起点{report['delta_initial']:+.9f}。"
            if post:conclusion+=f"更新后最高{post['target_score']:.9f}@{post['epoch']}。"
            conclusion+='分数为SW单目标，非TOTAL或完整OOF；无自动补折。'
        else:
            conclusion='状态='+status.get('status','prepared')+'；未形成最终恢复成绩，不预告增益。'
        nodes[ident]=dict(title=title,stage='backbone',status='recorded' if report else ('running_snapshot' if status.get('status')=='running' else status.get('status','unscored')),
            primary=primary,results=[] if primary is None else [primary],summary=summary,conclusion=conclusion,
            evidence=[source,'outputs/'+ident+'/config.json','outputs/'+ident+'/README.md'],highlight=False)
        edge=dict(source=parent,target=ident,type='evolution',evidence='outputs/'+ident+'/config.json',note='明确的相同训练折预训练来源；不是从零训练。')
        edges[:]=[e for e in edges if not (e['source']==parent and e['target']==ident)]
        edges.append(edge); recorded.append(ident)
    for version,right in [('v12','joint_utility'),('v13','tolerance_mass')]:
        src='pure_dl_independent_sw_'+version+'_likelihood80_fold5';dst='pure_dl_independent_sw_'+version+'_'+right+'80_fold5'
        if src in nodes and dst in nodes:
            edges[:]=[e for e in edges if not(e['source']==src and e['target']==dst)]
            edges.append(dict(source=src,target=dst,type='control',evidence='outputs/'+dst+'/config.json',note='同架构/起点/训练预算，仅辅助损失系数不同。'))
    final_reports=[]
    for mode in ('likelihood','tolerance_mass'):
        f=ROOT/'outputs'/('pure_dl_independent_sw_v13_'+mode+'80_fold5')/'fold_5_report.json'
        if f.exists():final_reports.append(json.loads(f.read_text(encoding='utf-8-sig')))
    if len(final_reports)==2 and all(r['delta_initial']<=0 for r in final_reports):
        ident='dead:independent_sw_tolerance_v13'
        dead=data.setdefault('deadends',[])
        dead[:]=[n for n in dead if n.get('id')!=ident]
        dead.append(dict(id=ident,title='强SW容差质量微调·本设置无增益',stage='backbone',
            parents=['pure_dl_independent_sw_v13_likelihood80_fold5','pure_dl_independent_sw_v13_tolerance_mass80_fold5'],
            summary='1281706参数、同起点两项18轮早停，均保留第0轮SW73.622385。',
            reason='更新后最高73.473302/73.457862，均低于起点。本系数与低学习率全参数微调未改善，不推广为所有评分对齐或独立模型无效。'))
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'updated':recorded},ensure_ascii=False))


if __name__=='__main__':update()
