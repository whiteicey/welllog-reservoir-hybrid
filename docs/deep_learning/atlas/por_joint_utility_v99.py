"""Loss-alignment study; parent or partial score is not a final gain."""
import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_joint_utility_v99_20260930'
    if not (study/'README.md').exists():return
    out=root/'outputs/pure_dl_por_joint_utility_v99_POR_fold5'
    read=lambda p:json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    runtime=read(out/'runtime_status.json');report=read(out/'fold_5_report.json')
    status='recorded' if report else ('running_snapshot' if runtime.get('status')=='running'
                                    else 'failed' if runtime.get('status')=='failed' else 'prepared')
    ident='pure_dl_por_joint_utility_v99_POR_fold5'
    conclusion='未训练，尚无成绩' if not runtime else '实际状态='+str(runtime.get('status'))+'；单折训练快照，不是完整OOF。'
    if report:conclusion='完成同容量joint减likelihood TOTAL='+str(report['predefined_ablation_delta_TOTAL'])+'；初始函数不算新收益。'
    node(ident,'v99 POR联合动作收益代理',
         '原v31全函数热启动，现有43635路由/位置参数；同初始化/AdamW3e-5，原似然对照原似然+0.5 joint softaction效用，temperature0.05。原九曲线、Q50和解码保持，仅训练側物理POR查表。',
         conclusion+'80.504197仅父组合；不修改v97固定配方或自动补折。',
         stage='optimization',status=status,resultText=conclusion,
         evidence=['results/deep_learning/por_joint_utility_v99_20260930/README.md'])
    edge('pure_dl_mechanism_v31_POR_fold5',ident,'evolution','实际constructor仍加载保留v31全函数，不加载v98负结果')
    edge('pure_dl_por_effective_sam_v98_POR_fold5',ident,'comparison','同容量/更新范围；改变损失机制，实际状态见本节点')
    for mode in ('likelihood','joint'):
        arm=next((a for a in report.get('arms',[]) if a['mode']==mode),None)
        history=read(out/mode/'validation_history.json') or []
        shown=dict(resultText='尚无训练成绩；epoch0父函数非新收益')
        if arm or history:
            value=arm or max(history,key=lambda a:a['target_score'])
            result=dict(value=value['reference_replacement_total'],scores={'POR':value['target_score'],'TOTAL':value['reference_replacement_total']},
                        kind='fifth' if arm else 'observed',fold=5,scope='原第五折固定其他目标组合'+('最终' if arm else '训练快照'),
                        source='outputs/pure_dl_por_joint_utility_v99_POR_fold5/'+mode+'/fold_5_report.json' if arm else 'outputs/pure_dl_por_joint_utility_v99_POR_fold5/'+mode+'/validation_history.json')
            shown=dict(primary=result,results=[result])
        node(ident+'/'+mode,'v99 POR '+mode,'同43635参数/初始化，仅联合动作收益系数0/.5不同。',
             '独立最佳选择与早停；只有真实最终恢复后比较才算消融收益。',stage='optimization',status='recorded' if arm else status,**shown)
        edge(ident,ident+'/'+mode,'comparison','预定义匹配损失对照')
    if report and not any(a['positive_gain_and_gate'] for a in report['arms']):
        end('dead:v99_por_joint_utility','v99 当前POR有效模块联合效用无收益','optimization',
            [ident+'/'+m for m in ('likelihood','joint')],
            '同43635参数/初始化/AdamW3e-5，仅系数0/.5，温度.05；18轮停止恢复epoch0，选定TOTAL差0，更新后joint-like=-0.000170132804。',
            '关闭本原函数、更新范围与损失配置的系数/温度邻域；不泛化为所有效用损失或其他通道无效。')
