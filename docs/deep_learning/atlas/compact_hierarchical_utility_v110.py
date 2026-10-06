import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_hierarchical_utility_v110_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_compact_hierarchical_utility_v110_fold5/hierarchical';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'assembled.json')
    text='已实现，未训练；无收益'
    if report:text='最终恢复SW='+str(report['target_score'])
    elif 'best' in state:text='训练快照SW='+str(state['best'])+'，epoch='+str(state['epoch'])+'；非最终收益'
    elif state:text='状态='+state.get('status','unknown')+'；无最终成绩'
    shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,
                 scope='原第五实际固定三通道；含源4依赖训练阶段',source=str((study/'assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v110 SW分层动作奖励代理','同845913参数随机local SW，仅将训练效用softmax改为连续动作内与默认/连续最大效用之间两级概率，temp.05/alpha1；原数值奖励核/验证argmax不改。复用已完成平坦控制；own SW/原206/Q50/80-18/SAM.1保持，全4阶段含源。',text,
         stage='optimization',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
         evidence=['results/deep_learning/compact_hierarchical_utility_v110_20261001/README.md'],**shown)
    edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','相同网络/初始化/预算，历史平坦奖励对照复用；只改训练损失代理')
    if assembled and assembled['predefined_hierarchical_minus_flat_SW']<=0:
        end('dead:v110_hierarchical_utility','v110 当前分层奖励代理无增益','optimization',[ident],
            '实际SW hierarchical-flat='+str(assembled['predefined_hierarchical_minus_flat_SW']),
            '关闭本分层/temp/alpha设置，不将动作数分析当实测收益，不泛化所有奖励代理。')
