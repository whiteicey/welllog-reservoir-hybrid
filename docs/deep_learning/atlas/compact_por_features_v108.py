import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_features_v108_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ablation=read(study/'ablation_report.json')
    for mode in ('curve_control','por_posterior'):
        ident='pure_dl_compact_por_features_v108_fold5/'+mode
        out=root/'outputs'/ident;state=read(out/'runtime_status.json');report=read(out/'report.json')
        text='未训练；特征缓存已完成，无收益证据'
        if report:text='最终恢复SW='+str(report['target_score'])
        elif 'best' in state:text='训练快照SW='+str(state['best'])+'，epoch='+str(state['epoch'])+'，非最终收益'
        elif state:text='状态='+state.get('status','unknown')+'；尚无最终成绩'
        assembled=read(study/(mode+'_assembled.json'));shown=dict(resultText=text)
        if assembled:
            score=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,
                       scope='原第五折实际固定三通道；4训练依赖阶段含来源和冻结POR老师',
                       source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=score,results=[score])
        node(ident,'v108 POR语义特征 / '+mode,
             '随机local65私有SW，原206+4项输入；控制为GR/DEN/CNL/RT，候选为冻结精简POR后验统计。同容量和其余初始化。老师计入第2阶段，全配方4阶段；训练侧统计拟合，无验证标签或输出投票。',text,
             stage='features',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             evidence=['results/deep_learning/compact_por_features_v108_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'evolution','保留已有效local结构，随机重建并添加4维输入')
        if mode=='por_posterior':edge('pure_dl_compact_mean_scope_v101_fold5/por',ident,'dependency','冻结POR作为输入特征老师，计入完整训练阶段')
    edge('pure_dl_compact_por_features_v108_fold5/curve_control','pure_dl_compact_por_features_v108_fold5/por_posterior','comparison','4维冗余曲线对照与4维POR语义统计；同容量预算')
    if ablation and ablation['predefined_semantic_feature_delta']<=0:
        end('dead:v108_por_semantic_inputs','v108 当前4项POR后验输入无匹配增益','features',
            ['pure_dl_compact_por_features_v108_fold5/por_posterior'],
            '实际恢复posterior-control SW='+str(ablation['predefined_semantic_feature_delta']),
            '关闭当前4统计和老师设置，不泛化全部目标间语义输入。')
