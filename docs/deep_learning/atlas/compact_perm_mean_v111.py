import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_perm_mean_v111_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('gate_route','mean_coadapt'):
        ident='pure_dl_compact_perm_mean_v111_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='已实现，未训练；无成绩'
        if report:text='最终恢复PERM='+str(report['target_score'])
        elif 'best' in state:text='训练快照PERM='+str(state['best'])+'，epoch='+str(state['epoch'])+'；非最终收益'
        elif state:text='状态='+state.get('status','unknown')+'；无最终成绩'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,
                     scope='原第五实际三通道；执行5阶段，选定有效依赖'+str(assembled['selected_effective_dependency_stages'])+'，含源',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v111 精简PERM更新范围 / '+mode,
             '同v100 PERM选定函数热启动，新lr3e-5/80-18，仅自身PERM损失；门控/侧头/路由共更新，候选额外训练原8个均值行2064变量。骨干/权重/尺度固定，非同训练容量；含源和父PERM阶段，全配方5训练阶段。',text,
             stage='optimization',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             evidence=['results/deep_learning/compact_perm_mean_v111_20261001/README.md'],**shown)
        edge('pure_dl_compact_v100_fold5/perm',ident,'evolution','已选紧凑PERM函数继续优化，初训练阶段完整计入')
    edge('pure_dl_compact_perm_mean_v111_fold5/gate_route','pure_dl_compact_perm_mean_v111_fold5/mean_coadapt','comparison','相同起点/其他更新与预算，是否额外开放均值；非同训练容量')
    ablation=read(study/'ablation_report.json')
    if ablation and ablation['predefined_mean_minus_gate_route_PERM']<=0:
        end('dead:v111_perm_mean_scope','v111 PERM新增均值更新无匹配收益','optimization',
            ['pure_dl_compact_perm_mean_v111_fold5/mean_coadapt'],
            '最终PERM mean-control='+str(ablation['predefined_mean_minus_gate_route_PERM']),
            '关闭本原均值更新范围，不自动扫学习率或解冻密度尺度/骨干。')
