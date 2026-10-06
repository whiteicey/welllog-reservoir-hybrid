import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_mixed_family_v113_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('t3_cauchy','cauchy_t3'):
        ident='pure_dl_compact_por_mixed_family_v113_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='已实现，未训练；无成绩'
        if report:text='最终恢复POR='+str(report['target_score'])
        elif 'best' in state:text='训练快照POR='+str(state['best'])+'，epoch='+str(state['epoch'])+'；非最终收益'
        elif state:text='状态='+state.get('status','unknown')+'；无最终成绩'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,
                     scope='原第五实际三通道；4依赖训练阶段含源',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v113 POR内部专家族 / '+mode,
             '同54483活跃参数/source初始raw/预算，expert0/1各8分量固定Student-t3/Cauchy或反向；似然/CDF同步、IQR匹配，初始函数不同，含Cauchy无有限均值/方差。自身POR/原数据/Q50/80-18保持，非外部投票或逐行选族，4阶段含源。',text,
             stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             evidence=['results/deep_learning/compact_por_mixed_family_v113_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_cauchy_v112_fold5/cauchy',ident,'evolution','沿同质Cauchy局部收益改变内部专家族；从原source重建，不加载已训练v112头')
        edge('pure_dl_compact_mean_scope_v101_fold5/por',ident,'comparison','历史同质t3控制，容量/source初始化/预算相同，初始函数不同')
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy','pure_dl_compact_por_mixed_family_v113_fold5/cauchy_t3','comparison','相同raw初始参数和预算，仅两内部专家的固定族赋值反向')
    results=[read(study/(m+'_assembled.json')) for m in ('t3_cauchy','cauchy_t3')]
    if all(results) and all(x['gain_vs_homogeneous_Cauchy_POR']<=0 for x in results):
        end('dead:v113_mixed_expert_families','v113 混合专家族未超过同质Cauchy','architecture',
            ['pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy','pure_dl_compact_por_mixed_family_v113_fold5/cauchy_t3'],
            '两项实际POR对同质Cauchy='+str([x['gain_vs_homogeneous_Cauchy_POR'] for x in results]),
            '关闭本固定混合族赋值；若对t3有局部正差单列，不将其冒称超过同质Cauchy或完整OOF收益。')
