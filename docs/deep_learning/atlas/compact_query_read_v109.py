import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_query_read_v109_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('geometry','content'):
        ident='pure_dl_compact_query_read_v109_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='已实现，未训练；无成绩'
        if report:text='最终恢复SW='+str(report['target_score'])
        elif 'best' in state:text='训练快照SW='+str(state['best'])+'，epoch='+str(state['epoch'])+'；非最终收益'
        elif state:text='状态='+state.get('status','unknown')+'；尚无最终成绩'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,
                     scope='原第五折实际固定三通道；4依赖训练阶段含源',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v109 SW逐点全局查询 / '+mode,
             '随机local65 SW，新增12672参数四头32维查询读取64个全局token，输出零初始化、残差±0.25。位置与内容query同容量；原插值/token汇聚/窗口/评分不变，原206/Q50/own SW/80-18/SAM.1保持，完整配方4阶段含源。',text,
             stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             evidence=['results/deep_learning/compact_query_read_v109_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'evolution','保留local结构，从随机SW重建并增加逐点读取全局记忆的残差')
    edge('pure_dl_compact_query_read_v109_fold5/geometry','pure_dl_compact_query_read_v109_fold5/content','comparison','同858585参数/初始化/预算，仅query读取位置或本点内容')
    ablation=read(study/'ablation_report.json')
    if ablation and ablation['predefined_query_input_delta']<=0:
        end('dead:v109_content_query','v109 本点内容查询无匹配增益','architecture',
            ['pure_dl_compact_query_read_v109_fold5/content'],
            '实际恢复content-geometry SW='+str(ablation['predefined_query_input_delta']),
            '关闭当前query来源与固定残差设置，不泛化所有全局记忆读取。')
