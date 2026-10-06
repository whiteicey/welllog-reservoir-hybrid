import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_cauchy_v112_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_compact_por_cauchy_v112_fold5/cauchy';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'assembled.json')
    text='已实现，未训练；无收益'
    if report:text='最终恢复POR='+str(report['target_score'])
    elif 'best' in state:text='训练快照POR='+str(state['best'])+'，epoch='+str(state['epoch'])+'；非最终收益'
    elif state:text='状态='+state.get('status','unknown')+'；无最终成绩'
    shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,
                 scope='原第五实际固定三通道；含源4依赖训练阶段',source=str((study/'assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v112 POR Cauchy位置读出','复用v101同容量/原始raw初始化位置读出，固定分量IQR匹配，Cauchy无有限均值/方差；自身POR似然和预测CDF同步更改，初始函数不同，不计作训练收益。原输入/Q50/80-18/动作评分人口不变，含源4阶段。',text,
         stage='optimization',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
         evidence=['results/deep_learning/compact_por_cauchy_v112_20261001/README.md'],**shown)
    edge('pure_dl_compact_mean_scope_v101_fold5/por',ident,'comparison','相同网络/raw初始化/预算，历史Student-t3控制复用；密度族同时影响训练和预测，初始函数不同')
    if assembled and assembled['predefined_cauchy_minus_t3_POR']<=0:
        end('dead:v112_cauchy_por','v112 当前Cauchy POR密度族无增益','optimization',[ident],
            '实际POR Cauchy-t3='+str(assembled['predefined_cauchy_minus_t3_POR']),
            '关闭本IQR匹配密度族设置，不无界扫尺度或族，不泛化所有重尾模型。')
