import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_attention_v107_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ablation=read(study/'ablation_report.json')
    for mode in ('dot','cosine'):
        ident='pure_dl_compact_attention_v107_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='尚未启动，无成绩'
        if report:text='最终SW='+str(report['target_score'])+'；最佳恢复完成'
        elif 'best' in state:text='训练快照SW='+str(state['best'])+'，epoch='+str(state['epoch'])+'，非最终收益'
        elif state:text='状态='+state.get('status','未知')+'，尚无成绩'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='原第五实际固定三通道，4依赖阶段，来源计入',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v107 SW注意力 '+mode,'从随机原206 local65 SW重建，不加载旧SW。dot与QK cosine均16可学习head温度参数，总845929；初温度1/sqrt24或10，差值属于归一化+初温度参数化。原数据/Q50/own SW/80-18/utility1/SAM.1/DropKey.4不变，串行，最终配方含源≤5阶段。',text,
             stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             evidence=['results/deep_learning/compact_attention_v107_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'evolution','仅保留已获整体改善的local结构，重新随机联合训练并改变注意力参数化')
    edge('pure_dl_compact_attention_v107_fold5/dot','pure_dl_compact_attention_v107_fold5/cosine','comparison','同845929容量、其他参数初始化与预算，QK归一化及对应温度初值改变')
    if ablation and ablation['predefined_attention_delta']<=0:
        end('dead:v107_cosine_attention','v107 当前余弦注意力无匹配收益','architecture',['pure_dl_compact_attention_v107_fold5/cosine'],
            '最终cosine-dot SW='+str(ablation['predefined_attention_delta']),
            '关闭本归一化/初始化/温度设置的扫描，不泛化全部注意力方法。')
