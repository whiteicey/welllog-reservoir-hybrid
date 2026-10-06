import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_local_density_v106_20261001'
    if not study.exists():return
    ident='pure_dl_compact_local_density_v106_fold5/density';out=root/'outputs'/ident
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'assembled_report.json')
    text='未启动，无成绩'
    if report:text='最终SW='+str(report['target_score'])+'；对父local='+str(report['target_score']-74.12175282815758)
    elif 'best' in state:text='训练快照SW='+str(state['best'])+'，epoch='+str(state['epoch'])+'，非最终增益'
    elif state:text='状态='+state.get('status','未知')+'，尚无成绩'
    shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='原第五固定三通道，执行5阶段；选定有效依赖'+str(assembled.get('selected_effective_training_dependencies',5)),source=str((study/'assembled_report.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v106 local SW私有密度（第5阶段）','仅从v104 local epoch37选定模型复制atom/density；atom冻结eval，density845913参数own SW NLL/SAM.1/1e-4/DropKey.4。冻结atom同批两SAM pass复用，无效用图；原数据/Q50/解码/80-18不变，5阶段含源。',text,
         stage='optimization',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
         highlight=bool(report and report['target_score']>74.12175282815758),evidence=['results/deep_learning/compact_local_density_v106_20261001/README.md'],**shown)
    edge('pure_dl_compact_frequency_v104_fold5/local',ident,'evolution','实际加载选定local SW，冻结atom，私有密度第五阶段')
    edge('pure_dl_compact_v100_fold5/sw_density',ident,'comparison','v100私有密度+0.062135机制迁移，未加载其训练权重')
