import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_source_features_v102_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    ablation=read(study/'ablation_report.json')
    for mode in ('random','random_block256','pretrained_block256'):
        ident='pure_dl_compact_source_features_v102_fold5/'+mode
        out=root/'outputs'/ident
        report=read(out/'report.json');state=read(out/'runtime_status.json')
        text='尚未启动，无成绩'
        if report:text='最终SW='+str(report['target_score'])+'，非完整TOTAL；匹配对照待两项结束'
        elif 'best' in state:text='训练快照SW='+str(state['best'])+'，非最终收益'
        elif state:text='状态='+state.get('status','未知')+'，尚无成绩'+('；1024源输入超出最大token，训练前失败' if mode=='random' else '')
        assembled=read(study/(mode+'_assembled.json'));shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='原第五固定三通道，4依赖阶段，第5尚未执行',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        if ablation:text+='；pretrained-random SW='+str(ablation['predefined_source_feature_delta'])
        node(ident,'v102 SW冻结源隐藏特征 '+mode,
             '原206特征追加两专家256隐藏维，逐点固定LayerNorm；同结构冻结随机/训练源编码器，仅参数来源不同，训练侧buffers相同。新SW骨干随机，own SW utility1/SAM.1；五阶段含源，单SW串行，无旧预测。',text,
             stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             highlight=bool(mode=='pretrained_block256' and ablation.get('predefined_source_feature_delta',0)>0),
             evidence=['results/deep_learning/compact_source_features_v102_20261001/README.md'],**shown)
        edge('pure_dl_compact_v100_fold5/sw_joint',ident,'evolution','同私有SW结构和训练机制，新增冻结隐藏输入；不加载旧SW权重')
    edge('pure_dl_compact_source_features_v102_fold5/random_block256','pure_dl_compact_source_features_v102_fold5/pretrained_block256','comparison','同容量/私有初始化/源buffer，冻结编码器参数随机vs原训练来源')
    if ablation and ablation['predefined_source_feature_delta']<=0:
        end('dead:v102_source_hidden','v102 当前源隐藏特征无匹配增益','architecture',['pure_dl_compact_source_features_v102_fold5/pretrained_block256'],
            '原第五最终pretrained-random SW='+str(ablation['predefined_source_feature_delta']),
            '关闭本256源块/1024私有SW/固定归一化设置的种子和LR扫描，不泛化所有源表征。')
