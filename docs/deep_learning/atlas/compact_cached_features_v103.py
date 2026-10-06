import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_cached_features_v103_20261001'
    if not study.exists():return
    ident='pure_dl_compact_cached_features_v103_fold5'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    caches={mode:read(study/(mode+'_cache.json')) for mode in ('random','pretrained')}
    ablation=read(study/'ablation_report.json')
    text='缓存状态：'+str({mode:dict(complete=c.get('complete',False),seconds=c.get('seconds')) for mode,c in caches.items()})+'；缓存上下文改变，非v102等价加速。'
    if ablation:text+='最终pretrained-random SW='+str(ablation['predefined_source_feature_delta'])
    node(ident,'v103 固定原源窗口隐藏缓存',
         '冻结源256点窗口/128核心，每点隐藏特征一次；新SW读取206+256特征，整井bank仍原206。新源局部锚点不同于v102移动SW锚点，因此是新表征方案而非等价缓存。五阶段含源/缓存成本，原数据/Q50/80-18/own SW损失不变。',
         text,
         stage='feature',status='recorded' if ablation else 'running_snapshot' if any(caches.values()) else 'prepared',resultText=text,
         evidence=['results/deep_learning/compact_cached_features_v103_20261001/README.md'])
    edge('pure_dl_compact_source_features_v102_fold5/pretrained_block256',ident,'evolution','准备降低冻结源计算重复；源上下文/坐标定义改变，不加载v102新SW训练权重')
    for mode in ('random','pretrained'):
        out=root/'outputs'/ident/mode;state=read(out/'runtime_status.json');report=read(out/'report.json')
        assembled=read(study/(mode+'_assembled.json'))
        result='尚未训练，无成绩'
        if report:result='最终SW='+str(report['target_score'])+'；实际最佳恢复完成'
        elif 'best' in state:result='训练快照SW='+str(state['best'])+'；epoch='+str(state['epoch'])+'，非最终收益'
        elif state:result='状态='+state.get('status','未知')+'，尚无成绩'
        shown=dict(resultText=result)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='原第五固定三通道，4依赖训练阶段，第5密度尚未执行',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident+'/'+mode,'v103 缓存源SW '+mode,'同容量864057私有参数/初始化，冻结随机或预训练编码器，输入源256锚点和128核心相同。own SW/SAM.1/utility1/80-18，显存实测后最多双并发。',result,
             stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             highlight=bool(mode=='pretrained' and ablation.get('predefined_source_feature_delta',0)>0),evidence=['results/deep_learning/compact_cached_features_v103_20261001/README.md'],**shown)
        edge(ident,ident+'/'+mode,'evolution','实际缓存输入机制，独立SW最佳checkpoint与早停')
    edge(ident+'/random',ident+'/pretrained','comparison','预定义同容量、同固定源窗口上下文，仅冻结编码器参数来源不同')
    if ablation and ablation['predefined_source_feature_delta']<=0:
        end('dead:v103_cached_source_hidden','v103 原源窗口缓存未取得来源增益','feature',[ident+'/pretrained'],
            '最终SW pretrained-random='+str(ablation['predefined_source_feature_delta']),
            '停止这两个固定上下文/归一化表征的源种子和LR扫描，不泛化所有缓存或所有源特征。')
