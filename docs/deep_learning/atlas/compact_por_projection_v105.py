import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_projection_v105_20261001'
    if not study.exists():return
    ident='pure_dl_compact_por_projection_v105_fold5/por';out=root/'outputs'/ident
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    state=read(out/'runtime_status.json');report=read(out/'report.json');ablation=read(study/'ablation_report.json')
    text='尚未训练，无成绩'
    if report:text='最终POR='+str(report['target_score'])+'；对v101更新范围差='+str(ablation.get('projection_scope_delta','待记录'))+'，非完整TOTAL'
    elif 'best' in state:text='训练快照POR='+str(state['best'])+'，epoch='+str(state['epoch'])+'，非最终增益'
    elif state:text='状态='+state.get('status','未知')+'，尚无成绩'
    node(ident,'v105 POR末端特征投影解冻','从原源重建v101 mean_scope，不加载已训练头；仅两成员combine[0]各64→128投影新增16640活动参数，3e-6；源保持eval，own POR，原数据/Q50/80-18。更新范围消融，非同活动容量；改变h也改变原固定head的有效分布。',text,
         stage='optimization',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
         highlight=bool(ablation.get('projection_scope_delta',0)>0),resultText=text,evidence=['results/deep_learning/compact_por_projection_v105_20261001/README.md'])
    edge('pure_dl_compact_mean_scope_v101_fold5/por',ident,'comparison','原源同初始函数/新head初始化，末端投影固定vs训练；不热启动已训练v101')
    if ablation and ablation['projection_scope_delta']<=0:
        end('dead:v105_final_projection','v105 当前POR末端投影未保留收益','optimization',[ident],
            '最终POR对v101='+str(ablation['projection_scope_delta']),
            '关闭本有限投影范围/学习率3e-6扫描，不扩大冻结层数范围。')
