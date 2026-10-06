import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_mean_scope_v101_20261001'
    if not study.exists():return
    ident='pure_dl_compact_mean_scope_v101_fold5/por';out=root/'outputs/pure_dl_compact_mean_scope_v101_fold5/por'
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    report=read(out/'report.json');ablation=read(study/'ablation_report.json');assembled=read(study/'assembled_report.json')
    state=read(out/'runtime_status.json');text='尚无训练成绩'
    if report:text='最终POR='+str(report['target_score'])+'；对v100范围消融='+str(ablation.get('predefined_update_scope_delta','待记录'))+'，非完整TOTAL'
    elif state.get('best'):text='训练快照POR='+str(state['best'])+'，非最终消融'
    shown=dict(resultText=text)
    if assembled:
        score=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='原第五实际紧凑组合，共享阶段计入5',source=str((study/'assembled_report.json').relative_to(root)))
        shown=dict(primary=score,results=[score])
    node(ident,'v101 紧凑POR固定源密度形状','同v100头/路由/初始功能，只冻结原混合权重与尺度；位置行仍更新。更新范围消融，活动参数不相等。共享PERM/SW完整计数5阶段，无旧18阶段权重。',text,
         stage='optimization',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'prepared',
         highlight=bool(ablation.get('predefined_update_scope_delta',0)>0),evidence=['results/deep_learning/compact_mean_scope_v101_20261001/README.md'],**shown)
    edge('pure_dl_compact_v100_fold5/por',ident,'comparison','最终更新范围对照，同初始官方分，冻结原混合权重/尺度')
    edge('pure_dl_compact_v100_fold5',ident,'evolution','相同初始化与模块；只有源位置参数训练，其他通道阶段共享并计入完整配方')
    if ablation and ablation['predefined_update_scope_delta']<=0:
        end('dead:v101_mean_scope','v101 原位置范围限制未保留收益','optimization',[ident],
            '原第五最终POR对v100='+str(ablation['predefined_update_scope_delta']),
            '仅关闭本源/同模块/预算的范围限制，不泛化全部位置头或冻结策略。')
