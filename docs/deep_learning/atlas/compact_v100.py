"""Curated compact-stage lineage and actual training snapshots."""
import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_five_stage_v100_20260930'
    if not study.exists():return
    ident='pure_dl_compact_v100_fold5'
    node(ident,'v100 五个总训练阶段','来源预训练计入1阶段；联合POR头、联合PERM头、随机完整SW联合SAM、固定本批atom的密度SAM共5阶段。只复用原同折来源，不加载旧18阶段后续权重。',
         '尚未完成本批三通道组合；初始/快照/最终分开，80.504197是历史对照，严格>80.8才考虑其他折。',stage='architecture',status='running_snapshot',
         resultText='尚无完整紧凑方案成绩',evidence=['results/deep_learning/compact_five_stage_v100_20260930/README.md'])
    edge('pure_dl_fullcv_v97',ident,'comparison','原OOF81.891019及18后续阶段历史对照；本批只有第五折，不构成完整OOF比较')
    edge('pure_dl_paired_local_context_router80_sequence',ident,'evolution','实际仅复用原第五来源epoch52；计入总阶段数')
    for stage in ('por','perm','sw_joint','sw_density'):
        out=root/'outputs/pure_dl_compact_v100_fold5'/stage
        read=lambda name:json.loads((out/name).read_text(encoding='utf-8')) if (out/name).exists() else {}
        report=read('report.json');state=read('runtime_status.json');history=read('validation_history.json')
        status='recorded' if report else 'failed' if state.get('status')=='failed' else 'running_snapshot' if state.get('status')=='running' else 'prepared'
        result='尚无训练成绩' if not history and not report else '通道选定分='+str(report.get('target_score',state.get('best')))+'；不当作三通道TOTAL'
        node(ident+'/'+stage,'v100 '+stage,'同一预定紧凑配方的独立通道阶段；每目标自身损失与最佳checkpoint。',result,
             stage='architecture' if stage!='sw_density' else 'optimization',status=status,resultText=result,
             evidence=['outputs/pure_dl_compact_v100_fold5/'+stage+'/config.json'])
        edge(ident,ident+'/'+stage,'evolution','预定阶段，非自动搜索')
    final=study/'assembled_report.json'
    if final.exists():
        report=json.loads(final.read_text(encoding='utf-8'));scores=report['scores']
        node(ident,'v100 五个总训练阶段','来源预训练+联合POR/PERM+SW直接联合+私有密度；旧后续权重未载入。',
             '完整本批第五三通道组合，非完整OOF；对v96差='+str(scores['TOTAL']-80.50419742447689),stage='architecture',status='recorded',
             primary=dict(value=scores['TOTAL'],scores=scores,kind='fifth',fold=5,scope='原第五实际紧凑组合',source=str(final.relative_to(root))),
             results=[dict(value=scores['TOTAL'],scores=scores,kind='fifth',fold=5,scope='原第五实际紧凑组合',source=str(final.relative_to(root)))])
