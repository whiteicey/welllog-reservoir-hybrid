import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_joint_density_width_v140_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_joint_density_width_v140_fold5/wide256';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'wide256_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target fifth assembly; five phases including source',source=str((study/'wide256_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v140 POR conditional density width256','One bounded width128 to256 capacity comparison after measured width64 to128 POR gain. Same frozen v113 initial predictor and bounds; no trained v134 head loaded. Replace PORhead phase, fixed PERM136/SW104, five total phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_joint_density_width_v140_20261001/README.md'],**shown)
    edge('pure_dl_por_joint_density_width_v134_fold5/wide128',ident,'comparison','Positive width128 control only, not warm-start; same frozen v113 predictor')
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen positive mixed-family parent, new head zero output')
