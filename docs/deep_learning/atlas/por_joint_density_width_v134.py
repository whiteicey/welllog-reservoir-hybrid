import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_joint_density_width_v134_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_joint_density_width_v134_fold5/wide128';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'wide128_assembled.json')
    text='Implemented, untrained, no score' if not state else str(state)
    shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 recipe phases including source',source=str((study/'wide128_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v134 POR conditional density width128','Frozen positivev113, new319->128->32 conditional densityhead,82720params vs width64 control33184; fullinitialfrozenv113functionsame, internalrandomfeature matrices/capacity differ; POR historical+0.317154 verified. Zero output same initial predictor, own mixedStudent3/Cauchy NLL/CDF; new delta bound one fixed referencewidth; five dependency phases inclsource, fixedPERM/SW. POR output route alongside independent backbone route.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_joint_density_width_v134_20261001/README.md'],**shown)
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen full positivev113 predictor, initial newheadzero')
    edge('pure_dl_compact_por_family_gate_v125_fold5/fixed',ident,'comparison','Best compact POR control only, not loaded')
    edge('pure_dl_por_joint_density_head_v131_fold5/location_weight',ident,'comparison','Positivewidth64 control, not warmstart; extra49536newheadparams')
