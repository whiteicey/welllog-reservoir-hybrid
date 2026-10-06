import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_output_scope_v126_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('gate','weight'):
        ident='pure_dl_compact_por_output_scope_v126_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 phases including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v126 POR original output factor scope / '+mode,'Warm v113 forward with same predictive function. Shared location/newhead/routes refinement plus original default gate258 rows or component weights2064 rows;3e-5 groups; fixed body/reference widths/hard t3-Cauchy assignment. Own properNLL/CDF, update scope not matched capacity; fixed other targets,5 phases, POR focus until historical gain>0.25.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_por_output_scope_v126_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Exact retained positive v113 checkpoint and function')
        edge('pure_dl_compact_por_scale_v114_fold5/location_only',ident,'comparison','Historical compact POR refinement control; not loaded')



    results=[read(study/(m+'_assembled.json')) for m in ('gate','weight')]
    if all(results) and all(v['gain_vs_v114_POR']<=0 for v in results):
        end('dead:v126_output_scope','v126 original gate/weight updates below location-only control','optimization',['pure_dl_compact_por_output_scope_v126_fold5/gate','pure_dl_compact_por_output_scope_v126_fold5/weight'],'Both best0 POR77.54045433 vs location-only77.54806949; all144665 replay exact.','Close these original linear output-factor scopes, not all gating/mixture mechanisms.')
