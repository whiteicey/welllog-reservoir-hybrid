import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_ranked_probability_v151_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_ranked_probability_v151_fold5/joint_rps';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'joint_rps_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual original-fifth assembly; five phases including source',source=str((study/'joint_rps_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v151 effective256 readout plus joint RPS','Same230944parameters and initialization as effectivev140; NLL plus joint atom/continuous rankedCDF error on original255finite edges, own default/numeric labels. No added head module/trainedv140 weights/actionutility.',str(st),stage='optimization',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_ranked_probability_v151_20261001/README.md'],**shown)
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozen parent')
    edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Same architecture/capacity/initialization, add joint RPS supervision')
