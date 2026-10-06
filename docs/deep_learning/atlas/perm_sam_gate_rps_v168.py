import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_sam_gate_rps_v168_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_sam_gate_rps_v168_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v168 PERM SAM gate/readout','Carry measuredSAM0.05 benefit into originalPERM gate rows3e-5 plus effective128head3e-4/RPS1. Same SOURCE initialfunction/80-14, active+258 vs pureSAM, no trained165 head loaded, fivephases.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_sam_gate_rps_v168_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function, active update scope and patience differ; GaussianRPS1 positivecontrol')

    edge('pure_dl_perm_sam_rps_v165_fold5/sam',ident,'comparison','Measured local SAM benefit; original gate update scope only, no checkpoint warmstart')
