import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_gate_coadapt_v161_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_gate_coadapt_v161_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v161 PERM gate/readout coadaptation','Original PERM gate rows lowLR3e-5 with positive128conditional readout/RPS1. Total architecture unchanged, +258 activeparams; original body/router/scale frozen. Same initial function, user80/14 vs historical80/18.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_gate_coadapt_v161_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function, active update scope and patience differ; GaussianRPS1 positivecontrol')

    if a and (study/'checkpoint_deletion_manifest.json').exists():
        end('dead:v161_gate','v161 PERM gate coadapt no gain','architecture',[ident],'PERM89.344760174753; deltaRPS -0.060804682166; best0/stop14, both gradients positive, replay exact, weights cleaned.','Close this gate update configuration, test depth-local representation; not all gate mechanisms rejected.')
