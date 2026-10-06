import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_ranked_probability_v156_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_ranked_probability_v156_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v156 PERM positive jointRPS coefficient1->2','Same82720parameters/initialization/Gaussian/default/log10; only coefficient2 changes. Evidence-positive PERM1 control retained; no trainedhead warmstart, no negative module stacking.',str(st),stage='optimization',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_ranked_probability_v156_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Samecapacity/initialization, coefficient1->2 only')
