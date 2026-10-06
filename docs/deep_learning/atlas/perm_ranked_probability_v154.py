import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_ranked_probability_v154_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_ranked_probability_v154_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five phases including source',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v154 PERM effective128 plus jointRPS','Transfer evidence-positive POR supervision, ownPERM Gaussian/log10/default value; same82720parameters/initialization as v136, onlyNLL+jointRPS1 loss. No trainedv136 head loaded. FixedPOR151/SW104 assembly.',str(st),stage='optimization',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_ranked_probability_v154_20261001/README.md'],**shown)
    edge('pure_dl_perm_joint_density_v136_fold5/perm_conditional',ident,'comparison','Samecapacity/initialization, add jointCDF supervision, no head weight warmstart')
    edge('pure_dl_por_ranked_probability_v151_fold5/joint_rps',ident,'comparison','Evidence-positive supervision mechanism transfer; not target model or weight reuse')
