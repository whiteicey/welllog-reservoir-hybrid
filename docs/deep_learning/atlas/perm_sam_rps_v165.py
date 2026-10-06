import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_sam_rps_v165_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('adamw','sam'):
        ident='pure_dl_perm_sam_rps_v165_fold5/'+mode;out=root/'outputs'/ident
        st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/(mode+'_assembled.json'))
        shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
        if a:
            row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly, five source-counted phases',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v165 PERM optimizer '+mode,'Matched82720params/commoninitialization/80-14, AdamW vs SAM0.05 on positivePERM128/RPS1; original SOURCE frozen, no trained154 head loaded. SAM double passes: report equal updates and unequal compute.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_sam_rps_v165_20261001/README.md'],**shown)
        edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Historical positive PERM RPS80/18')
        edge('pure_dl_paired_local_context_router80_sequence/fold_5',ident,'warm_start','Same frozenparent')
    edge('pure_dl_perm_sam_rps_v165_fold5/adamw','pure_dl_perm_sam_rps_v165_fold5/sam','comparison','Matched capacity/init/update budget, SAM two passes vs one; original decoder unchanged')
