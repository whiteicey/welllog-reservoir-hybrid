import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_expert_interaction_v152_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('additive','product'):
        ident='pure_dl_por_expert_interaction_v152_fold5/'+mode;out=root/'outputs'/ident
        st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/(mode+'_assembled.json'));shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
        if a:
            row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/(mode+'_assembled.json')).relative_to(root)));shown=dict(primary=row,results=[row])
        node(ident,'v152 effective256 plus '+mode+' expert features','Preserve common effective256 density readout initialization; two128->32 expert projections, matched addition/product, zero32->256 branch.247712activeparameters, samecapacity twoarms; no trainedv140 head loaded.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_expert_interaction_v152_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen parent')
        edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Augment proven structure, unequal capacity absolute comparison')
    edge('pure_dl_por_expert_interaction_v152_fold5/additive','pure_dl_por_expert_interaction_v152_fold5/product','comparison','Matched capacity/initialization, addition vs multiplication only')
    if (study/'checkpoint_cleanup.json').exists():
        end('dead:v152_expert_interaction','v152 expert addition/product below useful direct readout','architecture',['pure_dl_por_expert_interaction_v152_fold5/additive','pure_dl_por_expert_interaction_v152_fold5/product'],'Restored POR additive77.857373/product77.848495; product minus additive-0.008878, both below direct25677.912103 and positiveRPS77.926381. Exact replay/assembly, negativeweights cleaned.','Close rank32 additive/product feature configurations; do not stack into positive RPS recipe or reject all expert interaction mechanisms.')
