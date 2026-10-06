import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_ranked_probability_v153_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_ranked_probability_v153_fold5/joint_rps';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'joint_rps_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual original-fifth assembly; five source-counted phases',source=str((study/'joint_rps_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v153 increase useful jointRPS coefficient1->2','Same effective256/230944parameters, seed and frozenparent; only NLL+2jointRPS instead of1. No trainedv151/v140 head loaded; five phases. Expand evidence-positive supervision without negative module stacking.',str(st),stage='optimization',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_ranked_probability_v153_20261001/README.md'],**shown)
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozen parent')
    edge('pure_dl_por_ranked_probability_v151_fold5/joint_rps',ident,'comparison','Positive control77.926381, matched capacity/initialization, coefficient only')
    if a and (study/'checkpoint_cleanup.json').exists():
        end('dead:v153_RPS2','v153 coefficient2 below positive coefficient1','optimization',[ident],'Restored POR77.901655, coefficient2 minus1-0.024726, below NLL-only control by0.010448; exact replay/assembly and negative cleanup complete.','Keep positive coefficient1, close this coefficient2 configuration; no automatic coefficient grid.')
