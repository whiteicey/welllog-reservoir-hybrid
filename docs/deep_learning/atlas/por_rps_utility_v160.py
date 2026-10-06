import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_rps_utility_v160_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_rps_utility_v160_fold5/joint_rps';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'joint_rps_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'joint_rps_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v160 POR RPS plus tolerance utility','Matched230944 parameters; own POR tolerance utility0.1 plus properNLL/RPS1, frozenv113, unchanged decoder; userpatience14 via boundary request.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_rps_utility_v160_20261001/README.md'],**shown)
    edge('pure_dl_por_ranked_probability_v151_fold5/joint_rps',ident,'comparison','Augment useful mechanism, no trainedhead warmstart')
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozen parent')

    if a and (study/'checkpoint_deletion_manifest.json').exists():
        end('dead:v160_utility','v160 utility0.1 below RPS1','architecture',[ident],'POR77.873725032741; deltaRPS1 -0.052655727983, best30/stop45. bad14 boundary request at44, latency one epoch; actual assembly verified, weights cleaned.','Close this utility coefficient configuration, not all utility losses.')
