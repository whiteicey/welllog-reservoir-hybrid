import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_bounded_gate_rps_v157_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_bounded_gate_rps_v157_fold5/joint_rps';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'joint_rps_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'joint_rps_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v157 positive POR RPS plus small gate correction','Effective256/location/logweight/RPS1 retained; add257parameters for defaultlogodds correction bounded0.125. Frozenv113, zero initial function; no trainedv151 head loaded. Unequalcapacity by257 vs positivecontrol.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_bounded_gate_rps_v157_20261001/README.md'],**shown)
    edge('pure_dl_por_ranked_probability_v151_fold5/joint_rps',ident,'comparison','Augment useful mechanism, no trainedhead warmstart')
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozen parent')

    if a and (study/'checkpoint_deletion_manifest.json').exists():
        end('dead:por_bounded_gate_rps_v157','por_bounded_gate_rps_v157 no gain','architecture',[ident],'Verified target delta vs positive RPS control -0.163101953158; actual assembly, weights cleaned.','Close this bounded correction configuration only; retain positive RPS readout.')
