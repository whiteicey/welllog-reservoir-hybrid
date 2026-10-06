import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_smooth_cdf_readout_v150_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_smooth_cdf_readout_v150_fold5/bernstein8';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'bernstein8_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'bernstein8_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v150 smooth conditional CDF readout','Eight positive Beta basis weights define monotone W(F); proper density p(y)*Wprime(F(y)). Smooth low-dimensional shape, unchanged actions/density grid. Frozenv113 and zero outputs;233000active parameters, v140capacity unequal.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_smooth_cdf_readout_v150_20261001/README.md'],**shown)
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozen parent, no other trained head loaded')
    edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Absolute POR77.912103 control')
    edge('pure_dl_por_density_tilt_v148_fold5/tilt',ident,'comparison','Negative free interval tilt prompted low-dimensional smooth CDF hypothesis, no weights loaded')
    if a and (study/'checkpoint_cleanup.json').exists():
        end('dead:v150_smoothCDF','v150 smoothCDF below effective readout','architecture',[ident],'Restored POR77.769405, delta256-0.142698, best17/stop35; exact replay/actual assembly, negative checkpoints cleaned.','Close this eight-basis/bound0.5 configuration, return to augmenting proven256 readout; not all CDF parameterizations rejected.')
