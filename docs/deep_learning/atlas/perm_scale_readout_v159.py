import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_scale_readout_v159_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_scale_readout_v159_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v159 PERM bounded scale conditional readout','Gaussian location/weight/raw-scale head; bounded raw-scale delta0.5, +2064 parameters, same initial function; own properNLL/RPS1, original gate/body frozen.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_scale_readout_v159_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function; capacity differs2064; GaussianRPS1 positive control')

    if a and (study/'checkpoint_deletion_manifest.json').exists():
        end('dead:perm_scale_readout_v159','perm_scale_readout_v159 no gain','architecture',[ident],'Verified target delta vs positive RPS control -0.060804682166; actual assembly, weights cleaned.','Close this bounded correction configuration only; retain positive RPS readout.')
