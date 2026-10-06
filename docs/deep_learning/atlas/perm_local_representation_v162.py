import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_local_representation_v162_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_local_representation_v162_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v162 PERM depth-local representation adaptation','Two zero-init bounded128-32-depth7-128 feature adapters; original parameters frozen, new representation and positive128/RPS readout train. +17216 parameters, same initial function; user80/14 vs historical80/18.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_local_representation_v162_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function, capacity and patience differ; GaussianRPS1 positivecontrol')

    if a and (study/'checkpoint_deletion_manifest.json').exists():
        end('dead:v162_local_adapter','v162 no gain under80/14','architecture',[ident],'PERM89.344760174753 initialbest/stop14; delta historicalRPS -0.060804682166; replay exact, negative weights cleaned.','Scope limited to this config and80/14; historical positive RPS firstgain16, no general adapter rejection.')
