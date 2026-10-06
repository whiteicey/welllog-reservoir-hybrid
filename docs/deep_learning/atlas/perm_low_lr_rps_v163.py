import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_low_lr_rps_v163_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_low_lr_rps_v163_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v163 PERM lowerLR RPS readout','Same82720 positiveGaussianRPS readout, headLR1e-4; original gate/scale/body frozen. Historical154 firstgain16, userpat14 truncates it; newbudget80/14 vs control80/18, no automatic resume.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_low_lr_rps_v163_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function, active update scope and patience differ; GaussianRPS1 positivecontrol')

    if a and (study/'checkpoint_deletion_manifest.json').exists():
        end('dead:v163_lowLR','v163 lowLR no gain under14','optimization',[ident],'PERM89.344760174753 best0/stop14; delta positiveRPS -0.060804682166; replay/assembly exact, weights cleaned.','No further LR grid; scope limited to this1e-4/80-14 config.')
