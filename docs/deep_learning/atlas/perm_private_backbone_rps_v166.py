import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_private_backbone_rps_v166_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_private_backbone_rps_v166_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v166 PERM privatebody/RPS coadaptation','Original private two-view body LR1e-5 plus positive128/RPS1 readout3e-4; original output/router parameters frozen, gate/scale function may adapt. Dropout eval, real body gradients, original SOURCE init; user80/14, fivephases.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_private_backbone_rps_v166_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function, active update scope and patience differ; GaussianRPS1 positivecontrol')

    if a and (study/'perm_conditional_checkpoint_deletion_manifest.json').exists():
        end('dead:v166_privatebody','v166 unanchoredbody no gain','architecture',[ident],'PERM89.344760174753 best0/stop14, own gradients positive, actual assembly verified, negative weights cleaned.','Close this unanchored1e-5/80-14 config, test train-only hidden anchor; not all backbone finetuning rejected.')
