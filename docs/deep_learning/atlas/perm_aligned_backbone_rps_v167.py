import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_aligned_backbone_rps_v167_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_aligned_backbone_rps_v167_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v167 PERM anchored privatebody/RPS','Same activecapacity/init/80-14 as166, add0.1 train-only SOURCE128x2 hiddenMSE. Teacher frozen, no teacherpredictionloss/inference, no extra trainingphase; body1e-5/head3e-4.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_aligned_backbone_rps_v167_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Same initial function, active update scope and patience differ; GaussianRPS1 positivecontrol')

    edge('pure_dl_perm_private_backbone_rps_v166_fold5/perm_conditional',ident,'comparison','Matched capacity/init/updatebudget, additional frozen teacher compute only duringtraining')
