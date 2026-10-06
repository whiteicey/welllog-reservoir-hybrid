import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/joint_private_readout_backbone_v141_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_joint_private_readout_backbone_v141_fold5/private_readouts';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');verified=read(study/'private_readouts_verified.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if verified:
        row=dict(value=verified['scores']['TOTAL'],scores=verified['scores'],kind='fifth',fold=5,scope='Actual three-target original fifth; two source-counted phases',source=str((study/'private_readouts_verified.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v141 private nonlinear target readouts','Random joint aligned student, three separate96->192GLU96->25 nonlinear readouts instead of shared linear75 output. Same initial distribution priors, original dense/local/global-attention body; extra capacity, no trained student weights loaded. Teacher hidden features training-only.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/joint_private_readout_backbone_v141_20261001/README.md'],**shown)
    edge('pure_dl_joint_representation_backbone_v132r1_fold5/aligned',ident,'comparison','Retain measured alignment; change prediction branches, not warm-start')
