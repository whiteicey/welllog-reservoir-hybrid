import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_joint_density_v136_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_joint_density_v136_fold5/perm_conditional';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'perm_conditional_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actualcombinedoriginalfifth;5phases inclsource/fixedPORv134/SWv104',source=str((study/'perm_conditional_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v136 independent PERM conditionaldensityhead','OriginalsourceGaussianPERM predictor fullyfrozen; new319->128->32 pointGLU loc/logweighthead,82720params, no oldPERMdownstreamweights. OwnPERMlabels/NLL/CDF/argmax/Q, fixedpositivePOR/SW. Transfer positivearchitecture afterPOR+0.25 milestone;5finalrecipephases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_joint_density_v136_20261001/README.md'],**shown)
    edge('pure_dl_compact_v100_fold5/perm',ident,'comparison','RetainedPERMcontrol; not loaded')
    edge('pure_dl_por_joint_density_width_v134_fold5/wide128',ident,'comparison','Architecturetransfer only, no PORparameters/labels enter privatePERMloss')
