import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_joint_density_width_v138_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_joint_density_width_v138_fold5/wide256';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'wide256_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly with fixed positive POR/SW; five total phases',source=str((study/'wide256_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v138 PERM conditional density width256','Widen point location/logweight head128 to256, same original-source frozen predictor and zero-output initial function. No v136 trained weights loaded. Capacity comparison, own Gaussian loss/decoder; replaces PERM phase, five total including source.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_joint_density_width_v138_20261001/README.md'],**shown)
    edge('pure_dl_perm_joint_density_v136_fold5/perm_conditional',ident,'comparison','Positive width128 control; same frozen source, capacity change only, no warm-start')
    if assembled and (study/'checkpoint_cleanup.json').exists():
        end('dead:v138_PERM_width256','v138 wider PERM readout has no gain','architecture',[ident],'PERM89.344760 best0/stop18 vs width12889.387936, delta-0.043176; exact replay/frozen check. Negative best/last cleaned with SHA manifest.','Close PERM width expansion; retain width128 positive model.')
