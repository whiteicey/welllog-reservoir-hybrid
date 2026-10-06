import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/sw_joint_density_transfer_v143_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_sw_joint_density_transfer_v143_fold5/sw_conditional';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'sw_conditional_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual original-fifth combination with fixed POR140/PERM136; five total phases',source=str((study/'sw_conditional_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v143 SW conditional location/weight head transfer','Transfer successful nonlinear point density head to frozen original-source SW Gaussian predictor. 319->128 GLU32 zero-output head, same bounds, own SW NLL only. No SW104 trained weights, no Student3 family change. Compare source gain and retained SW, five total phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/sw_joint_density_transfer_v143_20261001/README.md'],**shown)
    edge('pure_dl_por_joint_density_width_v134_fold5/wide128',ident,'comparison','Mechanism transfer to SW only; no POR weights loaded')
    edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Retained SW control, different backbone/capacity, not warm-start')
    if assembled and (study/'checkpoint_cleanup.json').exists():
        end('dead:v143_SW_conditional_head','v143 SW conditional density head has no source gain','architecture',[ident],'SW73.622385 best0/stop18, exact unchanged source score; below retained SW74.121753. Exact replay/frozen integrity, negative weights cleaned.','Close this frozen-source Gaussian/NLL-only transfer; not proof that all output-head mechanisms fail on SW.')
