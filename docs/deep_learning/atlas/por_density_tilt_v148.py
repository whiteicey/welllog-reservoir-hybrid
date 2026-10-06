import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_density_tilt_v148_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_density_tilt_v148_fold5/tilt';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'tilt_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual original-fifth assembly; five source-counted training phases',source=str((study/'tilt_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v148 POR residual log-density readout','Frozen v113 parent. Shared256hidden head predicts location/logweights and bounded256interval log-density tilt. Proper conditional PDF and inference interval masses use the same normalization Z. Original grid/actions unchanged; no trained v140 head loaded. 296736 active parameters, unequal capacity comparison.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_density_tilt_v148_20261001/README.md'],**shown)
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen source predictor, zero new output initialization')
    edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Verified location/logweight readout POR77.912103, not a trained warm start')
    if assembled and assembled.get('gain_vs_width256_POR',0)<0 and (study/'checkpoint_cleanup.json').exists():
        end('dead:v148_density_tilt','v148 interval density tilt below best readout','architecture',[ident],f"Completed POR{assembled['scores']['POR']:.6f}, delta256{assembled['gain_vs_width256_POR']:+.6f}; exact replay and actual assembly. Negative checkpoints cleaned.",'Close this bounded interval-tilt configuration; does not rule out other distribution readouts.')
