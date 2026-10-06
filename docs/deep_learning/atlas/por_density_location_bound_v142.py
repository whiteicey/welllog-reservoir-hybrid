import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_density_location_bound_v142_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_density_location_bound_v142_fold5/locbound2';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'locbound2_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual fixed three-target assembly; five phases including source',source=str((study/'locbound2_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v142 same-capacity POR location bound2','Same width128/82720 active parameters and initial raw predictor as v134, no trained v134 head loaded. Only location delta formula changes s*tanh(u/s) to2s*tanh(u/(2s)); logweight bound1/defaultgate/source widths frozen. Hypothesis of restricted conditional correction, no claimed gain yet.',text,stage='loss',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_density_location_bound_v142_20261001/README.md','results/deep_learning/por_joint_density_width_v140_20261001/GAIN_ANALYSIS.md'],**shown)
    edge('pure_dl_por_joint_density_width_v134_fold5/wide128',ident,'comparison','Same capacity/init/loss/sample/budget, locationbound1 control; not warm-start')
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen positive parent, zero initial newhead')
    if assembled and (study/'checkpoint_cleanup.json').exists():
        end('dead:v142_locationbound2','v142 larger location bound loses same-capacity comparison','loss',[ident],'POR77.831487 vs bound1 width12877.869090, delta-0.037603; best34/stop52, exact replay/frozen integrity; negative weights cleaned.','Close this location-bound expansion; retain positive bound1. Do not generalize to all residual distributions.')
