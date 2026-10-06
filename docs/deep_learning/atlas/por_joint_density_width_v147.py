import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_joint_density_width_v147_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_joint_density_width_v147_fold5/wide384';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'wide384_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual original-fifth three-target assembly; five total source-counted phases',source=str((study/'wide384_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v147 POR conditional density width384','Userrequested intermediate capacity after512stop. 319->384/GLU768/32point outputs, same frozenv113 initial predictor andbounds/loss. No trained128/256/512 heads loaded; capacity comparison,444704activeparameters.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_joint_density_width_v147_20261001/README.md'],**shown)
    edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Verified25677.912103 control, not warm-start')
    edge('pure_dl_por_joint_density_width_v144_fold5/wide512',ident,'comparison','512completed77.854566 and user stopped extension; capacity not monotonic')
    edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozenparent, zerooutput new head')
    if assembled and assembled.get('gain_vs_width256_POR',0)<0 and (study/'checkpoint_cleanup.json').exists():
        end('dead:v147_width384','v147 width384 below verified256 control','architecture',[ident],'POR77.791188 best22/stop40, delta256-0.120915; exact144665/frozen replay, negativeweights cleaned.','Close this width384 configuration; capacity did not yield monotonic gains.')
