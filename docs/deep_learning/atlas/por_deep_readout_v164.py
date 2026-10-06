import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_deep_readout_v164_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('linear','nonlinear'):
        ident='pure_dl_por_deep_readout_v164_fold5/'+mode;out=root/'outputs'/ident
        st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/(mode+'_assembled.json'))
        shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
        if a:
            row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly, five source-counted phases',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v164 POR residual depth '+mode,'Matched296864 params/commoninitialization; only bottleneck activation differs. Frozenv113, positive256/RPS1, user80/14, no trained151 head loaded.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_deep_readout_v164_20261001/README.md'],**shown)
        edge('pure_dl_por_ranked_probability_v151_fold5/joint_rps',ident,'comparison','Historical positive RPS smaller capacity and80/18 budget')
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Same frozenparent')
    edge('pure_dl_por_deep_readout_v164_fold5/linear','pure_dl_por_deep_readout_v164_fold5/nonlinear','comparison','Matched capacity/init/budget, only GELU activation differs')

    if (study/'control_delta.json').exists():
        end('dead:v164_residual_depth','v164 residual depth no final gain','architecture',['pure_dl_por_deep_readout_v164_fold5/linear','pure_dl_por_deep_readout_v164_fold5/nonlinear'],'Linear77.891844743355/nonlinear77.885830334051; nonlinear-linear -0.006014409304. Both below positive RPS; best30/stop44, actual assembly, negative weights cleaned.','Early prefix positive reversed at final. Close this bottleneck/bound configuration, not all deep heads.')
