import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_component_readout_v149_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('shared','communicate'):
        ident='pure_dl_por_component_readout_v149_fold5/'+mode;out=root/'outputs'/ident
        st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/(mode+'_assembled.json'))
        shown=dict(resultText=str(st) if st else 'Prepared, no score')
        if a:
            row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five phases including source',source=str((study/(mode+'_assembled.json')).relative_to(root)));shown=dict(primary=row,results=[row])
        node(ident,'v149 component readout '+mode,'Matched16x16 component token readouts, shared decoder for bounded location/logweight; only diagonal/full attention differs. Same frozenv113 parent and zero-output initialization; no v140 head loaded.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_component_readout_v149_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen parent')
        edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Absolute POR77.912103 control, unequal structure/capacity')
    edge('pure_dl_por_component_readout_v149_fold5/shared','pure_dl_por_component_readout_v149_fold5/communicate','comparison','Matched parameter and initialization; change only component attention connectivity')
    if (study/'checkpoint_cleanup.json').exists():
        end('dead:v149_component_readout','v149 shared component and communication readouts below direct readout','architecture',['pure_dl_por_component_readout_v149_fold5/shared','pure_dl_por_component_readout_v149_fold5/communicate'],'Restored POR shared77.672952 / communicate77.645462; communication delta-0.027491, both below direct25677.912103. Exact replay and actual assembly complete; negative checkpoints cleaned.','Close these two component-token configurations, not a universal rejection of attention or distribution readout.')
