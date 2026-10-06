import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_sw_well_sampling_v120_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('sqrt_well','equal_well'):
        ident='pure_dl_compact_sw_well_sampling_v120_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        if mode=='equal_well' and (study/'decisions.json').exists():text='NOT TRAINED, no score: stronger redistribution not launched after sqrt negative; not a measured failure.'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v120 SW well anchor sampling / '+mode,'Same random v104 local SW845913 parameters/init/loss/budget; only train-only well-count powered anchor sampling sqrt/equal. All training anchors eligible, original official row score unchanged.5 stages including source and2 POR phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_sw_well_sampling_v120_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Historical row-uniform control; same random initialization/model/loss, no trained SW loaded')
    if all(results) and all(v['gain_vs_local_SW']<=0 for v in results):
        end('dead:v120_SW_sampling','v120 well sampling below row uniform','architecture',['pure_dl_compact_sw_well_sampling_v120_fold5/sqrt_well','pure_dl_compact_sw_well_sampling_v120_fold5/equal_well'],'Both restored original fifth scores below retained row-uniform local model','Close these train-only sampling probabilities, not all group-generalization training.')
    result=read(study/'sqrt_well_assembled.json')
    if result and result['gain_vs_local_SW']<0:
        end('dead:v120_sqrt_SW_well_sampling','v120 square-root well sampling below row uniform','architecture',['pure_dl_compact_sw_well_sampling_v120_fold5/sqrt_well'],
            'Restored original fifth SW73.86117968 versus74.12175283; negative0.26057315. Equal arm untrained, no negative score invented.',
            'Close this sampling distribution under current model/Q50; not all well-generalization training.')
