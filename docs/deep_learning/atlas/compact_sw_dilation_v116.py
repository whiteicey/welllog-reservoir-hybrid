import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_sw_dilation_v116_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('dilation2','dilation4'):
        ident='pure_dl_compact_sw_dilation_v116_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v116 fixed-capacity SW dilation / '+mode,'Same random SW/845913 active parameters/budget/zero residual initial function; only65-tap depthwise filter dilation2 or4. Gaussian NLL+utility1/SAM.1/DropKey.4 unchanged,5 stages including source and2 POR phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_sw_dilation_v116_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Historical dilation1 control; same random initialization, no trained SW loaded')
    if all(results) and all(v['gain_vs_local_SW']<=0 for v in results):
        end('dead:v116_SW_dilation','v116 larger filter span not beneficial','architecture',['pure_dl_compact_sw_dilation_v116_fold5/dilation2','pure_dl_compact_sw_dilation_v116_fold5/dilation4'],'Both restored original fifth scores below retained local dilation1','Close this specific learned filter dilation scope; not all multi-scale models.')
