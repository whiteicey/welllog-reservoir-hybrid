import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_sw_readout_split_v121_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('shared','split'):
        ident='pure_dl_compact_sw_readout_split_v121_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v121 SW residual readout connectivity / '+mode,'Two zero residual128-64-128 branches; same879577 parameters/initial function, shared branch mean or private gate/density branches; backbone shared. Original local SW GaussianNLL+utility1/SAM.1/DropKey.4/sampling unchanged,5 stages including source/2POR phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_sw_readout_split_v121_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Original local SW architecture control has lower capacity; cold random SW/no trained weights loaded')
    if all(results) and all(v['gain_vs_local_SW']<=0 for v in results):
        end('dead:v121_SW_readout','v121 added readouts below original local SW','architecture',['pure_dl_compact_sw_readout_split_v121_fold5/shared','pure_dl_compact_sw_readout_split_v121_fold5/split'],'Both restored original fifth scores below retained local dilation1','Both absolute scores below original; matched split-shared contrast recorded separately, preserve any positive local connectivity evidence.')
    edge('pure_dl_compact_sw_readout_split_v121_fold5/shared','pure_dl_compact_sw_readout_split_v121_fold5/split','comparison','Same879577 trainable parameters and zero residual initialization, only residual-to-output connectivity changes')
