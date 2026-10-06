import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_feature_adapt_v119_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('calibration','features'):
        ident='pure_dl_compact_por_feature_adapt_v119_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state);shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth,5 executed stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)));shown=dict(primary=row,results=[row])
        node(ident,'v119 POR feature update scope / '+mode,'Same v113 checkpoint/function, readouts3e-5 plus calibration-only or whole non-head expert features1e-5. Eval mode with gradients; original input router and coefficient tensors frozen, predictions may change via hidden features. Proper mixed-family own likelihood/original CDF/grid/argmax unchanged.5 stages incl source/parent. Update scope, not equal capacity.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_por_feature_adapt_v119_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'evolution','Actual selected warm source; refinement phase replaces v114, does not append6th stage')
        edge('pure_dl_compact_por_scale_v114_fold5/location_only',ident,'comparison','Same initial checkpoint/function and update scope, only expert feature update scope added')
    if all(results) and all(v['gain_vs_likelihood_control_POR']<=0 for v in results):
        end('dead:v119_POR_feature_scope','v119 expert feature updates below frozen control','architecture',['pure_dl_compact_por_feature_adapt_v119_fold5/calibration','pure_dl_compact_por_feature_adapt_v119_fold5/features'],str([v['gain_vs_likelihood_control_POR'] for v in results]),'Close this fixed-init/learning-rate/eval-mode feature scope; no general claim against backbone adaptation.')
