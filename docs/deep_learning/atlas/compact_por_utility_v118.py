import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_utility_v118_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('alpha025','alpha050'):
        ident='pure_dl_compact_por_utility_v118_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state);shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth,5 executed stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)));shown=dict(primary=row,results=[row])
        node(ident,'v118 own POR utility loss / '+mode,'Same v113 checkpoint/function/54483 active parameters/lr3e-5. Proper mixed-family likelihood plus own original POR reward via soft action temp.05. Original grid/CDF/argmax unchanged.5 total stages including source and initial POR. v114 likelihood control reused.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_por_utility_v118_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'evolution','Actual selected warm source; refinement phase replaces v114, does not append6th stage')
        edge('pure_dl_compact_por_scale_v114_fold5/location_only',ident,'comparison','Same initial checkpoint/function and update scope, only utility term added')
    if all(results) and all(v['gain_vs_likelihood_control_POR']<=0 for v in results):
        end('dead:v118_POR_utility','v118 fixed-head POR utility loss below likelihood','architecture',['pure_dl_compact_por_utility_v118_fold5/alpha025','pure_dl_compact_por_utility_v118_fold5/alpha050'],str([v['gain_vs_likelihood_control_POR'] for v in results]),'Close these bounded mixed-family own-head utility strengths; no general claim against utility loss.')
