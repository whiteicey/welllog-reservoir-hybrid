import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_scale_v114_20261001'
    if not study.exists():return
    for mode in ('location_only','location_scale'):
        ident='pure_dl_compact_por_scale_v114_fold5/'+mode;out=root/'outputs'/ident
        p=out/'runtime_status.json';state=json.loads(p.read_text()) if p.exists() else {}
        ap=study/(mode+'_assembled.json');assembled=json.loads(ap.read_text()) if ap.exists() else {}
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth; five executed stages including source',source=str(ap.relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v114 mixed POR width scope / '+mode,'Same v113 forward checkpoint/function; lr3e-5; only candidate also updates original width rows. Width also changes detached location bounds. Five training stages including source and parent POR. Original data/scoring/families fixed.',text,stage='architecture',status='recorded' if assembled else 'running_snapshot' if state.get('status')=='running' else 'prepared',evidence=['results/deep_learning/compact_por_scale_v114_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'evolution','Actual warm checkpoint; additional training phase counted')
    edge('pure_dl_compact_por_scale_v114_fold5/location_only','pure_dl_compact_por_scale_v114_fold5/location_scale','comparison','Same initial function and budget; update-scope ablation, not equal active capacity')
    end('dead:v114_width_scope','v114 width update did not improve retained function','architecture',
        ['pure_dl_compact_por_scale_v114_fold5/location_scale'],
        'Width arm selected epoch0 POR77.540454; position control POR77.548069; scale-minus-location=-0.007615168154.',
        'Close this bounded warm mixed-family width update scope; does not disprove all width optimization.')
