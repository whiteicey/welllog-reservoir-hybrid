import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_por_family_gate_v125_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for mode in ('fixed','learned'):
        ident='pure_dl_compact_por_family_gate_v125_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 phases including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v125 POR per-component tail-family probability / '+mode,'Warm v113 forward;16 shared location/width Student3+Cauchy components; same0.95/0.05 initial family probabilities. Fixed versus learned16 logits, update scope not equal capacity. Own proper NLL and synchronized CDF. Other targets fixed, five phases incl source; concentratePOR until historical gain>0.25.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_por_family_gate_v125_20261001/README.md'],**shown)
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Exact retained positive v113 checkpoint; smooth family initialization changes function')
        edge('pure_dl_compact_por_scale_v114_fold5/location_only',ident,'comparison','Historical compact POR refinement control; not loaded')
    edge('pure_dl_compact_por_family_gate_v125_fold5/fixed','pure_dl_compact_por_family_gate_v125_fold5/learned','comparison','Matched initialization; only16 family logits update')

    ablation=read(study/'ablation_report.json')
    if ablation and ablation['learned_minus_fixed_POR']<0:
        end('dead:v125_learned_family','v125 learned family logits below matched fixed control','optimization',['pure_dl_compact_por_family_gate_v125_fold5/learned'],'Matched learned-minus-fixed POR -0.006719430811; fixed smoothing retains local positive gain.','Close this learned tail-logit scope/LR; not all heavy-tail families.')
