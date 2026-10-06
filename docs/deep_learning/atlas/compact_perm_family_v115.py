import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_perm_family_v115_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('t3','cauchy'):
        ident='pure_dl_compact_perm_family_v115_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v115 compact PERM family / '+mode,'Fixed proper t3/Cauchy likelihood and CDF; source Gaussian component IQR matched, same raw init/43431 active params/budget, initial functions differ. Gate/routes/side updates only; original density/backbone frozen.5 stages incl source and2 POR phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_perm_family_v115_20261001/README.md'],**shown)
        edge('pure_dl_compact_v100_fold5/perm',ident,'comparison','Gaussian historical control; cold original source, no trained PERM downstream loaded')
    if all(results) and all(v['gain_vs_Gaussian_PERM']<=0 for v in results):
        end('dead:v115_PERM_heavy_tail','v115 fixed PERM tails below Gaussian','architecture',['pure_dl_compact_perm_family_v115_fold5/t3','pure_dl_compact_perm_family_v115_fold5/cauchy'],'Both restored original fifth scores below retained Gaussian','Close this fixed-IQR/frozen-density family scope, not all heavy-tail models.')
