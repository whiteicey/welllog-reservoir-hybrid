import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_perm_density_head_v117_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('joint_density',):
        ident='pure_dl_compact_perm_density_head_v117_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v117 nonlinear PERM density / '+mode,'Bounded per-expert191-32-24 density residual, zero output; same source initial function, added capacity. Gate/side/routes plus new weights/location/width residual trained together, original density/backbone frozen; proper Gaussian NLL/CDF.5 total training stages including source and2 POR phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_perm_density_head_v117_20261001/README.md'],**shown)
        edge('pure_dl_compact_v100_fold5/perm',ident,'comparison','Original gate-route-only historical control; same initial function and budget, capacity changed')
    if results and results[0] and results[0]['gain_vs_Gaussian_PERM']<=0:
        end('dead:v117_joint_PERM_density','v117 nonlinear joint-density residual below original head','architecture',['pure_dl_compact_perm_density_head_v117_fold5/joint_density'],
            'Restored PERM89.34735488 versus89.37839421; negative0.03103933 with144665 exact replay.',
            'Close this bounded cold-source joint-density correction; no general claim against all density networks.')
