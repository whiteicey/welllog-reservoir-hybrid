import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_multiorder_interaction_v192_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for stage in ('add','multiply'):
        ident=f'pure_dl_perm_multiorder_interaction_v192_fold5/{stage}'
        out=root/'outputs'/ident;state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/f'{stage}_assembled.json')
        status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','prepared')
        shown=dict(resultText=str(state)+'; own PERM snapshot, pair gain unverified' if state else 'Prepared/untrained, no score')
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/f'{stage}_assembled.json').relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v192 '+stage,'Masked1D multi-order detail residual, split16/48/64 dilation1/2/3; SiLU gate/value addition versus multiplication, same parameter initialization, zero project and fixed residual0.1. Preserve positive128/SAM/well sampling; no trained185 weights.',str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
        edge('pure_dl_perm_private_width128_v185_fold5/perm_conditional',ident,'comparison','Definition inheritance, capacity added, no pretrained185 weights; same initial output and common core')
    edge('pure_dl_perm_multiorder_interaction_v192_fold5/add','pure_dl_perm_multiorder_interaction_v192_fold5/multiply','comparison','Same parameters and zero-projection initial hidden function; addition versus multiplication. Both final: multiply-minus-add+0.007156523218, same1162673; local positive singlefold/seed, below185/154.')
