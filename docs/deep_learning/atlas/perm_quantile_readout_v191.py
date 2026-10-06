import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_quantile_readout_v191_20261001'
    if not study.exists():return
    ident='pure_dl_perm_quantile_readout_v191_fold5/iqf';out=root/'outputs'/ident
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'iqf_assembled.json')
    status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','prepared')
    shown=dict(resultText=str(state)+'; own PERM snapshot, initial89.34524668852337 differs from SOURCE89.3447601747529; no final gain' if state else 'Prepared/untrained, no score')
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'iqf_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    description='SOURCE-referenced32monotoneIQF knots plus bound exponential standardized tails, same82720 context head and source feature trunk. ConditionalRPS1/finite utility0.1/SAM0.05/80-14; no trained174 head. Reference255grid+dynamic8sigma bounds/fourlocal bisections; initial CDF/PDF differs. Same-SAM-step source cache only, cleared each step/eval; physical PERM moments not asserted.'
    node(ident,'v191 SOURCE IQF32',description,str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
    edge('pure_dl_perm_conditional_cdf_v174_fold5/perm_conditional',ident,'comparison','Same active capacity/source hidden/optimizer/loss coefficients/data/budget; Gaussian toIQF representation, initial predictive function and computation differ, not exact-init ablation')
