import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_adaptive_mass_v193_20261001'
    if not study.exists():return
    ident='pure_dl_perm_adaptive_mass_v193_fold5/adaptive';out=root/'outputs'/ident
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'adaptive_assembled.json')
    status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','prepared')
    shown=dict(resultText=str(state)+'; own PERM snapshot, no final mechanism gain' if state else 'Prepared/untrained, no score; no automatic launch')
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'adaptive_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v193 SOURCE adaptive probability mass','Same82720 head/trunk,32outputs parameterize16interval widths/masses in SOURCE conditionalCDF coordinates; newCDF=T(F0),density=f0*Tprime with exactJacobian; min width/mass0.001 and logits+-1. Uniform identity initialization to be checked; no trained174 weights, no quantile inversion or changed tail; conditionalRPS1/utility0.1/SAM0.05/80-14.',str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
    edge('pure_dl_perm_conditional_cdf_v174_fold5/perm_conditional',ident,'comparison','SameSOURCE/source features/capacity/trunk/loss coefficients/SAM/budget; Gaussian means/logweights versus adaptive probability mass transport')
    edge('pure_dl_perm_quantile_readout_v191_fold5/iqf',ident,'inspiration','After quantile replacement failed, preserveSOURCE density/tails through identity-initialized probability transport; no191 trained-weight inheritance')
