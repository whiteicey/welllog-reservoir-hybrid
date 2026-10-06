import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_additive_crps_v195_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_additive_crps_v195_fold5/perm_conditional';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'perm_conditional_assembled.json')
    status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','prepared')
    shown=dict(resultText=str(state)+'; own PERM snapshot only, no learned gain yet' if state else 'Prepared/untrained, no score')
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v195 additive analyticCRPS','Retain entire154 Gaussian location/logweight readout,128head/82720params/NLL/BCE/jointRPS1; add only conditional analytic mixtureCRPS0.1 with defaultrows zero, no new parameters. SameSOURCEinit/no154trainedhead/AdamW/80-14; original decoder and five phases. Latest user small additive repair.',str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'inspiration','Preserve positive structure and jointRPS, add one term; original154 frozen, no trained-weight warmstart. Historical154 patience18 differs.')
    edge('pure_dl_perm_sam_rps_v165_fold5/adamw',ident,'comparison','Matched80-14/current budget/same initial prediction/capacity/AdamW/noSAM/loss except additiveCRPS0.1')
