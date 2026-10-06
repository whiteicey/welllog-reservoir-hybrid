import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_wavelet_detail_v190_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for stage in ('wavelet','long29'):
        ident=f'pure_dl_perm_wavelet_detail_v190_fold5/{stage}';out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/f'{stage}_assembled.json')
        status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','prepared')
        shown=dict(resultText=str(state)+'; own PERM snapshot only, not pair gain' if state else 'Prepared/untrained, no score; no automatic queue')
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/f'{stage}_assembled.json').relative_to(root)))
            shown=dict(primary=row,results=[row])
        description='Private128 first detail depthwise only; base5 plus two-level Haar low/high band5 convolution and synthesis, zero new filters/gains0.1' if stage=='wavelet' else 'Exactly matched3840 first detail parameters;29tap DW with original5 center and zero extra weights'
        description+='; same base128 random initialization/hidden and predictive initial function/3072 extra capacity/SAM0.05/mixed-well sampling/jointRPS/80-14, no trained185 weights; fixed row pairing, masks before/after DW'
        node(ident,'v190 '+stage,description,str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
        edge('pure_dl_perm_private_width128_v185_fold5/perm_conditional',ident,'comparison','Extend positive185 backbone definition, no trained-weight inheritance; same core initialization/loss/SAM/sampling, detail capacity+3072')
    edge('pure_dl_perm_wavelet_detail_v190_fold5/long29','pure_dl_perm_wavelet_detail_v190_fold5/wavelet','comparison','Matched3840 first-detail parameters and initial algebraic function, frequency decomposition versus long29; Both arms final/restored; wavelet minus long29=-0.017784499445, no frequency benefit in this setting')
