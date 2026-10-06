import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_curve_core_v194_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for stage in ('mean','weighted'):
        ident=f'pure_dl_perm_curve_core_v194_fold5/{stage}';out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/f'{stage}_assembled.json')
        status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','held' if (study/'HOLD.json').exists() else 'prepared')
        shown=dict(resultText=str(state)+'; own PERM snapshot, no completed pair gain' if state else 'Prepared/untrained, no score; no automatic queue')
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/f'{stage}_assembled.json').relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v194 local curve core '+stage,'SOFTS-inspired local16point ninecurve64tokens/32core, mean versus featurewise softmax weighted aggregate and redistribute, identical fourMLPs; zero output128 plus original206 fullresolution path. No stochastic pooling/pretrained185/extra phase; exact curve names, same SAM/mixedsampling/80-14.',str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
        edge('pure_dl_perm_private_width128_v185_fold5/perm_conditional',ident,'comparison','Private128 definition and positiveSAM/sampling inherited; capacity added, no trained185 weights; zero side projection preserves algebraic initial function')
    edge('pure_dl_perm_curve_core_v194_fold5/mean','pure_dl_perm_curve_core_v194_fold5/weighted','comparison','Same parameter tensors, input curve patches and fullresolution backbone; fixed mean versus content-selected core. No gain until final both arms.')
