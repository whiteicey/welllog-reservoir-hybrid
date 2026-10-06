import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_frozen_positive_patch_v197_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for stage in ('mean','weighted'):
        ident=f'pure_dl_perm_frozen_positive_patch_v197_fold5/{stage}';out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/f'{stage}_assembled.json')
        status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else state.get('status','prepared')
        shown=dict(resultText=str(state)+'; starts at verified154 PERM89.405564856919, pair alternatives, no final gain yet' if state else 'Prepared/untrained, no score; no automatic queue')
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; six source-counted phases, at most one added stage per head',source=str((study/f'{stage}_assembled.json').relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v197 frozen positive '+stage,'Load/freeze all learned154/SOURCE weights; only20056parameter local9curve16patch/16token/8core residual trained, zero128 projection, fixed0.1. OriginalGaussian structure/NLL/BCE/jointRPS kept. Mean/weighted alternative, not sequential; own head/80-14/lr1e-4, six total stages under user8cap and one extra per head.',str(state),stage='architecture',status=status,evidence=[str((study/'README.md').relative_to(root))],**shown)
        edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'initialization','Actual verified learned154 checkpoint SHA loaded and frozen; one added PERM stage, starts same functional score; no old benefit discarded')
    edge('pure_dl_perm_frozen_positive_patch_v197_fold5/mean','pure_dl_perm_frozen_positive_patch_v197_fold5/weighted','comparison','Same parent/extra20056 capacity/initial tensors/function/data/loss/budget; mean versus content-weighted core; alternatives never stacked')
