import json

def annotate(root,node,edge,end):
    arms=[('por_joint_density_width_v144','wide512','v144 POR density head width512','Width512/GLU1024/32-output head vs verified width256. Same frozen v113 predictor and bounds, no trained v140 head loaded; capacity comparison, five source-counted phases.','architecture'),('por_residual_distribution_v145','residual256','v145 POR residual distribution correction','Width256/48 outputs jointly correct16 locations/logweights/softplus scale parameters. Scale raw delta bounded0.5, correct proper NLL andCDF, defaultgate and oldpredictor frozen. Extra4112params, no trained v140 head loaded, not identical capacity.','loss')]
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for suffix,mode,title,desc,category in arms:
        study=root/'results/deep_learning'/(suffix+'_20261001')
        if not study.exists():continue
        ident='pure_dl_'+suffix+'_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual original-fifth assembly with fixed PERM136/SW104; five phases',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,title,desc,text,stage=category,status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=[str((study/'README.md').relative_to(root))],**shown)
        edge('pure_dl_por_joint_density_width_v140_fold5/wide256',ident,'comparison','Userrequested extension of measured POR head gain; not checkpoint warm-start')
        edge('pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy',ident,'warm_start','Frozen original positive parent, new head zero output')
        if assembled and assembled.get('gain_vs_width256_POR',0)<0 and (study/'checkpoint_cleanup.json').exists():
            end('dead:'+suffix,'POR density extension below verified256',category,[ident],str(assembled['scores']),'Close this exact extension; source/config/results retained, negativeweights cleaned. No broader rejection of output-head refinement.')
