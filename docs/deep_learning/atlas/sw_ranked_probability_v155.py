import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/sw_ranked_probability_v155_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_sw_ranked_probability_v155_fold5/local';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'local_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'local_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v155 useful SW local model plus jointRPS','Same845913parameters/random initial SW104 local Gaussian/utility1/SAM.1, add own SW jointCDF supervision coefficient1. No trainedSW/PERM/POR weights used inSWnetwork; fixedPOR151/PERM136 assembly.',str(st),stage='optimization',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/sw_ranked_probability_v155_20261001/README.md'],**shown)
    edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Same architecture/capacity/initialization, add jointRPS only, no checkpoint warmstart')
    edge('pure_dl_por_ranked_probability_v151_fold5/joint_rps',ident,'comparison','Evidence-positive supervision transfer, not model weight reuse')
    if a and (study/'checkpoint_cleanup.json').exists():
        end('dead:v155_priority_stop','v155 SW priority stop, incomplete budget','optimization',[ident],'User refocuses POR/PERM. Saved complete epoch21/best20 replay SW73.670352 below retained74.121753; earlyterminated, not a convergence comparison. Negative checkpoint artifacts cleaned after actual assembly.','Close current SW budget allocation, no auto-resume; does not prove full-budget SW RPS mechanism negative.')
