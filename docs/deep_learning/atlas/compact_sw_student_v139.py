import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_sw_student_v139_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_compact_sw_student_v139_fold5/student3';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'student3_assembled.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five phases including source and two POR stages',source=str((study/'student3_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v139 SW Student-t3 conditional density','Same random compact-local SW architecture and active capacity as Gaussian v104. Student3 own SW NLL/utility/inference CDF synchronized, finite SD scale/sqrt3; original action grid/score/Q/SAM unchanged. No old SW weights; initial raw same but initial predictive family differs.',text,stage='loss',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_sw_student_v139_20261001/README.md'],**shown)
    edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Same architecture, Gaussian versus Student3 family; no trained checkpoint warm-start')
    if assembled and assembled['gain_vs_local_SW']<=0 and (study/'checkpoint_cleanup.json').exists():
        end('dead:v139_SW_student3','v139 SW Student3 below Gaussian control','loss',[ident],str(assembled['scores']),'Close this exact SW Student3 configuration; no df/utility/decoder sweep.')
