import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_sw_scaled_sam_v123_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text()) if p.exists() else {}
    results=[]
    for mode in ('rms_direction',):
        ident='pure_dl_compact_sw_scaled_sam_v123_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        if report.get('early_terminated'):text='EARLY TERMINATED at completed48, best38 POR/SW target='+str(report['target_score'])+'; best replay verified, not normal patience completion.'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v123 same-radius RMS SAM direction / '+mode,'Same local SW845913 parameters/init/loss/budget. d=(tensorRMS+0.01)*gradient then normalized to same EuclideanL2rho0.1. Only perturbation direction changes, NOT standardASAM. Original2-pass clone restoration/RNG replay/decoder preserved.5 stages including source/2POR phases.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_sw_scaled_sam_v123_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Historical standard SAM.1 matched model/init/budget control; no trained SW loaded')
    if results and results[0] and results[0]['gain_vs_local_SW']<0:
        end('dead:v123_RMS_SAM_direction','v123 same-radius tensor-RMS SAM direction below control','optimization',['pure_dl_compact_sw_scaled_sam_v123_fold5/rms_direction'],
            'Early terminated after48 full epochs, best38SW73.89718825 vs74.12175283, negative0.22456458; full144665 best replay exact.',
            'Close this fixed floor/radius/direction, not all adaptive optimizers. Timing includes Modern Standby exit interval, no active-compute speed claim.')
