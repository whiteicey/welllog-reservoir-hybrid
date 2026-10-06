import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_sw_cdf_brier_v124_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    decision=read(study/'decision.json')
    results=[]
    for mode in ('cdf1','cdf4'):
        ident='pure_dl_compact_sw_cdf_brier_v124_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'));results.append(assembled)
        text='Implemented, untrained, no score' if not state else str(state)
        if mode=='cdf4' and decision.get('cdf4_skipped'):
            text='NOT TRAINED; skipped after cdf1 negative result, no measured score or inferred failure.'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;5 stages including source',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v124 finite-threshold SW CDF Brier / '+mode,'Same localSW845913 parameters/init/model/loss/budget; add own-label finite255-threshold CDFBrier weight1/4 to GaussianNLL+utility1. OriginalCDF/grid/argmax/Q/SAM.1/DropKey.4 unchanged;5 phases incl source/2POR phases. Temporary keepawake and explicit epoch-boundary early stop.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/compact_sw_cdf_brier_v124_20261001/README.md'],**shown)
        edge('pure_dl_compact_frequency_v104_fold5/local',ident,'comparison','Historical no-CDF same model/init/budget control; no trainedSW loaded')
    if results[0] and results[0]['gain_vs_local_SW']<0 and decision.get('cdf4_skipped'):
        end('dead:v124_SW_cdf','v124 fixed CDF calibration trial below original loss','optimization',['pure_dl_compact_sw_cdf_brier_v124_fold5/cdf1'],'cdf1 restored SW74.05171626 vs74.12175283; cdf4 untrained and skipped, not measured negative.','Close this fixed loss extension; do not generalize to all probability calibration.')
