import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/por_convgru_backbone_v127_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_por_convgru_backbone_v127_fold5/convgru';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/'convgru_assembled.json')
    text='Implemented, untrained, no score' if not state else str(state)
    shown=dict(resultText=text)
    if assembled:
        row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='Actual combined original fifth;4 recipe phases including source',source=str((study/'convgru_assembled.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v127 new POR ConvGRU backbone','Random privatePOR269+relative inputs; three96-wide masked ConvNeXt-GLU blocks and packed BiGRU64x2, input-only well FiLM. All network weights new, own Student3NLL, originalCDF/grid/argmax/Q; fixedPERM/SW, four dependency phases including source. Backbone route independent of old POR outputs.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/por_convgru_backbone_v127_20261001/README.md'],**shown)
    edge('pure_dl_compact_por_family_gate_v125_fold5/fixed',ident,'comparison','Best compact POR control only; NO trained POR/source network weights loaded')
    if assembled and assembled['gain_vs_historical_POR']<0:
        end('dead:v127_ConvGRU','v127 private POR ConvGRU below retained champion','architecture',[ident],'Original-fifth restored POR below historical champion; not an inference about all recurrent backbones.','Close this concrete configuration; no automatic width/LR sweep.')
