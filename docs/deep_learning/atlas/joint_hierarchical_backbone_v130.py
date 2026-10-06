import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/joint_hierarchical_backbone_v130_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_joint_hierarchical_backbone_v130_fold5/joint';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');verified=read(study/'joint_verified.json')
    text='Implemented, untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if verified:
        row=dict(value=verified['scores']['TOTAL'],scores=verified['scores'],kind='fifth',fold=5,scope='Actual new3-target backbone original fifth;2 dependency phases incl oldsourceQ',source=str((study/'joint_verified.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v130 new joint hierarchical attention backbone','All random weights937515; local ConvNeXt96 +8point pooling +4Transformer96/head4/FF384 +detail; input-only wholewell FiLM. Original206+relative, own3traininglabels/properStudent3-Gaussian likelihoods; real3output TOTAL selects best. No oldnetworkweights, oldsource retained fortrainonlyQ count, standalone2phases/futureprivatePORrecipe5.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/joint_hierarchical_backbone_v130_20261001/README.md'],**shown)
    edge('pure_dl_por_convgru_backbone_v127_fold5/convgru',ident,'comparison','Architecture and trainingrepresentation change after weak privatePOR; not a onefactor ablation or weight inheritance')

    if verified and (study/'checkpoint_cleanup.json').exists():
        end('dead:v130_joint_plain_backbone','v130 plain-label shared backbone below retained targets','architecture',[ident],'ActualfifthTOTAL78.95535249, POR73.73051180/PERM88.65145391/SW73.73768594, all below retainedcontrols; best18/stop36, full144665x3 replay exact.','Close this purelabelconfiguration, not all Transformers/sharedtraining. Train-onlyrepresentation auxiliary is a separate newhypothesis.')
