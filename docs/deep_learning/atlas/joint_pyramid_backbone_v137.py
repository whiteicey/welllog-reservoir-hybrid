import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/joint_pyramid_backbone_v137_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_joint_pyramid_backbone_v137_fold5/pyramid';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');verified=read(study/'pyramid_verified.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if verified:
        row=dict(value=verified['scores']['TOTAL'],scores=verified['scores'],kind='fifth',fold=5,scope='Actual three-target original fifth; two phases including own-fold teacher/Q',source=str((study/'pyramid_verified.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v137 masked convolutional pyramid backbone','Replace temporal Transformer with convolutional context64/32/16, mask-weighted pooling and gated skip decoder. Random student, original dense207 stem, positive train-only hidden alignment retained. No teacher prediction loss or inference. Architecture and capacity change.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/joint_pyramid_backbone_v137_20261001/README.md'],**shown)
    edge('pure_dl_joint_representation_backbone_v132r1_fold5/aligned',ident,'comparison','Retain positive alignment; replace temporal communication, no trained student weights inherited')
    if verified and (study/'control_delta.json').exists():
        end('dead:v137_joint_pyramid_TOTAL','v137 convolutional pyramid loses joint TOTAL','architecture',[ident],'TOTAL79.154301 vs aligned79.240236; POR+0.138371, PERM-0.323087, SW-0.041047. Exact replay; best retained only as local POR positive control, redundant last cleaned.','Close this joint three-target configuration; retain scoped POR gain evidence, no blanket rejection of convolutional pyramid.')
