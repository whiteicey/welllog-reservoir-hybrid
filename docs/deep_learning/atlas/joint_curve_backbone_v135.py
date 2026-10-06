import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/joint_curve_backbone_v135_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_joint_curve_backbone_v135_fold5/curve_tokens';out=root/'outputs'/ident
    state=read(out/'runtime_status.json');report=read(out/'report.json');verified=read(study/'curve_tokens_verified.json')
    text='Prepared/untrained, no score' if not state else str(state);shown=dict(resultText=text)
    if verified:
        row=dict(value=verified['scores']['TOTAL'],scores=verified['scores'],kind='fifth',fold=5,scope='Actual3-target newstudent original fifth;2dependencyphases inclsource teacher/Q',source=str((study/'curve_tokens_verified.json').relative_to(root)))
        shown=dict(primary=row,results=[row])
    node(ident,'v135 newbackbone ninecurve inputtokens','Ninecurve-specificfeatureMLPs32 +2perpointcrosscurveTransformer/head4, all206featuresretained via denseshortcut. New randomstudent, positivev132r1 train-onlyteacherhiddenalignment retained; original3labels/NLL/CDF/argmax/Q, no teacher predictions/pseudolabels/inference. Representation+capacity change, not identicalcapacity; standalone2phases inclsource.',text,stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/joint_curve_backbone_v135_20261001/README.md'],**shown)
    edge('pure_dl_joint_representation_backbone_v132r1_fold5/aligned',ident,'comparison','Retain measuredalignmentgain, change inputencoding; no trainedstudentweights inherited')
    if verified and (study/'control_delta.json').exists():
        end('dead:v135_joint_curve_tokens','v135 curve tokens lose joint TOTAL','architecture',[ident],'Final TOTAL79.147765 vs aligned79.240236, delta-0.092471; POR-0.416136, PERM+0.031821, SW+0.060665. Training1161.755s/peak3199.237MiB; exact replay.','Close this joint input architecture; retain best as small PERM/SW positive comparison evidence. Not a blanket rejection of curve encoding.')
