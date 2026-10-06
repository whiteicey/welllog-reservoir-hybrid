import json

def annotate(root,node,edge,end):
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    for version in ('v132','v132r1'):
        study=root/('results/deep_learning/joint_representation_backbone_'+version+'_20261001')
        if not study.exists():continue
        ident='pure_dl_joint_representation_backbone_'+version+'_fold5/aligned';out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');verified=read(study/'aligned_verified.json')
        text='Prepared/untrained, no score' if not state else str(state)
        if version=='v132':text='FAILED before first optimizer step: teacher512 exceeds native256 tokenlimit;0completedepochs, initialTOTAL68.41007972 only, NO trainedscore/gain. Source/config/trace retained; initialstate cleaned.'
        shown=dict(resultText=text)
        if verified:
            row=dict(value=verified['scores']['TOTAL'],scores=verified['scores'],kind='fifth',fold=5,scope='Actual3-target student original fifth;2dependencyphases inclteacher source',source=str((study/'aligned_verified.json').relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,version+' new backbone train-only hidden alignment','Student new random hierarchicalattention937515 predictionparams +24832auxprojectionparams, frozen ownfoldteacher hidden vectors only duringtraining. Student512/teacher256/common128core in r1; original3labels/properNLL/Q/scorer, no teacher predictions/pseudolabels/inference. Same studentcapacity/init as v130, auxiliaryobjective/trainingparams differ.2phases inclsource; potentialprivatePORrecipe5.',text,stage='architecture',status='failed' if version=='v132' else 'recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',evidence=[str((study/'README.md').relative_to(root))],**shown)
        edge('pure_dl_joint_hierarchical_backbone_v130_fold5/joint',ident,'comparison','Same randomstudentarchitecture/init, add train-onlyrepresentationauxiliary; no sourceweight inheritance into student')
        if version=='v132r1':edge('pure_dl_joint_representation_backbone_v132_fold5/aligned',ident,'continuation','NEW trial after zero-stepfailure; teacher nativecontext repair, not checkpointresume')
