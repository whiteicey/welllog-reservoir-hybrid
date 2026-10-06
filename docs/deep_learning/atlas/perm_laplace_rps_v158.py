import json
def annotate(root,node,edge,end):
    study=root/'results/deep_learning/perm_laplace_rps_v158_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ident='pure_dl_perm_laplace_rps_v158_fold5/perm_conditional';out=root/'outputs'/ident
    st=read(out/'runtime_status.json');r=read(out/'report.json');a=read(study/'perm_conditional_assembled.json');shown=dict(resultText=str(st) if st else 'Prepared/untrained, no score')
    if a:
        row=dict(value=a['scores']['TOTAL'],scores=a['scores'],kind='fifth',fold=5,scope='Actual three-target assembly; five source-counted phases',source=str((study/'perm_conditional_assembled.json').relative_to(root)));shown=dict(primary=row,results=[row])
    node(ident,'v158 PERM Laplace conditional readout','PERM-specific family hypothesis, same128head/82720rawparameters, sigma/sqrt2 Laplace scale; properNLL/CDF/jointRPS1. Initial function differs fromGaussian; report separately, no trainedv154 head loaded.',str(st),stage='architecture',status='recorded' if r else 'running_snapshot' if st.get('status')=='running' else 'failed' if st.get('status')=='failed' else 'prepared',evidence=['results/deep_learning/perm_laplace_rps_v158_20261001/README.md'],**shown)
    edge('pure_dl_perm_ranked_probability_v154_fold5/perm_conditional',ident,'comparison','Samecapacity/rawinit, family change changes initial function; GaussianRPS1 positivecontrol')

    if a and a['gain_vs_Gaussian_RPS_PERM']<0:
        end('dead:v158_Laplace','v158 PERM Laplace closed','architecture',[ident],'PERM89.332375289863; deltaGaussianRPS1-0.073189567056, best0/stop18, actual assembled replay verified, negative weights cleaned.','Close this configuration; no general family conclusion.')
