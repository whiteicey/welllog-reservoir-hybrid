"""Explicit finite v34 launcher, status and cross-fitted feature assembly."""
from pathlib import Path
from datetime import datetime
import argparse,ast,hashlib,json,subprocess,sys
import numpy as np
import psutil

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/cross_target_v34_20260929'
EXPS=ROOT/'experiments/pure_dl'

def jobs():
    result=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            z=p.info;a=z.get('cmdline') or []
            if 'python' not in (z.get('name') or '').lower() or '-c' in a:continue
            if any('pure_dl' in s and (Path(s.replace('\\','/')).name.startswith(('train_','benchmark_','verify_')) or 'auxiliary_crossfit_v34.py' in s) for s in a):result.append(z)
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result

def append(p,marker,text):
    old=p.read_text(encoding='utf-8-sig') if p.exists() else ''
    if marker not in old:
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+text+'\n')

def launch(kind,number=None):
    if len(jobs())>=2:raise RuntimeError('Max2 GPU jobs')
    if kind=='aux':
        name=f'pure_dl_cross_target_v34_aux{number}';script=EXPS/'auxiliary_crossfit_v34.py';args=['--group',str(number)]
    else:
        name='pure_dl_cross_target_v34_SW_fold5';script=EXPS/'train_cross_target_v34.py';args=['--target','SW','--out',str(ROOT/'outputs'/name)]
    out=ROOT/'outputs'/name
    if out.exists():raise FileExistsError('Do not relaunch existing run '+str(out))
    out.mkdir()
    for f in ('AGENTS.md','PROJECTS.md'):(out/f).write_text((STUDY/f).read_text(encoding='utf-8'),encoding='utf-8')
    (out/'README.md').write_text('# '+name+'\n\nSee results/deep_learning/cross_target_v34_20260929/PROJECTS.md.\nThis '+kind+' run is part of the fixed cross-target study. Auxiliary NLL is not a fifth-fold score.\n',encoding='utf-8')
    command=[sys.executable,'-X','utf8','-u',str(script),*args]
    with (out/'train.log').open('wb') as log,(out/'train.err').open('wb') as err:
        proc=subprocess.Popen(command,cwd=str(ROOT),stdout=log,stderr=err,stdin=subprocess.DEVNULL,close_fds=True,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
    rec=dict(pid=proc.pid,kind=kind,group=number,started_at=datetime.now().astimezone().isoformat(),command=command)
    (out/'launch.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    for p in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md'):
        append(p,'## v34 launch '+name,'## v34 launch '+name+'\n'+json.dumps(rec,ensure_ascii=False)+'\nNo automatic queue or restart.')
    print('LAUNCHED',json.dumps(rec),flush=True)

def assemble_features():
    if jobs():raise RuntimeError('Finish auxiliary jobs before assembling features')
    if (STUDY/'feature_manifest.json').exists():raise RuntimeError('Feature cache is immutable')
    manifest=json.loads((STUDY/'auxiliary_manifest.json').read_text())
    with np.load(STUDY/'row_mapping.npz') as z:
        group=z['heldout_group'];well=z['raw_well'];fit=z['train_raw'];val=z['validation_raw']
    values=np.empty((len(group),3,6),np.float32);outer=np.zeros((int((group<0).sum()),3,6),np.float64)
    provenance=[]
    for g in range(4):
        out=ROOT/'outputs'/f'pure_dl_cross_target_v34_aux{g}'
        report=json.loads((out/'auxiliary_report.json').read_text(encoding='utf-8'));cfg=report['config']
        assert cfg['source_sha256']==hashlib.sha256((EXPS/'auxiliary_crossfit_v34.py').read_bytes()).hexdigest()
        query=np.unique(well[group==g]);assert sorted(query.tolist())==sorted(cfg['oof_wells'])
        assert set(query).isdisjoint(cfg['fit_wells']) and set(query).isdisjoint(cfg['stop_wells'])
        assert set(cfg['outer_wells']).isdisjoint(set(cfg['fit_wells'])|set(cfg['stop_wells']))
        a=np.load(out/'all_raw_summaries.npy',mmap_mode='r');assert a.shape==values.shape and np.isfinite(a).all()
        values[group==g]=a[group==g];outer+=a[group<0]/4
        provenance.append(dict(group=g,fit_wells=cfg['fit_wells'],stop_wells=cfg['stop_wells'],query_wells=cfg['oof_wells'],models=report['models'],
            checkpoint=str((out/'model_best.pt').relative_to(ROOT)),checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),seconds=report['seconds'],peak_allocated_MiB=report['peak_allocated_MiB']))
    values[group<0]=outer.astype(np.float32);assert np.isfinite(values).all()
    qs=np.quantile(values[fit],[.25,.5,.75],axis=0);center=qs[1];scale=np.maximum(qs[2]-qs[0],1e-3)
    z=np.clip((values-center)/scale,-8,8).astype(np.float16)
    own=np.concatenate([z[:,2],z[:,2]],-1);cross=np.concatenate([z[:,0],z[:,1]],-1)
    shuffled=np.empty_like(cross);rng=np.random.default_rng(20261134);shuffle_checks=[]
    for g in (-1,0,1,2,3):
        rows=np.flatnonzero(group==g);perm=rows[rng.permutation(len(rows))];shuffled[rows]=cross[perm]
        assert (group[rows]==group[perm]).all()
        shuffle_checks.append(dict(group=g,rows=len(rows),exact_self_matches=int((rows==perm).sum())))
    np.savez(STUDY/'feature_views.npz',self=own,cross=cross,shuffled=shuffled)
    np.savez(STUDY/'raw_auxiliary_summaries.npz',summaries=values)
    meta=dict(created_at=datetime.now().astimezone().isoformat(),raw_points=len(group),fit_points=len(fit),validation_points=len(val),
        side_features=12,modes=['self','cross','shuffled'],fit_wells=np.unique(well[fit]).tolist(),monitor_wells=np.unique(well[val]).tolist(),
        summaries_per_target=6,feature_labels=['default_probability','transformed_mean','log_transformed_sd','transformed_q10','transformed_q50','transformed_q90'],
        median=center.tolist(),scale=scale.tolist(),scaler_fit='original64 OOF summaries only, unsupervised scaling',
        provenance=provenance,shuffle_checks=shuffle_checks,
        auxiliary_query_wells_excluded_from_fit_and_stopping=True,auxiliary_preprocessing_fit40_only=True,
        old_supervised_network_used_for_auxiliary=False,original_fifth_labels_used=False,
        validation_summaries='mean of four models summaries; not exact quantiles of their mixture',
        caveat='Auxiliary predictions are fold-excluded; main SW parent was previously fitted on original64 and selected on originalfifth. This is not independently validated full-pipeline OOF.',
        main_predictor_unchanged=True,negative_control_eligible_for_adoption=False)
    (STUDY/'feature_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('CROSSFIT_FEATURES_READY',json.dumps({k:meta[k] for k in ['raw_points','fit_points','validation_points','side_features','auxiliary_query_wells_excluded_from_fit_and_stopping']}),flush=True)

def make_head_trainer():
    source=(EXPS/'train_crosscurve_v32.py').read_text(encoding='utf-8-sig')
    changes={
        'Matched redundant/cross-curve feature corrections on retained v31.':'Matched own/cross-target/shuffled OOF prediction features on frozen v31 SW.',
        "module_name='crosscurve_v32'":"module_name='cross_target_v34'",
        "choices=('POR','PERM','SW')":"choices=('SW',)",
        "assert ids[0].isdisjoint(ids[1])":"assert all(ids[i].isdisjoint(ids[j]) for i in range(len(ids)) for j in range(i))",
        "assert common_equal":"assert common_equal and all(all(torch.equal(initial[MODES[0]][k],initial[m][k]) for k in common) for m in MODES)",
        "('prepare_crosscurve_v32.py','por_reparam_v31.py'":"('auxiliary_crossfit_v34.py','run_cross_target_v34.py','crosscurve_v32.py','por_reparam_v31.py'",
        'extra_view_features=18,cached_views=2':'extra_view_features=12,cached_views=3',
        'v32_feature_head_plus_frozen_v31_dependencies':'v34_cross_fitted_target_context_plus_frozen_SW'}
    for old,new in changes.items():
        if source.count(old)!=1:raise RuntimeError('Source template mismatch '+old)
        source=source.replace(old,new)
    ast.parse(source);dest=EXPS/'train_cross_target_v34.py'
    if dest.exists():assert dest.read_text(encoding='utf-8')==source
    else:dest.write_text(source,encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--aux-groups',nargs='+',type=int,choices=range(4));ap.add_argument('--assemble-features',action='store_true');ap.add_argument('--head',action='store_true');ap.add_argument('--status',action='store_true');a=ap.parse_args()
    if a.status:
        for name in [f'pure_dl_cross_target_v34_aux{g}' for g in range(4)]+['pure_dl_cross_target_v34_SW_fold5']:
            out=ROOT/'outputs'/name
            for f in ('runtime_status.json','failure.json'):
                if (out/f).exists():print(name,f,(out/f).read_text(encoding='utf-8'))
            if (out/'train.err').exists() and (out/'train.err').stat().st_size:print('STDERR',name,(out/'train.err').read_text(encoding='utf-8',errors='replace')[-1200:])
        print('ACTIVE',json.dumps(jobs()));return
    marker='## v34 已授权跨目标协同试验'
    text=marker+'\n原64训练井内部4组交叉拟合辅助模型（每组40拟合/8早停/16留出），来源井同时不参与拟合和选择；主SW固定。只开展self/cross/shuffled三项同容量头，负对照不部署；旧原其他折/交付不改。详见cross_target_v34_20260929/PROJECTS.md。'
    for p in (ROOT.parent/'AGENTS.md',ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md'):append(p,marker,text)
    append(ROOT/'EXPLORATIONS.md','cross_target_v34_20260929','- [v34 按井交叉拟合的跨目标协同](results/deep_learning/cross_target_v34_20260929/PROJECTS.md)：12个微型辅助模型与三个冻结SW修正对照，原第五折不进入辅助拟合或早停。')
    if a.aux_groups:
        if len(set(a.aux_groups))!=len(a.aux_groups) or len(jobs())+len(a.aux_groups)>2:raise RuntimeError('Duplicate groups or max2 jobs exceeded')
        for g in a.aux_groups:launch('aux',g)
    if a.assemble_features:assemble_features()
    if a.head:
        if not (STUDY/'feature_manifest.json').exists():raise RuntimeError('Missing OOF features')
        make_head_trainer();launch('head')

if __name__=='__main__':main()
