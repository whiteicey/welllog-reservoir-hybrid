"""Bounded progress and final disk replay for the two standalone SW candidates."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/independent_sw_scratch_v10_20260928'
MODES = ('tcn', 'bigru')
PREFIX = 'pure_dl_independent_sw_v10_'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def active_trainers():
    found = []
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            info = p.info; argv = info.get('cmdline') or []
            if 'python' in (info.get('name') or '').lower() and 'base64' not in ' '.join(argv):
                if any('pure_dl/' in a.replace('\\','/') and
                       Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in argv):
                    found.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def snapshot():
    rows = []
    for mode in MODES:
        out = ROOT/'outputs'/(PREFIX+mode+'80_fold5')
        history = read(out/'validation_history.json') or []
        rows.append({'mode':mode, 'id':out.name, 'status':read(out/'runtime_status.json'),
                     'config':read(out/'config.json'), 'initial':read(out/'initial_verification.json'),
                     'report':read(out/'fold_5_report.json'), 'last':history[-1] if history else None,
                     'best_after_update':max(history, key=lambda x:x['target_score']) if history else None,
                     'history_count':len(history), 'failure':read(out/'failure.json')})
    return {'observed_at':datetime.now().astimezone().isoformat(), 'runs':rows,
            'all_complete':all(x['report'] is not None for x in rows),
            'reference':'outputs/pure_dl_independent_v8_selected_fold5/model_manifest.json',
            'scope':'individualSW scores;replacementTOTAL diagnostic until actual assembly',
            'active_training_pids':[x['pid'] for x in active_trainers()]}


def record(snap):
    from train_dual8222_heads80 import atomic_json
    atomic_json(STUDY/'progress.json', snap)
    curpath = ROOT/'docs/deep_learning/atlas/curation.json'
    cur = read(curpath)
    for row in snap['runs']:
        report, last = row['report'], row['last']
        status = (row['status'] or {}).get('status','unknown')
        if report:
            desc = f"完成SW={report['target_score']:.9f};相对旧SW{report['delta_old_SW']:+.9f};最佳{report['best_epoch']}/停止{report['stopped_epoch']}轮。"
        elif last:
            desc = f"运行快照epoch{last['epoch']}，bestSW={last['best_target_score']:.9f}；尚未最终恢复。"
        else:
            desc = status+'；尚无成绩。'
        cur['nodes'][row['id']] = {'title':'SW从头训练·'+row['mode'],
                'summary':'独立SW骨干及门控/256格分布；不加载旧网络；同原数据/评分，参数规模不同。',
                'conclusion':desc, 'stage':'loss', 'highlight':status=='running'}
        if report:
            single_target_result = {'scope':'第五折 · SW单目标', 'value':report['target_score'],
                'scores':{'SW':report['target_score']}, 'source':'outputs/'+row['id']+'/fold_5_report.json',
                'kind':'target', 'target':'SW', 'fold':5}
            cur['nodes'][row['id']].update(status='recorded', primary=single_target_result,
                results=[single_target_result], evidence=['outputs/'+row['id']+'/fold_5_report.json',
                'outputs/'+row['id']+'/README.md', 'outputs/'+row['id']+'/config.json',
                'experiments/pure_dl/independent_sw_scratch_v10.py'])
        entry = '\n\n## 最近状态 '+snap['observed_at']+'\n'+desc+'\n'
        for fn in ['PROJECTS.md','README.md']:
            with (ROOT/'outputs'/row['id']/fn).open('a',encoding='utf-8') as f:
                f.write(entry)
    edges = [
        {'source':'pure_dl_independent_sw_v9_selected_fold5', 'target':snap['runs'][0]['id'],
         'type':'evolution', 'note':'研究方向延续，未继承预训练权重；改为无先验的完整SW学习。',
         'evidence':'results/deep_learning/independent_sw_scratch_v10_20260928/METHOD.md'},
        {'source':snap['runs'][0]['id'], 'target':snap['runs'][1]['id'], 'type':'control',
         'note':'同输入/头/损失/步数，卷积与双向循环网络；非严格同参或同精度消融。',
         'evidence':'results/deep_learning/independent_sw_scratch_v10_20260928/METHOD.md'}]
    for edge in edges:
        if not any(e['source']==edge['source'] and e['target']==edge['target'] and e['type']==edge['type'] for e in cur['edges']):
            cur['edges'].append(edge)
    atomic_json(curpath, cur)


def finalize(snap):
    if not snap['all_complete']:
        raise RuntimeError('Both candidates must finish before fixed final selection')
    if active_trainers():
        raise RuntimeError('Do not add replay while training is active')
    if (STUDY/'assembled_report.json').exists():
        raise RuntimeError('Final report exists; preserve it instead of rerunning')
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from train_dual8222_heads80 import prepare_data, SOURCE, atomic_json, atomic_array
    from train_independent_channels_v8 import REFERENCE, Evaluator as PriorEvaluator
    from independent_channels_v8 import Network as PriorModel
    from independent_sw_scratch_v10 import Network
    from train_independent_sw_v9 import Evaluator
    from gpu_job_guard_v2 import check_slot
    import train_invariant as base
    print('GPU_ADMISSION', check_slot(min_free_gib=3.), flush=True)
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    t0=time.perf_counter()
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    data=prepare_data(master)
    old=np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/fold_5_pred.npy')
    truth=data['truth'][data['validation']]
    wells=data['wells'][data['validation']]
    baseline=base.metric(truth,old)
    evaluator=Evaluator(data,master,2,old)
    reports=[];preds={};models={}
    for row in snap['runs']:
        path=ROOT/'outputs'/row['id']/'model_best.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        net=Network(row['mode'],cp['state_dict']['centers'],cp['target_median'],cp['target_scale'])
        net.load_state_dict(cp['state_dict'],strict=True)
        net.eval().requires_grad_(False).cuda()
        torch.cuda.synchronize();tick=time.perf_counter()
        pred,score,total=evaluator(net)
        torch.cuda.synchronize();duration=time.perf_counter()-tick
        assert np.array_equal(pred,np.load(path.parent/'channel_pred.npy'))
        assert abs(score-row['report']['target_score'])<1e-10
        # Trained model boundary checks: changing invalid values cannot affect
        # valid results. A zero-initialized head would make this test vacuous.
        gen=torch.Generator(device='cuda').manual_seed(20260928)
        x=torch.randn((3,32,206),generator=gen,device='cuda')
        relative=torch.linspace(-1,1,32,device='cuda')[None].expand(3,-1)
        mask=torch.zeros((3,32),dtype=torch.bool,device='cuda')
        mask[0,:]=True;mask[1,7:25]=True
        changed=x.clone();changed[~mask]=1e4
        with torch.no_grad():
            a=net(x,relative,mask)
            b=net(changed,relative,mask)
        maxdiff=max(float((u[mask]-v[mask]).abs().max()) for u,v in zip(a,b))
        empty_zero=all(bool((u[2]==0).all()) and bool(torch.isfinite(u).all()) for u in a)
        assert maxdiff<1e-6 and empty_zero
        combined=old.copy();combined[:,2]=pred
        default=base.dmask(truth)[:,2]
        populations={}
        for label,m in [('true_default',default),('true_nondefault',~default)]:
            s0=base.metric(truth[m],old[m])['SW'];s1=base.metric(truth[m],combined[m])['SW']
            populations[label]={'n':int(m.sum()),'old_SW':s0,'new_SW':s1,
                               'SW_delta':s1-s0,'TOTAL_contribution':float(.35*m.mean()*(s1-s0))}
        perwell=[]
        for w in np.unique(wells):
            m=wells==w
            before=base.metric(truth[m],old[m]);after=base.metric(truth[m],combined[m])
            perwell.append({'well':int(w),'n':int(m.sum()),'SW_delta':after['SW']-before['SW'],
                            'TOTAL_delta':after['TOTAL']-before['TOTAL']})
        n=np.array([x['n'] for x in perwell]);d=np.array([x['TOTAL_delta'] for x in perwell])
        ix=np.random.default_rng(20260928).integers(0,len(n),(10000,len(n)))
        boot=(n[ix]*d[ix]).sum(1)/n[ix].sum(1)
        reports.append({'mode':row['mode'],'report':row['report'],'scores_with_fixed_POR_PERM':base.metric(truth,combined),
                        'all144665predictions_reloaded_exactly':True,'padding_invariance_max_error':maxdiff,
                        'fully_empty_window_finite_zero':empty_zero,'population_decomposition':populations,
                        'per_well':perwell,'well_wins':int((d>1e-12).sum()),'well_losses':int((d< -1e-12).sum()),
                        'exploratory_selection_unadjusted_95':np.quantile(boot,[.025,.975]).tolist(),
                        'seconds_full_validation':duration})
        preds[row['mode']]=pred;models[row['mode']]=net.cpu()
    winner=max(snap['runs'],key=lambda x:x['report']['target_score'])
    improved=bool(winner['report']['target_score']>baseline['SW'])
    manifest=read(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/model_manifest.json')
    choices=[dict(x) for x in manifest['choices']]
    if improved:
        choices[2]={'target':'SW','architecture':winner['mode'],'model_module':'independent_sw_scratch_v10',
                    'checkpoint':'outputs/'+winner['id']+'/model_best.pt',
                    'target_score':winner['report']['target_score'],'best_epoch':winner['report']['best_epoch']}
    # No repetitive replay of unchanged old assembly when both new candidates
    # lose: it was independently verified in v8 and v9 and remains immutable.
    actual=old.copy();assembly_proof='reused unchanged verified v8 assembly'
    if improved:
        previous=torch.load(REFERENCE/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
        candidate=models[winner['mode']]
        ptrs={p.data_ptr() for p in candidate.parameters()}
        stored=[]; private_models=[]
        for c in (0,1):
            choice=choices[c];cp=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False)
            model=PriorModel(master,previous,choice['target'],choice['architecture'])
            model.load_state_dict(cp['state_dict'],strict=True)
            addresses={p.data_ptr() for p in model.parameters()}
            assert not addresses&ptrs
            for previous_ptrs in stored:
                assert not addresses&previous_ptrs
            stored.append(addresses)
            private_models.append(model.eval().requires_grad_(False))
        # Keep all CPU models alive during the storage check; transferring them
        # earlier would allow an allocator to reuse a freed address.
        for c,model in enumerate(private_models):
            model.cuda()
            pred,_,_=PriorEvaluator(data,master,c,old)(model)
            assert np.array_equal(pred,old[:,c]);actual[:,c]=pred
            model.cpu()
        actual[:,2]=preds[winner['mode']]
        assembly_proof='three distinct models loaded;old POR/PERM and selected newSW replayed'
    assert np.array_equal(actual[:,:2],old[:,:2])
    score=base.metric(truth,actual)
    delta_between=.35*(snap['runs'][1]['report']['target_score']-snap['runs'][0]['report']['target_score'])
    n=np.array([x['n'] for x in reports[0]['per_well']])
    delta=np.array([b['TOTAL_delta']-a['TOTAL_delta'] for a,b in zip(reports[0]['per_well'],reports[1]['per_well'])])
    ix=np.random.default_rng(20260928).integers(0,len(n),(10000,len(n)))
    interval=np.quantile((n[ix]*delta[ix]).sum(1)/n[ix].sum(1),[.025,.975]).tolist()
    launches=read(STUDY/'launches.json')
    first=min(datetime.fromisoformat(x['started_at']) for x in launches)
    last=max(datetime.fromisoformat(x['report']['finished_at']) for x in snap['runs'])
    report={'scores':score,'reference_scores':baseline,'delta_reference':score['TOTAL']-baseline['TOTAL'],
            'selected_new_SW':improved,'best_new_SW_architecture':winner['mode'],
            'best_new_SW_score':winner['report']['target_score'],'choices':choices,
            'assembly_proof':assembly_proof,'POR_PERM_unchanged_exactly':True,
            'candidate_diagnostics':reports,'architecture_comparison':{'bigru_minus_tcn_TOTAL_equivalent':delta_between,
                'well_wins':int((delta>1e-12).sum()),'well_losses':int((delta< -1e-12).sum()),
                'exploratory_selection_unadjusted_95':interval,
                'limitations':'different parameter count,precision and receptive range;single seed;reused developer fold'},
            'timing':{'two_runs_launch_to_last_finish_wall_seconds':(last-first).total_seconds(),
                     'parallel_jobs':2,'single_run_seconds_include':'preprocessing,training,validation,restore',
                     'final_replay_and_diagnostic_seconds':time.perf_counter()-t0},
            'complete_five_folds':False,'remaining_folds_started':False,'old_delivery_unchanged':True,
            'finished_at':datetime.now().astimezone().isoformat()}
    atomic_json(STUDY/'assembled_report.json',report)
    recipe={'scores':score,'choices':choices,'shared_parameters_across_targets':False,
            'selected_new_SW':improved,'preprocessing_source':str(SOURCE.relative_to(ROOT)),
            'development_selection':True,'complete_five_folds':False,'old_delivery_unchanged':True}
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    if improved:
        dest=ROOT/'outputs/pure_dl_independent_sw_v10_selected_fold5'
        dest.mkdir(exist_ok=False)
        atomic_array(dest/'fold_5_pred.npy',actual)
        atomic_json(dest/'fold_5_report.json',dict(report,fold=5,screening_fold=5))
        atomic_json(dest/'CURRENT_RECIPE.json',recipe)
        atomic_json(dest/'runtime_status.json',{'status':'complete','atlas_current_trial':False})
        (dest/'AGENTS.md').write_text('# 固定三独立模型\n维护PROJECTS.md；不逐井选模型，不自动补折，不覆盖旧交付。\n',encoding='utf-8')
        (dest/'README.md').write_text('# 固定三模型验证\n证据见fold_5_report.json；开发集选择，非完整OOF。\n',encoding='utf-8')
        (dest/'PROJECTS.md').write_text('# 状态\n已完成：独立模型合并验证。未完成：跨折泛化、OOF>82。计划：明确授权后验证冻结配方；不自动补折。\n',encoding='utf-8')
    lines=['# SW从头训练：结果','', '|结构|参数|初始SW|最好SW|最佳/停止轮次|替换诊断TOTAL|',
           '|---|---:|---:|---:|---|---:|']
    for x in snap['runs']:
        r=x['report']
        lines.append(f"|{x['mode']}|{r['trainable_parameters']}|{r['initial_target_score']:.6f}|{r['target_score']:.6f}|{r['best_epoch']}/{r['stopped_epoch']}|{r['reference_replacement_total']:.6f}|")
    lines+=['',f"当前固定配方TOTAL={score['TOTAL']:.12f}，相对旧配方{report['delta_reference']:+.12f}；使用新SW={improved}。",
            '两个新网络均从磁盘重载，所有144665点精确复现。训练后有效段/填充检查通过。',
            '从训练集先验初始化的提升不能等同于超过旧SW；旧模型使用预训练，不构成随机起点匹配对照。',
            '两结构的参数、精度和感受范围存在差别，不把分差全归因于某一算子。每井/默认段分解与未校正选择的探索性区间见assembled_report.json。',
            '不追加本批搜索或恢复旧路线。完整OOF>82与独立新井验证仍未完成。',
            '后续应先依据错误分解决定目标专用表示或状态/连续头分离，不能默认更宽或更多训练一定有效。',
            '图谱说明由本report脚本的record维护；若重建主curate.py，须再运行本脚本--record再build_atlas.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines),encoding='utf-8')
    note='\n\n## '+report['finished_at']+' SW独立冷启动v10完成\n'+f"固定配方TOTAL={score['TOTAL']:.12f}，相对旧{report['delta_reference']:+.12f}；采用新SW={improved}。"+'\n两候选真实训练/恢复及磁盘核验完成；无新补折或交付变更。结果：results/deep_learning/independent_sw_scratch_v10_20260928/FINAL.md。尚未证明独立新井增益。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:
            f.write(note)
    print('FINAL_SUMMARY',json.dumps({k:v for k,v in report.items() if k!='candidate_diagnostics'},ensure_ascii=False),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--final',action='store_true')
    args=ap.parse_args();snap=snapshot()
    if args.record:
        record(snap)
    if args.final:
        finalize(snap)
    else:
        for x in snap['runs']:
            x.pop('config',None)
        print(json.dumps(snap,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
