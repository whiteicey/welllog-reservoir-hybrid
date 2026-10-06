"""Traceable SW shared/split comparison; --final independently reloads candidates."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,json,sys,time,traceback
from pathlib import Path
from datetime import datetime
import numpy as np
from launch_independent_sw_separation_v11 import ROOT,WORK,STUDY,MODES,active


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else None


def snapshot():
    rows=[]
    for mode in MODES:
        p=ROOT/'outputs'/f'pure_dl_independent_sw_v11_{mode}80_fold5'
        h=read(p/'validation_history.json') or []
        rows.append({'mode':mode,'id':p.name,'status':read(p/'runtime_status.json'),
                     'config':read(p/'config.json'),'initial':read(p/'initial_verification.json'),
                     'final':read(p/'fold_5_report.json'),'last':h[-1] if h else None,
                     'best_trained':max(h,key=lambda x:x['target_score']) if h else None,
                     'failure':read(p/'failure.json')})
    return {'observed_at':datetime.now().astimezone().isoformat(),'runs':rows,
            'all_complete':all(r['final'] is not None for r in rows),
            'active_pids':[r['pid'] for r in active()]}


def record(snap):
    from train_dual8222_heads80 import atomic_json
    atomic_json(STUDY/'progress.json',snap)
    path=ROOT/'docs/deep_learning/atlas/curation.json'
    cur=read(path)
    for row in snap['runs']:
        final,last,status=row['final'],row['last'],row['status'] or {}
        if final:
            conclusion=f"完成SW={final['target_score']:.9f}，旧SW差{final['delta_old_SW']:+.9f}；best{final['best_epoch']}/stop{final['stopped_epoch']}。"
        elif last:
            conclusion=f"训练中epoch{last['epoch']}，bestSW={last['best_target_score']:.9f}；非最终恢复成绩。"
        else:conclusion=status.get('status','prepared')+'，尚无成绩。'
        node={'title':'SW内部表征·'+row['mode'],'summary':'两套48维TCN和92273参数，相同初始权重，仅改变门控/密度读共享均值还是各自编码器。',
              'stage':'loss','highlight':status.get('status')=='running','conclusion':conclusion}
        if final:
            result={'scope':'第五折 · SW单目标','value':final['target_score'],'scores':{'SW':final['target_score']},
                    'source':'outputs/'+row['id']+'/fold_5_report.json','kind':'target','target':'SW','fold':5}
            node.update(status='recorded',primary=result,results=[result],evidence=[result['source'],'outputs/'+row['id']+'/README.md'])
        cur['nodes'][row['id']]=node
    edges=[{'source':'pure_dl_independent_sw_v10_tcn80_fold5','target':snap['runs'][0]['id'],'type':'evolution',
            'note':'研究延续，无权重继承；双48维编码器共享表示对照。'},
           {'source':snap['runs'][0]['id'],'target':snap['runs'][1]['id'],'type':'control',
            'note':'同参数/初始化/输入/损失/预算，仅变化任务与表征的连接方式。'}]
    for edge in edges:
        edge['evidence']='results/deep_learning/independent_sw_separation_v11_20260928/METHOD.md'
        if not any(e['source']==edge['source'] and e['target']==edge['target'] and e['type']==edge['type'] for e in cur['edges']):cur['edges'].append(edge)
    atomic_json(path,cur)


def finalize(snap):
    if not snap['all_complete'] or active():raise RuntimeError('Finish both runs before evaluation replay')
    if (STUDY/'comparison.json').exists():raise RuntimeError('Final comparison already exists')
    import torch
    from train_dual8222_heads80 import prepare_data,SOURCE,atomic_json,atomic_array
    from train_independent_sw_v9 import Evaluator,FIXED
    from independent_sw_separation_v11 import Network
    from multiscale_sequence import window_indices
    from gpu_job_guard_v2 import check_slot
    import train_invariant as base
    print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    started=time.perf_counter()
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    data=prepare_data(master)
    old=np.load(FIXED);truth=data['truth'][data['validation']];wells=data['wells'][data['validation']]
    default=base.dmask(truth)[:,2];baseline=base.metric(truth,old)
    ev=Evaluator(data,master,2,old)
    def sw_score(y,p):
        return float(100*np.maximum(0,1-np.abs(p-y)/(.05*(np.abs(y)+1e-5))).mean())
    @torch.no_grad()
    def replay(net):
        n=len(truth)
        pred=np.full(n,np.nan);gates=np.full(n,np.nan);conditional=np.full(n,np.nan)
        coverage=np.zeros(n,np.int8)
        net.eval()
        for begin in range(0,len(ev.centers),32):
            cc=torch.tensor(ev.centers[begin:begin+32],device='cuda')
            anchor=torch.tensor(ev.anchors[begin:begin+32],device='cuda')
            ix,valid=window_indices(cc,data['lower'],data['upper'],anchors=anchor)
            relative=torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=net(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
            rows=data['raw_to_valid'][ix[:,64:192].cpu().numpy()]
            use=valid[:,64:192].cpu().numpy() & (rows>=0)
            loc=ev.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            mask=torch.zeros_like(valid);mask[:,64:192]=torch.as_tensor(use,device='cuda')
            g,p,_,_=net.distributions(output,mask)
            joint=torch.cat([g[:,None],(1-g[:,None])*p],-1)
            pred[loc]=ev.actions[(joint@ev.utilities).argmax(-1).cpu().numpy()]
            conditional[loc]=ev.actions[(p@ev.utilities[1:]).argmax(-1).cpu().numpy()]
            gates[loc]=g.cpu().numpy();coverage[loc]+=1
        assert (coverage==1).all() and np.isfinite(pred).all() and np.isfinite(gates).all()
        return pred,gates,conditional
    reports=[];models={};predictions={}
    for row in snap['runs']:
        path=ROOT/'outputs'/row['id']/'model_best.pt'
        cp=torch.load(path,map_location='cpu',weights_only=False)
        net=Network(row['mode'],cp['state_dict']['centers'],cp['target_median'],cp['target_scale'])
        net.load_state_dict(cp['state_dict'],strict=True);net.eval().requires_grad_(False).cuda()
        torch.cuda.synchronize();tick=time.perf_counter()
        pred,g,conditional=replay(net)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        assert np.array_equal(pred,np.load(path.parent/'channel_pred.npy'))
        score=sw_score(truth[:,2],pred)
        assert abs(score-row['final']['target_score'])<1e-10
        # Trained (not zero-init) boundary checks.
        gen=torch.Generator(device='cuda').manual_seed(20261111)
        x=torch.randn((3,32,206),device='cuda',generator=gen)
        relative=torch.linspace(-1,1,32,device='cuda')[None].expand(3,-1)
        mask=torch.zeros((3,32),device='cuda',dtype=torch.bool);mask[0]=True;mask[1,7:25]=True
        modified=x.clone();modified[~mask]=1e4
        with torch.no_grad():a=net(x,relative,mask);b=net(modified,relative,mask)
        err=max(float((u[mask]-v[mask]).abs().max()) for u,v in zip(a,b))
        empty=all(bool((u[2]==0).all()) and bool(torch.isfinite(u).all()) for u in a)
        assert err<1e-6 and empty
        combined=old.copy();combined[:,2]=pred
        populations={}
        for label,m in [('true_default',default),('true_nondefault',~default)]:
            before=sw_score(truth[m,2],old[m,2]);after=sw_score(truth[m,2],pred[m])
            populations[label]={'n':int(m.sum()),'old_SW':before,'new_SW':after,
                               'SW_delta':after-before,'TOTAL_contribution':float(.35*m.mean()*(after-before))}
        perwell=[]
        for w in np.unique(wells):
            m=wells==w;delta=sw_score(truth[m,2],pred[m])-sw_score(truth[m,2],old[m,2])
            perwell.append({'well':int(w),'n':int(m.sum()),'SW_delta':delta,'TOTAL_delta':.35*delta})
        eps=1e-7;gc=np.clip(g,eps,1-eps)
        state={'default_probability_BCE':float(-(default*np.log(gc)+(~default)*np.log1p(-gc)).mean()),
               'default_probability_Brier':float(((g-default)**2).mean()),
               'default_threshold_05_accuracy':float(((g>=.5)==default).mean()),
               'nondefault_recall_threshold_05':float((g[~default]<.5).mean()),
               'gate_mean_true_nondefault':float(g[~default].mean()),
               'final_default_action_rate_true_nondefault':float(np.isclose(pred[~default],99.9,atol=1e-6,rtol=0).mean()),
               'conditional_decoder_SW_on_true_nondefault':sw_score(truth[~default,2],conditional[~default]),
               'conditional_decoder_note':'diagnostic only:gate mass set0,labels used solely to restrict evaluation subset;not deployable label-based routing'}
        atomic_json(path.parent/'reload_and_state_diagnostic.json',{'all144665predictions_exact':True,'padding_error':err,
                    'empty_window_finite_zero':empty,'state_and_numeric_diagnostic':state,'seconds':seconds})
        reports.append({'mode':row['mode'],'report':row['final'],'replacement_scores':base.metric(truth,combined),
                        'populations':populations,'per_well':perwell,'state_and_numeric_diagnostic':state,
                        'well_wins':sum(v['SW_delta']>1e-12 for v in perwell),'well_losses':sum(v['SW_delta']< -1e-12 for v in perwell),
                        'reload_exact':True,'seconds_full_validation':seconds})
        predictions[row['mode']]=pred;models[row['mode']]=net.cpu()
    shared,split=reports
    delta=.35*(split['report']['target_score']-shared['report']['target_score'])
    wd=np.array([b['TOTAL_delta']-a['TOTAL_delta'] for a,b in zip(shared['per_well'],split['per_well'])])
    counts=np.array([x['n'] for x in shared['per_well']])
    ix=np.random.default_rng(20261111).integers(0,len(counts),(10000,len(counts)))
    ci=np.quantile((counts[ix]*wd[ix]).sum(1)/counts[ix].sum(1),[.025,.975]).tolist()
    assert len({r['config']['initialization']['initial_state_sha256'] for r in snap['runs']})==1
    assert len({r['config']['trainable_parameters'] for r in snap['runs']})==1
    winner=max(snap['runs'],key=lambda r:r['final']['target_score'])
    improved=bool(winner['final']['target_score']>baseline['SW'])
    manifest=read(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/model_manifest.json')
    choices=[dict(x) for x in manifest['choices']]
    combined=old.copy();assembly='reuse unchanged verified v8 assembly;newSW candidates lost'
    if improved:
        from independent_channels_v8 import Network as PriorModel
        from train_independent_channels_v8 import REFERENCE,Evaluator as PriorEvaluator
        previous=torch.load(REFERENCE/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
        choices[2]={'target':'SW','architecture':winner['mode'],'checkpoint':'outputs/'+winner['id']+'/model_best.pt',
                    'model_module':'independent_sw_separation_v11','best_epoch':winner['final']['best_epoch']}
        three=[None,None,models[winner['mode']]]
        for c in (0,1):
            item=choices[c];cp=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)
            net=PriorModel(master,previous,item['target'],item['architecture'])
            net.load_state_dict(cp['state_dict'],strict=True);three[c]=net.eval().requires_grad_(False)
        pointers=[{p.data_ptr() for p in m.parameters()} for m in three]
        assert not pointers[0]&pointers[1] and not pointers[0]&pointers[2] and not pointers[1]&pointers[2]
        for c in (0,1):
            net=three[c].cuda();p,_,_=PriorEvaluator(data,master,c,old)(net)
            assert np.array_equal(p,old[:,c]);combined[:,c]=p;net.cpu()
        combined[:,2]=predictions[winner['mode']]
        assembly='three independently loaded private models;exact predictions;no shared parameter storage'
    assert np.array_equal(combined[:,:2],old[:,:2])
    scores=base.metric(truth,combined)
    launches=read(STUDY/'launches.json')
    start=min(datetime.fromisoformat(x['started_at']) for x in launches)
    finish=max(datetime.fromisoformat(x['final']['finished_at']) for x in snap['runs'])
    report={'reference_scores':baseline,'kept_scores':scores,'selected_new_SW':improved,'best_new_architecture':winner['mode'],
            'choices':choices,'assembly':assembly,'candidates':reports,
            'matched_split_minus_shared':{'SW':delta/.35,'TOTAL_equivalent':delta,'well_wins':int((wd>1e-12).sum()),
                'well_losses':int((wd< -1e-12).sum()),'selection_unadjusted_exploratory95':ci,
                'same_parameters':True,'same_initial_state':True},
            'timing':{'two_runs_wall_seconds':(finish-start).total_seconds(),'max_parallel_training':2,
                'individual_seconds_include':'preprocessing/training/validation/bestrestore/traininggradientdiagnostic',
                'final_diagnostics_seconds':time.perf_counter()-started},
            'complete_five_folds':False,'remaining_folds_started':False,'old_delivery_unchanged':True,
            'limitations':'reused developer fold;single seed;fixed oldQ and grids;gradient sample128trainingwindows;no resume-equivalence test',
            'finished_at':datetime.now().astimezone().isoformat()}
    atomic_json(STUDY/'comparison.json',report)
    recipe={'scores':scores,'choices':choices,'selected_new_SW':improved,'independent_targets':True,
            'preprocessing_source':str(SOURCE.relative_to(ROOT)),'complete_five_folds':False,'old_delivery_unchanged':True}
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    if improved:
        dest=ROOT/'outputs/pure_dl_independent_sw_v11_selected_fold5';dest.mkdir(exist_ok=False)
        atomic_array(dest/'fold_5_pred.npy',combined);atomic_json(dest/'fold_5_report.json',dict(report,fold=5,screening_fold=5,scores=scores))
        atomic_json(dest/'CURRENT_RECIPE.json',recipe);atomic_json(dest/'runtime_status.json',{'status':'complete','atlas_current_trial':False})
        (dest/'AGENTS.md').write_text('# 固定三模型\n维护PROJECTS.md；固定配方不逐井选模型，原数据及交付不改，无自动补折。\n',encoding='utf-8')
        (dest/'README.md').write_text('# v11固定三模型\n本目录仅开发验证结果，来源见fold_5_report.json；非完整OOF。\n',encoding='utf-8')
        (dest/'PROJECTS.md').write_text('# 进度\n已完成共同恢复，未完成完整OOF与独立泛化；无自动补折，达成后再创建独立交付。\n',encoding='utf-8')
    lines=['# SW内部共享/独立表征v11：结果','', '|方案|参数|SW|相对旧SW|最佳/停止轮次|','|---|---:|---:|---:|---|']
    for row in snap['runs']:
        r=row['final'];lines.append(f"|{row['mode']}|{r['trainable_parameters']}|{r['target_score']:.9f}|{r['delta_old_SW']:+.9f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['',f"匹配split-shared: SW {delta/.35:+.9f},等价TOTAL {delta:+.9f}。",
            f"保留TOTAL={scores['TOTAL']:.12f}，采用新SW={improved}；{assembly}。",
            '两套TCN同参数和逐字节初始状态。共享组两头均使用两个编码器平均，分离组各使用一个。有效容量分配随连接变化，不能把差别唯一归因于梯度冲突。',
            '梯度诊断仅固定训练样本，见各fold_5_report.json。默认/数值与条件解码诊断、逐井差异见comparison.json；条件解码使用真标签筛子集，仅诊断，未选为推理方案。',
            '两个新模型从磁盘重载，全部144665点精确复现，训练后填充/空窗口检查通过。',
            '未自动启动第三项、旧训练或补折，未更改旧交付。完整OOF>82未完成。',
            '图谱新增节点由本报告--record维护curation.json后build_atlas.py；主curate.py若重跑，需再执行本记录器。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines),encoding='utf-8')
    note='\n\n## '+report['finished_at']+' SW共享/分離v11完成\n'+f"sharedSW={shared['report']['target_score']:.9f}，splitSW={split['report']['target_score']:.9f}；保留TOTAL={scores['TOTAL']:.12f}，使用新SW={improved}。同参/初始化已核对；诊断与磁盘恢复完成，非独立OOF结论。\n详见results/deep_learning/independent_sw_separation_v11_20260928/FINAL.md。无新补折或交付变更；下一步尚未启动。\n"
    for p in [WORK/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    for row in snap['runs']:
        for fn in ['PROJECTS.md','README.md']:
            with (ROOT/'outputs'/row['id']/fn).open('a',encoding='utf-8') as f:f.write(note)
    print('FINAL',json.dumps({k:v for k,v in report.items() if k!='candidates'},ensure_ascii=False),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');ap.add_argument('--final',action='store_true')
    args=ap.parse_args();snap=snapshot()
    if args.record:record(snap)
    if args.final:finalize(snap)
    else:
        for r in snap['runs']:r.pop('config',None)
        print(json.dumps(snap,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if STUDY.exists():
            (STUDY/('report_failure_'+datetime.now().strftime('%H%M%S')+'.json')).write_text(
                json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),'training_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')
        raise
