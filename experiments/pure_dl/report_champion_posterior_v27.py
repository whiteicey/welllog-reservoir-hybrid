"""Status and final original-population comparison for the six v27 candidates."""
from pathlib import Path
from datetime import datetime
import argparse,gc,hashlib,json,time
import numpy as np
import torch
from champion_posterior_v27 import ROOT,STUDY,PREVIOUS,TARGETS,MODES,SELECTED,Network,Evaluator,load_data,reference_prediction
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train_champion_posterior_v27 import factor_scope
from launch_champion_posterior_v27 import jobs,append_once
from gpu_job_guard_v2 import check_slot
import train_invariant as base

WEIGHTS={'POR':.30,'PERM':.35,'SW':.35}

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def status():
    rows=[]
    for target in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_champion_posterior_v27_{target}_fold5'
        state=read(out/'runtime_status.json') if (out/'runtime_status.json').exists() else dict(status='not_launched')
        report=read(out/'fold_5_report.json') if (out/'fold_5_report.json').exists() else None
        rows.append(dict(target=target,id=out.name,state=state,report=report))
    result=dict(observed_at=datetime.now().astimezone().isoformat(),reference_scores=read(PREVIOUS/'CURRENT_RECIPE.json')['scores'],
        all_complete=all(r['report'] is not None and r['state']['status']=='complete' for r in rows),rows=rows)
    atomic_json(STUDY/'progress.json',result)
    for r in rows:
        brief=dict(target=r['target'],status=r['state']['status'],epoch=r['state'].get('last_completed_epoch'),modes=r['state'].get('modes'))
        if r['report']:brief['scores']=[{k:a[k] for k in ['mode','target_score','best_epoch','stopped_epoch','best_after_update']} for a in r['report']['arms']]
        print(json.dumps(brief,ensure_ascii=False),flush=True)
    return result


def finish(progress):
    if not progress['all_complete']:raise RuntimeError('All six trials must complete before final selection')
    if jobs():raise RuntimeError('Final replay requires no active training jobs')
    prior=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in prior['scores'].items())
    records=[];selections=[];matched=[];combined=reference.copy();starts=[];ends=[];costs={}
    for group in progress['rows']:
        t=group['target'];c=TARGETS.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        starts.append(datetime.fromisoformat(read(out/'launch.json')['started_at']));ends.append(datetime.fromisoformat(group['report']['finished_at']))
        costs[t]=dict(seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            epoch_median_seconds=float(np.median([e['seconds'] for e in group['report']['epoch_timing']])),
            scope='both candidates share frozen source computation in this target process; not per-arm elapsed time')
        target_records=[]
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json')
            pred=np.load(arm/'channel_pred.npy');updated=np.load(arm/'best_updated_pred.npy')
            d=diagnostics(truth,reference,pred,wells,t);u=diagnostics(truth,reference,updated,wells,t)
            assert abs(d['scores'][t]-rep['target_score'])<1e-9
            assert abs(u['scores'][t]-rep['best_after_update']['target_score'])<1e-9
            if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,c])
            row=dict(target=t,mode=m,factor=SELECTED[t],report=rep,selected=d,post_update=u,
                checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),prediction=str((arm/'channel_pred.npy').relative_to(ROOT)))
            records.append(row);target_records.append(row)
        a,b=target_records
        matched.append(dict(target=t,pair_minus_self_selected=b['selected']['channel_delta']-a['selected']['channel_delta'],
            pair_minus_self_post_update=b['post_update']['channel_delta']-a['post_update']['channel_delta'],
            note='self/pair matched information contrast; score relative to v26 also includes a new correction head'))
        winner=max(target_records,key=lambda row:row['selected']['scores'][t])
        adopt=winner['selected']['scores'][t]>refs[t]+1e-9
        selections.append(dict(target=t,mode=winner['mode'] if adopt else None,adopted=adopt,
            target_score=winner['selected']['scores'][t],checkpoint=winner['checkpoint'] if adopt else next(c['checkpoint'] for c in prior['choices'] if c['target']==t),
            best_epoch=winner['report']['best_epoch'],additional_parameters=winner['report']['trainable_parameters'] if adopt else 0,
            verification='in-run full restoration; unchanged columns reuse prior disk-replay evidence'))
        combined[:,c]=np.load(ROOT/winner['prediction'])
    replay_start=time.perf_counter();choices=[dict(c) for c in prior['choices']]
    if any(s['adopted'] for s in selections):
        torch.set_num_threads(3);check_slot(min_free_gib=3.)
        torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for s in selections:
            if not s['adopted']:continue
            t=s['target'];m=s['mode'];c=TARGETS.index(t)
            cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            model=Network(t,master,modes=(m,)).cuda();model.heads[m].load_state_dict(cp['delta_state'],strict=True)
            pred,score,_=Evaluator(data,master,c,reference).many(model,(m,))[m]
            assert np.array_equal(pred,combined[:,c]) and abs(score-s['target_score'])<1e-9
            s.update(all144665_fresh_disk_replay_exact=True,verification='fresh source+retained v26 head+new delta reconstruction',factor_scope=factor_scope(model,data,m))
            combined[:,c]=pred
            for choice in choices:
                if choice['target']==t:
                    choice.update(key='v27_'+t+'_'+m,mode=m,factor=SELECTED[t],architecture='posterior_conditioned_frozen_factor',
                        checkpoint=s['checkpoint'],model_module='champion_posterior_v27',
                        builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state])',
                        target_score=score,best_epoch=s['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
            model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert abs(scores['TOTAL']-sum(WEIGHTS[s['target']]*s['target_score'] for s in selections))<1e-9
    perwell=[]
    for w in np.unique(wells):
        mask=wells==w;delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],reference[mask])['TOTAL']
        perwell.append(dict(well=int(w),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([r['n'] for r in perwell]);deltas=np.array([r['TOTAL_delta'] for r in perwell])
    indices=np.random.default_rng(20261010).integers(0,len(perwell),(10000,len(perwell)))
    boot=(counts[indices]*deltas[indices]).sum(1)/counts[indices].sum(1)
    result=native(dict(fold=5,rows=144665,reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=records,selections=selections,matched_comparisons=matched,per_well=perwell,
        well_wins=sum(x['TOTAL_delta']>0 for x in perwell),well_losses=sum(x['TOTAL_delta']<0 for x in perwell),
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]),
        all_six_complete=True,all_selected_restored_in_run=True,actual_combined_prediction_scored=True,
        new_adopted_targets_disk_replayed=True,unchanged_targets_not_fresh_replayed=True,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(prior);recipe.update(scores=native(scores),choices=native(choices),reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(maximum_parallel_target_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),targets=costs,
        final_replay_seconds=time.perf_counter()-replay_start,scope='process intervals include preparation/training/restoration and staged-launch gaps; implementation time excluded')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v27：基于80.30的后验条件头优化','',f"完成时间：{result['finished_at']}",'',
        f"基线TOTAL={refs['TOTAL']:.12f}，实际最终TOTAL={scores['TOTAL']:.12f}，差值={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 六项已完成结果','',
        '|目标|模式|新增参数|训练后最好通道分|最终通道分|最佳/停止轮次|',
        '|---|---|---:|---:|---:|---|']
    for r in records:
        t=r['target'];rep=r['report']
        lines.append(f"|{t}|{r['mode']}|{rep['trainable_parameters']}|{r['post_update']['scores'][t]:.9f}|{r['selected']['scores'][t]:.9f}|{rep['best_epoch']}/{rep['stopped_epoch']}|")
    lines+=['','第0轮表示保留基线，不记新增收益。每目标self/pair的参数与初始化一致，差异仅为第二份后验描述来自自身或另一专家。',
        'pair相对self是匹配信息对照；相对v26还增加了一个新小头，不能将全部差值归因于跨专家信息。完整原有参数、已采用的v25/v26修正、路由和原评分保持不变。','',
        '## 最终保留与收益','']
    for s in selections:
        lines.append(f"{s['target']}：{'新增'+s['mode'] if s['adopted'] else '保留v26'}，分数{s['target_score']:.12f}，新增推理参数{s['additional_parameters']}。")
    lines+=['',f"最终{result['well_wins']}井提高、{result['well_losses']}井下降；探索性未校正区间{result['exploratory_selection_unadjusted_TOTAL_95']}。",
        '同一反复开发第五折，六项有限选模，未作独立重复确认，不自动补其他折。不同目标使用固定整模型，不按点/按井选方案。','',
        '## 已验证与未执行','',
        '六项初始预测精确复现当前父配方，实际训练有新头梯度，旧参数不变；训练内最佳恢复全部144665点精确。采用的新目标额外磁盘重建，未变目标复用原验证，实际组合数组调用原metric评分。',
        '其他输出因子、冷启动/扩容、其他折或旧队列没有启动。负结果不进一步扫宽度、系数或种子；仅关闭本次信息输入和小头设置，不泛化为全部后验建模无效。',
        '原80.30及历史交付不覆盖。完整五折>82仍未完成，严格第五折>80.5门槛保持。','',
        '## 文件','',
        'METHOD.md包含预注册结构/损失/信息来源；progress.json、assembled_report.json、timing.json和CURRENT_RECIPE.json保存实际结果。新模块为champion_posterior_v27.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    note=f"## v27基于80.30研究完成\n六项完成，TOTAL={scores['TOTAL']:.12f}，相对父配方{result['delta_reference_TOTAL']:+.12f}。采用"+','.join(s['target']+':'+str(s['mode']) for s in selections)+"。原权重/交付保持，无其他折；完整记录champion_posterior_v27_20260929/FINAL.md。"
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(p,'## v27基于80.30研究完成',note)
    current='# v27已完成\n\n目标：在80.30原模型上检验三个有效头的跨专家后验条件。\n\n已完成：六项训练、当前基线一致性、冻结范围、最终恢复及必要的新目标磁盘重载和实际组合。\n\n'+note+'\n\n未完成：独立种子/数据划分和其他四折确认；不自动延伸本批负候选。\n\n达成后：固定有效配方，再按用户授权讨论完整OOF；原交付保留。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for r in records:
        arm=(ROOT/r['checkpoint']).parent;t=r['target'];rep=r['report']
        part=f"\n\n## 最终结果\n{t}/{r['mode']}：更新后最好{r['post_update']['scores'][t]:.12f}，选定{rep['target_score']:.12f}，最佳{rep['best_epoch']}轮/停止{rep['stopped_epoch']}轮。相对基线{r['selected']['channel_delta']:+.12f}。验证范围与后续边界见研究目录FINAL.md。\n"
        append_once(arm/'README.md','## 最终结果',part)
        (arm/'PROJECTS.md').write_text(current+part,encoding='utf-8')
        (arm.parent/'PROJECTS.md').write_text(current,encoding='utf-8')
    policy='## v27最终研究配方\n'+f"原第五折TOTAL={scores['TOTAL']:.12f}，相对v26{result['delta_reference_TOTAL']:+.12f}。六项有限比较已完成，当前配方见results/deep_learning/champion_posterior_v27_20260929/CURRENT_RECIPE.json。旧模型/交付不改，不恢复负路线或其他折。"
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:append_once(p,'## v27最终研究配方',policy)
    print('FINAL',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,matched=matched,timing=timing)),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--finish',action='store_true');args=ap.parse_args()
    progress=status()
    if args.finish:finish(progress)
