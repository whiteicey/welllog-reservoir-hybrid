"""Finish six routing trials, inspect trained scope, replay only adopted targets."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,sys,time
import numpy as np
import torch
from champion_route_v28 import ROOT,STUDY,PREVIOUS,TARGETS,MODES,Network,Evaluator,load_data,reference_prediction,routing_scope
from run_champion_route_v28 import record
from launch_champion_posterior_v27 import jobs,append_once
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from gpu_job_guard_v2 import check_slot
import train_invariant as base

WEIGHTS={'POR':.3,'PERM':.35,'SW':.35}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    progress=record()
    if not progress['all_complete']:raise RuntimeError('All prespecified trials must finish first')
    if jobs():raise RuntimeError('No final reconstruction while training')
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    ref=base.metric(truth,reference)
    assert all(abs(ref[k]-v)<1e-9 for k,v in previous['scores'].items())
    records=[];selections=[];comparisons=[];combined=reference.copy();starts=[];ends=[];costs={}
    for group in progress['rows']:
        t=group['target'];c=TARGETS.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        starts.append(datetime.fromisoformat(read(out/'launch.json')['started_at']))
        ends.append(datetime.fromisoformat(group['report']['finished_at']))
        costs[t]=dict(process_seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            epoch_median_seconds=float(np.median([x['seconds'] for x in group['report']['epoch_timing']])),
            scope='two candidates share fixed forward; not independent per-candidate time')
        own=[]
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json')
            pred=np.load(arm/'channel_pred.npy');updated=np.load(arm/'best_updated_pred.npy')
            d=diagnostics(truth,reference,pred,wells,t);u=diagnostics(truth,reference,updated,wells,t)
            assert abs(d['scores'][t]-rep['target_score'])<1e-9
            assert abs(u['scores'][t]-rep['best_after_update']['target_score'])<1e-9
            assert abs(d['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
            if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,c])
            row=dict(target=t,mode=m,report=rep,selected=d,post_update=u,
                checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),prediction=str((arm/'channel_pred.npy').relative_to(ROOT)))
            records.append(row);own.append(row)
        winner=max(own,key=lambda row:row['selected']['scores'][t])
        adopt=winner['selected']['scores'][t]>ref[t]+1e-9
        selections.append(dict(target=t,mode=winner['mode'] if adopt else None,adopted=adopt,
            target_score=winner['selected']['scores'][t],best_epoch=winner['report']['best_epoch'],
            checkpoint=winner['checkpoint'] if adopt else next(c['checkpoint'] for c in previous['choices'] if c['target']==t),
            added_parameters=winner['report']['trainable_parameters'] if adopt else 0))
        combined[:,c]=np.load(ROOT/winner['prediction'])
        comparisons.append(dict(target=t,conditional_minus_joint_selected=own[1]['selected']['channel_delta']-own[0]['selected']['channel_delta'],
            conditional_minus_joint_after_update=own[1]['post_update']['channel_delta']-own[0]['post_update']['channel_delta']))
    # Small trained-model scope checks for all arms, full disk prediction only
    # for positive adopted targets. Does not use validation labels for training.
    start=time.perf_counter();torch.set_num_threads(3);check_slot(min_free_gib=3.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    data,master=load_data();assert np.array_equal(data['truth'][data['validation']],truth)
    assert np.array_equal(data['wells'][data['validation']],wells)
    choices=[dict(x) for x in previous['choices']];replayed=[]
    for t in TARGETS:
        model=Network(t,master).cuda()
        for row in [r for r in records if r['target']==t]:
            path=(ROOT/row['checkpoint']).parent/'model_best_updated.pt'
            cp=torch.load(path,map_location='cpu',weights_only=False)
            assert cp['target']==t and cp['mode']==row['mode']
            model.heads[row['mode']].load_state_dict(cp['delta_state'],strict=True)
            row['trained_checkpoint_scope']=routing_scope(model,data,row['mode'])
        s=next(s for s in selections if s['target']==t)
        if s['adopted']:
            m=s['mode'];cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            model.heads[m].load_state_dict(cp['delta_state'],strict=True)
            pred,score,_=Evaluator(data,master,TARGETS.index(t),reference).many(model,(m,))[m]
            assert np.array_equal(pred,combined[:,TARGETS.index(t)]) and abs(score-s['target_score'])<1e-9
            combined[:,TARGETS.index(t)]=pred;s['all144665_fresh_disk_replay_exact']=True
            s['selected_scope']=routing_scope(model,data,m);replayed.append(t)
            for choice in choices:
                if choice['target']==t:
                    choice.update(key='v28_'+t+'_'+m,mode=m,factor='expert_routing',architecture='bounded_internal_expert_routing',
                        checkpoint=s['checkpoint'],model_module='champion_route_v28',
                        builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state])',
                        target_score=score,best_epoch=s['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        else:s['verification']='unchanged target uses prior full disk replay; only small trained-v28 scope check repeated'
        model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
    del data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert abs(scores['TOTAL']-sum(WEIGHTS[s['target']]*s['target_score'] for s in selections))<1e-9
    perwell=[]
    for w in np.unique(wells):
        mask=wells==w
        delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],reference[mask])['TOTAL']
        perwell.append(dict(well=int(w),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([x['n'] for x in perwell]);deltas=np.array([x['TOTAL_delta'] for x in perwell])
    ix=np.random.default_rng(20261010).integers(0,16,(10000,16))
    boot=(counts[ix]*deltas[ix]).sum(1)/counts[ix].sum(1)
    result=native(dict(fold=5,rows=144665,reference_scores=ref,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-ref['TOTAL'],
        candidates=records,selections=selections,matched_comparisons=comparisons,per_well=perwell,
        well_wins=sum(x['TOTAL_delta']>0 for x in perwell),well_losses=sum(x['TOTAL_delta']<0 for x in perwell),
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]),
        all_six_complete=True,actual_combined_prediction_scored=True,new_target_full_disk_replays=replayed,
        all_six_trained_checkpoints_small_scope_checked=True,complete_five_folds=False,remaining_folds_started=False,
        finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=native(choices),reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(max_parallel_target_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),targets=costs,
        final_scopes_and_replay_seconds=time.perf_counter()-start,
        scope='target process times include preparation/training/restore; overall interval includes staged gaps; implementation and final diagnostics separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v28：内部专家路由更新范围对照','',f"完成：{result['finished_at']}",'',
        f"原配方TOTAL={ref['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 六项实测结果','',
        '|目标|模式|新增参数|训练后最高通道分|最终通道分|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
    for row in records:
        t=row['target'];rep=row['report']
        lines.append(f"|{t}|{row['mode']}|{rep['trainable_parameters']}|{row['post_update']['scores'][t]:.9f}|{rep['target_score']:.9f}|{rep['best_epoch']}/{rep['stopped_epoch']}|")
    lines+=['','第0轮回退只表示保留当前源模型；不计新增训练收益。各模式同结构同初始权重，但默认分布是否参与更新会改变梯度，比较的是这项范围选择。','',
        '## 实际采用与收益分解','']
    for s in selections:
        t=s['target'];lines.append(f"{t}：{'采用'+s['mode'] if s['adopted'] else '保留v26'}，通道{s['target_score']:.12f}，新增推理参数{s['added_parameters']}。")
        if s['adopted']:
            d=next(x for x in records if x['target']==t and x['mode']==s['mode'])['selected']
            lines.append('默认/非默认贡献：'+json.dumps(native(d['populations']),ensure_ascii=False))
    lines+=['',f"组合{result['well_wins']}井升、{result['well_losses']}井降；未校正选择的探索性区间{result['exploratory_selection_unadjusted_TOTAL_95']}。",
        '同一第五折已反复开发，六项选模，无独立种子或新划分确认；微小分数变化不自动触发补折。','',
        '## 结构与验证','',
        '每目标只增加319→32→1的10273参数小头，读取两个128维隐藏向量和63幅值。零输出起点，log(a1/a2)增量限制±0.25。joint同时改变最终默认概率与连续条件专家占比；conditional只改变后者。各专家输出参数、旧修正和原路由全部固定。',
        '六项初始全144665点精确，实际训练有新头梯度且旧状态固定。每项选定预测训练内全量恢复；六份更新后检查点另做少量训练窗口范围检查。conditional最终门控精确固定，混合分量位置/尺度不变。',
        '新采纳目标独立磁盘重建并全量推理，未变目标复用已验证预测，实际三列合并调用原metric；新重载目标：'+str(replayed)+'。',
        '前向调用复用v27的原v26加载器，不执行v27负候选头；外部B/D投票、测试标签和后处理阈值拟合均未使用。','',
        '## 不继续与未执行','',
        '冷启动、扩容、重复后验条件头、其他折和旧任务均未启动。负路由候选不继续扫幅度、宽度或种子；结论仅限当前头与训练协议。当前原五折>82仍未完成，旧交付保持不变。',
        '下一步只依据采用方案和逐子集结果讨论，不自动叠加同通道多个路由。','',
        '## 记录','',
        'METHOD.md、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json；源码champion_route_v28.py和train_champion_route_v28.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v28路由对照完成'
    note=marker+'\n'+f"六项完成；原第五折TOTAL={scores['TOTAL']:.12f}，相对v26{result['delta_reference_TOTAL']:+.12f}。"+', '.join(s['target']+':'+str(s['mode']) for s in selections)+'。原模型/交付不变，无其他折。记录champion_route_v28_20260929/FINAL.md。'
    for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(path,marker,note)
    current='# v28已完成\n\n目标：当前80.30基础上检验专家占比分配。\n\n已完成：六项训练、初始与冻结验证、实际组合及新采用目标磁盘重载。\n\n'+note+'\n\n未完成：独立重复与其他四折。当前不自动延伸负候选。\n\n达成后：仅以固定有效方案讨论完整OOF，旧交付保留。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for row in records:
        arm=(ROOT/row['checkpoint']).parent;rep=row['report'];t=row['target']
        detail=f"\n\n## 最终结果\n{t}/{row['mode']}更新后最高{row['post_update']['scores'][t]:.12f}，最终{rep['target_score']:.12f}，最佳{rep['best_epoch']}/停止{rep['stopped_epoch']}；相对v26{row['selected']['channel_delta']:+.12f}。\n"
        append_once(arm/'README.md','## 最终结果',detail)
        (arm/'PROJECTS.md').write_text(current+detail,encoding='utf-8');(arm.parent/'PROJECTS.md').write_text(current,encoding='utf-8')
    policy='## v28最新完成研究配方\n'+f"TOTAL={scores['TOTAL']:.12f}，当前引用results/deep_learning/champion_route_v28_20260929/CURRENT_RECIPE.json。六项已结束，旧来源不变，无其他折和旧队列。"
    for path in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:append_once(path,'## v28最新完成研究配方',policy)
    print('FINAL',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,comparisons=comparisons,timing=timing)),ensure_ascii=False))
    print('TRAINED',[dict(target=r['target'],mode=r['mode'],scope=r['trained_checkpoint_scope']) for r in records])


if __name__=='__main__':main()
