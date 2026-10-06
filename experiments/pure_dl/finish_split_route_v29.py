"""Score every fixed candidate; rebuild newly adopted private models from disk."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from champion_split_route_v29 import ROOT,STUDY,PREVIOUS,TARGETS,MODES,Network,Evaluator,load_data,reference_prediction,routing_scope
from run_split_route_v29 import record
from launch_champion_posterior_v27 import jobs,append_once
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from gpu_job_guard_v2 import check_slot
import train_invariant as base

W={'POR':.3,'PERM':.35,'SW':.35}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    progress=record()
    if not progress['all_complete']:raise RuntimeError('Six declared trials must complete before selection')
    if jobs():raise RuntimeError('Active training prevents final replay')
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');refpred=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==refpred.shape==(144665,3)
    refs=base.metric(truth,refpred)
    assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    records=[];selections=[];comparisons=[];starts=[];ends=[];costs={};combined=refpred.copy()
    for group in progress['rows']:
        t=group['target'];c=TARGETS.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for filename,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/filename).read_bytes()).hexdigest()==digest,filename
        starts.append(datetime.fromisoformat(read(out/'launch.json')['started_at']))
        ends.append(datetime.fromisoformat(group['report']['finished_at']))
        costs[t]=dict(process_seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            epoch_median_seconds=float(np.median([x['seconds'] for x in group['report']['epoch_timing']])),
            scope='one target process shares only frozen inference between candidates')
        own=[];preds={}
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json')
            pred=np.load(arm/'channel_pred.npy');up=np.load(arm/'best_updated_pred.npy');preds[m]=pred
            d=diagnostics(truth,refpred,pred,wells,t);u=diagnostics(truth,refpred,up,wells,t)
            assert abs(d['scores'][t]-rep['target_score'])<1e-9
            assert abs(d['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
            assert abs(u['scores'][t]-rep['best_after_update']['target_score'])<1e-9
            if rep['best_epoch']==0:assert np.array_equal(pred,refpred[:,c])
            row=dict(target=t,mode=m,report=rep,selected=d,post_update=u,
                checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),prediction=str((arm/'channel_pred.npy').relative_to(ROOT)))
            own.append(row);records.append(row)
        coupled=refpred.copy();coupled[:,c]=preds['coupled']
        comparisons.append(dict(target=t,split_minus_coupled=diagnostics(truth,coupled,preds['split'],wells,t),
            note='each arm selects its best checkpoint on the same development fold; total capacity equal, allocation and gradients differ'))
        winner=max(own,key=lambda x:x['selected']['scores'][t]);adopt=winner['selected']['scores'][t]>refs[t]+1e-9
        selected_mode=winner['mode'] if adopt else None
        oldchoice=next(x for x in previous['choices'] if x['target']==t)
        selections.append(dict(target=t,mode=selected_mode,adopted=adopt,
            target_score=winner['selected']['scores'][t],best_epoch=winner['report']['best_epoch'],
            checkpoint=winner['checkpoint'] if adopt else oldchoice['checkpoint'],
            added_parameters=winner['report']['trainable_parameters'] if adopt else 0,
            parent_v28_checkpoint=cfg['retained_v28_route'],parent_v28_mode=cfg['retained_v28_mode']))
        combined[:,c]=preds[winner['mode']]
    start=time.perf_counter();choices=[dict(x) for x in previous['choices']];replayed=[]
    if any(s['adopted'] for s in selections):
        torch.set_num_threads(3);check_slot(min_free_gib=3.)
        torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for s in selections:
            if not s['adopted']:continue
            t=s['target'];m=s['mode'];c=TARGETS.index(t)
            cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            assert cp['target']==t and cp['mode']==m
            model=Network(t,master,modes=(m,)).cuda()
            model.heads[m].load_state_dict(cp['delta_state'],strict=True)
            pred,score,_=Evaluator(data,master,c,refpred).many(model,(m,))[m]
            assert np.array_equal(pred,combined[:,c]) and abs(score-s['target_score'])<1e-9
            combined[:,c]=pred;s['all144665_fresh_disk_replay_exact']=True
            s['selected_scope']=routing_scope(model,data,m);replayed.append(t)
            for choice in choices:
                if choice['target']==t:
                    choice.update(key='v29_'+t+'_'+m,mode=m,factor='atom_conditional_routing',architecture='coupled_or_split_residual_routes',
                        checkpoint=s['checkpoint'],model_module='champion_split_route_v29',
                        builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state])',
                        target_score=score,best_epoch=s['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                        retained_v28_checkpoint=s['parent_v28_checkpoint'])
            model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert abs(scores['TOTAL']-sum(W[s['target']]*s['target_score'] for s in selections))<1e-9
    perwell=[]
    for well in np.unique(wells):
        mask=wells==well
        delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],refpred[mask])['TOTAL']
        perwell.append(dict(well=int(well),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([x['n'] for x in perwell]);changes=np.array([x['TOTAL_delta'] for x in perwell])
    indices=np.random.default_rng(20261010).integers(0,len(perwell),(10000,len(perwell)))
    boot=(counts[indices]*changes[indices]).sum(1)/counts[indices].sum(1)
    result=native(dict(fold=5,rows=144665,reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=records,selections=selections,comparisons=comparisons,per_well=perwell,
        well_wins=sum(x['TOTAL_delta']>0 for x in perwell),well_losses=sum(x['TOTAL_delta']<0 for x in perwell),
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boot,[.025,.975]),
        all_six_complete=True,actual_combined_prediction_scored=True,new_target_full_disk_replays=replayed,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=native(choices),reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(max_parallel_target_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),targets=costs,
        final_replay_seconds=time.perf_counter()-start,scope='process times include imports, setup, training, restoration; overall time includes explicit launch gaps; source implementation excluded')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v29：当前强模型上的默认/条件路由解耦','',f"完成：{result['finished_at']}",'',
        f"原TOTAL={refs['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；变化={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 全部六项结果','',
        '|目标|模式|新增参数|训练后最高通道分|最终通道分|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
    for row in records:
        t=row['target'];p=row['report']
        lines.append(f"|{t}|{row['mode']}|{p['trainable_parameters']}|{row['post_update']['scores'][t]:.9f}|{p['target_score']:.9f}|{p['best_epoch']}/{p['stopped_epoch']}|")
    lines+=['','第0轮回退不计训练增益。六项均由相同v28父函数开始；coupled/split张量初始相同，候选间不共享可训练参数。','',
        '## 最终选择与子集来源','']
    for s in selections:
        t=s['target'];lines.append(f"{t}：{'采用'+s['mode'] if s['adopted'] else '保留v28'}，通道{s['target_score']:.12f}，TOTAL贡献{W[t]*(s['target_score']-refs[t]):+.12f}，新增推理参数{s['added_parameters']}。")
        if s['adopted']:
            d=next(r for r in records if r['target']==t and r['mode']==s['mode'])['selected']
            lines.append('子集分解：'+json.dumps(native(d['populations']),ensure_ascii=False))
    lines+=['','## 同容量比较','']
    for c in comparisons:
        d=c['split_minus_coupled']
        lines.append(f"{c['target']} split-coupled：通道{d['channel_delta']:+.12f}，TOTAL{d['TOTAL_delta']:+.12f}；默认段{d['populations']['default']['channel_delta']:+.9f}，数值段{d['populations']['numeric']['channel_delta']:+.9f}。")
    lines+=['',f"组合{result['well_wins']}井升、{result['well_losses']}井降；探索性未校正区间{result['exploratory_selection_unadjusted_TOTAL_95']}。",
        '六项选择及早停均在反复开发的同一第五折进行；该区间没有校正选择影响，不是独立泛化确认。','',
        '## 机制与边界','',
        '每候选两条319→32→1非线性小头共20546参数，输入投影分别随机初始化，输出层零初始化；coupled平均后同时修正默认/密度路由，split分配给各自路由。±0.125限制的是新增log-odds量，原v28路由保留。',
        '固定总容量下，coupled两个任务共用两支路，split每个任务用一支路；包含表征分配、梯度和有效任务容量差异，不能把结果唯一归因于梯度冲突。',
        '旧专家门控、分量权重、位置、尺度和已有有效侧头均冻结，未从头重训或扩容骨干。初始144665点等于v28；旧状态终场检查一致，实际反向有新增梯度。',
        'split输出的默认与条件分布对另一支路梯度为零的检查由训练输入完成，数值记录保留各fold_5_report.json。原似然/3、Q50和解码不变。','',
        '## 验证与未完成','',
        '六项最佳状态在训练进程内全量恢复，新采用目标从磁盘独立重建并全量预测，最后实际合并：'+str(replayed)+'。未变目标沿用之前验证，不虚称新增重载。',
        '没有补其他四折、其他种子或更换监控井；没有覆盖原80.324397及交付。全OOF>82和第五折>80.5仍按实际结果判断，本批不自动延伸。',
        '负候选仅关闭本次设置，不否定所有路由解耦。不会自动重复宽度/修正范围扫描。','',
        '## 文件','',
        'METHOD.md、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json和trainer_provenance.json。模型champion_split_route_v29.py，训练train_champion_split_route_v29.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v29路由解耦完成'
    note=marker+'\n'+f"TOTAL={scores['TOTAL']:.12f}，相对v28{result['delta_reference_TOTAL']:+.12f}。"+', '.join(s['target']+':'+str(s['mode']) for s in selections)+'。原模型与交付不改，无其他折。证据champion_split_route_v29_20260929/FINAL.md。'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(p,marker,note)
    current='# v29已完成\n\n目标：保留已有有效模型，检验默认/数值路由解耦。\n\n已完成：六项训练与最佳恢复、固定参数检查、实际组合及采用目标磁盘重载。\n\n'+note+'\n\n未完成：独立重复、其他分折；本批不自动延伸。\n\n达成后：固定正收益配方，由用户决定完整验证；原交付保留。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for row in records:
        arm=(ROOT/row['checkpoint']).parent;t=row['target'];p=row['report']
        detail=f"\n## 最终结果\n{t}/{row['mode']}训练后最高{row['post_update']['scores'][t]:.12f}，选定{p['target_score']:.12f}，最佳{p['best_epoch']}/停止{p['stopped_epoch']}。相对v28{row['selected']['channel_delta']:+.12f}。\n"
        append_once(arm/'README.md','## 最终结果',detail)
        (arm/'PROJECTS.md').write_text(current+detail,encoding='utf-8');(arm.parent/'PROJECTS.md').write_text(current,encoding='utf-8')
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:
        append_once(p,'## v29最新研究配方','## v29最新研究配方\n'+f"当前TOTAL={scores['TOTAL']:.12f}，相对v28{result['delta_reference_TOTAL']:+.12f}。引用results/deep_learning/champion_split_route_v29_20260929/CURRENT_RECIPE.json；六项完成，无其他折和旧队列，原交付保留。")
    print('FINAL',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,comparisons=comparisons,timing=timing)),ensure_ascii=False))


if __name__=='__main__':main()
