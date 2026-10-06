"""Finalize v30's six predeclared candidates and preserve immutable sources."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from champion_coadapt_v30 import ROOT,STUDY,PREVIOUS,TARGETS,MODES,Network,Evaluator,load_data,reference_prediction,factor_scope
from run_coadapt_v30 import record
from launch_champion_posterior_v27 import jobs,append_once
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from gpu_job_guard_v2 import check_slot
import train_invariant as base

W={'POR':.30,'PERM':.35,'SW':.35}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    progress=record()
    if not progress['all_complete']:raise RuntimeError('All six candidates must finish')
    if jobs():raise RuntimeError('Final replay requires no active trainers')
    prior=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in prior['scores'].items())
    records=[];selections=[];comparisons=[];starts=[];ends=[];costs={};combined=reference.copy()
    for group in progress['rows']:
        t=group['target'];c=TARGETS.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for name,digest in cfg['source_hashes'].items():
            assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
        starts.append(datetime.fromisoformat(read(out/'launch.json')['started_at']))
        ends.append(datetime.fromisoformat(group['report']['finished_at']))
        costs[t]=dict(process_seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            median_epoch_seconds=float(np.median([x['seconds'] for x in group['report']['epoch_timing']])),
            scope='one process for two alternatives with shared frozen backbone computation')
        own=[];preds={}
        for m in MODES:
            arm=out/m;rep=read(arm/'fold_5_report.json')
            selected=np.load(arm/'channel_pred.npy');up=np.load(arm/'best_updated_pred.npy');preds[m]=selected
            d=diagnostics(truth,reference,selected,wells,t);u=diagnostics(truth,reference,up,wells,t)
            assert abs(d['scores'][t]-rep['target_score'])<1e-9
            assert abs(d['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
            assert abs(u['scores'][t]-rep['best_after_update']['target_score'])<1e-9
            if rep['best_epoch']==0:assert np.array_equal(selected,reference[:,c])
            row=dict(target=t,mode=m,report=rep,selected=d,post_update=u,
                checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),prediction=str((arm/'channel_pred.npy').relative_to(ROOT)))
            records.append(row);own.append(row)
        ctrl=reference.copy();ctrl[:,c]=preds['route_only']
        comparisons.append(dict(target=t,coadapt_minus_route_only=diagnostics(truth,ctrl,preds['coadapt'],wells,t),
            best_updated_target_difference=own[1]['post_update']['channel_delta']-own[0]['post_update']['channel_delta'],
            note='same architecture/start and separate clipping; comparison changes the set of trainable retained parameters'))
        winner=max(own,key=lambda r:r['selected']['scores'][t]);adopt=winner['selected']['scores'][t]>refs[t]+1e-9
        old=next(x for x in prior['choices'] if x['target']==t)
        selections.append(dict(target=t,mode=winner['mode'] if adopt else None,adopted=adopt,
            target_score=winner['selected']['scores'][t],best_epoch=winner['report']['best_epoch'],
            checkpoint=winner['checkpoint'] if adopt else old['checkpoint'],parent_checkpoint=old['checkpoint'],
            updated_parameters=winner['report']['trainable_parameters'] if adopt else 0,added_architecture_parameters=0))
        combined[:,c]=preds[winner['mode']]
    start=time.perf_counter();choices=[dict(c) for c in prior['choices']];replayed=[]
    if any(s['adopted'] for s in selections):
        torch.set_num_threads(3);check_slot(min_free_gib=3.)
        torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for s in selections:
            if not s['adopted']:continue
            t=s['target'];c=TARGETS.index(t);m=s['mode']
            cp=torch.load(ROOT/s['checkpoint'],map_location='cpu',weights_only=False)
            assert cp['target']==t and cp['mode']==m
            model=Network(t,master,modes=(m,)).cuda();model.heads[m].load_state_dict(cp['delta_state'],strict=True)
            pred,score,_=Evaluator(data,master,c,reference).many(model,(m,))[m]
            assert np.array_equal(pred,combined[:,c]) and abs(score-s['target_score'])<1e-9
            combined[:,c]=pred;s['all144665_fresh_disk_replay_exact']=True
            s['scope_replayed']=factor_scope(model,data,m);replayed.append(t)
            for choice in choices:
                if choice['target']==t:
                    choice.update(key='v30_'+t+'_'+m,mode=m,factor=model.factor,architecture='retained_readout_route_coadaptation',
                        checkpoint=s['checkpoint'],model_module='champion_coadapt_v30',
                        builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state])',
                        target_score=score,best_epoch=s['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                        retained_v29_checkpoint=s['parent_checkpoint'])
            model.close();del model,cp;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert abs(scores['TOTAL']-sum(W[s['target']]*s['target_score'] for s in selections))<1e-9
    perwell=[]
    for w in np.unique(wells):
        mask=wells==w;delta=base.metric(truth[mask],combined[mask])['TOTAL']-base.metric(truth[mask],reference[mask])['TOTAL']
        perwell.append(dict(well=int(w),n=int(mask.sum()),TOTAL_delta=delta))
    counts=np.array([x['n'] for x in perwell]);deltas=np.array([x['TOTAL_delta'] for x in perwell])
    ix=np.random.default_rng(20261010).integers(0,len(perwell),(10000,len(perwell)))
    boots=(counts[ix]*deltas[ix]).sum(1)/counts[ix].sum(1)
    result=native(dict(fold=5,rows=144665,reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=records,selections=selections,comparisons=comparisons,per_well=perwell,
        well_wins=sum(x['TOTAL_delta']>0 for x in perwell),well_losses=sum(x['TOTAL_delta']<0 for x in perwell),
        exploratory_selection_unadjusted_TOTAL_95=np.quantile(boots,[.025,.975]),all_six_complete=True,
        actual_combined_prediction_scored=True,new_target_full_disk_replays=replayed,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(prior);recipe.update(scores=native(scores),choices=native(choices),reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
        original_fifth_development_selection=True,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(max_parallel_target_jobs=2,first_launch_to_last_finish_seconds=(max(ends)-min(starts)).total_seconds(),targets=costs,
        final_replay_seconds=time.perf_counter()-start,scope='overall launch interval includes staged gaps and preparation, training and restoration; excludes code implementation/final replay')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v30：保留结构的已有模块协同微调','',f"完成：{result['finished_at']}",'',
        f"父模型TOTAL={refs['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 六项实测结果','',
        '|目标|更新范围|可训练参数|训练后最高通道分|最终通道分|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
    for row in records:
        p=row['report'];t=row['target']
        lines.append(f"|{t}|{row['mode']}|{p['trainable_parameters']}|{row['post_update']['scores'][t]:.9f}|{p['target_score']:.9f}|{p['best_epoch']}/{p['stopped_epoch']}|")
    lines+=['','第0轮表示保留父模型，不算训练收益。两候选同结构同初始状态；参数量列为更新数量，不是新增加的模型容量。','',
        '## 最终采用与来源','']
    for s in selections:
        t=s['target'];lines.append(f"{t}：{'采用'+s['mode'] if s['adopted'] else '保留v29'}，{s['target_score']:.12f}，TOTAL贡献{W[t]*(s['target_score']-refs[t]):+.12f}。")
        if s['adopted']:
            row=next(r for r in records if r['target']==t and r['mode']==s['mode'])
            lines.append('子集分解：'+json.dumps(native(row['selected']['populations']),ensure_ascii=False))
    lines+=['','## 协同更新的匹配比较','']
    for comp in comparisons:
        d=comp['coadapt_minus_route_only']
        lines.append(f"{comp['target']} coadapt-route_only：通道{d['channel_delta']:+.12f}，TOTAL{d['TOTAL_delta']:+.12f}；默认段{d['populations']['default']['channel_delta']:+.9f}，数值段{d['populations']['numeric']['channel_delta']:+.9f}。")
    lines+=['',f"实际组合{result['well_wins']}井升、{result['well_losses']}井降；探索性未校正区间{result['exploratory_selection_unadjusted_TOTAL_95']}。",
        '该折长期参与开发，六项候选与早停都依赖它。没有跨种子/独立分折确认，区间未作选择校正。','',
        '## 机制、验证与未完成','',
        '不增加新的修正头或放宽修正界；沿用v25/v26的已选因子、v28/v29的已选路由。原骨干、原线性/point32、原路由及非选定输出因子保持冻结。coadapt比route_only多开放一个已有因子的小头，分别按路由/因子梯度组clip2。',
        '采用同一低学习率3e-5重置AdamW，其他原训练数据/Q50/损失/评分和80-18预算不变；不是原训练器断点续训，不把更多更新本身解释为新结构收益。',
        '全部初始144665点复现v29，训练有实际梯度，所有被冻结状态（包括候选读出复制体）终场检查不变。六项最佳参数训练内恢复通过；新采用目标独立磁盘重建完整预测：'+str(replayed)+'。随后实际合并评分。',
        '为复用冻结骨干，新训练器重算已存在的读出，保存完整小模块副本；不是额外加入一条模型预测路径。不以训练包装器的重复存储量宣称网络扩容。',
        '原80.335083与之前交付保留，其他四折未运行。未测试更多学习率、更多因子或新模块，本批负候选不继续网格扫描。完整OOF>82及原第五折严格>80.5仍按真实分数判定。','',
        '## 记录','',
        'METHOD.md、progress.json、assembled_report.json、CURRENT_RECIPE.json、timing.json、trainer_provenance.json。源码champion_coadapt_v30.py/train_coadapt_v30.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v30已有模块微调完成'
    note=marker+'\n'+f"实际TOTAL={scores['TOTAL']:.12f}，相对v29{result['delta_reference_TOTAL']:+.12f}。"+', '.join(s['target']+':'+str(s['mode']) for s in selections)+'。无新结构、其他折或旧队列；原交付保留。记录champion_coadapt_v30_20260929/FINAL.md。'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(p,marker,note)
    current='# v30已完成\n\n目标：已有有效模块有限微调。\n\n已完成：六项训练与恢复、冻结检查、实际组合、新采用模型独立重建。\n\n'+note+'\n\n未完成：独立重复、其他分折；不自动继续本批负候选。\n\n达成后：固定有收益配方，在用户授权下再进行完整OOF；交付保留。\n'
    (STUDY/'PROJECTS.md').write_text(current,encoding='utf-8')
    for r in records:
        arm=(ROOT/r['checkpoint']).parent;t=r['target'];p=r['report']
        part=f"\n## 最终结果\n{t}/{r['mode']}：训练后最高{r['post_update']['scores'][t]:.12f}，保留{p['target_score']:.12f}，最佳{p['best_epoch']}/停止{p['stopped_epoch']}；相对v29{r['selected']['channel_delta']:+.12f}。\n"
        append_once(arm/'README.md','## 最终结果',part)
        (arm/'PROJECTS.md').write_text(current+part,encoding='utf-8');(arm.parent/'PROJECTS.md').write_text(current,encoding='utf-8')
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:
        append_once(p,'## v30当前研究配方','## v30当前研究配方\n'+f"TOTAL={scores['TOTAL']:.12f}，相对v29{result['delta_reference_TOTAL']:+.12f}。当前配方results/deep_learning/champion_coadapt_v30_20260929/CURRENT_RECIPE.json。六项已结束，旧模型/交付保留，无其他折及旧队列。")
    compact=dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,
        comparisons=[dict(target=c['target'],channel_delta=c['coadapt_minus_route_only']['channel_delta'],TOTAL_delta=c['coadapt_minus_route_only']['TOTAL_delta']) for c in comparisons],
        well_wins=result['well_wins'],well_losses=result['well_losses'],interval=result['exploratory_selection_unadjusted_TOTAL_95'],timing=timing)
    print('FINAL',json.dumps(native(compact),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
