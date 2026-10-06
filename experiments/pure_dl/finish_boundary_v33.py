"""Finalize four boundary-context trials with exact old-population scoring."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from boundary_context_v33 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction,factor_scope
from run_boundary_v33 import TARGETS,jobs,append_once
from prepare_boundary_v33 import kernels
from train import metric,dmask
from train_dual8222_heads80 import atomic_json,atomic_array

ALL=('POR','PERM','SW')
W=np.array([.3,.35,.35])
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def native(x):
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [native(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x

def main():
    started=time.perf_counter()
    groups=[]
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_boundary_v33_{t}_fold5'
        if not (out/'fold_5_report.json').exists():raise RuntimeError('Not complete: '+t)
        groups.append(dict(target=t,id=out.name,report=read(out/'fold_5_report.json')))
    if jobs():raise RuntimeError('Active training; do not begin final GPU replay')
    prev=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=cache['truth'][cache['validation']];wells=cache['wells'][cache['validation']]
    raw=np.flatnonzero(cache['raw_to_valid']>=0)[cache['validation']]
    with np.load(STUDY/'boundary_arrays.npz') as z:distance=z['distance'][raw]
    del cache;gc.collect()
    refs=metric(truth,reference);assert truth.shape==reference.shape==(144665,3)
    assert max(abs(refs[k]-v) for k,v in prev['scores'].items())<1e-9
    parent_u=kernels(truth,reference);dm=dmask(truth)
    masks={'near_0_5':distance<=5,'middle_6_25':(distance>5)&(distance<=25),'far_gt25':distance>25}
    combined=reference.copy();records=[];selections=[];comparisons=[];events=[];costs={}
    for group in groups:
        t=group['target'];c=ALL.index(t);out=ROOT/'outputs'/group['id'];cfg=read(out/'config.json')
        for file,sha in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/file).read_bytes()).hexdigest()==sha,file
        assert cfg['metadata']['feature_manifest_sha256']==hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest()
        preds={};updated={};own=[]
        for mode in MODES:
            arm=out/mode;rep=read(arm/'fold_5_report.json')
            pred=np.load(arm/'channel_pred.npy');up=np.load(arm/'best_updated_pred.npy')
            assert pred.shape==up.shape==(144665,)
            trial=reference.copy();trial[:,c]=pred;score=metric(truth,trial)
            uptrial=reference.copy();uptrial[:,c]=up;upscore=metric(truth,uptrial)
            assert abs(score[t]-rep['target_score'])<1e-9 and abs(upscore[t]-rep['best_after_update']['target_score'])<1e-9
            assert rep['all_frozen_state_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
            if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,c])
            du=100*(kernels(truth,uptrial)[:,c]-parent_u[:,c]);population=[]
            for gn,gm in masks.items():
                for pn,pm in [('all',np.ones(len(truth),bool)),('default',dm[:,c]),('numeric',~dm[:,c])]:
                    sel=gm&pm;n=int(sel.sum())
                    population.append(dict(group=gn,population=pn,n=n,channel_delta=float(du[sel].mean()) if n else None,TOTAL_contribution=float(W[c]*du[sel].sum()/len(truth))))
            row=dict(target=t,mode=mode,report=rep,selected_scores=score,post_update_scores=upscore,
                post_update_boundary_decomposition=population,checkpoint=str((arm/'model_best.pt').relative_to(ROOT)))
            own.append(row);records.append(row);preds[mode]=pred;updated[mode]=up
        best=max(own,key=lambda z:z['selected_scores'][t]);adopt=best['selected_scores'][t]>refs[t]+1e-9
        selections.append(dict(target=t,adopted=adopt,mode=best['mode'] if adopt else None,score=best['selected_scores'][t],
            checkpoint=best['checkpoint'] if adopt else next(z['checkpoint'] for z in prev['choices'] if z['target']==t),
            best_epoch=best['report']['best_epoch'],added_parameters=best['report']['trainable_parameters'] if adopt else 0))
        if adopt:combined[:,c]=preds[best['mode']]
        comparisons.append(dict(target=t,selected_segmented_minus_fixed=own[1]['selected_scores'][t]-own[0]['selected_scores'][t],
            post_update_segmented_minus_fixed=own[1]['post_update_scores'][t]-own[0]['post_update_scores'][t],
            scope='Matched parameters, initialization, scaler, trainer; compares each early-stopped best, same development fold.'))
        launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(group['report']['finished_at'])
        events.extend([launch,end]);costs[t]=dict(seconds=group['report']['seconds'],peak_allocated_MiB=group['report']['peak_allocated_MiB'],
            median_epoch_seconds=float(np.median([z['seconds'] for z in group['report']['epoch_timing']])))
    replayed=[];choices=[dict(z) for z in prev['choices']]
    if any(s['adopted'] for s in selections):
        torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']])
        for selected in selections:
            if not selected['adopted']:continue
            t=selected['target'];m=selected['mode'];c=ALL.index(t)
            saved=torch.load(ROOT/selected['checkpoint'],map_location='cpu',weights_only=False)
            model=Network(t,master,modes=(m,)).cuda();model.heads[m].load_state_dict(saved['delta_state'],strict=True)
            pred,score,_=Evaluator(data,master,c,reference).many(model,(m,))[m]
            assert np.array_equal(pred,combined[:,c]) and abs(score-selected['score'])<1e-9
            selected.update(fresh_disk_replay_exact=True,factor_scope=factor_scope(model,data,m));replayed.append(t)
            for choice in choices:
                if choice['target']==t:choice.update(key='v33_'+t+'_'+m,mode=m,model_module='boundary_context_v33',
                    checkpoint=selected['checkpoint'],target_score=score,best_epoch=selected['best_epoch'],factor=model.factor,
                    parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                    feature_manifest=str((STUDY/'feature_manifest.json').relative_to(ROOT)),
                    builder='Network(target,master,modes=(mode,)); heads[mode].load_state_dict(checkpoint[delta_state]); boundary_context_v33.load_data()')
            model.close();del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);perwell=[]
    for w in np.unique(wells):
        sel=wells==w
        perwell.append(dict(well=int(w),n=int(sel.sum()),TOTAL_delta=float(metric(truth[sel],combined[sel])['TOTAL']-metric(truth[sel],reference[sel])['TOTAL'])))
    result=native(dict(finished_at=datetime.now().astimezone().isoformat(),reference_scores=refs,selected_scores=scores,
        delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],all_four_complete=True,actual_combined_prediction_scored=True,
        new_target_full_disk_replays=replayed,selections=selections,candidates=records,comparisons=comparisons,per_well=perwell,
        well_wins=sum(z['TOTAL_delta']>0 for z in perwell),well_losses=sum(z['TOTAL_delta']<0 for z in perwell),
        rows=144665,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True))
    recipe=dict(prev);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(processes=costs,parallel_target_processes=2,maximum_allowed=2,launch_to_last_finish_seconds=(max(events)-min(events)).total_seconds(),
        seconds_for_this_finalization=time.perf_counter()-started,
        scope='Per-target process includes two independently optimized heads with shared frozen forward; wall includes setup/training/evaluation/recovery but excludes earlier diagnostics and code authoring.')
    atomic_json(STUDY/'timing.json',timing)
    diag=read(STUDY/'diagnostic.json')
    lines=['# v33 输入边界上下文对照：完成结果','',f"观察完成时间：{result['finished_at']}",
        f"原TOTAL={refs['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 探索重点与输入诊断','',
        '固定51点统计可能混合边界两侧。边界仅依据九条输入曲线左右5点均值差，以训练IQR标准化后计算均方根；训练有效点95%分位阈值，峰间距11点。未用标签定义或挑选边界。',
        '近边界定义为距离不超过5个采样点；只是一种输入变化定义，不是标注地质界面。',
        json.dumps(diag['summary'],ensure_ascii=False),
        f"验证近边界占比={diag['near_fraction']:.6%}；统计来源diagnostic.json与boundary_error_groups.csv。",'',
        '## 已完成的四项训练','',
        '|目标|侧输入|参数|更新后最高通道分|最终通道分|最佳/停止轮次|','|---|---|---:|---:|---:|---|']
    for row in records:
        q=row['report'];t=row['target']
        lines.append(f"|{t}|{row['mode']}|{q['trainable_parameters']}|{row['post_update_scores'][t]:.9f}|{q['target_score']:.9f}|{q['best_epoch']}/{q['stopped_epoch']}|")
    lines+=['','两组相同337→32→16小头、相同初始权重和缩放、相同训练预算；旧模型全部冻结。第0轮回退不计训练增益。固定组提供冗余51点统计；分段组只缩短跨输入边界的窗口，不改变原269维路径、原标签或解码。',
        '', '## 相同模型下的特征对照','']
    for c in comparisons:lines.append(f"{c['target']}：segmented-fixed选定通道差{c['selected_segmented_minus_fixed']:+.9f}；排除第0轮的最高分差{c['post_update_segmented_minus_fixed']:+.9f}。")
    lines+=['','## 取得的收益及适用范围','',
        f"实际组合增益{result['delta_reference_TOTAL']:+.12f}。新采用磁盘重建目标：{replayed}；不变目标复用已验证预测。所有训练内最佳恢复与冻结检查通过，三列实际评分完成。",
        f"相对原版本{result['well_wins']}井升、{result['well_losses']}井降。原第五折已反复开发，输入难点关联与选模结果都不是独立泛化确认。",
        '', '## 不继续的设置与未完成事项','',
        '如候选没有超越旧预测，不扩大当前阈值、统计窗口和小头搜索。边界附近失分集中不意味着截断统计能解决，不能推广为所有边界结构都无效。',
        '尚未做输入与标签边界的深度错位诊断、不同训练种子或新井确认；不自动开展。原其他四折、完整OOF>82及新交付均未完成。',
        '', '## 数据与验证记录','',
        '准备脚本第一次将有效标签索引当作原始行索引，行序断言在评分前阻止输出；随后按raw_to_valid映射修正、重新准备，正确精确复现原分。该错误只发生于本次新诊断，未改变旧模型或训练标签。',
        '源码和日志保留。finalization只对入选的新模型做额外完整磁盘推理；其他模型训练内恢复已核对，不重复未变化推理。',
        '', '## 文件','',
        'diagnostic.json、boundary_error_groups.csv、boundary_error_by_well.csv、feature_manifest.json、assembled_report.json、CURRENT_RECIPE.json、timing.json；源代码prepare_boundary_v33.py、boundary_context_v33.py、train_boundary_v33.py。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    note='## v33输入边界对照完成\n'+f"四项完成，TOTAL={scores['TOTAL']:.12f}，增量={result['delta_reference_TOTAL']:+.12f}；新采用={replayed}。输入边界与失分相关，统计截断的实际结果见boundary_context_v33_20260929/FINAL.md。无其他折、旧任务或交付覆盖。"
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(p,'## v33输入边界对照完成',note)
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_boundary_v33_{t}_fold5'
        for n in ['README.md','PROJECTS.md']:append_once(out/n,'## v33输入边界对照完成',note)
    append_once(STUDY/'README.md','## v33输入边界对照完成',note)
    notes=dict(observed_at=datetime.now().astimezone().isoformat(),training_failures=[],active_jobs=jobs(),
        read_command_failures=['shell quoting failure; no command executed','encoded inspection had misspelled root; no writes'],
        diagnostic_failure='valid/raw index mix-up caught before scoring; corrected before training',
        all_four_complete=True,old_training_sources_unchanged=True,old_delivery_unchanged=True)
    for t in TARGETS:
        p=ROOT/'outputs'/f'pure_dl_boundary_v33_{t}_fold5'/'failure.json'
        if p.exists():notes['training_failures'].append(read(p))
    atomic_json(STUDY/'execution_notes.json',native(notes))
    print('FINAL_V33',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],selections=selections,comparisons=comparisons,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
