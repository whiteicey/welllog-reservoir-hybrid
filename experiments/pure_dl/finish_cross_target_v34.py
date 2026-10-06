"""Finish v34 using actual predictions. Negative control is never deployable."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from scipy.stats import spearmanr
from cross_target_v34 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction,factor_scope
from run_cross_target_v34 import jobs,append
from train_dual8222_heads80 import atomic_json,atomic_array
from prepare_boundary_v33 import kernels
from train import metric,dmask
import train_invariant as base

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def native(v):
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [native(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def main():
    started=time.perf_counter();out=ROOT/'outputs/pure_dl_cross_target_v34_SW_fold5'
    if not (out/'fold_5_report.json').exists() or jobs():raise RuntimeError('All auxiliary/head training must finish')
    reports=read(out/'fold_5_report.json');cfg=read(out/'config.json');meta=read(STUDY/'feature_manifest.json')
    for name,digest in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((STUDY/'feature_manifest.json').read_bytes()).hexdigest()==cfg['metadata']['feature_manifest_sha256']
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    val=cache['validation'];truth=cache['truth'][val];wells=cache['wells'][val]
    reference=reference_prediction();refs=metric(truth,reference);previous=read(PREVIOUS/'CURRENT_RECIPE.json')
    assert reference.shape==(144665,3) and max(abs(refs[k]-previous['scores'][k]) for k in refs)<1e-9
    parent_util=kernels(truth,reference)[:,2];dm=dmask(truth)[:,2]
    records=[];selected_arrays={};updated_arrays={}
    for m in MODES:
        arm=out/m;rep=read(arm/'fold_5_report.json')
        pred=np.load(arm/'channel_pred.npy');up=np.load(arm/'best_updated_pred.npy')
        p=reference.copy();p[:,2]=pred;sp=metric(truth,p)
        pu=reference.copy();pu[:,2]=up;su=metric(truth,pu)
        assert abs(sp['SW']-rep['target_score'])<1e-9 and abs(su['SW']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all_frozen_state_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,2])
        du=100*(kernels(truth,pu)[:,2]-parent_util)
        pops={key:dict(n=int(mask.sum()),SW_delta=float(du[mask].mean()),TOTAL_contribution=float(.35*du[mask].sum()/len(truth))) for key,mask in [('default',dm),('numeric',~dm)]}
        records.append(dict(mode=m,report=rep,selected_scores=sp,post_update_scores=su,
            post_update_populations=pops,checkpoint=str((arm/'model_best.pt').relative_to(ROOT)),deployable=m!='shuffled'))
        selected_arrays[m]=pred;updated_arrays[m]=up
    by_mode={r['mode']:r for r in records}
    best=max((by_mode[m] for m in ('self','cross')),key=lambda r:r['selected_scores']['SW'])
    adopt=best['selected_scores']['SW']>refs['SW']+1e-9
    cross_evidence=(by_mode['cross']['selected_scores']['SW']>max(refs['SW'],by_mode['self']['selected_scores']['SW'],by_mode['shuffled']['selected_scores']['SW'])+1e-9)
    combined=reference.copy();choice=None;replayed=[]
    if adopt:
        m=best['mode'];torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
        data,master=load_data();assert np.array_equal(truth,data['truth'][data['validation']])
        model=Network('SW',master,modes=(m,)).cuda()
        cp=torch.load(ROOT/best['checkpoint'],map_location='cpu',weights_only=False)
        model.heads[m].load_state_dict(cp['delta_state'],strict=True)
        pred,score,_=Evaluator(data,master,2,reference).many(model,(m,))[m]
        assert np.array_equal(pred,selected_arrays[m]) and abs(score-best['selected_scores']['SW'])<1e-9
        combined[:,2]=pred;replayed=['SW'];scope=factor_scope(model,data,m)
        choice=dict(target='SW',key='v34_SW_'+m,mode=m,model_module='cross_target_v34',checkpoint=best['checkpoint'],
            factor='conditional_mixture_weights',target_score=score,best_epoch=best['report']['best_epoch'],
            feature_manifest=str((STUDY/'feature_manifest.json').relative_to(ROOT)),
            builder='Network(SW,master,modes=(mode,)); heads[mode].load_state_dict(delta_state); cross_target_v34.load_data()',
            auxiliary_sources=meta['provenance'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),factor_scope=scope)
        model.close();del model,data,master,cp;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);perwell=[]
    for w in np.unique(wells):
        mask=wells==w;perwell.append(dict(well=int(w),n=int(mask.sum()),TOTAL_delta=float(metric(truth[mask],combined[mask])['TOTAL']-metric(truth[mask],reference[mask])['TOTAL'])))
    counts=np.array([x['n'] for x in perwell]);delta=np.array([x['TOTAL_delta'] for x in perwell]);ix=np.random.default_rng(20261134).integers(0,len(counts),(10000,len(counts)))
    interval=np.quantile((counts[ix]*delta[ix]).sum(1)/counts[ix].sum(1),[.025,.975])
    comparison=dict(cross_minus_self=by_mode['cross']['selected_scores']['SW']-by_mode['self']['selected_scores']['SW'],
        cross_minus_shuffled=by_mode['cross']['selected_scores']['SW']-by_mode['shuffled']['selected_scores']['SW'],
        post_update_cross_minus_self=by_mode['cross']['post_update_scores']['SW']-by_mode['self']['post_update_scores']['SW'],
        post_update_cross_minus_shuffled=by_mode['cross']['post_update_scores']['SW']-by_mode['shuffled']['post_update_scores']['SW'])
    # Describe the quality of the auxiliary signals only after the experiment is fixed.
    with np.load(STUDY/'row_mapping.npz') as z:training_raw=z['train_raw'];validation_raw=z['validation_raw']
    with np.load(STUDY/'raw_auxiliary_summaries.npz') as z:aux=z['summaries']
    yt=cache['truth'][cache['fold']!=4];dmt=dmask(yt);quality=[]
    for c,t in enumerate(('POR','PERM','SW')):
        v=aux[training_raw,c];numeric=~dmt[:,c];y=base.TRANSFORM[c](yt[numeric,c]);mu=v[numeric,1]
        quality.append(dict(target=t,OOF_rows=len(yt),numeric_n=int(numeric.sum()),
            gate_brier=float(np.mean((v[:,0]-dmt[:,c])**2)),numeric_mean_spearman=float(spearmanr(y,mu).statistic),
            predictive_interval_coverage=float(np.mean((y>=v[numeric,3])&(y<=v[numeric,5]))),
            scope='Original64 auxiliary OOF descriptive diagnostics; not a new total score; not used to select feature variants'))
    aux_reports=[];events=[]
    for g in range(4):
        p=ROOT/'outputs'/f'pure_dl_cross_target_v34_aux{g}';report=read(p/'auxiliary_report.json');config=report['config']
        assert set(config['oof_wells']).isdisjoint(set(config['fit_wells'])|set(config['stop_wells']))
        assert set(config['outer_wells']).isdisjoint(set(config['fit_wells'])|set(config['stop_wells']))
        assert config['source_sha256']==hashlib.sha256((ROOT/'experiments/pure_dl/auxiliary_crossfit_v34.py').read_bytes()).hexdigest()
        aux_reports.append(dict(group=g,fit_wells=config['fit_wells'],stop_wells=config['stop_wells'],oof_wells=config['oof_wells'],
            models=[{k:v for k,v in x.items() if k!='history'} for x in report['models']],seconds=report['seconds'],peak_allocated_MiB=report['peak_allocated_MiB']))
        events.extend([(datetime.fromisoformat(read(p/'launch.json')['started_at']),1),(datetime.fromisoformat(report['finished_at']),-1)])
    events.extend([(datetime.fromisoformat(read(out/'launch.json')['started_at']),1),(datetime.fromisoformat(reports['finished_at']),-1)])
    current=peak=0
    for _,v in sorted(events):current+=v;peak=max(peak,current)
    result=native(dict(finished_at=datetime.now().astimezone().isoformat(),reference_scores=refs,selected_scores=scores,
        delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,comparisons=comparison,
        adopted=adopt,selected_mode=best['mode'] if adopt else None,cross_target_mechanism_positive=cross_evidence,
        all_three_heads_complete=True,all_twelve_auxiliary_models_complete=True,negative_control_not_deployed=True,
        auxiliary_models=aux_reports,auxiliary_quality=quality,
        new_target_full_disk_replays=replayed,actual_combined_prediction_scored=True,
        per_well=perwell,well_wins=sum(r['TOTAL_delta']>0 for r in perwell),well_losses=sum(r['TOTAL_delta']<0 for r in perwell),
        exploratory_selection_unadjusted_TOTAL_95=interval,rows=144665,screening_fold=5,
        original_other_folds_started=False,independent_full_pipeline_OOF=False,old_delivery_unchanged=True))
    choices=[choice if choice is not None and c['target']=='SW' else c for c in previous['choices']]
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    timing=dict(auxiliary_process_seconds={str(r['group']):r['seconds'] for r in aux_reports},
        head_three_arm_process_seconds=reports['seconds'],head_peak_allocated_MiB=reports['peak_allocated_MiB'],
        maximum_actual_process_overlap=peak,first_launch_to_last_finish_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),
        input_preparation_seconds=read(STUDY/'auxiliary_manifest.json')['seconds'],finalization_seconds=time.perf_counter()-started,
        scope='Auxiliary fitting/inference included per group. Total wall includes explicit launch gaps; code authoring/input prep/finalization not part of first-to-last interval. No speedup claim.')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v34 跨目标预测协同完成结果','',f"完成时间：{result['finished_at']}",
        f"原TOTAL={refs['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 探索重点与真实完成范围','',
        '原模型保持，12个新辅助MLP（4组×3目标）完成，随后训练3个SW权重修正头。原POR/PERM输出未变化。所有辅助模型不使用当前强模型权重；这是微型辅助预测试验，不是当前冠军POR/PERM的完整交叉拟合重训。',
        '每组40井拟合，8井NLL早停，16井接收OOF预测；查询井同时排除于拟合和早停，特征统计只拟合40井。原第五折不进入辅助拟合、统计或停止选择。',
        '', '## 原第五折SW对照','',
        '|方案|参数|更新后最高SW|最终SW|最佳/停止轮次|','|---|---:|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{r['trainable_parameters']}|{row['post_update_scores']['SW']:.9f}|{row['selected_scores']['SW']:.9f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','第0轮回退不记作训练收益。shuffled只作负对照，不允许部署。',
        '机制对照：'+json.dumps(comparison,ensure_ascii=False),
        f"跨目标机制是否通过预设比较={cross_evidence}；新采用={best['mode'] if adopt else '无，保留原模型'}。",
        '', '## 来源、信息范围与限制','',
        '每个辅助目标输出六维描述，self重复SW自身描述，cross拼接POR/PERM，shuffled在同一OOF组内打乱cross。三个头均331→32→16、11152参数、同初始化、lr3e-4和80/18；只改SW16分量权重。',
        '辅助输入来自预测，没有读取真实其他目标值；打乱也不跨留出组，避免间接换入见过查询井的模型输出。原第五折使用四模型描述均值，与训练单模型OOF存在分布差异；平均分位数不等于混合分位数。',
        '当前SW主模型曾在原64井训练，原第五折长期参与选模，因此本批只有辅助预测来源是井外OOF，整个系统并非独立OOF验证。',
        '辅助质量诊断：'+json.dumps(quality,ensure_ascii=False),
        '', '## 验证与保留','',
        f"三个头训练内最佳恢复、冻结状态及原评分行序均通过。新采用模型额外磁盘重建：{replayed}；无采用则不重复原模型推理。三列按原行序合并实测。",
        f"组合{result['well_wins']}井升/{result['well_losses']}井降；探索性未校正区间{result['exploratory_selection_unadjusted_TOTAL_95']}。预测未变时[0,0]不构成新增证据。",
        '', '## 不继续和未完成','',
        '如没有同时超过原模型和两个对照，不扩展本批MLP宽度、描述项或学习率网格；结果只约束本次微型辅助来源与固定权重头，不否定全部跨目标方法。',
        '当前强POR/PERM完整骨干的嵌套交叉拟合、跨井检索、状态序列模型、新井确认均未实施。原其他四折继续暂停，旧交付未改。',
        '', '## 证据文件','',
        'METHOD.md、auxiliary_manifest.json、4份aux_config、feature_manifest.json、assembled_report.json、CURRENT_RECIPE.json、timing.json；完整运行与最佳参数位于outputs/pure_dl_cross_target_v34_*。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (STUDY/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v34跨目标协同完成'
    note=marker+'\n'+f"12个井外辅助模型及3项SW头完成，TOTAL={scores['TOTAL']:.12f}，增量={result['delta_reference_TOTAL']:+.12f}。采用={result['selected_mode']}；微型辅助信号，不是强模型全部OOF重训。无原其他折/交付变更。详见cross_target_v34_20260929/FINAL.md。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md',STUDY/'PROJECTS.md'):append(p,marker,note)
    for g in range(4):
        p=ROOT/'outputs'/f'pure_dl_cross_target_v34_aux{g}'
        for file in ('README.md','PROJECTS.md'):append(p/file,marker,note)
    for file in ('README.md','PROJECTS.md'):append(out/file,marker,note)
    for p in (ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md'):append(p,'## v34当前配方','## v34当前配方\n'+f"当前TOTAL={scores['TOTAL']:.12f}；cross_target_v34_20260929/CURRENT_RECIPE.json。辅助OOF与整体泛化区别见报告，有限候选完成，不自动开其他方向/原其他折。")
    failures=[]
    for p in [STUDY]+[ROOT/'outputs'/f'pure_dl_cross_target_v34_aux{g}' for g in range(4)]+[out]:
        if (p/'failure.json').exists():failures.append(read(p/'failure.json'))
    atomic_json(STUDY/'execution_notes.json',dict(active_jobs=jobs(),failures=failures,auxiliary_source_scopes_checked=True,
        all_three_heads_finished=True,all_twelve_auxiliary_finished=True,negative_control_not_deployable=True,
        no_new_external_data=True,no_original_other_folds=True))
    print('FINAL_V34',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],comparisons=comparison,cross_evidence=cross_evidence,
        adopted=result['selected_mode'],candidates=[dict(mode=r['mode'],post=r['post_update_scores']['SW'],final=r['selected_scores']['SW']) for r in records],timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
