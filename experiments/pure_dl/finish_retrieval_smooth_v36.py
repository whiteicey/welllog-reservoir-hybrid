"""Finalize gate diagnosis and valid smooth-gate retrieval comparison."""
from pathlib import Path
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
import retrieval_smooth_v36 as module
from run_retrieval_smooth_v36 import ROOT,STUDY,OUT,jobs,append
from train import metric,dmask
from prepare_boundary_v33 import kernels
from train_dual8222_heads80 import atomic_json,atomic_array

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def native(v):
    if isinstance(v,dict):return {k:native(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [native(x) for x in v]
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    return v

def main():
    start=time.perf_counter()
    if jobs() or not (OUT/'fold_5_report.json').exists():raise RuntimeError('Both candidates must finish first')
    report=read(OUT/'fold_5_report.json');cfg=read(OUT/'config.json')
    for name,digest in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    previous=read(module.PREVIOUS/'CURRENT_RECIPE.json');reference=module.reference_prediction()
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=cache['truth'][cache['validation']];wells=cache['wells'][cache['validation']]
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert max(abs(refs[k]-v) for k,v in previous['scores'].items())<1e-9
    memory=module.MEMORY_STUDY
    with np.load(memory/'memory.npz') as z:
        rw=z['well'].copy();rawref=z['raw_rows'].copy();perm=z['permutation'].copy()
    assert np.isin(rawref,cache['train_raw'].numpy()).all() and np.array_equal(rw,rw[perm])
    rawwell=np.repeat(np.arange(80),np.diff(cache['starts']))
    ids=np.load(memory/'neighbor_ids.npy',mmap_mode='r')
    for lo in range(0,len(ids),8192):assert not (rw[ids[lo:lo+8192]]==rawwell[lo:lo+8192,None]).any()
    del cache,ids;gc.collect()
    pu=kernels(truth,reference)[:,2];dm=dmask(truth)[:,2]
    rows=[];predictions={}
    for rep in report['arms']:
        m=rep['mode'];p=np.load(OUT/m/'candidate_pred.npy');up=np.load(OUT/m/'best_updated_pred.npy')
        temp=reference.copy();temp[:,2]=p;scores=metric(truth,temp)
        updated=reference.copy();updated[:,2]=up;upscore=metric(truth,updated)
        assert abs(scores['SW']-rep['candidate_score'])<1e-9 and abs(upscore['SW']-rep['best_after_update']['target_score'])<1e-9
        assert rep['candidate_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        du=100*(kernels(truth,temp)[:,2]-pu)
        pops={name:dict(n=int(mask.sum()),SW_delta=float(du[mask].mean()),TOTAL_contribution=float(.35*du[mask].sum()/len(truth))) for name,mask in [('default',dm),('numeric',~dm)]}
        rows.append(dict(mode=m,report=rep,candidate_scores=scores,post_update_scores=upscore,populations=pops,
            changed_predictions=int((p!=reference[:,2]).sum()),initial=read(OUT/m/'initial_verification.json')))
        predictions[m]=p
    matched=next(x for x in rows if x['mode']=='matched');shuffled=next(x for x in rows if x['mode']=='shuffled')
    above_parent=matched['candidate_scores']['SW']>refs['SW']+1e-9
    above_control=matched['candidate_scores']['SW']>shuffled['candidate_scores']['SW']+1e-9
    adopted=above_parent and above_control
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];replay=[]
    if adopted:
        torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
        data,master=module.load_data();model=module.Network('SW',master,modes=('matched',)).cuda()
        checkpoint=OUT/'matched/model_best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.heads['matched'].load_state_dict(saved['delta_state'],strict=True)
        p,score,_=module.Evaluator(data,master,2,reference).many(model,('matched',))['matched']
        assert np.array_equal(p,predictions['matched']) and abs(score-matched['candidate_scores']['SW'])<1e-9
        combined[:,2]=p;replay=['SW']
        choice=dict(target='SW',key='v36_SW_matched',architecture='frozen_SW_plus_smooth_retrieved_conditional_density',model_module='retrieval_smooth_v36',mode='matched',
            checkpoint=str(checkpoint.relative_to(ROOT)),target_score=score,best_epoch=saved['epoch'],
            parent_recipe=str((module.PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),memory_manifest=str((memory/'memory_manifest.json').relative_to(ROOT)),
            builder='Network(SW,master,modes=(matched,)); heads[matched].load_state_dict(delta_state); forward requires raw rows for neighbor IDs',
            additional_trainable_parameters=11266,original_default_probability_fixed=True)
        choices=[choice if c['target']=='SW' else c for c in choices]
        model.close();del model,saved,data,master;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);perwell=[];pairwell=[]
    for w in np.unique(wells):
        mask=wells==w;parent=metric(truth[mask],reference[mask])['TOTAL'];n=int(mask.sum())
        perwell.append(dict(well=int(w),n=n,TOTAL_delta=metric(truth[mask],combined[mask])['TOTAL']-parent))
        a=reference[mask].copy();a[:,2]=predictions['matched'][mask]
        b=reference[mask].copy();b[:,2]=predictions['shuffled'][mask]
        pairwell.append(dict(well=int(w),n=n,TOTAL_delta=metric(truth[mask],a)['TOTAL']-metric(truth[mask],b)['TOTAL']))
    rng=np.random.default_rng(20261136);sample=rng.integers(0,16,(10000,16));n=np.array([x['n'] for x in perwell]);d=np.array([x['TOTAL_delta'] for x in perwell]);dp=np.array([x['TOTAL_delta'] for x in pairwell])
    interval=np.quantile((n[sample]*d[sample]).sum(1)/n[sample].sum(1),[.025,.975])
    pair_interval=np.quantile((n[sample]*dp[sample]).sum(1)/n[sample].sum(1),[.025,.975])
    diag=read(memory/'gate_diagnostic.json')
    result=native(dict(finished_at=datetime.now().astimezone().isoformat(),reference_scores=refs,selected_scores=scores,
        delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,adopted=adopted,selected_mode='matched' if adopted else None,
        matched_minus_shuffled_SW=matched['candidate_scores']['SW']-shuffled['candidate_scores']['SW'],
        mechanism_positive_development_only=adopted,all_two_complete=True,actual_combined_prediction_scored=True,
        new_target_full_disk_replays=replay,whole_query_well_exclusion_checked=True,memory_labels_training_only=True,
        per_well=perwell,well_wins=sum(x['TOTAL_delta']>0 for x in perwell),well_losses=sum(x['TOTAL_delta']<0 for x in perwell),
        exploratory_selection_unadjusted_TOTAL_95=interval,matched_minus_shuffled_TOTAL_95=pair_interval,
        v35_record='inactive last-state retrieval; not sufficient negative mechanism evidence',
        rows=144665,screening_fold=5,complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((module.PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    launch=read(OUT/'launch.json');timing=dict(training_and_eval_seconds=report['seconds'],peak_allocated_MiB=report['peak_allocated_MiB'],
        launch_to_completion_seconds=(datetime.fromisoformat(report['finished_at'])-datetime.fromisoformat(launch['started_at'])).total_seconds(),
        target_processes=1,candidates=2,scope='One process, two independent heads sharing frozen forward; existing memory reused, no new indexing. Excludes code authoring and finalization.',
        finalization_seconds=time.perf_counter()-start)
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v36：跨井检索平滑门控对照','',f"完成：{result['finished_at']}",
        f"原TOTAL={refs['TOTAL']:.12f}；实际组合TOTAL={scores['TOTAL']:.12f}；增量={result['delta_reference_TOTAL']:+.12f}。",'',
        '## 先纠正v35实验解释','',
        'v35已结束；末态64固定训练锚点的7973有效点上，两组proposal全部负，硬clip使alpha与全部分支梯度为0。初始输出层有梯度，查询/键无梯度；18轮验证分数恒定。这是学习通路失活，不能当作充分训练后的检索负证据。权重变化不代表有效梯度学习。旧源码/权重不改，gate_diagnostic.json单独保存。','',
        '## 本批改变与不变','',
        '仅替换参与门控为alpha=.25*sigmoid(u+logit(.04))，初始1%，输出最大25%。不使用替代梯度。初始候选分布不等于父分布；关闭检索时才逐点复现父模型。参数、参考库4096、Top16、27维距离、32维重排序、带宽机制与原NLL/Q50/80-18均保持。两个11266参数头同初始化，仅参考标签在其井内部是否打乱不同。','',
        '## 实测候选','',
        '|候选|初始SW|训练后最高SW|候选最佳轮|停止轮|相对原模型TOTAL|',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        q=row['report'];lines.append(f"|{row['mode']}|{q['initial_candidate_score']:.9f}|{q['best_after_update']['target_score']:.9f}|{q['candidate_epoch']}|{q['stopped_epoch']}|{row['candidate_scores']['TOTAL']-refs['TOTAL']:+.9f}|")
    lines+=['',f"matched-shuffled通道差={result['matched_minus_shuffled_SW']:+.12f}；采用={result['selected_mode']}。shuffled仅负对照不可部署。",'',
        '## 学习是否生效','']
    for row in rows:
        q=row['report'];lines.append(row['mode']+' 首次梯度：'+json.dumps(q['first_gradient'],ensure_ascii=False)+'；最后梯度：'+json.dumps(q['last_gradient'],ensure_ascii=False))
        lines.append('最佳候选固定训练探针：'+json.dumps(q['best_candidate_probe'],ensure_ascii=False))
        lines.append('验证子集变化：'+json.dumps(row['populations'],ensure_ascii=False)+f"；实际预测变化点数{row['changed_predictions']}。")
    lines+=['','## 收益边界与停止方向','',
        f"最终{result['well_wins']}井升/{result['well_losses']}井降；未校正选择的整井重采样TOTAL区间{result['exploratory_selection_unadjusted_TOTAL_95']}，matched-shuffled区间{result['matched_minus_shuffled_TOTAL_95']}。这仍是反复使用的开发折，无独立新井/种子确认。",
        '只满足数值上的正比较不代表有稳定收益；不据微小差值扩大K、库规模、带宽或初始比例扫描。固定候选池内学习重排序，不是全库学习度量。若需要新路线，应从来源质量与训练/推理误差差异重新提出独立假设。',
        '当前预测器在64井拟合过，参考仅排除查询整井，不代表整个训练系统完成了交叉拟合。对条件分布混合，默认概率固定不等于最终默认动作次数固定。',
        '', '## 已完成验证与未做','',
        f"候选最佳恢复全144665点一致，原冻结状态未改。入选SW额外磁盘重建目标={replay}，三列实际组合评分完成。所有参考原行位于64训练井；全730268行的候选均排除查询整井，打乱保持来源井。",
        '原其他四折、强模型完整交叉拟合、全库学习检索、状态序列路线均未启动。旧交付保留。',
        '', '## 文件','',
        'assembled_report.json、CURRENT_RECIPE.json、timing.json；v35/memory_manifest.json与gate_diagnostic.json；outputs/pure_dl_retrieval_smooth_v36_SW_fold5中的梯度、状态与候选权重。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8');(STUDY/'README.md').write_text(text,encoding='utf-8')
    marker='## v36检索门控对照完成'
    note=marker+'\n'+f"两项已完成，TOTAL={scores['TOTAL']:.12f}，增量={result['delta_reference_TOTAL']:+.12f}；采用={result['selected_mode']}。v35失活证据和v36有效训练区别已记录。所有旧模型/交付保留，无其他折；见retrieval_smooth_gate_v36_20260929/FINAL.md。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md',STUDY/'PROJECTS.md',OUT/'PROJECTS.md',OUT/'README.md'):append(p,marker,note)
    for p in (ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md'):append(p,'## v36当前配方','## v36当前配方\n'+f"当前开发TOTAL={scores['TOTAL']:.12f}；配方retrieval_smooth_gate_v36_20260929/CURRENT_RECIPE.json。v35门控失活不作为充分负证据，v36已真实训练。无自动后续扫描或其他折，旧交付不变。")
    old_final='# v35 收尾：门控失活\n\n原两项均第18轮完成，全部验证分数保持73.6630827970656。后续末态检查确认硬截断门控全关闭且梯度为0；不能将其作为充分训练后的跨井检索负证据。详见gate_diagnostic.json。历史源和checkpoint未改。v36用平滑门控作独立修正，详见../retrieval_smooth_gate_v36_20260929/FINAL.md。\n'
    (memory/'FINAL.md').write_text(old_final,encoding='utf-8');append(memory/'PROJECTS.md','## v35最终解释',old_final.replace('# v35 收尾：门控失活','## v35最终解释'))
    failures=[]
    if (OUT/'failure.json').exists():failures.append(read(OUT/'failure.json'))
    atomic_json(STUDY/'execution_notes.json',dict(observed_at=datetime.now().astimezone().isoformat(),active_jobs=jobs(),failures=failures,
        all_two_complete=True,historical_training_sources_unchanged=True,old_delivery_unchanged=True,reference_memory_row_exclusion_checked=True))
    print('FINAL_V36',json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],adopted=adopted,matched_minus_shuffled_SW=result['matched_minus_shuffled_SW'],
        candidates=[dict(mode=r['mode'],initial=r['report']['initial_candidate_score'],best_after_update=r['post_update_scores']['SW'],populations=r['populations'],probe=r['report']['best_candidate_probe']) for r in rows],interval=interval,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
