"""Actual fixed three-column scoring, replay positive SW, recorded negative cleanup."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from sw_frozen_direct_v91 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,SOURCE
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    tick=time.perf_counter();out=ROOT/'outputs/pure_dl_sw_frozen_direct_v91_SW_fold5'
    if jobs():raise RuntimeError('Wait for active training to complete')
    assert read(out/'runtime_status.json')['status']=='complete'
    group=read(out/'fold_5_report.json');cfg=read(out/'config.json')
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['parent_sha256']==cfg['metadata']['sha256']
    n=cfg['target_normalization'];assert hashlib.sha256((ROOT/n['checkpoint']).read_bytes()).hexdigest()==n['sha256']
    assert hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest()==cfg['scoring_kernel_sha256']
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=np.load(PREVIOUS/'assembled_pred.npy')
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    candidates=[];predictions={}
    for mode in MODES:
        arm=out/mode;rep=read(arm/'fold_5_report.json');pred=np.load(arm/'channel_pred.npy')
        selected=diagnostics(truth,reference,pred,wells,'SW')
        updated=diagnostics(truth,reference,np.load(arm/'best_updated_pred.npy'),wells,'SW')
        assert abs(selected['scores']['SW']-rep['target_score'])<1e-9
        assert abs(selected['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
        assert abs(updated['scores']['SW']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all_frozen_state_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        assert rep['trainable_parameters']==2425 and rep['gradient_scope']['decision_gradient_norm']>0
        initial=read(arm/'initial_verification.json')
        assert initial['all144665_exact'] and abs(initial['score']-refs['SW'])<1e-9
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,2])
        candidates.append(dict(mode=mode,report=rep,selected=selected,post_update=updated,initial=initial))
        predictions[mode]=pred
    ctrl=reference.copy();ctrl[:,2]=predictions['reference']
    contrast=diagnostics(truth,ctrl,predictions['well'],wells,'SW')
    updated_delta=candidates[1]['post_update']['channel_delta']-candidates[0]['post_update']['channel_delta']
    matched_modes=list(MODES) if updated_delta>1e-9 else []
    best=max(candidates,key=lambda r:r['selected']['scores']['SW'])
    adopted=best['selected']['scores']['SW']>refs['SW']+1e-9
    retained=[r['mode'] for r in candidates if r['selected']['scores']['SW']>refs['SW']+1e-9]
    if updated_delta>1e-9 and 'well' not in retained:retained.append('well')
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];verification=[]
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    for mode in sorted(set(retained)|set(matched_modes)):
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']])
        assert np.array_equal(wells,data['wells'][data['validation']])
        use_updated=mode in matched_modes
        cp=torch.load(out/mode/('model_best_updated.pt' if use_updated else 'model_best.pt'),map_location='cpu',weights_only=False)
        model=Network('SW',master).cuda()
        model.heads[mode].load_state_dict(cp['delta_state'],strict=True)
        pred,score,total=Evaluator(data,master,2,reference).many(model,(mode,))[mode]
        expected=np.load(out/mode/'best_updated_pred.npy') if use_updated else predictions[mode]
        assert np.array_equal(pred,expected)
        verification.append(dict(mode=mode,rows=144665,fresh_disk_replay_exact=True,score=score,updated_checkpoint=use_updated))
        if adopted and mode==best['mode'] and np.array_equal(pred,predictions[mode]):
            combined[:,2]=pred
            choices[2].update(key='v91_SW_'+mode,mode=mode,checkpoint=str((out/mode/'model_best.pt').relative_to(ROOT)),
                model_module='sw_frozen_direct_v91',best_epoch=cp.get('best_epoch',cp.get('epoch')),target_score=score,
                parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                builder='Network(SW,master); heads[mode].load_state_dict(delta_state); Evaluator.many(model,(mode,))')
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert np.array_equal(combined[:,:2],reference[:,:2])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    manifest=STUDY/'deletion_manifest.json'
    deleted=read(manifest).get('deleted',[]) if manifest.exists() else [];plan=[]
    for mode in MODES:
        for name in ('model_best.pt','model_best_updated.pt'):
            if name=='model_best.pt' and mode in retained:continue
            if name=='model_best_updated.pt' and mode in matched_modes:continue
            path=(out/mode/name).resolve();assert path.parent==out.resolve()/mode
            if path.exists():plan.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    last=(out/'last_state.pt').resolve();assert last.parent==out.resolve()
    if last.exists():plan.append(dict(path=str(last.relative_to(ROOT)),bytes=last.stat().st_size,sha256=hashlib.sha256(last.read_bytes()).hexdigest()))
    atomic_json(STUDY/'deletion_plan.json',native(dict(planned=plan,status='planned_before_unlink',retained=retained,retained_updated=matched_modes)))
    for item in plan:
        path=(ROOT/item['path']).resolve();assert path==last or path.parent in [out.resolve()/m for m in MODES]
        path.unlink();deleted.append(item);atomic_json(manifest,native(dict(deleted=deleted,status='in_progress')))
    atomic_json(manifest,native(dict(deleted=deleted,bytes=sum(r['bytes'] for r in deleted),retained_modes=retained,retained_updated_modes=matched_modes,status='complete')))
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=candidates,well_minus_reference=contrast,post_update_well_minus_reference=updated_delta,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,retained=retained,retained_updated=matched_modes,verification=verification,
        actual_fixed_three_column_scoring=True,screening_fold=5,rows=144665,strict805_passed=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(group['finished_at'])
    timing=dict(shared_process_seconds=group['seconds'],launch_to_last_completion_seconds=(end-launch).total_seconds(),
        launch_to_verified_seconds=(datetime.now().astimezone()-launch).total_seconds(),peak_training_processes=1,
        peak_allocated_MiB=group['peak_allocated_MiB'],peak_reserved_MiB=group['peak_reserved_MiB'],epoch_timing=group['epoch_timing'],finalize_seconds=time.perf_counter()-tick,
        scope='two independent2425heads sequentially update in one GPU process, frozen extraction shared; no invented per-arm wall time; loading/warmup/eval/save included, authoring/atlas separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v91 SW冻结全井直接头结果','',f"当前独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|方案|SW|固定POR/PERM TOTAL|最佳/停止轮|训练后最高SW|','|---|---:|---:|---:|---:|']
    for r in candidates:
        p=r['report'];lines.append(f"|{r['mode']}|{p['target_score']:.12f}|{p['reference_replacement_total']:.12f}|{p['best_epoch']}/{p['stopped_epoch']}|{p['best_after_update']['target_score']:.12f}|")
    lines+=['','## 探索与验证','同2425零Linear96→25头、同当前Gaussian初始全预测；原v82最佳25轮完整SW模型839481参数固定，原FiLM两组读本井。新支路只差训练64井参考code/当前本井code，两code都计算；共享冻结前向eval无DropKey。原9曲线206/输入bank/分折/Q50/原256支持800动作不改，仅SW BCE+GaussianNLL+效用1/3，温度.05，新head1e-4/cos至1e-5/80上限pat18。与v90全网络训练不同，不能将两批差归因单因素。',
        '实际第二batch头/效用梯度非零，来源/预处理/source hash、冻结无梯度/终态不变、初始和最佳144665点恢复、原行序和三列评价通过；正候选磁盘重建='+str(verification)+'。',
        '', '## 收益与不继续',f"well-reference含初始最终SW {contrast['channel_delta']:+.12f}；训练后最高差 {updated_delta:+.12f}。采用={result['selected_mode']}，保留={retained}/训练后对照={matched_modes}。epoch0恢复不算新收益，完整公平训练后最佳差分开报告，不替代当前模型选择；正增益两组更新后head/pred必须磁盘重载。"]
    for r in candidates:
        lines.append(r['mode']+' 默认/数值/井变化：'+json.dumps(native(r['post_update']['populations']),ensure_ascii=False)+f"；{r['post_update']['well_wins']}胜/{r['post_update']['well_losses']}负。")
    lines+=['若无绝对或预定匹配收益，关闭当前冻结Gaussian/共享96code/零25输出/1e-4/预算组合，不临时扫宽度LR强度，不泛化全部全井机制；单开发折选择非独立泛化确认。',
        '', '## 成本、清理、未完成与目标后',json.dumps(native(timing),ensure_ascii=False,indent=2),
        f"清理{len(deleted)}个无亮点/冗余状态，共{sum(r['bytes'] for r in deleted)}字节；正源和交付不改。",
        '未运行其他折或旧队列。原第五严格>80.5且明确收益后才考虑其余四折；完整原OOF>82未完成。目标后独立井验证和新目录交付。']
    final='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(final,encoding='utf-8')
    for path in (STUDY/'README.md',out/'README.md'):
        append(path,'## 最终结果','## 最终结果\n'+final)
    progress='目标：纯DL原完整OOF>82；当前仅固定第五。\n已做：两项真实训练、匹配对照、最佳恢复、实际组合、必要重载与清理。\n'+final
    for path in (STUDY/'PROJECTS.md',out/'PROJECTS.md'):
        path.write_text(progress,encoding='utf-8')
    note=f"## v91 SW冻结全井直接头完成\n当前TOTAL={scores['TOTAL']:.12f}，变化{result['delta_reference_TOTAL']:+.12f}；well-reference SW {contrast['channel_delta']:+.12f}，更新后最高差{updated_delta:+.12f}。当前配方results/deep_learning/sw_frozen_direct_v91_20260930/CURRENT_RECIPE.json；无其他折。"
    for path in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append(path,'## v91 SW冻结全井直接头完成',note)
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):
        append(path,'## v91当前研究配方',note.replace('## v91 SW冻结全井直接头完成','## v91当前研究配方'))
    append(ROOT/'EXPLORATIONS.md','v91完成：','- v91完成：[结果](results/deep_learning/sw_frozen_direct_v91_20260930/FINAL.md)，'+note)
    print(json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],matched_delta=contrast['channel_delta'],updated_delta=updated_delta,retained=retained,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
