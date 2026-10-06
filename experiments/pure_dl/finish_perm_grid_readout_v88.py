"""Actual fixed three-column scoring, replay positive PERM, recorded negative cleanup."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from perm_grid_readout_v88 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,SOURCE
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    tick=time.perf_counter();out=ROOT/'outputs/pure_dl_perm_grid_readout_v88_PERM_fold5'
    if jobs():raise RuntimeError('Wait for active training to complete')
    assert read(out/'runtime_status.json')['status']=='complete'
    group=read(out/'fold_5_report.json');cfg=read(out/'config.json')
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['preprocessing_sha256']
    source=ROOT/cfg['metadata']['source_checkpoint']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['metadata']['source_sha256']
    previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=np.load(PREVIOUS/'assembled_pred.npy')
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    candidates=[];predictions={}
    for mode in MODES:
        arm=out/mode;rep=read(arm/'fold_5_report.json');pred=np.load(arm/'channel_pred.npy')
        selected=diagnostics(truth,reference,pred,wells,'PERM')
        updated=diagnostics(truth,reference,np.load(arm/'best_updated_pred.npy'),wells,'PERM')
        assert abs(selected['scores']['PERM']-rep['target_score'])<1e-9
        assert abs(selected['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
        assert abs(updated['scores']['PERM']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all_frozen_state_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        assert rep['trainable_parameters']==37185 and rep['gradient_scope']['decision_gradient_norm']>0
        initial=read(arm/'initial_verification.json')
        assert initial['all144665_exact'] and abs(initial['score']-refs['PERM'])<1e-9
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,1])
        candidates.append(dict(mode=mode,report=rep,selected=selected,post_update=updated,initial=initial))
        predictions[mode]=pred
    ctrl=reference.copy();ctrl[:,1]=predictions['likelihood']
    contrast=diagnostics(truth,ctrl,predictions['joint'],wells,'PERM')
    updated_delta=candidates[1]['post_update']['channel_delta']-candidates[0]['post_update']['channel_delta']
    matched_modes=list(MODES) if updated_delta>1e-9 else []
    best=max(candidates,key=lambda r:r['selected']['scores']['PERM'])
    adopted=best['selected']['scores']['PERM']>refs['PERM']+1e-9
    retained=[r['mode'] for r in candidates if r['selected']['scores']['PERM']>refs['PERM']+1e-9]
    if updated_delta>1e-9 and 'joint' not in retained:retained.append('joint')
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];verification=[]
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    for mode in sorted(set(retained)|set(matched_modes)):
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']])
        assert np.array_equal(wells,data['wells'][data['validation']])
        use_updated=mode in matched_modes
        cp=torch.load(out/mode/('model_best_updated.pt' if use_updated else 'model_best.pt'),map_location='cpu',weights_only=False)
        model=Network('PERM',master,modes=(mode,)).cuda()
        model.heads[mode].load_state_dict(cp['delta_state'],strict=True)
        pred,score,total=Evaluator(data,master,1,reference).many(model,(mode,))[mode]
        expected=np.load(out/mode/'best_updated_pred.npy') if use_updated else predictions[mode]
        assert np.array_equal(pred,expected)
        verification.append(dict(mode=mode,rows=144665,fresh_disk_replay_exact=True,score=score,updated_checkpoint=use_updated))
        if adopted and mode==best['mode'] and np.array_equal(pred,predictions[mode]):
            combined[:,1]=pred
            choices[1].update(key='v88_PERM_'+mode,mode=mode,checkpoint=str((out/mode/'model_best.pt').relative_to(ROOT)),
                model_module='perm_grid_readout_v88',best_epoch=cp.get('best_epoch',cp.get('epoch')),target_score=score,
                parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                builder='Network(PERM,master,modes=(mode,)); heads[mode].load_state_dict(delta_state)')
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert np.array_equal(combined[:,[0,2]],reference[:,[0,2]])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    deleted=[]
    for mode in MODES:
        for name in ('model_best.pt','model_best_updated.pt'):
            if name=='model_best.pt' and mode in retained:continue
            if name=='model_best_updated.pt' and mode in matched_modes:continue
            path=(out/mode/name).resolve()
            assert path.parent==out.resolve()/mode
            if path.exists():
                deleted.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                path.unlink()
    last=(out/'last_state.pt').resolve();assert last.parent==out.resolve()
    if last.exists():
        deleted.append(dict(path=str(last.relative_to(ROOT)),bytes=last.stat().st_size,sha256=hashlib.sha256(last.read_bytes()).hexdigest()));last.unlink()
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(r['bytes'] for r in deleted),retained_modes=retained,retained_updated_modes=matched_modes,status='complete'))
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=candidates,joint_minus_likelihood=contrast,post_update_joint_minus_likelihood=updated_delta,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,retained=retained,retained_updated=matched_modes,verification=verification,
        actual_fixed_three_column_scoring=True,screening_fold=5,rows=144665,strict805_passed=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(group['finished_at'])
    timing=dict(shared_process_seconds=group['seconds'],launch_to_last_completion_seconds=(end-launch).total_seconds(),
        launch_to_verified_seconds=(datetime.now().astimezone()-launch).total_seconds(),peak_training_processes=1,
        peak_allocated_MiB=group['peak_allocated_MiB'],epoch_timing=group['epoch_timing'],finalize_seconds=time.perf_counter()-tick,
        scope='two independent37185heads sequentially update in one GPU process, frozen extraction shared; no invented per-arm wall time; loading/warmup/eval/save included, authoring/atlas separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v88 PERM条件概率格读出最终结果','',f"当前独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|方案|PERM|固定POR/SW TOTAL|最佳/停止轮|训练后最高PERM|','|---|---:|---:|---:|---:|']
    for r in candidates:
        p=r['report'];lines.append(f"|{r['mode']}|{p['target_score']:.12f}|{p['reference_replacement_total']:.12f}|{p['best_epoch']}/{p['stopped_epoch']}|{p['best_after_update']['target_score']:.12f}|")
    lines+=['','## 探索与验证','同37185小格头参数、同初始原函数，原PERM私有预测器冻结。log质量界0.5/gate界0.25，先验零格不能新增质量；原9曲线206+已有幅值63/分折/Q50/256支持800动作不改。标签为本目标训练log10双邻格软CE，效用0/1、温度.05；新head1e-4/cos至1e-5/80上限pat18。',
        '实际第二batch效用梯度非零，来源/预处理/source hash、冻结状态、最佳恢复、144665点原行序和三列评价通过；正候选磁盘重建='+str(verification)+'。',
        '', '## 收益与不继续',f"joint-likelihood含初始最终PERM {contrast['channel_delta']:+.12f}；训练后最高差 {updated_delta:+.12f}。采用={result['selected_mode']}，保留={retained}。epoch0恢复不算新收益，更新后最高为同容量完成对照，不替代当前模型选择；若正增益，两组更新后head/pred必须磁盘重载。"]
    for r in candidates:
        lines.append(r['mode']+' 默认/数值/井变化：'+json.dumps(native(r['post_update']['populations']),ensure_ascii=False)+f"；{r['post_update']['well_wins']}胜/{r['post_update']['well_losses']}负。")
    lines+=['若无绝对或预定匹配收益，关闭此来源/路由范围/系数1/温度.05组合，不泛化全部效用学习；单开发折选择非独立泛化确认。',
        '', '## 成本、清理、未完成与目标后',json.dumps(native(timing),ensure_ascii=False,indent=2),
        f"清理{len(deleted)}个无亮点/冗余状态，共{sum(r['bytes'] for r in deleted)}字节；正源和交付不改。",
        '未运行其他折或旧队列。原第五严格>80.5且明确收益后才考虑其余四折；完整原OOF>82未完成。目标后独立井验证和新目录交付。']
    final='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(final,encoding='utf-8')
    for path in (STUDY/'README.md',out/'README.md'):
        append(path,'## 最终结果','## 最终结果\n'+final)
    progress='目标：纯DL原完整OOF>82；当前仅固定第五。\n已做：两项真实训练、匹配对照、最佳恢复、实际组合、必要重载与清理。\n'+final
    for path in (STUDY/'PROJECTS.md',out/'PROJECTS.md'):
        path.write_text(progress,encoding='utf-8')
    note=f"## v88 PERM训练目标比较完成\n当前TOTAL={scores['TOTAL']:.12f}，变化{result['delta_reference_TOTAL']:+.12f}；joint-likelihood PERM {contrast['channel_delta']:+.12f}，更新后最高差{updated_delta:+.12f}。当前配方results/deep_learning/perm_grid_readout_v88_20260930/CURRENT_RECIPE.json；无其他折。"
    for path in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append(path,'## v88 PERM训练目标比较完成',note)
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):
        append(path,'## v88当前研究配方',note.replace('## v88 PERM训练目标比较完成','## v88当前研究配方'))
    print(json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],matched_delta=contrast['channel_delta'],updated_delta=updated_delta,retained=retained,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
