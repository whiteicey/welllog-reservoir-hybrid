"""Actual fixed three-column scoring, replay positive POR, recorded negative cleanup."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from por_joint_utility_v86 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,SOURCE
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    tick=time.perf_counter();out=ROOT/'outputs/pure_dl_por_joint_utility_v86_POR_fold5'
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
        selected=diagnostics(truth,reference,pred,wells,'POR')
        updated=diagnostics(truth,reference,np.load(arm/'best_updated_pred.npy'),wells,'POR')
        assert abs(selected['scores']['POR']-rep['target_score'])<1e-9
        assert abs(selected['scores']['TOTAL']-rep['reference_replacement_total'])<1e-9
        assert abs(updated['scores']['POR']-rep['best_after_update']['target_score'])<1e-9
        assert rep['all_frozen_state_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        assert rep['trainable_parameters']==30819 and rep['gradient_scope']['decision_gradient_norm']>0
        initial=read(arm/'initial_verification.json')
        assert initial['all144665_exact'] and abs(initial['score']-refs['POR'])<1e-9
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,0])
        candidates.append(dict(mode=mode,report=rep,selected=selected,post_update=updated,initial=initial))
        predictions[mode]=pred
    ctrl=reference.copy();ctrl[:,0]=predictions['likelihood']
    contrast=diagnostics(truth,ctrl,predictions['joint'],wells,'POR')
    updated_delta=candidates[1]['post_update']['channel_delta']-candidates[0]['post_update']['channel_delta']
    best=max(candidates,key=lambda r:r['selected']['scores']['POR'])
    adopted=best['selected']['scores']['POR']>refs['POR']+1e-9
    retained=[r['mode'] for r in candidates if r['selected']['scores']['POR']>refs['POR']+1e-9]
    if contrast['channel_delta']>1e-9 and 'joint' not in retained:retained.append('joint')
    combined=reference.copy();choices=[dict(c) for c in previous['choices']];verification=[]
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    for mode in retained:
        data,master=load_data()
        assert np.array_equal(truth,data['truth'][data['validation']])
        assert np.array_equal(wells,data['wells'][data['validation']])
        cp=torch.load(out/mode/'model_best.pt',map_location='cpu',weights_only=False)
        model=Network('POR',master,modes=(mode,)).cuda()
        model.heads[mode].load_state_dict(cp['delta_state'],strict=True)
        pred,score,total=Evaluator(data,master,0,reference).many(model,(mode,))[mode]
        assert np.array_equal(pred,predictions[mode])
        verification.append(dict(mode=mode,rows=144665,fresh_disk_replay_exact=True,score=score))
        if adopted and mode==best['mode']:
            combined[:,0]=pred
            choices[0].update(key='v86_POR_'+mode,mode=mode,checkpoint=str((out/mode/'model_best.pt').relative_to(ROOT)),
                model_module='por_joint_utility_v86',best_epoch=cp['best_epoch'],target_score=score,
                parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
                builder='Network(POR,master,modes=(mode,)); heads[mode].load_state_dict(delta_state)')
        model.close();del model,cp,data,master;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined)
    assert np.array_equal(combined[:,1:],reference[:,1:])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    deleted=[]
    for mode in MODES:
        if mode in retained:continue
        for name in ('model_best.pt','model_best_updated.pt'):
            path=(out/mode/name).resolve()
            assert path.parent==out.resolve()/mode
            if path.exists():
                deleted.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                path.unlink()
    last=(out/'last_state.pt').resolve();assert last.parent==out.resolve()
    if last.exists():
        deleted.append(dict(path=str(last.relative_to(ROOT)),bytes=last.stat().st_size,sha256=hashlib.sha256(last.read_bytes()).hexdigest()));last.unlink()
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(r['bytes'] for r in deleted),retained_modes=retained,status='complete'))
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],
        candidates=candidates,joint_minus_likelihood=contrast,post_update_joint_minus_likelihood=updated_delta,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,retained=retained,verification=verification,
        actual_fixed_three_column_scoring=True,screening_fold=5,rows=144665,strict805_passed=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result);atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(group['finished_at'])
    timing=dict(shared_process_seconds=group['seconds'],launch_to_last_completion_seconds=(end-launch).total_seconds(),
        launch_to_verified_seconds=(datetime.now().astimezone()-launch).total_seconds(),peak_training_processes=1,
        peak_allocated_MiB=group['peak_allocated_MiB'],epoch_timing=group['epoch_timing'],finalize_seconds=time.perf_counter()-tick,
        scope='two independent heads sequentially update in one process, frozen extraction shared; no invented per-arm wall time; loading/warmup/eval/save included, authoring/atlas separate')
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v86 POR联合动作效用最终结果','',f"当前独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|方案|POR|固定PERM/SW TOTAL|最佳/停止轮|训练后最高POR|','|---|---:|---:|---:|---:|']
    for r in candidates:
        p=r['report'];lines.append(f"|{r['mode']}|{p['target_score']:.12f}|{p['reference_replacement_total']:.12f}|{p['best_epoch']}/{p['stopped_epoch']}|{p['best_after_update']['target_score']:.12f}|")
    lines+=['','## 探索与验证','同30819既有anchored路由、同起点，原专家/骨干/位置头冻结，无新增容量。效用只读POR训练标签，t3安全尾CDF/原动作/0.08容差/log1p逆变换保持；0/1系数、温度.05、Q50、80上限/pat18，新3e-5及cos到1e-5。',
        '实际第二batch效用梯度非零，来源/预处理/source hash、冻结状态、最佳恢复、144665点原行序和三列评价通过；正候选磁盘重建='+str(verification)+'。',
        '', '## 收益与不继续',f"joint-likelihood最终POR {contrast['channel_delta']:+.12f}；训练后最高差 {updated_delta:+.12f}。采用={result['selected_mode']}，保留={retained}。epoch0恢复不算新收益，更新后最高为独立诊断，不替代最终选择。"]
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
    note=f"## v86 POR训练目标比较完成\n当前TOTAL={scores['TOTAL']:.12f}，变化{result['delta_reference_TOTAL']:+.12f}；joint-likelihood POR {contrast['channel_delta']:+.12f}，更新后最高差{updated_delta:+.12f}。当前配方results/deep_learning/por_joint_utility_v86_20260930/CURRENT_RECIPE.json；无其他折。"
    for path in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append(path,'## v86 POR训练目标比较完成',note)
    for path in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):
        append(path,'## v86当前研究配方',note.replace('## v86 POR训练目标比较完成','## v86当前研究配方'))
    print(json.dumps(native(dict(scores=scores,delta=result['delta_reference_TOTAL'],matched_delta=contrast['channel_delta'],updated_delta=updated_delta,retained=retained,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
