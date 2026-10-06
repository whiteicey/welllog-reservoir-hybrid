"""Finalize internal PERM adapters, verify retained models, and remove negatives."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from perm_cdfwarp_v70 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training still active')
    tick=time.perf_counter();previous=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    rows=[];preds={};updates={};costs=[];hashes=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_perm_cdfwarp_v70_PERM_{mode}_fold5';cfg=read(out/'config.json');rep=read(out/'fold_5_report.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        pred=np.load(out/'channel_pred.npy');updated=np.load(out/'best_updated_pred.npy')
        diag=diagnostics(truth,reference,pred,wells,'PERM');up=diagnostics(truth,reference,updated,wells,'PERM')
        assert abs(diag['scores']['PERM']-rep['target_score'])<1e-9 and abs(up['scores']['PERM']-rep['best_after_update']['target_score'])<1e-9
        assert read(out/'initial_verification.json')['all144665_exact']
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        assert rep['gradient_scope']['frozen_gradients_none'] and rep['last_parameter_change_L2']>0
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,1])
        rows.append(dict(mode=mode,report=rep,selected=diag,post_update=up));preds[mode]=pred;updates[mode]=updated
        hashes.append(cfg['initial_trainable_sha256'])
        launch=read(out/'launch.json');costs.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]==hashes[1]
    ctrl=reference.copy();ctrl[:,1]=preds['affine'];contrast=diagnostics(truth,ctrl,preds['nonlinear'],wells,'PERM')
    upctrl=reference.copy();upctrl[:,1]=updates['affine'];upcontrast=diagnostics(truth,upctrl,updates['nonlinear'],wells,'PERM')
    best=max(rows,key=lambda x:x['report']['target_score']);adopted=bool(best['report']['target_score']>refs['PERM']+1e-9)
    retained=set()
    if adopted:retained.add(best['mode'])
    if contrast['channel_delta']>0:retained.add('nonlinear')
    verification=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;data,master=load_data()
        for mode in sorted(retained):
            out=ROOT/f'outputs/pure_dl_perm_cdfwarp_v70_PERM_{mode}_fold5'
            model=Network('PERM',master,mode).cuda();saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
            model.load_trainable_state(saved['backbone_state']);pred,score,total=Evaluator(data,master,1,reference).many(model,(mode,))[mode]
            assert np.array_equal(pred,preds[mode]) and abs(score-saved['target_score'])<1e-9
            verification.append(dict(mode=mode,rows=len(pred),full_raw_disk_replay_exact=True,target_score=score,TOTAL=total,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            model.close();del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    combined=reference.copy();choices=[dict(c) for c in previous['choices']]
    if adopted:
        mode=best['mode'];combined[:,1]=preds[mode]
        choice=dict(target='PERM',key='v70_PERM_'+mode,mode=mode,architecture='frozen_dual_PERM_polynomial_CDF_warp',model_module='perm_cdfwarp_v70',
            checkpoint=best['report']['checkpoint'],best_epoch=best['report']['best_epoch'],target_score=best['report']['target_score'],
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),added_parameters=best['report']['trainable_parameters'],
            builder='load_data(); Network(PERM,master,mode); load_trainable_state(checkpoint[backbone_state]); Evaluator.many dict[mode]')
        choices=[choice if c['target']=='PERM' else c for c in choices]
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,retained=sorted(retained),verification=verification,
        nonlinear_minus_affine_selected=contrast,nonlinear_minus_affine_post_update=upcontrast,initial_trainable_hash_equal=True,
        actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=sorted([(datetime.fromisoformat(x['start']),1) for x in costs]+[(datetime.fromisoformat(x['end']),-1) for x in costs]);live=peak=0
    for _,change in events:live+=change;peak=max(peak,live)
    timing=dict(candidates=costs,first_launch_to_last_completion_seconds=(events[-1][0]-events[0][0]).total_seconds(),peak_process_concurrency=peak,
        finalization_and_replay_seconds=time.perf_counter()-tick,scope='individual includes load/warmup/train/eval; actual batch includes launch gaps; authoring/finalization separate')
    atomic_json(STUDY/'timing.json',timing)
    deletion_path=STUDY/'deletion_manifest.json'
    deleted=read(deletion_path)['deleted'] if deletion_path.exists() else []
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_perm_cdfwarp_v70_PERM_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        assert read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,reason='adopted or positive matched nonlinear access gain' if mode in retained else 'no improvement over parent and no demonstrated nonlinear-access gain',no_automatic_resume=True))
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained)))
    lines=['# v70 PERM条件CDF校准结果','',f"第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高PERM|含起点PERM|最佳/停止轮|','|---|---:|---:|---|']
    for row in rows:
        r=row['report'];lines.append(f"|{row['mode']}|{r['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与验证',
        '同10306新参数、319→32→2条件CDF小头，affine中间恒等/nonlinear GELU，相同张量与零输出。原v29 coupled PERM全冻结。G(F)=F+uF(1-F)+vF(1-F)(1-2F)，u/v各界±0.25，端点固定且导数≥0.5；训练原混合Gaussian logpdf加log导数，原门控不更新。验证原256格概率累计+校正差保持零头逐位一致，原800动作/评分不变。',
        '原9曲线/206+已有63侧特征/Q50/原第五/256中心128/80-18保持，新增头统一3e-4。实际新参数梯度有效、原参数无梯度且状态不变，初始及最佳144665点、来源哈希和实际三列评分核验。',
        f"保留者原始输入磁盘重建：{verification}。",'', '## 收益与不继续',
        f"nonlinear-affine含起点PERM={contrast['channel_delta']:+.12f}，训练后最佳差={upcontrast['channel_delta']:+.12f}；采用={result['selected_mode']}，保留={sorted(retained)}。",
        '起点回退不是学习收益。若nonlinear无配对收益，不扫头宽度/激活；若两组均无收益，不扫当前多项式阶数/幅度；仅限本条件CDF变换，不泛化全部分布校准。重复开发折仍无独立新井确认。',
        '', '## 耗时、清理和未完成',json.dumps(timing,ensure_ascii=False,indent=2),
        f"本轮删除{len(deleted)}个负检查点，共{sum(x['bytes'] for x in deleted)}字节，清单先写后删；源/配置/日志/预测保留。",
        '实际并发峰值见timing；单项与总墙钟含加载/预热/训练/评估和启动间隔，代码和收尾另计。第五严格>80.5与完整原五折OOF>82未完成，无其他折/旧队列/交付修改；目标后独立确认及新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v70最终结果';note=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}，nonlinear-affine PERM={contrast['channel_delta']:+.12f}（训练后差{upcontrast['channel_delta']:+.12f}）。采用={result['selected_mode']}，保留{sorted(retained)}，负权重已清理。原模型/交付不变，无其他折/队列；目标未完成，见perm_cdfwarp_v70_20260929/FINAL.md。"
    for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_perm_cdfwarp_v70_PERM_{mode}_fold5'
        for name in ('README.md','PROJECTS.md'):append(out/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v70完成状态',note.replace(marker,'## v70完成状态'))
    append(ROOT/'EXPLORATIONS.md','v70完成：','- v70完成：'+note+' [结果](results/deep_learning/perm_cdfwarp_v70_20260929/FINAL.md)')
    print(json.dumps(native(dict(scores=scores,adopted=adopted,nonlinear_minus_affine=contrast['channel_delta'],post_update_delta=upcontrast['channel_delta'],retained=sorted(retained),timing=timing))),flush=True)

if __name__=='__main__':main()
