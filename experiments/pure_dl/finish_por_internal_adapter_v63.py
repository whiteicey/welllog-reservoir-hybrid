"""Finalize internal POR adapters, verify retained models, and remove negatives."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from por_internal_adapter_v63 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
        out=ROOT/f'outputs/pure_dl_por_internal_adapter_v63_POR_{mode}_fold5';cfg=read(out/'config.json');rep=read(out/'fold_5_report.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        pred=np.load(out/'channel_pred.npy');updated=np.load(out/'best_updated_pred.npy')
        diag=diagnostics(truth,reference,pred,wells,'POR');up=diagnostics(truth,reference,updated,wells,'POR')
        assert abs(diag['scores']['POR']-rep['target_score'])<1e-9 and abs(up['scores']['POR']-rep['best_after_update']['target_score'])<1e-9
        assert read(out/'initial_verification.json')['all144665_exact']
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        assert rep['gradient_scope']['frozen_gradients_none'] and rep['last_parameter_change_L2']>0
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,0])
        rows.append(dict(mode=mode,report=rep,selected=diag,post_update=up));preds[mode]=pred;updates[mode]=updated
        hashes.append(cfg['initial_trainable_sha256'])
        launch=read(out/'launch.json');costs.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]==hashes[1]
    ctrl=reference.copy();ctrl[:,0]=preds['point'];contrast=diagnostics(truth,ctrl,preds['local'],wells,'POR')
    upctrl=reference.copy();upctrl[:,0]=updates['point'];upcontrast=diagnostics(truth,upctrl,updates['local'],wells,'POR')
    best=max(rows,key=lambda x:x['report']['target_score']);adopted=best['report']['target_score']>refs['POR']+1e-9
    retained=set()
    if adopted:retained.add(best['mode'])
    if contrast['channel_delta']>0:retained.add('local')
    verification=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;data,master=load_data()
        for mode in sorted(retained):
            out=ROOT/f'outputs/pure_dl_por_internal_adapter_v63_POR_{mode}_fold5'
            model=Network('POR',master,mode).cuda();saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
            model.load_trainable_state(saved['backbone_state']);pred,score,total=Evaluator(data,master,0,reference).many(model,(mode,))[mode]
            assert np.array_equal(pred,preds[mode]) and abs(score-saved['target_score'])<1e-9
            verification.append(dict(mode=mode,rows=len(pred),full_raw_disk_replay_exact=True,target_score=score,TOTAL=total,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            model.close();del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,master;gc.collect();torch.cuda.empty_cache()
    combined=reference.copy();choices=[dict(c) for c in previous['choices']]
    if adopted:
        mode=best['mode'];combined[:,0]=preds[mode]
        choice=dict(target='POR',key='v63_POR_'+mode,mode=mode,architecture='frozen_dual_POR_internal_adapters',model_module='por_internal_adapter_v63',
            checkpoint=best['report']['checkpoint'],best_epoch=best['report']['best_epoch'],target_score=best['report']['target_score'],
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),added_parameters=best['report']['trainable_parameters'],
            builder='load_data(); Network(POR,master,mode); load_trainable_state(checkpoint[backbone_state]); Evaluator.many dict[mode]')
        choices=[choice if c['target']=='POR' else c for c in choices]
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,retained=sorted(retained),verification=verification,
        local_minus_point_selected=contrast,local_minus_point_post_update=upcontrast,initial_trainable_hash_equal=True,
        actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=sorted([(datetime.fromisoformat(x['start']),1) for x in costs]+[(datetime.fromisoformat(x['end']),-1) for x in costs]);live=peak=0
    for _,change in events:live+=change;peak=max(peak,live)
    timing=dict(candidates=costs,first_launch_to_last_completion_seconds=(events[-1][0]-events[0][0]).total_seconds(),peak_process_concurrency=peak,
        finalization_and_replay_seconds=time.perf_counter()-tick,scope='explicit serial execution after prior study OOM; individual includes load/warmup/train/eval; total includes launch gaps; authoring/finalization separate')
    atomic_json(STUDY/'timing.json',timing)
    deleted=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_por_internal_adapter_v63_POR_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        assert read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,reason='adopted or positive matched local access gain' if mode in retained else 'no improvement over parent and no demonstrated local-access gain',no_automatic_resume=True))
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained)))
    lines=['# v63 POR内部适配器结果','',f"第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高POR|含起点POR|最佳/停止轮|','|---|---:|---:|---|']
    for row in rows:
        r=row['report'];lines.append(f"|{row['mode']}|{r['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与验证','两组18个rank8内部适配器，39888新参数，旧骨干/路由/头冻结；point将3个核权重用于同点，local读取邻点，同初始张量/完整函数，函数空间不同。0.1仅单块新增分支界限。',
        '原v31锚点路由参数冻结、允许输入梯度，数值函数不变。原9曲线/206+已有63侧视图、Q50、POR Student-t3损失、分折/解码/评分保持。真实第二步新参数梯度、旧参数无梯度、冻结状态不变、初始/最佳144665点、源哈希和三列评分已核验。',
        f"保留者原始输入磁盘重建：{verification}。",'', '## 收益与不继续',
        f"local-point含起点POR={contrast['channel_delta']:+.12f}，训练后最高POR差={upcontrast['channel_delta']:+.12f}；采用={result['selected_mode']}，保留={sorted(retained)}。",
        '起点回退不算学习收益；若local无最终配对收益，关闭当前rank/核宽度/边界扫描，结论仅限本插入方式，不泛化所有适配器。重复开发折结果尚未经独立井确认。',
        '', '## 耗时、清理与未做',json.dumps(timing,ensure_ascii=False,indent=2),
        f"已删除负检查点{len(deleted)}个、{sum(x['bytes'] for x in deleted)}字节；保留源码、配置、结果与删除清单。",
        '第五严格>80.5及完整原五折OOF>82仍未完成；无其他折、旧队列或交付修改。目标达成后可做独立确认及新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v63最终结果';note=marker+f"\n两项串行完成，TOTAL={scores['TOTAL']:.12f}，local-point POR={contrast['channel_delta']:+.12f}（训练后差{upcontrast['channel_delta']:+.12f}）。采用={result['selected_mode']}，保留{sorted(retained)}，负权重已清理。原模型/交付不变，无其他折/队列；目标未完成，见por_internal_adapter_v63_20260929/FINAL.md。"
    for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_por_internal_adapter_v63_POR_{mode}_fold5'
        for name in ('README.md','PROJECTS.md'):append(out/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v63完成状态',note.replace(marker,'## v63完成状态'))
    append(ROOT/'EXPLORATIONS.md','v63完成：','- v63完成：'+note+' [结果](results/deep_learning/por_internal_adapter_v63_20260929/FINAL.md)')
    print(json.dumps(native(dict(scores=scores,adopted=adopted,local_minus_point=contrast['channel_delta'],post_update_delta=upcontrast['channel_delta'],retained=sorted(retained),timing=timing))),flush=True)

if __name__=='__main__':main()
