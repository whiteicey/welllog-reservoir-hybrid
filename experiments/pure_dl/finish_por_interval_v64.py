"""Complete interval-loss comparison against the immutable historical control."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from por_interval_v64 import ROOT,STUDY,PREVIOUS,CONTROL,Network,Evaluator,load_data,reference_prediction
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
    out=ROOT/'outputs/pure_dl_por_interval_v64_POR_interval_fold5'
    rows=[];configs=[];preds={};updates={}
    for mode,folder in [('point_control',CONTROL),('interval',out)]:
        cfg=read(folder/'config.json');rep=read(folder/'fold_5_report.json');configs.append(cfg)
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        pred=np.load(folder/'channel_pred.npy');updated=np.load(folder/'best_updated_pred.npy')
        diag=diagnostics(truth,reference,pred,wells,'POR');up=diagnostics(truth,reference,updated,wells,'POR')
        assert abs(diag['scores']['POR']-rep['target_score'])<1e-9 and abs(up['scores']['POR']-rep['best_after_update']['target_score'])<1e-9
        assert read(folder/'initial_verification.json')['all144665_exact']
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        assert rep['gradient_scope']['frozen_gradients_none'] and rep['last_parameter_change_L2']>0
        if rep['best_epoch']==0:assert np.array_equal(pred,reference[:,0])
        rows.append(dict(mode=mode,report=rep,selected=diag,post_update=up));preds[mode]=pred;updates[mode]=updated
    matched_fields=['initial_trainable_sha256','trainable_names','trainable_parameters','total_parameters','parent_choice',
        'epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','quality','decoder']
    for field in matched_fields:assert configs[0][field]==configs[1][field],field
    ctrl=reference.copy();ctrl[:,0]=preds['point_control'];contrast=diagnostics(truth,ctrl,preds['interval'],wells,'POR')
    upctrl=reference.copy();upctrl[:,0]=updates['point_control'];upcontrast=diagnostics(truth,upctrl,updates['interval'],wells,'POR')
    rep=rows[1]['report'];adopted=bool(rep['target_score']>refs['POR']+1e-9);retained=bool(adopted or contrast['channel_delta']>0)
    verification=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;data,master=load_data()
        model=Network('POR',master,'interval').cuda();saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
        model.load_trainable_state(saved['backbone_state']);pred,score,total=Evaluator(data,master,0,reference).many(model,('interval',))['interval']
        assert np.array_equal(pred,preds['interval']) and abs(score-saved['target_score'])<1e-9
        verification.append(dict(rows=len(pred),full_raw_disk_replay_exact=True,target_score=score,TOTAL=total,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
        model.close();del model,saved,data,master;gc.collect();torch.cuda.empty_cache()
    combined=reference.copy();choices=[dict(c) for c in previous['choices']]
    if adopted:
        combined[:,0]=preds['interval']
        choice=dict(target='POR',key='v64_POR_interval',mode='interval',architecture='frozen_dual_POR_internal_adapters',model_module='por_interval_v64',
            checkpoint=rep['checkpoint'],best_epoch=rep['best_epoch'],target_score=rep['target_score'],
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),added_parameters=rep['trainable_parameters'],
            builder='load_data(); Network(POR,master,interval); load_trainable_state(checkpoint[backbone_state]); Evaluator.many dict[interval]')
        choices=[choice if c['target']=='POR' else c for c in choices]
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,
        adopted=adopted,retained=retained,verification=verification,interval_minus_control_selected=contrast,interval_minus_control_post_update=upcontrast,
        matched_fields=matched_fields,historical_control_reused=True,bitwise_retraining_reproducibility_claimed=False,
        actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,strict805_qualified=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(previous);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    launch=read(out/'launch.json');timing=dict(new_run_seconds=rep['seconds'],new_launch_to_completion_seconds=(datetime.fromisoformat(rep['finished_at'])-datetime.fromisoformat(launch['started_at'])).total_seconds(),
        historical_control_seconds=rows[0]['report']['seconds'],historical_control_retrained=False,peak_allocated_MiB=rep['peak_allocated_MiB'],peak_process_concurrency=1,
        finalization_and_replay_seconds=time.perf_counter()-tick,scope='one new run includes loading/warmup/training/evaluation; historical control separate; authoring/finalization separate')
    atomic_json(STUDY/'timing.json',timing)
    manifest_path=STUDY/'deletion_manifest.json'
    deleted=read(manifest_path)['deleted'] if manifest_path.exists() else []
    assert read(out/'runtime_status.json')['status']=='complete'
    if not retained:
        for name in ('model_best.pt','last_state.pt'):
            file=(out/name).resolve();assert file.parent==out.resolve() and out.resolve().parent==(ROOT/'outputs').resolve()
            if file.exists():
                deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                atomic_json(manifest_path,dict(deleted=deleted,retained=retained,status='deletion_planned'));file.unlink()
    atomic_json(out/'candidate_status.json',dict(retained=retained,reason='positive selected gain' if retained else 'no selected gain over historical control or current parent',no_automatic_resume=True))
    known_bytes=sum(x['bytes'] for x in deleted if x['bytes'] is not None);unknown_sizes=sum(x['bytes'] is None for x in deleted)
    atomic_json(manifest_path,dict(deleted=deleted,known_bytes=known_bytes,unknown_file_sizes=unknown_sizes,retained=retained,status='complete'))
    lines=['# v64 POR 容差区间质量监督结果','',f"原第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高POR|含起点POR|最佳/停止轮|','|---|---:|---:|---|']
    for row in rows:
        r=row['report'];lines.append(f"|{row['mode']}|{r['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与已经验证',
        '复用已经完成的v63 point对照；同18个rank8内部适配器、39888新参数、初始张量及完整函数，仅加入0.5容差区间质量负对数。区间为训练真实POR的±0.08*(abs(y)+1e-5)，原log1p归一化和Student-t3 CDF；只监督非default条件质量，原Q50和BCE/NLL保留。这是区间代理损失，不等于精确官方三角效用。',
        '原9曲线/206输入与已有63侧特征、分折、训练井、Q、解码和评分不改。原权重冻结；实际梯度、初始和最佳144665点恢复、源哈希、匹配配置及三列评分均通过。历史控制复用不声称同步双跑或CUDA逐位重训保证。',
        f'保留者原始输入磁盘重建：{verification}。','', '## 收益与不继续',
        f"interval-control含起点POR={contrast['channel_delta']:+.12f}；训练后最高POR差={upcontrast['channel_delta']:+.12f}；采用={adopted}，保留={retained}。",
        '回退起点不是学习收益；若含起点无收益，不继续扫描当前区间宽度/辅助系数。结论仅限此POR适配器和监督配方，不泛化全部效用损失。反复使用的开发折尚无独立井确认。',
        '', '## 耗时、清理和未完成',json.dumps(timing,ensure_ascii=False,indent=2),
        f"删除本轮负检查点{len(deleted)}个；已知大小合计{known_bytes}字节，大小未记录{unknown_sizes}个。首次收尾在删除后遇到numpy bool JSON序列化错误，训练/评分成功；失败证据与恢复清单保留，未重训。源、配置、日志、预测保留，历史对照未改。",
        '第五严格>80.5及完整原五折OOF>82尚未完成；无其他折、旧队列或旧交付修改。目标达成后可做独立确认与新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v64最终结果';note=marker+f"\n一项新训练完成，复用v63 point对照。TOTAL={scores['TOTAL']:.12f}；interval-control POR={contrast['channel_delta']:+.12f}（训练后差{upcontrast['channel_delta']:+.12f}）；采用={adopted}，保留={retained}。负权重清理，来源/交付不改；无其他折或队列，目标未完成，见por_interval_v64_20260929/FINAL.md。"
    for path in (STUDY/'README.md',STUDY/'PROJECTS.md',out/'README.md',out/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    append(ROOT.parent/'AGENTS.md','## v64完成状态',note.replace(marker,'## v64完成状态'))
    append(ROOT/'EXPLORATIONS.md','v64完成：','- v64完成：'+note+' [结果](results/deep_learning/por_interval_v64_20260929/FINAL.md)')
    print(json.dumps(native(dict(scores=scores,adopted=adopted,delta=contrast['channel_delta'],post_update_delta=upcontrast['channel_delta'],retained=retained,timing=timing))),flush=True)

if __name__=='__main__':main()
