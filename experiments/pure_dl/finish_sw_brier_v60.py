"""Verify raw-input reconstruction, assemble fixed target choices, and clean caches."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_brier_v60 import ROOT,STUDY,PREVIOUS,CACHE,MODES,Network,LiveEvaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training active')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    refs=metric(truth,reference);assert truth.shape==reference.shape==(144665,3)
    assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    records=[];preds={};hashes=[];timings=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_brier_v60_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json');initial=read(out/'initial.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert cfg['cache_manifest']['training_rows']==578891 and cfg['cache_manifest']['validation_rows']==144665
        assert initial['all144665_parent_exact'] and rep['frozen_core_unchanged'] and rep['all144665_selected_predictions_restored_exactly']
        assert rep['gradient_scope']['frozen_parameters_no_gradient']
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        records.append(dict(mode=mode,report=rep,diagnostics=diag,best_after_update=max(hist,key=lambda x:x['target_score'])));preds[mode]=pred
        hashes.append((cfg['initial_state_sha256'],cfg['initial_head_sha256'],initial['initial_prediction_sha256']))
        launch=read(out/'launch.json');timings.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]==hashes[1]
    ctrl=reference.copy();ctrl[:,2]=preds['likelihood'];contrast=diagnostics(truth,ctrl,preds['brier'],wells,'SW')
    best=max(records,key=lambda x:x['report']['target_score']);adopted=best['report']['target_score']>refs['SW']+1e-9
    retained=set()
    if adopted:retained.add(best['mode'])
    if contrast['channel_delta']>0:retained.add('brier')
    verification=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
        d,meta=load_data()
        for mode in sorted(retained):
            out=ROOT/f'outputs/pure_dl_sw_brier_v60_SW_{mode}_fold5'
            saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False);model=Network(mode,meta).cuda()
            model.load_state_dict(saved['state_dict'],strict=True)
            p,score,scores=LiveEvaluator(d,meta,2,reference)(model)
            assert np.array_equal(p,preds[mode]) and abs(score-saved['target_score'])<1e-9
            verification.append(dict(mode=mode,raw_input_full144665_exact=True,score=score,scores=scores,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            del model,saved;gc.collect();torch.cuda.empty_cache()
        del d,meta;gc.collect();torch.cuda.empty_cache()
    combined=reference.copy();choices=[dict(c) for c in old['choices']]
    if adopted:
        mode=best['mode'];combined[:,2]=preds[mode]
        choice=dict(target='SW',mode=mode,key='v60_SW_'+mode,architecture='frozen_SW_RQS_Brier_readout',model_module='sw_brier_v60',
            checkpoint=best['report']['checkpoint'],target_score=best['report']['target_score'],best_epoch=best['report']['best_epoch'],
            builder='load_data own-well input banks; Network(mode,metadata); full state_dict; LiveEvaluator on raw inputs; training cache not required',
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        choices=[choice if c['target']=='SW' else c for c in choices]
    scores=metric(truth,combined)
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,brier_minus_likelihood=contrast,retained=sorted(retained),verification=verification,
        post_update_brier_minus_likelihood=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
        initial_state_head_prediction_equal=True,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
        strict805_qualified=scores['TOTAL']>80.5,complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,
        finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    cache_manifest=read(STUDY/'cache_manifest.json');prep=read(STUDY/'cache_status.json');first=read(STUDY/'cache_failure_01.json') if (STUDY/'cache_failure_01.json').exists() else prep
    begin=min(datetime.fromisoformat(t['start']) for t in timings);end=max(datetime.fromisoformat(t['end']) for t in timings)
    events=sorted([(datetime.fromisoformat(t['start']),1) for t in timings]+[(datetime.fromisoformat(t['end']),-1) for t in timings]);live=peak=0
    for _,change in events:live+=change;peak=max(peak,live)
    timing=dict(candidates=timings,cache_preparation_seconds=cache_manifest['seconds'],cache_preparation_peak_MiB=cache_manifest['peak_allocated_MiB'],
        pair_launch_to_completion_seconds=(end-begin).total_seconds(),first_preparation_including_failed_attempt_to_completion_seconds=(end-datetime.fromisoformat(first['started_at'])).total_seconds(),
        successful_preparation_to_completion_seconds=(end-datetime.fromisoformat(prep['started_at'])).total_seconds(),
        gap_after_cache_to_first_launch_seconds=(begin-datetime.fromisoformat(prep['finished_at'])).total_seconds(),
        peak_process_concurrency=peak,failed_cache_attempt=(STUDY/'cache_failure_01.json').exists(),
        finalization_and_raw_replay_seconds=time.perf_counter()-tick,scope='training includes load/warmup/evaluation; cache shared once; wall times include launch/authoring gaps')
    atomic_json(STUDY/'timing.json',timing)
    deleted=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_brier_v60_SW_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        assert read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                p=(out/name).resolve();assert p.parent==out
                if p.exists():
                    deleted.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));p.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,reason='adopted or positive matched Brier gain' if mode in retained else 'below current SW with no positive tested Brier gain',no_automatic_resume=True))
    if CACHE.exists():
        assert CACHE.resolve().parent==STUDY.resolve() and CACHE.name=='frozen_features.pt'
        deleted.append(dict(path=str(CACHE.relative_to(ROOT)),bytes=CACHE.stat().st_size,sha256=cache_manifest['cache_sha256'],reason='regenerable shared frozen features; training complete; full raw replay done for retained models'));CACHE.unlink()
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained)))
    lines=['# v60 SW概率平方损失结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高SW|含起点SW|最佳/停止轮|','|---|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与完成验证','同840771总参数/4515门控密度参数与完整初始函数；冻结v58骨干，仅门控与RQS输出更新。目标为原连续似然，或加0.5*(gate Brier+16*非default条件256格Brier)。固定canonical训练上下文是共同变化，不能把历史绝对差全部归因于辅助项。',
        '训练578891点、验证144665点分包，Q50与原RQS公式/官方评分保持；初始源预测、参数与头哈希相同，冻结状态和真实梯度已核验。保留候选均从原始输入重载完整模型逐点验证。',
        '沿用已修正的训练/验证分包与行映射；Brier只用训练标签按原edges映射类别，不使用验证拟合。',
        '', '## 收益与不继续的点',f"Brier-likelihood SW {contrast['channel_delta']:+.12f}；训练后最高差{result['post_update_brier_minus_likelihood']:+.12f}；采用{result['selected_mode']}，保留{sorted(retained)}。",
        '初始回退不算训练收益。若Brier无配对收益，关闭当前256格平方损失系数扫描；结论不泛化所有概率评分。正局部收益仍是重复开发折证据，未经独立井确认。',
        '', '## 耗时、清理和未做',json.dumps(timing,ensure_ascii=False,indent=2),f"删除负权重与可再生缓存{len(deleted)}项，共{sum(x['bytes'] for x in deleted)}字节。保留模型不依赖训练缓存推理，训练恢复需先重建同缓存。",
        '目标仍是第五折严格>80.5及完整原五折OOF>82；本批无其他折、旧任务恢复或交付覆盖。未做独立新井确认。目标完成后可新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v60最终结果';note=marker+f"\n实际TOTAL={scores['TOTAL']:.12f}，Brier-likelihood SW={contrast['channel_delta']:+.12f}，采用{result['selected_mode']}，保留{sorted(retained)}。两项完成，保留者原始输入144665点重建一致；负权重及可再生缓存清理，耗时如实记录。详见sw_brier_v60_20260929/FINAL.md。目标未完成，无其他折/自动队列/交付修改。"
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,marker,note)
    append(ROOT/'EXPLORATIONS.md','v60完成：','- v60完成：'+note+' [结果](results/deep_learning/sw_brier_v60_20260929/FINAL.md)')
    append(ROOT.parent/'AGENTS.md','## v60完成状态',note.replace(marker,'## v60完成状态'))
    print(json.dumps(native(dict(scores=scores,adopted=adopted,delta=contrast['channel_delta'],retained=sorted(retained),timing=timing))),flush=True)

if __name__=='__main__':main()
