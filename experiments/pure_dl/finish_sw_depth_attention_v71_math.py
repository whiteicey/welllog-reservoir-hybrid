"""Finalize matched full-well context versus fixed reference and score assembly."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_depth_attention_v71_math import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def main():
    if jobs():raise RuntimeError('Training active')
    start=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    refs=metric(truth,reference);assert truth.shape==reference.shape==(144665,3)
    assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    records=[];preds={};histories={};timings=[];hashes=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_depth_attention_v71_math_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert read(out/'initial.json')['all144665_parent_exact']
        assert rep['all144665_selected_predictions_restored_exactly']
        assert all(rep['gradient_scope']['gradient_norms'][g]>0 for g in ('body.global_blocks.0.coefficients','body.global_blocks.1.coefficients','body.global_blocks.2.coefficients','body.global_blocks.3.coefficients'))
        assert rep['all_original_state_unchanged'] and rep['gradient_scope']['frozen_gradients_none']
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        updated=max(hist,key=lambda x:x['target_score'])
        records.append(dict(mode=mode,report=rep,diagnostics=diag,best_after_update=updated));preds[mode]=pred;histories[mode]=hist;hashes.append(dict(state=cfg['initial_state_sha256'],core=cfg['initial_core_sha256'],head=cfg['initial_head_sha256'],prediction=read(out/'initial.json')['initial_prediction_sha256']))
        launch=read(out/'launch.json');timings.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert hashes[0]['state']==hashes[1]['state'] and hashes[0]['core']==hashes[1]['core'] and hashes[0]['head']==hashes[1]['head'] and hashes[0]['prediction']==hashes[1]['prediction']
    best=max(records,key=lambda x:x['report']['target_score']);adopted=bool(best['report']['target_score']>refs['SW']+1e-9)
    combined=reference.copy();choices=[dict(c) for c in old['choices']];replay=False
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        d,meta=load_data();mode=best['mode'];model=Network(mode,meta).cuda();saved=torch.load(ROOT/best['report']['checkpoint'],map_location='cpu',weights_only=False)
        model.load_state_dict(saved['state_dict'],strict=True);p,score,_=Evaluator(d,meta,2,reference)(model)
        assert np.array_equal(p,preds[mode]) and abs(score-best['report']['target_score'])<1e-9
        combined[:,2]=p;replay=True
        choice=dict(target='SW',mode=mode,key='v71_SW_'+mode,architecture='private_SW_physical_depth_attention',model_module='sw_depth_attention_v71_math',
            checkpoint=best['report']['checkpoint'],target_score=score,best_epoch=best['report']['best_epoch'],
            builder='load_data builds input-only banks and train64 reference; Network(mode,metadata); load_state_dict(state_dict); Evaluator supplies own-well bank',
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        choices=[choice if c['target']=='SW' else c for c in choices]
        del model,saved,d,meta;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=preds['symmetric'];contrast=diagnostics(truth,ctrl,preds['directional'],wells,'SW')
    by={x['epoch']:x for x in histories['symmetric']}
    matched=[dict(epoch=x['epoch'],directional_minus_symmetric_SW=x['target_score']-by[x['epoch']]['target_score']) for x in histories['directional'] if x['epoch'] in by]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,new_target_full_disk_replay=replay,directional_minus_symmetric=contrast,
        post_update_directional_minus_symmetric=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
        matched_trajectory=matched,initial_state_hash_equal=True,initial_core_hash_equal=True,initial_head_hash_equal=True,initial_prediction_hash_equal=True,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
        strict805_qualified=scores['TOTAL']>80.5,complete_five_folds=False,remaining_folds_started=False,old_deliveries_unchanged=True,
        finished_at=datetime.now().astimezone().isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for t in timings:events.extend([(datetime.fromisoformat(t['start']),1),(datetime.fromisoformat(t['end']),-1)])
    live=peak=0
    for _,d in sorted(events):live+=d;peak=max(peak,live)
    timing=dict(candidates=timings,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,
        finalization_seconds=time.perf_counter()-start,scope='loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate')
    failure=read(STUDY/'backend_failure.json')
    failed_out=ROOT/failure['failed_directory'];failed_launch=read(failed_out/'launch.json')
    timing.update(failed_attempt=failure,first_failed_launch_to_last_completion_seconds=(max(t for t,_ in events)-datetime.fromisoformat(failed_launch['started_at'])).total_seconds())
    atomic_json(STUDY/'timing.json',timing)
    deleted=[];manifest=STUDY/'deletion_manifest.json'
    if manifest.exists():deleted=read(manifest)['deleted']
    retained={best['mode']} if adopted else set()
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_depth_attention_v71_math_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='adopted after disk replay' if mode in retained else 'no selected improvement'))
    assert read(failed_out/'runtime_status.json')['status']=='failed' and failed_out.resolve().parent==(ROOT/'outputs').resolve()
    for name in ('model_best.pt','last_state.pt'):
        file=(failed_out/name).resolve();assert file.parent==failed_out.resolve()
        if file.exists():
            deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),reason='failed initial backend replaced by completed math implementation'))
            atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v71 SW深度关系注意力结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高SW|含起点SW|最佳/停止轮|','|---|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与验证',
        '原1m/4m方向差特征已被历史尝试覆盖，未重复训练该特征。本轮冻结v56全SW，仅4层×4头×8个零初始系数。物理token深度两两差/10m/tanh，symmetric取绝对值、directional保留正负，各自8个归一化RBF基；系数0.5*tanh凸组合令偏置绝对值≤0.5。相同128参数和完整起点，不声称两种函数空间相同。原数据/Q50/损失/1024中心128/解码和评分保持，新系数3e-3。',
        '原始版本首次反向出现CUDA LSE strideH对齐错误、无完成训练轮；原失败来源/config/evidence冻结。math版保留原布尔mask注意力，仅新增偏置与零偏置差分用显式FP32 QK/softmax/V；零系数差为0，各自初始144665点与父模型逐位一致。该结构每块三次attention，成本实报。',
        '四层实际系数梯度有效，全部原参数无梯度且状态哈希不变；同初始状态/骨干/头/预测、来源哈希、最佳144665恢复与真实三列评分核验；新采用者原始输入磁盘重建='+str(replay)+'。',
        '', '## 收益与不继续',f"directional-symmetric含起点SW差{contrast['channel_delta']:+.12f}；训练后最佳差{result['post_update_directional_minus_symmetric']:+.12f}；采用={result['selected_mode']}。",
        '起点回退不是学习收益。若方向关系无最终配对收益，不扫距离尺度/RBF数量；两组均无当前收益则不扫偏置幅度。仅限粗token物理关系，不泛化全部位置编码；重复开发折仍无独立确认。',
        '', '## 耗时、失败、清理与未完成',json.dumps(timing,ensure_ascii=False,indent=2),
        f"删除本轮负或已被替代失败检查点{len(deleted)}个，共{sum(x['bytes'] for x in deleted)}字节，清单先写后删；所有源码/配置/失败/日志/预测保留。",
        'math两项与含失败/修复间隔总墙钟分别记录；单项含加载/预热/三次attention训练/评估，代码/收尾另计。第五严格>80.5和完整原五折OOF>82未完成；无其他折/旧队列/交付修改。目标后独立确认和新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v71最终结果';note=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}；directional-symmetric SW={contrast['channel_delta']:+.12f}，训练后差={result['post_update_directional_minus_symmetric']:+.12f}；采用={result['selected_mode']}。负权重清理，源/交付不改，无其他折；目标未完成，见sw_depth_attention_v71_20260929/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_depth_attention_v71_math_SW_{mode}_fold5'
        for name in ('PROJECTS.md','README.md'):append(out/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v71完成状态',note.replace(marker,'## v71完成状态'))
    append(ROOT/'EXPLORATIONS.md','v71完成：','- v71完成：'+note+' [结果](results/deep_learning/sw_depth_attention_v71_20260929/FINAL.md)')
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=result['selected_mode'],post_update_delta=result['post_update_directional_minus_symmetric'],timing=timing))),flush=True)

if __name__=='__main__':main()
