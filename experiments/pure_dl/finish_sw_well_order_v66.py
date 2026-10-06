"""Finalize matched full-well context versus fixed reference and score assembly."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_well_order_v66 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction
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
        out=ROOT/f'outputs/pure_dl_sw_well_order_v66_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert read(out/'initial.json')['all144665_parent_exact']
        assert rep['all144665_selected_predictions_restored_exactly']
        assert all(rep['gradient_scope']['gradient_norms'][g]>0 for g in ('condition.conv.','condition.output.'))
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
        choice=dict(target='SW',mode=mode,key='v66_SW_'+mode,architecture='private_SW_ordered_well_residual',model_module='sw_well_order_v66',
            checkpoint=best['report']['checkpoint'],target_score=score,best_epoch=best['report']['best_epoch'],
            builder='load_data builds input-only banks and train64 reference; Network(mode,metadata); load_state_dict(state_dict); Evaluator supplies own-well bank',
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
        choices=[choice if c['target']=='SW' else c for c in choices]
        del model,saved,d,meta;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,2]=preds['permuted'];contrast=diagnostics(truth,ctrl,preds['ordered'],wells,'SW')
    by={x['epoch']:x for x in histories['permuted']}
    matched=[dict(epoch=x['epoch'],ordered_minus_permuted_SW=x['target_score']-by[x['epoch']]['target_score']) for x in histories['ordered'] if x['epoch'] in by]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,new_target_full_disk_replay=replay,ordered_minus_permuted=contrast,
        post_update_ordered_minus_permuted=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
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
    atomic_json(STUDY/'timing.json',timing)
    deleted=[];manifest=STUDY/'deletion_manifest.json'
    if manifest.exists():deleted=read(manifest)['deleted']
    retained={best['mode']} if adopted else set()
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_well_order_v66_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='adopted after disk replay' if mode in retained else 'no selected improvement'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v66 SW整井层序残差结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|训练后最高SW|含回退SW|最佳/停止轮|','|---|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与验证',
        '同924697总参数、85216可训练参数和完整初始函数；原v56全预测器冻结。原point隐表示经kernel3 Conv64→32、均值/标准差池化、零初始Linear64→1216生成六层FiLM增量，0.1*tanh边界。permuted读固定随机次序，ordered读深度次序；输入点集合、参数、初始化相同。原主损失/Q50/1024上下文/解码不变。旧0.5与新0.1边界分别记录。',
        '完整初始状态/核心/输出头/144665点预测相同；实际新增Conv/输出梯度有效，全部原参数无梯度且训练结束哈希不变。最佳恢复、源哈希和三列评分已核验。保留者原始输入磁盘重建='+str(replay)+'。',
        '', '## 收益与不继续',f"ordered-permuted含回退SW差{contrast['channel_delta']:+.12f}；训练后最佳差{result['post_update_ordered_minus_permuted']:+.12f}；采用={result['selected_mode']}。",
        '回退第0轮不是学习收益。若ordered无最终配对收益，不继续本整井卷积核宽度/残差幅度扫描；结论只限当前128点稀疏层序支路，不泛化全部层序信息。重复开发折未经独立新井确认。',
        '', '## 耗时、清理与未完成',json.dumps(timing,ensure_ascii=False,indent=2),
        f"本轮删除{len(deleted)}个负检查点，共{sum(x['bytes'] for x in deleted)}字节，清单先写后删除；源、配置、日志、预测与曲线保留。",
        '并发峰值与真实墙钟见timing；单项含加载/预热/训练/评估，总墙钟含显式启动间隔，代码和收尾另计。CUDA反向允许非确定，不声称逐位重训复现。第五严格>80.5及完整原五折OOF>82仍未完成；无其他折、旧队列或交付修改。达成后独立确认及新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v66最终结果';note=marker+f"\n两项完成，TOTAL={scores['TOTAL']:.12f}；ordered-permuted SW={contrast['channel_delta']:+.12f}，训练后差={result['post_update_ordered_minus_permuted']:+.12f}；采用={result['selected_mode']}。负权重清理，源/交付不改，无其他折；目标未完成，见sw_well_order_v66_20260929/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_well_order_v66_SW_{mode}_fold5'
        for name in ('PROJECTS.md','README.md'):append(out/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v66完成状态',note.replace(marker,'## v66完成状态'))
    append(ROOT/'EXPLORATIONS.md','v66完成：','- v66完成：'+note+' [结果](results/deep_learning/sw_well_order_v66_20260929/FINAL.md)')
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=result['selected_mode'],post_update_delta=result['post_update_ordered_minus_permuted'],timing=timing))),flush=True)

if __name__=='__main__':main()
