"""Finalize matched full-well context versus fixed reference and score assembly."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from perm_curvature_v69 import ROOT,STUDY,PREVIOUS,MODES,Network,Evaluator,load_data,reference_prediction,CONTROL
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
    cfgs=[]
    for mode in ('sum','curvature'):
        out=CONTROL if mode=='sum' else ROOT/f'outputs/pure_dl_perm_curvature_v69_PERM_{mode}_fold5'
        cfg=read(out/'config.json');cfgs.append(cfg);rep=read(out/'fold_5_report.json');hist=read(out/'validation_history.json')
        for n,h in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/n).read_bytes()).hexdigest()==h,n
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert read(out/'initial.json')['all144665_parent_exact'] and not read(out/'initial.json')['new_density_initial_score']
        assert rep['all144665_selected_predictions_restored_exactly']
        assert all(rep['gradient_scope']['gradient_norms'][g]>0 for g in ('body.','output.','condition.','gate.'))
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'PERM')
        assert abs(diag['scores']['PERM']-rep['target_score'])<1e-9
        updated=max(hist,key=lambda x:x['target_score'])
        records.append(dict(mode=mode,report=rep,initial=read(out/'initial.json'),diagnostics=diag,best_after_update=updated));preds[mode]=pred;histories[mode]=hist;hashes.append(cfg['initial_state_sha256'])
        launch=read(out/'launch.json')
        if mode=='curvature':timings.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    matched_fields=['trainable_parameters','initial_state_sha256','initialization','quality','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','decoder','bank_manifest']
    for field in matched_fields:assert cfgs[0][field]==cfgs[1][field],field
    assert hashes[0]==hashes[1]
    assert records[0]['initial']['initial_prediction_sha256']==records[1]['initial']['initial_prediction_sha256']
    best=max(records,key=lambda x:x['report']['target_score']);adopted=bool(best['report']['target_score']>refs['PERM']+1e-9)
    ctrl=reference.copy();ctrl[:,1]=preds['sum'];contrast=diagnostics(truth,ctrl,preds['curvature'],wells,'PERM')
    retained={best['mode']} if adopted else set()
    if contrast['channel_delta']>0:retained.add('curvature')
    for row in records:
        if row['report']['target_score']>row['initial']['target_score']+1e-9:retained.add(row['mode'])
    assert retained<={'curvature'}
    combined=reference.copy();choices=[dict(c) for c in old['choices']];replay=False;verification=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        d,meta=load_data()
        for mode in sorted(retained):
            rep=next(x['report'] for x in records if x['mode']==mode)
            model=Network(mode,meta).cuda();saved=torch.load(ROOT/rep['checkpoint'],map_location='cpu',weights_only=False)
            model.load_state_dict(saved['state_dict'],strict=True);p,score,_=Evaluator(d,meta,1,reference)(model)
            assert np.array_equal(p,preds[mode]) and abs(score-rep['target_score'])<1e-9
            verification.append(dict(mode=mode,rows=len(p),full_raw_disk_replay_exact=True,PERM=score,checkpoint_sha256=hashlib.sha256((ROOT/rep['checkpoint']).read_bytes()).hexdigest()))
            if adopted and mode==best['mode']:
                combined[:,1]=p;replay=True
                choice=dict(target='PERM',mode=mode,key='v69_PERM_'+mode,architecture='private_PERM_smooth_density_curvature_regularization',model_module='perm_curvature_v69',
                    checkpoint=rep['checkpoint'],target_score=score,best_epoch=rep['best_epoch'],
                    builder='load_data builds input-only banks; Network(mode,metadata); load_state_dict(state_dict); Evaluator c=1 supplies bank and CDF decoding',
                    parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
                choices=[choice if c['target']=='PERM' else c for c in choices]
            del model,saved;gc.collect();torch.cuda.empty_cache()
        del d,meta;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);ctrl=reference.copy();ctrl[:,1]=preds['sum'];contrast=diagnostics(truth,ctrl,preds['curvature'],wells,'PERM')
    by={x['epoch']:x for x in histories['sum']}
    matched=[dict(epoch=x['epoch'],curvature_minus_sum_PERM=x['target_score']-by[x['epoch']]['target_score']) for x in histories['curvature'] if x['epoch'] in by]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=records,
        adopted=adopted,selected_mode=best['mode'] if adopted else None,new_target_full_disk_replay=replay,curvature_minus_sum=contrast,
        post_update_curvature_minus_sum=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'],
        matched_trajectory=matched,historical_control_reused=True,matched_fields=matched_fields,initial_state_hash_equal=True,initial_complete_function_equal=True,initial_curvature_minus_sum=records[1]['initial']['target_score']-records[0]['initial']['target_score'],retained=sorted(retained),verification=verification,actual_fixed_three_column_scoring=True,rows=144665,screening_fold=5,
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
        historical_control_seconds=records[0]['report']['seconds'],historical_control_retrained=False,finalization_seconds=time.perf_counter()-start,scope='loading/input-bank construction/warmup/training/evaluation plus launch gap; authoring/final replay separate')
    atomic_json(STUDY/'timing.json',timing)
    deleted=[];manifest=STUDY/'deletion_manifest.json'
    if manifest.exists():deleted=read(manifest)['deleted']
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_perm_curvature_v69_PERM_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='adopted or positive matched curvatureion or source gain after disk replay' if mode in retained else 'no absolute improvement and no retained mechanism gain'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v69 PERM分位间距曲率结果','',f"固定第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'',
        '|模式|初始PERM|训练后最高PERM|含起点PERM|最佳/停止轮|','|---|---:|---:|---:|---|']
    for row in records:
        r=row['report'];lines.append(f"|{row['mode']}|{row['initial']['target_score']:.12f}|{row['best_after_update']['target_score']:.12f}|{r['target_score']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','## 探索重点与验证',
        '复用v68 sum已完成控制；相同838211参数、完整v67 RQS第24轮函数、初始张量和两次autograd相加方式。仅密度目标增加固定0.05*log分位间距二阶差分平方均值，训练非default掩码、原Q加权/全部点平均/3；不是物理密度曲率精确积分，不强制单峰。原数据/评分/解码/3e-5及80/18预算保持。',
        '初始状态和144665点预测、关键配置一致；真实骨干/门控/密度/FiLM梯度有效，最佳恢复、来源哈希和三列评分核验。历史控制复用不声称同步双跑或逐位重训。正候选磁盘重建：'+str(verification)+'。',
        '', '## 收益与不继续',f"curvature-sum含起点PERM差{contrast['channel_delta']:+.12f}；训练后最高差{result['post_update_curvature_minus_sum']:+.12f}。采用={result['selected_mode']}，保留={sorted(retained)}。",
        '起点回退不是新学习收益；源模型与当前冠军差分开。若无最终配对或源模型增益，不继续当前正则系数/差分阶数扫描；只限本间距约束，不泛化全部密度正则。原v67正候选保留，开发折仍需独立确认。',
        '', '## 耗时、清理与未完成',json.dumps(timing,ensure_ascii=False,indent=2),
        f"本轮删除{len(deleted)}个负检查点，共{sum(x['bytes'] for x in deleted)}字节，清单先写后删；控制和原v67文件不修改，源配置日志预测保留。",
        '单个新任务的加载/预热/训练/评估墙钟单列；历史控制秒数含其当时并发环境，不冒称本轮两项总墙钟；代码/收尾另计。第五严格>80.5及完整原五折OOF>82尚未完成，无其他折/旧队列/交付修改。达成目标后独立确认、新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v69最终结果';note=marker+f"\n一个新候选完成，复用历史sum控制，TOTAL={scores['TOTAL']:.12f}；curvature-sum PERM={contrast['channel_delta']:+.12f}，训练后差={result['post_update_curvature_minus_sum']:+.12f}；采用={result['selected_mode']}，保留={sorted(retained)}。负权重清理，源/交付不改，无其他折；目标未完成，见perm_curvature_v69_20260929/FINAL.md。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_perm_curvature_v69_PERM_{mode}_fold5'
        for name in ('PROJECTS.md','README.md'):append(out/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v69完成状态',note.replace(marker,'## v69完成状态'))
    append(ROOT/'EXPLORATIONS.md','v69完成：','- v69完成：'+note+' [结果](results/deep_learning/perm_curvature_v69_20260929/FINAL.md)')
    print(json.dumps(native(dict(scores=scores,adopted=adopted,mode=result['selected_mode'],post_update_delta=result['post_update_curvature_minus_sum'],timing=timing))),flush=True)

if __name__=='__main__':main()
