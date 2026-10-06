"""Finalize the predefined population allocation study against global alpha1."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_population_utility_v83 import ROOT,STUDY,PREVIOUS,CONTROL,MODES,SOURCE,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric
from fold5_screening_805 import qualifies

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def source_proof(cfg):
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert cfg['initialization']['sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert cfg['scoring_kernel_sha256']==hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest()
    item=cfg['target_normalization']
    assert item['sha256']==hashlib.sha256((ROOT/item['checkpoint']).read_bytes()).hexdigest()

def main():
    if jobs():raise RuntimeError('Training still active')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    hist_cfg=read(CONTROL/'config.json');hist_initial=read(CONTROL/'initial.json');hist_report=read(CONTROL/'fold_5_report.json')
    source_proof(hist_cfg)
    assert np.array_equal(np.load(CONTROL/'channel_pred.npy'),reference[:,2])
    assert abs(hist_report['target_score']-refs['SW'])<1e-9
    zero_out=ROOT/'outputs/pure_dl_sw_full_utility_v81_SW_likelihood_fold5'
    zero_pred=np.load(zero_out/'channel_pred.npy');zero_reference=reference.copy();zero_reference[:,2]=zero_pred
    assert abs(metric(truth,zero_reference)['SW']-read(zero_out/'fold_5_report.json')['target_score'])<1e-9
    rows=[];preds={};configs=[];times=[]
    common=('trainable_parameters','input_features','window_length','central_supervised_points','validation_stride','validation_batch','quality','decoder','density_components','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','training_rows','validation_rows','bank_manifest','target_normalization','decision_temperature','initial_state_sha256')
    for mode,spec in MODES.items():
        out=ROOT/f'outputs/pure_dl_sw_population_utility_v83_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json')
        assert read(out/'runtime_status.json')['status']=='complete'
        source_proof(cfg)
        assert cfg['utility_weight']==spec['utility_weight'] and cfg['trainable_parameters']==839481
        pop=cfg['utility_population']
        assert pop['training_rows']==cfg['training_rows']==578891
        assert pop['default_rows']+pop['numeric_rows']==pop['training_rows']
        assert not pop['validation_population_used'] and pop['numeric_share']==spec['numeric_share']
        assert abs(pop['default_fraction']-pop['default_rows']/pop['training_rows'])<1e-12
        assert abs(pop['numeric_fraction']-pop['numeric_rows']/pop['training_rows'])<1e-12
        assert abs(pop['default_weight']*pop['default_fraction']+pop['numeric_weight']*pop['numeric_fraction']-1)<1e-12
        assert cfg['all_backbone_and_head_trainable'] and cfg['only_own_target_labels'] and cfg['screening_fold']==5
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert not initial['all144665_parent_exact'] and initial['initial_head_prior_only']
        assert initial['initial_prediction_sha256']==hist_initial['initial_prediction_sha256']
        for key in common:assert cfg[key]==hist_cfg[key],key
        for field in ('gradient_norms','decision_gradient_norms'):
            assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope'][field].values())
        assert rep['all144665_selected_predictions_restored_exactly']
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        assert abs(diag['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
        zero_diag=diagnostics(truth,zero_reference,pred,wells,'SW')
        history=read(out/'validation_history.json')
        rows.append(dict(mode=mode,numeric_share=spec['numeric_share'],population=cfg['utility_population'],report=rep,initial=initial,against_global=diag,against_alpha0=zero_diag,best_after_update=max(history,key=lambda x:x['target_score'])))
        preds[mode]=pred;configs.append(cfg)
        launch=read(out/'launch.json');times.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert all(k in ('mode','training_objective_mode','utility_weight','utility_population','initialization') or configs[0][k]==configs[1][k] for k in configs[0])
    best=max(rows,key=lambda row:row['report']['target_score']);adopt=best['report']['target_score']>refs['SW']+1e-9
    retained={row['mode'] for row in rows if row['report']['target_score']>refs['SW']+1e-9}
    combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data()
        for mode in sorted(retained):
            rep=next(row['report'] for row in rows if row['mode']==mode);file=ROOT/rep['checkpoint']
            model=Network(mode,meta).cuda();saved=torch.load(file,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['state_dict'],strict=True);pred,score,_=Evaluator(data,meta,2,reference)(model)
            assert np.array_equal(pred,preds[mode]) and abs(score-rep['target_score'])<1e-9
            proof.append(dict(mode=mode,raw_disk_replay_exact=True,rows=len(pred),checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
            if adopt and mode==best['mode']:
                combined[:,2]=pred
                choice=dict(target='SW',mode=mode,key='v83_SW_'+mode,model_module='sw_population_utility_v83',checkpoint=rep['checkpoint'],target_score=score,best_epoch=rep['best_epoch'],builder='load_data; Network(mode,metadata); saved state_dict; original Evaluator; no inference dropout',parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
                choices=[choice if c['target']=='SW' else c for c in choices]
            del model,saved;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);finished=datetime.now().astimezone()
    population=configs[0]['utility_population']
    grid=[dict(mode='global',numeric_share=population['numeric_fraction'],SW=refs['SW'],TOTAL=refs['TOTAL'],source=str((CONTROL/'fold_5_report.json').relative_to(ROOT)),historical=True)]
    grid += [dict(mode=r['mode'],numeric_share=r['numeric_share'],SW=r['report']['target_score'],TOTAL=r['against_global']['scores']['TOTAL'],source=str((ROOT/f"outputs/pure_dl_sw_population_utility_v83_SW_{r['mode']}_fold5/fold_5_report.json").relative_to(ROOT)),historical=False) for r in rows]
    grid.sort(key=lambda r:r['numeric_share'])
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,population_grid=grid,
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,initial_state_and_prediction_equal_to_global_alpha1=True,
        actual_fixed_three_column_scoring=True,screening_fold=5,rows=144665,strict805_qualified=qualifies(scores['TOTAL']),complete_five_folds=False,remaining_folds_started=False,finished_at=finished.isoformat()))
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    events=[]
    for item in times:
        events.extend([(datetime.fromisoformat(item['start']),1),(datetime.fromisoformat(item['end']),-1)])
        item['launch_to_completion_seconds']=(datetime.fromisoformat(item['end'])-datetime.fromisoformat(item['start'])).total_seconds()
    live=peak=0;previous=None;overlap=0.
    for at,change in sorted(events):
        if previous is not None and live==2:overlap+=(at-previous).total_seconds()
        live+=change;peak=max(peak,live);previous=at
    timing=dict(candidates=times,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),first_launch_to_verified_wall_seconds=(finished-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,process_overlap_seconds=overlap,sum_individual_launch_to_completion_seconds=sum(x['launch_to_completion_seconds'] for x in times),finalization_seconds=time.perf_counter()-tick,
        scope='loading/preprocessing/warmup/training/evaluation/persistence and explicit launch gap included; authoring before first launch and atlas after score verification separate; observed admission may require serial runs; no throughput benchmark')
    atomic_json(STUDY/'timing.json',timing)
    deleted=[];manifest=STUDY/'deletion_manifest.json'
    if manifest.exists():deleted=read(manifest)['deleted']
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_population_utility_v83_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='positive population allocation gain over verified global alpha1' if mode in retained else 'no new population allocation gain; global alpha1 control preserved'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v83 SW效用强度结果','',f"当前独立通道原第五折TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'', '|系数|SW|固定POR/PERM组合TOTAL|来源|','|---:|---:|---:|---|']
    for row in grid:lines.append(f"|{row['numeric_share']}|{row['SW']:.12f}|{row['TOTAL']:.12f}|{'历史完成对照' if row['historical'] else '本批新训练'}|")
    lines+=['','## 探索、验证与实际收益','仅联合效用系数改变；同v56迁移骨干及原64井重置SW先验、839481参数、40%DropKey、原输入/Q50/种子/评分/解码和1e-4/80-18预算。两项初始状态/全点预测与既有.5相同，关键配置匹配；共同可微损失图与门控/密度/骨干实际梯度。训练源码/评分kernel/预处理哈希、最佳恢复、真实三列评分通过。历史CUDA重训不声称逐位轨迹相同。', '正候选原始数据磁盘重建：'+str(proof)]
    for row in rows:
        rep=row['report'];d=row['against_global'];z=row['against_alpha0']
        lines.append(f"{row['mode']}：最佳/停止{rep['best_epoch']}/{rep['stopped_epoch']}；对.5 SW{d['channel_delta']:+.12f}/TOTAL{d['TOTAL_delta']:+.12f}；对0 SW{z['channel_delta']:+.12f}；对.5默认{d['populations']['default']['channel_delta']:+.12f}/数值{d['populations']['numeric']['channel_delta']:+.12f}；井胜/负{d['well_wins']}/{d['well_losses']}。")
    lines+=['','## 不继续与收益边界', '两项均未超过.5则关闭本温度/架构的辅助系数扫描；对0重复已知机制收益不当新系数收益。若有系数改善，只记录该有限网格证据，不自动加新点。开发折选择非独立泛化证明，代理效用非精确官方argmax；只固定一个SW模型，无行级选择。', '', '## 成本、留存与未完成',json.dumps(timing,ensure_ascii=False,indent=2),f"保留={sorted(retained)}，采用={result['selected_mode']}；删除{len(deleted)}无新增收益权重，共{sum(x['bytes'] for x in deleted)}字节，清单先写后删；旧.5/原来源/交付保留。",'无其他折/旧队列。第五严格>80.5且明确收益后才考虑其余四折，完整原五折OOF>82仍未完成；目标后独立井确认和新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v83最终结果';note=marker+f"\n两项完成，当前独立通道原第五TOTAL={scores['TOTAL']:.12f}，相对.5组合{scores['TOTAL']-refs['TOTAL']:+.12f}；系数网格见sw_population_utility_v83_20260930/FINAL.md，采用={result['selected_mode']}，保留={sorted(retained)}。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        for name in ('PROJECTS.md','README.md'):append(ROOT/f'outputs/pure_dl_sw_population_utility_v83_SW_{mode}_fold5'/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v83完成状态',note.replace(marker,'## v83完成状态'))
    append(ROOT/'EXPLORATIONS.md','v83完成：','- v83完成：'+note+' [结果](results/deep_learning/sw_population_utility_v83_20260930/FINAL.md)')
    print(json.dumps(result|dict(candidates='see assembled_report.json',timing=timing)),flush=True)

if __name__=='__main__':main()
