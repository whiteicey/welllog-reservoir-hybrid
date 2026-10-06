"""Finalize matched fixed-gate/private-density loss trials, preserving the source function."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_frozen_gate_utility_v84r1 import ROOT,STUDY,PREVIOUS,CONTROL,MODES,SOURCE,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from train import metric
from fold5_screening_805 import qualifies

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def source_proof(cfg):
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert cfg['initialization']['sha256']==hashlib.sha256((ROOT/cfg['initialization']['checkpoint']).read_bytes()).hexdigest()
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
    fixed_source=torch.load(SOURCE,map_location='cpu',weights_only=False)['state_dict']
    fixed_expected_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in fixed_source.values())).hexdigest()
    rows=[];preds={};configs=[];times=[]
    common=('trainable_parameters','input_features','window_length','central_supervised_points','validation_stride','validation_batch','quality','decoder','density_components','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','eta_min','weight_decay','clip_norm','training_rows','validation_rows','bank_manifest','target_normalization','decision_temperature')
    for mode,spec in MODES.items():
        out=ROOT/f'outputs/pure_dl_sw_frozen_gate_utility_v84r1_SW_{mode}_fold5'
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
        assert not cfg['all_backbone_and_head_trainable'] and cfg['private_density_backbone_and_head_trainable'] and cfg['only_own_target_labels'] and cfg['screening_fold']==5
        assert len(cfg['bank_manifest']['train_wells'])==64 and not cfg['bank_manifest']['validation_fit']
        assert initial['all144665_parent_exact'] and not initial['initial_head_prior_only']
        assert initial['initial_prediction_sha256']==hashlib.sha256(reference[:,2].tobytes()).hexdigest()
        assert cfg['initial_fixed_hash']==fixed_expected_hash and rep['fixed_current_and_selected_bitwise_unchanged']
        assert rep['gradient_scope']['frozen_gate_has_no_gradient'] and cfg['total_parameters']==1678962
        assert cfg['reset_output_prior'] is False and cfg['lr']==3e-5
        for key in common:assert cfg[key]==hist_cfg[key],key
        for field in ('gradient_norms','decision_gradient_norms'):
            assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope'][field].values())
        assert rep['all144665_selected_predictions_restored_exactly']
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9
        assert abs(diag['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
        history=read(out/'validation_history.json')
        rows.append(dict(mode=mode,alpha=spec['utility_weight'],population=cfg['utility_population'],report=rep,initial=initial,against_global=diag,best_after_update=max(history,key=lambda x:x['target_score'])))
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
                choice=dict(target='SW',mode=mode,key='v84r1_SW_'+mode,model_module='sw_frozen_gate_utility_v84r1',checkpoint=rep['checkpoint'],target_score=score,best_epoch=rep['best_epoch'],builder='load_data; Network(mode,metadata); saved state_dict; original Evaluator; no inference dropout',parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
                choices=[choice if c['target']=='SW' else c for c in choices]
            del model,saved;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);finished=datetime.now().astimezone()
    likelihood=reference.copy();likelihood[:,2]=preds['likelihood']
    contrast=diagnostics(truth,likelihood,preds['balanced'],wells,'SW')
    grid=[dict(mode='parent',alpha=1.,SW=refs['SW'],TOTAL=refs['TOTAL'],source=str((CONTROL/'fold_5_report.json').relative_to(ROOT)),historical=True)]
    grid += [dict(mode=r['mode'],alpha=r['alpha'],SW=r['report']['target_score'],TOTAL=r['against_global']['scores']['TOTAL'],source=str((ROOT/f"outputs/pure_dl_sw_frozen_gate_utility_v84r1_SW_{r['mode']}_fold5/fold_5_report.json").relative_to(ROOT)),historical=False) for r in rows]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,matched_loss_comparison=contrast,post_update_balanced_minus_likelihood=rows[1]['best_after_update']['target_score']-rows[0]['best_after_update']['target_score'],comparison_table=grid,
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,initial_state_equal_between_new_arms=True,initial_prediction_equal_to_retained_function=True,frozen_gate_parameter_hash_equal_to_source=True,
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
        out=(ROOT/f'outputs/pure_dl_sw_frozen_gate_utility_v84r1_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='positive full SW gain from private density' if mode in retained else 'no new private density gain; retained parent SW function preserved'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v84r1 固定SW默认网络/私有密度结果','',f"当前原第五固定三通道TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'','|模型|效用系数|SW|固定POR/PERM组合TOTAL|','|---|---:|---:|---:|']
    for row in grid:lines.append(f"|{row['mode']}|{row['alpha']}|{row['SW']:.12f}|{row['TOTAL']:.12f}|")
    lines+=['','## 探索与验证','固定一份当前v82完整SW网络的默认输出eval/no_grad，另一份私有网络更新骨干/FiLM/密度。1678962总参数，839481优化器参数，其中129个私有gate输出参数不用。两项同当前完整函数热启动，无输出重置，原输入/Q50/评分/解码不改；新3e-5/80-18且私有40%DropKey。只差均衡效用系数0/1，同一个损失图；BCE固定无gate梯度，数值NLL不改。人口比例原64训练井拟合，.5/.5份额均衡，推理无人口选模型。', '初始144665点与原函数相同，两个新完整状态相同。实际私有body/output/FiLM与辅助密度梯度、固定网络无梯度且当前/最佳hash一致；源码/评分/预处理/来源哈希、最佳恢复及实际三列评价通过；无独立数值预检。','正候选原始数据磁盘重建：'+str(proof)]
    for row in rows:
        rep=row['report'];d=row['against_global']
        lines.append(f"{row['mode']}最佳/停止{rep['best_epoch']}/{rep['stopped_epoch']}，对当前完整函数SW{d['channel_delta']:+.12f}/TOTAL{d['TOTAL_delta']:+.12f}，默认{d['populations']['default']['channel_delta']:+.12f}/数值{d['populations']['numeric']['channel_delta']:+.12f}，井胜/负{d['well_wins']}/{d['well_losses']}。更新后最高SW{row['best_after_update']['target_score']:.12f}与含起点成绩分开。")
    lines+=['','## 收益和不继续',f"均衡-似然匹配SW差{contrast['channel_delta']:+.12f}，更新后最高差{result['post_update_balanced_minus_likelihood']:+.12f}；采用={result['selected_mode']}，保留={sorted(retained)}。",'固定gate不保证默认点得分不变，密度也参与联合动作选择。开发折选择非独立泛化证明，初始及实际停止轮要与80上限分开。若两项均恢复起点，关闭此来源/私有密度/似然或均衡效用/3e-5组合，不继续扫LR或人口份额；不泛化全部分离模型。旧v62先前来源的负结果和本次不同来源/损失分开。','','## 原接口失败、耗时与留存','原v84缺少length导致初始评估失败，无初始分数或训练轮；原源码/配置/日志不改，本新目录只修复length1024/c2。不是算法负结果。',json.dumps(timing,ensure_ascii=False,indent=2),f"删除{len(deleted)}无新增收益/重复起点权重，共{sum(x['bytes'] for x in deleted)}字节；原正模型和交付保留。",'无其他折/旧队列；第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82仍未完成。目标后独立井确认和新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v84r1最终结果';note=marker+f"\n两项完成，当前独立通道原第五TOTAL={scores['TOTAL']:.12f}，相对.5组合{scores['TOTAL']-refs['TOTAL']:+.12f}；系数网格见sw_frozen_gate_utility_v84r1_20260930/FINAL.md，采用={result['selected_mode']}，保留={sorted(retained)}。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        for name in ('PROJECTS.md','README.md'):append(ROOT/f'outputs/pure_dl_sw_frozen_gate_utility_v84r1_SW_{mode}_fold5'/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v84r1完成状态',note.replace(marker,'## v84r1完成状态'))
    append(ROOT/'EXPLORATIONS.md','v84r1完成：','- v84r1完成：'+note+' [结果](results/deep_learning/sw_frozen_gate_utility_v84r1_20260930/FINAL.md)')
    print(json.dumps(result|dict(candidates='see assembled_report.json',timing=timing)),flush=True)

if __name__=='__main__':main()
