"""Finalize point/local readout variants with disclosed near-capacity comparison."""
import gc,hashlib,json,time
from datetime import datetime
import numpy as np
import torch
from sw_residual_readout_v85 import ROOT,STUDY,PREVIOUS,CONTROL,MODES,SOURCE,Network,Evaluator,load_data,reference_prediction
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
    common=('input_features','window_length','central_supervised_points','validation_stride','validation_batch','quality','decoder','density_components','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','training_rows','validation_rows','bank_manifest','target_normalization','decision_temperature')
    for mode,spec in MODES.items():
        out=ROOT/f'outputs/pure_dl_sw_residual_readout_v85_SW_{mode}_fold5'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json')
        assert read(out/'runtime_status.json')['status']=='complete'
        source_proof(cfg)
        assert cfg['added_readout_parameters']=={'point64':9881,'local64':10137}[mode]
        assert cfg['initial_base_state_sha256']==hist_cfg['initial_state_sha256']
        assert cfg['utility_weight']==spec['utility_weight'] and cfg['trainable_parameters']=={'point64':849362,'local64':849618}[mode]
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
        rows.append(dict(mode=mode,added_parameters=cfg['added_readout_parameters'],report=rep,initial=initial,against_linear=diag,against_alpha0=zero_diag,best_after_update=max(history,key=lambda x:x['target_score'])))
        preds[mode]=pred;configs.append(cfg)
        launch=read(out/'launch.json');times.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB']))
    assert all(k in ('mode','training_objective_mode','utility_weight','initialization','trainable_parameters','added_readout_parameters','initial_state_sha256','initial_head_sha256','readout_mode') or configs[0][k]==configs[1][k] for k in configs[0])
    point_reference=reference.copy();point_reference[:,2]=preds['point64']
    contrast=diagnostics(truth,point_reference,preds['local64'],wells,'SW')
    best=max(rows,key=lambda row:row['report']['target_score']);adopt=best['report']['target_score']>refs['SW']+1e-9
    retained={row['mode'] for row in rows if row['report']['target_score']>refs['SW']+1e-9}
    if contrast['channel_delta']>1e-9 and rows[1]['report']['best_epoch']>0:retained.add('local64')
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
                choice=dict(target='SW',mode=mode,key='v85_SW_'+mode,model_module='sw_residual_readout_v85',checkpoint=rep['checkpoint'],target_score=score,best_epoch=rep['best_epoch'],builder='load_data; Network(mode,metadata); saved state_dict; original Evaluator; no inference dropout',parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
                choices=[choice if c['target']=='SW' else c for c in choices]
            del model,saved;gc.collect();torch.cuda.empty_cache()
    scores=metric(truth,combined);finished=datetime.now().astimezone()
    grid=[dict(mode='linear',SW=refs['SW'],TOTAL=refs['TOTAL'],added_parameters=0,source=str((CONTROL/'fold_5_report.json').relative_to(ROOT)),historical=True)]
    grid += [dict(mode=r['mode'],SW=r['report']['target_score'],TOTAL=r['against_linear']['scores']['TOTAL'],added_parameters=configs[i]['added_readout_parameters'],source=str((ROOT/f"outputs/pure_dl_sw_residual_readout_v85_SW_{r['mode']}_fold5/fold_5_report.json").relative_to(ROOT)),historical=False) for i,r in enumerate(rows)]
    result=native(dict(reference_scores=refs,selected_scores=scores,delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,readout_comparison_table=grid,local_minus_point=contrast,post_update_local_minus_point=rows[1]['best_after_update']['target_score']-rows[0]['best_after_update']['target_score'],
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,initial_base_state_and_prediction_equal_to_linear_control=True,shared_backbone_initialization_equal=True,capacity_difference_parameters=256,
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
        out=(ROOT/f'outputs/pure_dl_sw_residual_readout_v85_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve() and read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                file=(out/name).resolve();assert file.parent==out
                if file.exists():
                    deleted.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
                    atomic_json(manifest,dict(deleted=deleted,status='planned'));file.unlink()
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='positive absolute SW gain or predefined local-point structure gain' if mode in retained else 'no new readout gain; original linear positive control preserved'))
    atomic_json(manifest,dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained)))
    lines=['# v85 SW最终残差输出结构结果','',f"当前原第五固定三通道TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'','|模型|新增参数|SW|固定POR/PERM组合TOTAL|','|---|---:|---:|---:|']
    for row in grid:lines.append(f"|{row['mode']}|{row['added_parameters']}|{row['SW']:.12f}|{row['TOTAL']:.12f}|")
    lines+=['','## 探索与验证','同原v56训练折内骨干/FiLM和SW先验重置，9881/10137新增输出参数，全网络更新。CPU fork_rng保护原随机态、共同64隐藏层初始化，零投影和局部中心恒等保证初始函数同历史线性头；基本状态hash和144665点初始预测同历史源。两者容量差256，近容量而非完全同参数，不将对照差完全归因于邻域。原输入/Q50/评分/解码/全人口效用1/40%DropKey/1e-4/80-18保持，不叠加人口重权或固定默认。','主body/FiLM/output和新hidden/project/局部卷积真实梯度、源码/评分/预处理/来源哈希、最佳恢复及实际三列评价通过；没有独立数值预检或审查。正候选原始输入磁盘重建：'+str(proof)]
    for row in rows:
        rep=row['report'];d=row['against_linear']
        lines.append(f"{row['mode']}最佳/停止{rep['best_epoch']}/{rep['stopped_epoch']}，对历史线性SW{d['channel_delta']:+.12f}/TOTAL{d['TOTAL_delta']:+.12f}，默认{d['populations']['default']['channel_delta']:+.12f}/数值{d['populations']['numeric']['channel_delta']:+.12f}，井胜/负{d['well_wins']}/{d['well_losses']}。")
    lines+=['','## 收益与不继续',f"local-point SW{contrast['channel_delta']:+.12f}，训练后最高差{result['post_update_local_minus_point']:+.12f}；绝对和结构比较分开，采用={result['selected_mode']}，保留={sorted(retained)}。",'若两项都无绝对或预定义结构收益，关闭这两个小读出，不无界扩头或临时扫参数。开发折选择非独立泛化证明，容量差和实际停止轮如实记录；不是完整OOF或已超最终目标。','','## 成本、清理与未完成',json.dumps(timing,ensure_ascii=False,indent=2),f"删除{len(deleted)}无新增收益权重，共{sum(x['bytes'] for x in deleted)}字节；原正模型/来源/交付保留。",'无其他折/旧队列；第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82仍未完成。目标后独立井确认与新目录交付。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    marker='## v85最终结果';note=marker+f"\n两项完成，当前独立通道原第五TOTAL={scores['TOTAL']:.12f}，相对线性组合{scores['TOTAL']-refs['TOTAL']:+.12f}；输出结构对照见sw_residual_readout_v85_20260930/FINAL.md，采用={result['selected_mode']}，保留={sorted(retained)}。无其他折/旧任务，负权重按清单清理，原来源/交付保留；完整OOF>82未完成。"
    for path in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',STUDY/'README.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(path,marker,note)
    for mode in MODES:
        for name in ('PROJECTS.md','README.md'):append(ROOT/f'outputs/pure_dl_sw_residual_readout_v85_SW_{mode}_fold5'/name,marker,note)
    append(ROOT.parent/'AGENTS.md','## v85完成状态',note.replace(marker,'## v85完成状态'))
    append(ROOT/'EXPLORATIONS.md','v85完成：','- v85完成：'+note+' [结果](results/deep_learning/sw_residual_readout_v85_20260930/FINAL.md)')
    print(json.dumps(result|dict(candidates='see assembled_report.json',timing=timing)),flush=True)

if __name__=='__main__':main()
