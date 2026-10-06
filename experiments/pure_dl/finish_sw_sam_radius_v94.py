"""Finalize bounded SAM radii against frozen complete.05/rho0 controls."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from sw_sam_radius_v94 import ROOT,STUDY,PREVIOUS,CONTROL,ZERO,MODES,SOURCE,Network,Evaluator,load_data,reference_prediction
from finish_sw_sam_v93 import source_proof
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from run_retrieval_smooth_v36 import jobs,append
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    if jobs():raise RuntimeError('Training remains active')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');previous=read(PREVIOUS/'assembled_report.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert reference.shape==truth.shape==(144665,3)
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    assert np.array_equal(reference[:,2],np.load(CONTROL/'channel_pred.npy'))
    controls={};histories={}
    for label,path in [('sam05',CONTROL),('rho0',ZERO)]:
        cfg=read(path/'config.json');source_proof(cfg);report=read(path/'fold_5_report.json');pred=np.load(path/'channel_pred.npy')
        proof=next(p for p in previous['verification'] if p['mode']==('sam005' if label=='sam05' else 'control'))
        assert hashlib.sha256((path/'model_best.pt').read_bytes()).hexdigest()==proof['sha256'] and proof['fresh_disk_replay_exact']
        controls[label]=dict(config=cfg,report=report,pred=pred);histories[label]={r['epoch']:r for r in read(path/'validation_history.json')}
    hist=controls['sam05']['config'];prior=read(CONTROL/'initial.json')
    zero_reference=reference.copy();zero_reference[:,2]=controls['rho0']['pred'];rows=[];configs=[];preds={};timing=[]
    keys=('trainable_parameters','initial_state_sha256','initial_head_sha256','input_features','window_length','central_supervised_points','validation_stride','validation_batch','quality','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','utility_weight','decision_temperature','training_rows','validation_rows','bank_manifest','target_normalization','forward_backward_evaluations_per_update','reuse_dropkey_mask','restore_parameters_by_exact_copy')
    for mode,spec in MODES.items():
        out=ROOT/f'outputs/pure_dl_sw_sam_radius_v94_SW_{mode}_fold5';assert read(out/'runtime_status.json')['status']=='complete'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json');source_proof(cfg)
        for k in keys:assert cfg[k]==hist[k],k
        assert cfg['sam_rho']==spec['rho'] and cfg['trainable_parameters']==839481
        assert initial['initial_prediction_sha256']==prior['initial_prediction_sha256']
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['gradient_scope']['second_pass_rng_consumption_exact']
        assert abs(rep['gradient_scope']['actual_perturbation_norm']-spec['rho'])<1e-6
        for f in ('gradient_norms','decision_gradient_norms'):assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope'][f].values())
        pred=np.load(out/'channel_pred.npy');post=np.load(out/'best_updated_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW');post0=diagnostics(truth,zero_reference,post,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9 and abs(diag['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
        assert abs(post0['scores']['SW']-rep['best_after_update']['target_score'])<1e-9
        if rep['best_epoch']>0:assert np.array_equal(pred,post)
        h={r['epoch']:r for r in read(out/'validation_history.json')};trajectory={}
        for label in controls:
            common=sorted(set(h)&set(histories[label]));ds=[h[e]['target_score']-histories[label][e]['target_score'] for e in common]
            trajectory[label]=dict(epochs=common,deltas=ds,positive_epochs=sum(d>0 for d in ds),negative_epochs=sum(d<0 for d in ds),mean_delta=float(np.mean(ds)),last_delta=float(ds[-1]))
        rows.append(dict(mode=mode,rho=spec['rho'],report=rep,against_sam05=diag,post_update_against_rho0=post0,trajectory=trajectory));configs.append(cfg);preds[mode]=pred
        timing.append(dict(mode=mode,start=read(out/'launch.json')['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB'],peak_scope=rep['peak_scope']))
    for k in configs[0]:assert k in ('mode','training_objective_mode','initialization','sam_rho') or configs[0][k]==configs[1][k],k
    best=max(rows,key=lambda r:r['report']['target_score']);adopt=bool(best['report']['target_score']>refs['SW']+1e-9)
    retained={r['mode'] for r in rows if r['against_sam05']['channel_delta']>1e-9 or r['post_update_against_rho0']['channel_delta']>1e-9}
    combined=reference.copy();choices=[dict(x) for x in old['choices']];proof=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data();assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for mode in sorted(retained):
            row=next(r for r in rows if r['mode']==mode);assert row['report']['best_epoch']>0
            out=ROOT/f'outputs/pure_dl_sw_sam_radius_v94_SW_{mode}_fold5';saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False);model=Network(mode,meta).cuda();model.load_state_dict(saved['state_dict'],strict=True)
            fresh,score,_=Evaluator(data,meta,2,reference)(model);assert np.array_equal(fresh,preds[mode]);proof.append(dict(mode=mode,rows=144665,fresh_disk_replay_exact=True,score=score,sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            if adopt and mode==best['mode']:
                combined[:,2]=fresh;choices[2].update(key='v94_SW_'+mode,mode=mode,model_module='sw_sam_radius_v94',checkpoint=str((out/'model_best.pt').relative_to(ROOT)),target_score=score,best_epoch=saved['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),builder='load_data; Network(mode,metadata); load_state_dict(saved[state_dict]); Gaussian Evaluator, no inference perturbation or DropKey')
            del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,meta;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined);assert np.array_equal(combined[:,:2],reference[:,:2]);assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    result=native(dict(reference_scores=refs,selected_scores=scores,candidates=rows,adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,
        historical_controls_reused_with_sha_and_prior_replay=True,same_initialization_capacity_compute_budget=True,screening_fold=5,rows=144665,actual_fixed_three_column_scoring=True,strict805_passed=bool(scores['TOTAL']>80.5),complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined);recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)));atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    starts=[datetime.fromisoformat(t['start']) for t in timing];ends=[datetime.fromisoformat(t['end']) for t in timing]
    assert max(starts)>=min(ends),'Unexpected concurrent v94 runs'
    cost=dict(candidates=timing,first_launch_to_last_completion_seconds=(max(ends)-min(starts)).total_seconds(),sum_candidate_seconds=sum(t['seconds'] for t in timing),first_launch_to_verified_seconds=(datetime.now().astimezone()-min(starts)).total_seconds(),actual_process_concurrency=1,overlap_seconds=0.,finalization_seconds=time.perf_counter()-tick,scope='loading/warmup/two-pass train/eval/save included; explicit serial launch gap in total wall time; authoring and atlas separately, no capacity claim')
    atomic_json(STUDY/'timing.json',native(cost))
    manifest=STUDY/'deletion_manifest.json';deleted=read(manifest).get('deleted',[]) if manifest.exists() else [];plan=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_sam_radius_v94_SW_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        for name in ('model_best.pt','model_best_updated.pt','last_state.pt'):
            if name=='model_best.pt' and mode in retained:continue
            p=(out/name).resolve();assert p.parent==out
            if p.exists():plan.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='positive absolute or complete same-capacity rho0 mechanism gain' if mode in retained else 'no new absolute or matched gain'))
    atomic_json(STUDY/'deletion_plan.json',native(dict(planned=plan,retained=sorted(retained),status='planned_before_unlink')))
    for item in plan:
        p=(ROOT/item['path']).resolve();assert p.parent.parent==(ROOT/'outputs').resolve();p.unlink();deleted.append(item);atomic_json(manifest,native(dict(deleted=deleted,status='in_progress')))
    atomic_json(manifest,native(dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained),status='complete')))
    lines=['# v94 SW有界SAM半径结果','',f"当前独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'','|半径|SW|固定POR/PERM TOTAL|最佳/停止|对.05 SW|训练后对rho0 SW|','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        p=r['report'];lines.append(f"|{r['rho']}|{p['target_score']:.12f}|{p['reference_replacement_scores']['TOTAL']:.12f}|{p['best_epoch']}/{p['stopped_epoch']}|{r['against_sam05']['channel_delta']:+.12f}|{r['post_update_against_rho0']['channel_delta']:+.12f}|")
    lines+=['','探索与验证：839481参数、同v56/训练先验重置，原9曲线206/bank/分折/Q50/Gaussian/效用1/DropKey.4/两次计算/原评分/1e-4/80-18，仅半径.025/.1变化；原.05/rho0已完整训练并重载，本批核对SHA复用，未重复不变的重载。实际梯度/扰动L2/RNG终态、初始hash、source/预处理/评分/144665最佳恢复与固定三列通过；新正绝对或完整匹配收益原始输入磁盘重载='+str(proof)+'。',f"收益：采用={result['selected_mode']}，保留={sorted(retained)}；对.05绝对和对rho0机制差分开，epoch0不当新增收益，不能称完整OOF或独立泛化。"]
    for r in rows:lines.append(r['mode']+' 默认/数值/井：'+json.dumps(native(r['against_sam05']['populations']),ensure_ascii=False)+f"；井升/降{r['against_sam05']['well_wins']}/{r['against_sam05']['well_losses']}；同轮轨迹："+json.dumps(native(r['trajectory']),ensure_ascii=False))
    lines+=['不继续：两项没有新的绝对增益则结束此半径范围，保留.05及真实机制亮点，不无界扫rho/LR/ASAM，不泛化SAM无效。', '', '成本：'+json.dumps(native(cost),ensure_ascii=False),f"删除{len(deleted)}个无亮点/重复状态{sum(x['bytes'] for x in deleted)}字节，完整计划先落盘，当前/正源/比较依赖/旧交付保留。",'已做：两项实测、选择/恢复、实际组合、必要重载、清理/记录；未做：其他折/独立泛化。原第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82未完成；目标后独立井确认和新目录交付。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8');(STUDY/'PROJECTS.md').write_text(text,encoding='utf-8');append(STUDY/'README.md','## 最终结果','## 最终结果\n'+text)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_sam_radius_v94_SW_{mode}_fold5';(out/'PROJECTS.md').write_text(text,encoding='utf-8');append(out/'README.md','## 最终结果','## 最终结果\n'+text)
    note=f"## v94 SW有界SAM半径完成\n当前TOTAL{scores['TOTAL']:.12f}，采用={result['selected_mode']}，保留={sorted(retained)}。配方results/deep_learning/sw_sam_radius_v94_20260930/CURRENT_RECIPE.json；两项真实串行，无其他折，原OOF>82未完成。"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v94 SW有界SAM半径完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v94当前研究配方',note.replace('## v94 SW有界SAM半径完成','## v94当前研究配方'))
    append(ROOT/'EXPLORATIONS.md','v94完成：','- v94完成：[结果](results/deep_learning/sw_sam_radius_v94_20260930/FINAL.md)，'+note)
    print(json.dumps(native(dict(scores=scores,selected_mode=result['selected_mode'],retained=sorted(retained),candidates=[dict(mode=r['mode'],SW=r['report']['target_score'],against05=r['against_sam05']['channel_delta'],against0=r['post_update_against_rho0']['channel_delta']) for r in rows],deleted_bytes=sum(x['bytes'] for x in deleted),timing=cost)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
