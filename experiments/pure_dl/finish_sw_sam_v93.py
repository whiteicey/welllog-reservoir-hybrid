"""Actual SW SAM compute-matched comparison, disk replay and planned cleanup."""
from datetime import datetime
import gc,hashlib,json,time,re
import numpy as np
import torch
from sw_sam_v93 import ROOT,STUDY,PREVIOUS,CONTROL,MODES,SOURCE,Network,Evaluator,load_data,reference_prediction
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def source_proof(cfg):
    for name,digest in cfg['source_hashes'].items():assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['initialization']['sha256']
    assert hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest()==cfg['scoring_kernel_sha256']
    n=cfg['target_normalization'];assert hashlib.sha256((ROOT/n['checkpoint']).read_bytes()).hexdigest()==n['sha256']

def main():
    if jobs():raise RuntimeError('Training remains active')
    tick=time.perf_counter();old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    hist=read(CONTROL/'config.json');source_proof(hist);hinitial=read(CONTROL/'initial.json')
    assert np.array_equal(np.load(CONTROL/'channel_pred.npy'),reference[:,2])
    keys=('initial_head_sha256','input_features','window_length','central_supervised_points','validation_stride','validation_batch','quality','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','utility_weight','decision_temperature','training_rows','validation_rows','bank_manifest','target_normalization')
    rows=[];cfgs=[];preds={};times=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{mode}_fold5'
        assert read(out/'runtime_status.json')['status']=='complete'
        cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');initial=read(out/'initial.json');source_proof(cfg)
        for k in keys:assert cfg[k]==hist[k],k
        assert cfg['trainable_parameters']==839481 and cfg['density_family']=='Gaussian'
        assert cfg['initial_state_sha256']==hist['initial_state_sha256']
        assert cfg['forward_backward_evaluations_per_update']==2 and cfg['reuse_dropkey_mask'] and cfg['restore_parameters_by_exact_copy']
        assert abs(rep['gradient_scope']['actual_perturbation_norm']-cfg['sam_rho'])<1e-6
        assert initial['initial_prediction_sha256']==hinitial['initial_prediction_sha256']
        assert rep['all144665_selected_predictions_restored_exactly'] and rep['best_epoch']>0
        for field in ('gradient_norms','decision_gradient_norms'):assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope'][field].values())
        pred=np.load(out/'channel_pred.npy');diag=diagnostics(truth,reference,pred,wells,'SW')
        assert abs(diag['scores']['SW']-rep['target_score'])<1e-9 and abs(diag['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
        post=np.load(out/'best_updated_pred.npy');assert np.array_equal(post,pred)
        rows.append(dict(mode=mode,report=rep,against_gaussian=diag,initial=initial));preds[mode]=pred;cfgs.append(cfg)
        launch=read(out/'launch.json');times.append(dict(mode=mode,start=launch['started_at'],end=rep['finished_at'],seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB']))
    allowed=('mode','training_objective_mode','initialization','sam_rho')
    for k in cfgs[0]:assert k in allowed or cfgs[0][k]==cfgs[1][k],k
    refarm=reference.copy();refarm[:,2]=preds['control'];contrast=diagnostics(truth,refarm,preds['sam005'],wells,'SW')
    post_delta=float(rows[1]['report']['best_after_update']['target_score']-rows[0]['report']['best_after_update']['target_score'])
    assert abs(post_delta-contrast['channel_delta'])<1e-9
    histories={m:{r['epoch']:r for r in read(ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{m}_fold5/validation_history.json')} for m in MODES}
    epochs=sorted(set(histories['control'])&set(histories['sam005']));deltas=[histories['sam005'][ep]['target_score']-histories['control'][ep]['target_score'] for ep in epochs]
    trajectory=dict(common_epochs=epochs,deltas=deltas,positive_epochs=sum(d>0 for d in deltas),negative_epochs=sum(d<0 for d in deltas),mean_delta=float(np.mean(deltas)),last_delta=float(deltas[-1]))
    best=max(rows,key=lambda r:r['report']['target_score']);adopt=bool(best['report']['target_score']>refs['SW']+1e-9)
    retained={r['mode'] for r in rows if r['against_gaussian']['channel_delta']>1e-9}
    if post_delta>1e-9:retained.update(MODES) # retain both sides of the predefined complete fair comparison
    combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    if retained:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data();assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        for mode in sorted(retained):
            out=ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{mode}_fold5';saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
            model=Network(mode,meta).cuda();model.load_state_dict(saved['state_dict'],strict=True)
            fresh,score,_=Evaluator(data,meta,2,reference)(model);assert np.array_equal(fresh,preds[mode])
            assert abs(score-next(r['report']['target_score'] for r in rows if r['mode']==mode))<1e-9
            proof.append(dict(mode=mode,rows=144665,fresh_disk_replay_exact=True,score=score,sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            if adopt and mode==best['mode']:
                combined[:,2]=fresh;choices[2].update(key='v93_SW_'+mode,mode=mode,model_module='sw_sam_v93',target_score=score,
                    checkpoint=str((out/'model_best.pt').relative_to(ROOT)),best_epoch=saved['best_epoch'],parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),builder='load_data; Network(mode,metadata); load_state_dict(state_dict); Gaussian Evaluator, no inference perturbation or DropKey')
            del model,saved;gc.collect();torch.cuda.empty_cache()
        del data,meta;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined);assert np.array_equal(combined[:,:2],reference[:,:2])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    result=native(dict(reference_scores=refs,selected_scores=scores,candidates=rows,sam_minus_control=contrast,post_update_sam_minus_control=post_delta,
        adopted=adopt,selected_mode=best['mode'] if adopt else None,retained=sorted(retained),verification=proof,
        same_capacity_and_initial_function=True,compute_matched_two_pass_control=True,trajectory_diagnostic=trajectory,initial_base_state_and_prediction_equal_to_gaussian=True,screening_fold=5,rows=144665,
        actual_fixed_three_column_scoring=True,strict805_passed=bool(scores['TOTAL']>80.5),complete_five_folds=False,remaining_folds_started=False,
        finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)));atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    segments=read(STUDY/'execution_segments.json')
    for item in segments:
        if item['status']=='running':
            item['end']=next(row['report']['finished_at'] for row in rows if row['mode']==item['mode'])
            item['status']='complete';item['end_basis']='final report timestamp; fresh process inventory empty'
    atomic_json(STUDY/'execution_segments.json',segments)
    for row in times:
        out=ROOT/f"outputs/pure_dl_sw_sam_v93_SW_{row['mode']}_fold5"
        peaks=[]
        for name in ('train.log','train.resume1.log'):
            f=out/name
            if f.exists():peaks.extend(float(x) for x in re.findall(r'"peak_allocated_MiB": ([0-9.]+)',f.read_text(encoding='utf-8')))
        row['peak_allocated_MiB_across_logged_segments']=max([row['peak_allocated_MiB']]+peaks)
        row['report_peak_scope']='final serial resume only; pre-failure allocated maximum from original epoch logs; pre-failure reserved maximum not recorded'
    events=[]
    for item in segments:
        start=datetime.fromisoformat(item['start']);end=datetime.fromisoformat(item['end']);events.extend([(start,1),(end,-1)])
        item['launch_to_terminal_seconds']=(end-start).total_seconds()
    for item in times:
        item['first_launch_to_final_completion_seconds']=(datetime.fromisoformat(item['end'])-datetime.fromisoformat(item['start'])).total_seconds()
    live=peak=0;last=None;overlap=0.
    for at,change in sorted(events):
        if last is not None and live==2:overlap+=(at-last).total_seconds()
        live+=change;peak=max(peak,live);last=at
    timing=dict(candidates=times,execution_segments=segments,first_launch_to_last_completion_seconds=(max(t for t,_ in events)-min(t for t,_ in events)).total_seconds(),
        first_launch_to_verified_seconds=(datetime.now().astimezone()-min(t for t,_ in events)).total_seconds(),peak_process_concurrency=peak,
        approximate_overlap_seconds=overlap,sum_process_segment_seconds=sum(t['launch_to_terminal_seconds'] for t in segments),finalization_seconds=time.perf_counter()-tick,
        concurrent_attempt_failed=True,recovered_serially=True,unpersisted_failed_batches_in_report_seconds=False,
        scope='actual segments use failure file and final report timestamps, not logical run lifespans; total includes failures/resume gaps; report seconds include only saved complete epochs/loading/warmup/eval/save; no safe concurrency/capacity claim')
    atomic_json(STUDY/'timing.json',timing)
    manifest=STUDY/'deletion_manifest.json';deleted=read(manifest).get('deleted',[]) if manifest.exists() else [];plan=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{mode}_fold5').resolve();assert out.parent==(ROOT/'outputs').resolve()
        for name in ('model_best.pt','model_best_updated.pt','last_state.pt'):
            if name=='model_best.pt' and mode in retained:continue
            # All candidates here have best_epoch>0 and identical normal/updated predictions; one best full checkpoint suffices for verified comparison.
            file=(out/name).resolve();assert file.parent==out
            if file.exists():plan.append(dict(path=str(file.relative_to(ROOT)),bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,no_automatic_resume=True,reason='absolute or complete predefined matched context gain; both comparison dependencies kept' if mode in retained else 'no absolute or matched gain'))
    # A durable full plan exists before any mutation; avoid the v89 metadata-loss failure.
    atomic_json(STUDY/'deletion_plan.json',native(dict(planned=plan,retained=sorted(retained),status='planned_before_unlink')))
    for item in plan:
        file=(ROOT/item['path']).resolve();assert file.parent.parent==(ROOT/'outputs').resolve();file.unlink();deleted.append(item)
        atomic_json(manifest,native(dict(deleted=deleted,status='in_progress')))
    atomic_json(manifest,native(dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),status='complete',retained=sorted(retained))))
    lines=['# v93 SW两次计算匹配SAM结果','',f"当前独立路线TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}。",'','|读出|SW|固定POR/PERM TOTAL|最佳/停止|','|---|---:|---:|---:|']
    for row in rows:
        r=row['report'];lines.append(f"|{row['mode']}|{r['target_score']:.12f}|{r['reference_replacement_scores']['TOTAL']:.12f}|{r['best_epoch']}/{r['stopped_epoch']}|")
    lines+=['','探索与验证：两项同839481参数、v56骨干/FiLM+训练先验重置，raw全state/旧头和144665点初始动作匹配历史Gaussian。原9曲线206/全井bank/分折/Q50/Gaussian/效用1/DropKey.4/1e-4/80-18保持。两项都两次前后向、同batch/RNG回放DropKey，首未clip全局L2梯度构造rho0/.05扰动，第二梯度AdamW，先clone精确恢复再更新；实际训练扰动半径/全网络和效用梯度/source/评分/预处理/最佳恢复/三列评价通过，没有独立预检或重复Q。历史单forward控制不声称同轨迹；本批rho0工作量匹配重跑。原始输入磁盘重建='+str(proof),'方法来源：[SAM论文](https://arxiv.org/abs/2010.01412)与官方google-research/sam仓库；自编AdamW变体，不声称原论文任务复现。旧v55 836921参数/3e-5/原模型warm-start SAM best0负证据见HISTORICAL.md；新配方不复活旧队列。']
    for row in rows:
        d=row['against_gaussian'];lines.append(f"{row['mode']}对Gaussian SW{d['channel_delta']:+.12f}/TOTAL{d['TOTAL_delta']:+.12f}，默认{d['populations']['default']['channel_delta']:+.12f}/数值{d['populations']['numeric']['channel_delta']:+.12f}，井升/降{d['well_wins']}/{d['well_losses']}。")
    lines+=['',f"收益：sam005-control SW{contrast['channel_delta']:+.12f}，完整训练后最佳匹配差{post_delta:+.12f}，采用={result['selected_mode']}；绝对和公平局部收益分开，局部小正不等同SOTA或独立泛化。",'不继续：若无绝对或完整匹配收益，关闭本v56-prior/Gaussian/效用1/DropKey.4/AdamW/rho.05/80-18变体，不临时扫rho/LR/ASAM；证据不泛化全部SAM。', '', '同轮轨迹：'+json.dumps(native(trajectory),ensure_ascii=False),'成本：'+json.dumps(timing,ensure_ascii=False),f"负项清理{len(deleted)}个权重，共{sum(x['bytes'] for x in deleted)}字节；完整删除计划先落盘，正源/比较依赖/旧交付保留。",'已做：真实两项、必要重载、实际组合、记录清理。未做：其他折和独立泛化；第五严格>80.5且明确收益才考虑其余四折，原完整OOF>82仍未达成。目标后独立井和新目录交付。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8')
    for p in (STUDY/'README.md',):append(p,'## 最终结果','## 最终结果\n'+text)
    (STUDY/'PROJECTS.md').write_text(text,encoding='utf-8')
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_sam_v93_SW_{mode}_fold5';append(out/'README.md','## 最终结果','## 最终结果\n'+text);(out/'PROJECTS.md').write_text(text,encoding='utf-8')
    note=f"## v93 SW两次计算匹配SAM完成\nsam005-control SW{post_delta:+.12f}；采用={result['selected_mode']}，当前TOTAL{scores['TOTAL']:.12f}，保留{sorted(retained)}。配方results/deep_learning/sw_sam_v93_20260930/CURRENT_RECIPE.json；无其他折，完整OOF>82未完成。"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v93 SW两次计算匹配SAM完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v93当前研究配方',note.replace('## v93 SW两次计算匹配SAM完成','## v93当前研究配方'))
    append(ROOT/'EXPLORATIONS.md','v93完成：','- v93完成：[结果](results/deep_learning/sw_sam_v93_20260930/FINAL.md)，'+note)
    print(json.dumps(native(dict(scores=scores,matched_delta=post_delta,selected_mode=result['selected_mode'],retained=sorted(retained),deleted_bytes=sum(x['bytes'] for x in deleted),timing=timing)),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
