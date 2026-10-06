"""Historical same-capacity Gaussian comparison, actual assembly and cleanup."""
from datetime import datetime
import gc,hashlib,json,time
import numpy as np
import torch
from sw_student4_v89 import ROOT,STUDY,PREVIOUS,CONTROL,SOURCE,Network,Evaluator,load_data,reference_prediction
from train_dual8222_heads80 import atomic_json,atomic_array
from finish_all_heads_v26 import diagnostics,native
from run_retrieval_smooth_v36 import append,jobs
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def source_proof(cfg):
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==cfg['initialization']['sha256']
    assert hashlib.sha256((ROOT/'package/src/utils.py').read_bytes()).hexdigest()==cfg['scoring_kernel_sha256']
    n=cfg['target_normalization'];assert hashlib.sha256((ROOT/n['checkpoint']).read_bytes()).hexdigest()==n['sha256']

def main():
    if jobs():raise RuntimeError('Training remains active')
    tick=time.perf_counter();out=ROOT/'outputs/pure_dl_sw_student4_v89_SW_student4_fold5'
    assert read(out/'runtime_status.json')['status']=='complete'
    cfg=read(out/'config.json');rep=read(out/'fold_5_report.json');hist=read(CONTROL/'config.json');source_proof(cfg);source_proof(hist)
    keys=('trainable_parameters','initial_state_sha256','initial_head_sha256','input_features','window_length','central_supervised_points','validation_stride','validation_batch','quality','epochs','patience','minimum_epochs','batch_size','steps_per_epoch','seed','lr','eta_min','weight_decay','clip_norm','utility_weight','decision_temperature','training_rows','validation_rows','bank_manifest','target_normalization')
    for k in keys:assert cfg[k]==hist[k],k
    assert cfg['trainable_parameters']==839481 and cfg['density_family']=='Student-t4'
    assert rep['all144665_selected_predictions_restored_exactly']
    assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope']['gradient_norms'].values())
    assert all(np.isfinite(v) and v>0 for v in rep['gradient_scope']['decision_gradient_norms'].values())
    old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        truth=z['truth'][z['fold']==4].copy();wells=z['well'][z['fold']==4].copy()
    assert truth.shape==reference.shape==(144665,3)
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    assert np.array_equal(np.load(CONTROL/'channel_pred.npy'),reference[:,2])
    pred=np.load(out/'channel_pred.npy');d=diagnostics(truth,reference,pred,wells,'SW')
    assert abs(d['scores']['SW']-rep['target_score'])<1e-9 and abs(d['scores']['TOTAL']-rep['reference_replacement_scores']['TOTAL'])<1e-9
    adopted=bool(d['channel_delta']>1e-9);combined=reference.copy();choices=[dict(c) for c in old['choices']];proof=[]
    if adopted:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
        data,meta=load_data();assert np.array_equal(truth,data['truth'][data['validation']]) and np.array_equal(wells,data['wells'][data['validation']])
        saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False);model=Network('student4',meta).cuda()
        model.load_state_dict(saved['state_dict'],strict=True);fresh,score,_=Evaluator(data,meta,2,reference)(model)
        assert np.array_equal(fresh,pred);combined[:,2]=fresh
        choices[2].update(key='v89_SW_student4',mode='student4',model_module='sw_student4_v89',target_score=score,
            checkpoint=str((out/'model_best.pt').relative_to(ROOT)),best_epoch=saved['best_epoch'],
            parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
            builder='load_data; Network(student4,metadata); load_state_dict(state_dict); Student-t4 Evaluator')
        proof.append(dict(rows=144665,fresh_disk_replay_exact=True,score=score))
        del model,saved,data,meta;gc.collect();torch.cuda.empty_cache()
    scores=base.metric(truth,combined);assert np.array_equal(combined[:,:2],reference[:,:2])
    assert abs(scores['TOTAL']-.3*scores['POR']-.35*scores['PERM']-.35*scores['SW'])<1e-9
    manifest_path=STUDY/'deletion_manifest.json'
    deleted=read(manifest_path).get('deleted',[]) if manifest_path.exists() else []
    if not adopted:
        assert out.resolve().parent==(ROOT/'outputs').resolve()
        plan=[]
        for name in ('model_best.pt','model_best_updated.pt','last_state.pt'):
            p=(out/name).resolve();assert p.parent==out.resolve()
            if p.exists():plan.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        if plan:
            atomic_json(STUDY/'deletion_plan.json',dict(planned=plan,adopted=False,status='planned_before_unlink'))
        for item in plan:
            p=(ROOT/item['path']).resolve();assert p.parent==out.resolve();p.unlink();deleted.append(item)
            atomic_json(manifest_path,dict(deleted=deleted,adopted=False,status='in_progress'))
    known_bytes=sum(x['bytes'] for x in deleted if x['bytes'] is not None)
    unknown_count=sum(x['bytes'] is None for x in deleted)
    deleted_bytes=None if unknown_count else known_bytes
    atomic_json(manifest_path,dict(deleted=deleted,bytes=deleted_bytes,known_bytes=known_bytes,unknown_size_count=unknown_count,
        adopted=adopted,status='complete_with_metadata_gap' if unknown_count else 'complete'))
    launch=datetime.fromisoformat(read(out/'launch.json')['started_at']);end=datetime.fromisoformat(rep['finished_at'])
    timing=dict(single_candidate_seconds=rep['seconds'],launch_to_completion_seconds=(end-launch).total_seconds(),
        launch_to_verified_seconds=(datetime.now().astimezone()-launch).total_seconds(),peak_training_processes=1,
        peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB'],finalization_seconds=time.perf_counter()-tick,
        gaussian_control_retrained=False,scope='candidate data/warmup/train/eval/save included; historical control reused, authoring/atlas separate')
    result=native(dict(reference_scores=refs,selected_scores=scores,against_gaussian=d,candidate_report=rep,
        adopted=adopted,verification=proof,matched_raw_initialization_capacity_budget=True,initial_predictive_distributions_equal=False,
        screening_fold=5,rows=144665,actual_fixed_three_column_scoring=True,strict805_passed=scores['TOTAL']>80.5,
        complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'timing.json',timing)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    text=f'''# v89 SW Student-t4结果

候选SW {rep['target_score']:.12f}，固定POR/PERM TOTAL {rep['reference_replacement_scores']['TOTAL']:.12f}；对Gaussian SW {d['channel_delta']:+.12f}、TOTAL {d['TOTAL_delta']:+.12f}。最佳/停止{rep['best_epoch']}/{rep['stopped_epoch']}，{rep['stop_reason']}。采用={adopted}，当前独立路线TOTAL仍为{scores['TOTAL']:.12f}。

探索与验证：同839481参数、raw全模型/输出初始hash、原训练侧先验/输入bank/预算/效用1/40%DropKey对齐历史Gaussian。t4标准差参数化与NLL/CDF同步；初始概率分布不同，但动作选择可能相同，不强行要求动作差异。实际梯度/源/预处理/评分hash、最佳144665恢复、原行序和固定三列评价通过。正候选重载={proof}；负项不重复原Q或独立预检。

默认/数值/井诊断：{json.dumps(native(d['populations']),ensure_ascii=False)}；{d['well_wins']}井升/{d['well_losses']}井降。

收益与不继续：未超同容量Gaussian则关闭当前df4/迁移骨干/输出先验重置/效用1/DropKey0.4组合，不扫df或scale；不把从自身先验提高当超越对照，不泛化所有重尾网络。旧共享t3负证据单独保留，不重启。

成本：{json.dumps(timing,ensure_ascii=False)}。
清理{len(deleted)}个冗余权重，总大小记录={deleted_bytes}（存在元数据缺口时不得推算；详见清单）；原正源/旧交付不改。

已完成：真实训练、匹配历史比较、诊断、最佳恢复、实际组合、必要重载/清理。未做：其他折和独立泛化；原第五严格>80.5且明确收益后才考虑其余四折，完整原OOF>82仍未完成。目标后：独立井确认和新目录交付。
'''
    (STUDY/'FINAL.md').write_text(text,encoding='utf-8')
    for p in (STUDY/'README.md',out/'README.md'):append(p,'## 最终结果','## 最终结果\n'+text)
    for p in (STUDY/'PROJECTS.md',out/'PROJECTS.md'):p.write_text(text,encoding='utf-8')
    note=f"## v89 SW Student-t4完成\nSW{rep['target_score']:.12f}，对Gaussian{d['channel_delta']:+.12f}，最佳/停止{rep['best_epoch']}/{rep['stopped_epoch']}；采用={adopted}，当前TOTAL{scores['TOTAL']:.12f}。配方results/deep_learning/sw_student4_v89_20260930/CURRENT_RECIPE.json；无其他折。"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v89 SW Student-t4完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v89当前研究配方',note.replace('## v89 SW Student-t4完成','## v89当前研究配方'))
    print(json.dumps(native(dict(scores=scores,candidate=d['scores'],deltaSW=d['channel_delta'],adopted=adopted,deleted_bytes=deleted_bytes,timing=timing)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
