"""Finalize completed v41 negative family comparison without changing old models."""
from datetime import datetime
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import perm_family_v41 as module
from run_retrieval_smooth_v36 import jobs,append
from train import metric
from train_dual8222_heads80 import atomic_json,atomic_array

ROOT=module.ROOT
STUDY=module.STUDY
OUT=ROOT/'outputs/pure_dl_perm_family_v41_fold5'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def main():
    start=time.perf_counter()
    assert not jobs(),'Wait for active training'
    report=read(OUT/'fold_5_report.json');cfg=read(OUT/'config.json')
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256((ROOT/'experiments/pure_dl'/name).read_bytes()).hexdigest()==digest,name
    previous=read(module.PREVIOUS/'CURRENT_RECIPE.json');reference=module.reference_prediction()
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=cache['truth'][cache['validation']]
    assert truth.shape==reference.shape==(144665,3)
    refs=metric(truth,reference)
    assert all(abs(refs[k]-v)<1e-9 for k,v in previous['scores'].items())
    rows=[]
    for rep in report['arms']:
        mode=rep['mode'];candidate=reference.copy();updated=reference.copy()
        candidate[:,1]=np.load(OUT/mode/'candidate_pred.npy')
        updated[:,1]=np.load(OUT/mode/'best_updated_pred.npy')
        selected=np.load(OUT/mode/'channel_pred.npy')
        sc=metric(truth,candidate);up=metric(truth,updated)
        assert np.isfinite(candidate).all() and np.isfinite(updated).all()
        assert abs(sc['PERM']-rep['candidate_score'])<1e-9
        assert abs(up['PERM']-rep['best_after_update']['target_score'])<1e-9
        assert rep['candidate_predictions_restored_exactly'] and rep['all_frozen_state_unchanged']
        assert rep['selected_is_parent'] and np.array_equal(selected,reference[:,1])
        assert rep['candidate_score']<=refs['PERM']+1e-9,'Positive requires separate fresh disk replay'
        rows.append(dict(mode=mode,report=rep,candidate_scores=sc,post_update_scores=up,
            initial=read(OUT/mode/'initial_verification.json'),
            initial_intervention_delta=rep['initial_candidate_score']-refs['PERM'],
            training_best_minus_initial=up['PERM']-rep['initial_candidate_score'],
            training_best_minus_parent=up['PERM']-refs['PERM']))
    combined=reference.copy();scores=metric(truth,combined)
    comparison=rows[1]['post_update_scores']['PERM']-rows[0]['post_update_scores']['PERM']
    result=dict(finished_at=datetime.now().astimezone().isoformat(),reference_scores=refs,selected_scores=scores,
        delta_reference_TOTAL=scores['TOTAL']-refs['TOTAL'],candidates=rows,adopted=False,
        post_update_student3_minus_gaussian_PERM=comparison,all_two_complete=True,
        actual_combined_prediction_scored=True,rows=144665,screening_fold=5,
        complete_five_folds=False,remaining_folds_started=False,old_delivery_unchanged=True)
    atomic_array(STUDY/'assembled_pred.npy',combined);atomic_json(STUDY/'assembled_report.json',result)
    recipe=dict(previous);recipe.update(reference_recipe=str((module.PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    launch=read(OUT/'launch.json')
    timing=dict(training_and_evaluation_seconds=report['seconds'],peak_allocated_MiB=report['peak_allocated_MiB'],
        launch_to_completion_seconds=(datetime.fromisoformat(report['finished_at'])-datetime.fromisoformat(launch['started_at'])).total_seconds(),
        processes=1,candidates=2,scope='Two independent heads, sequential head updates with shared frozen forward in one GPU process. Launch wall time includes load/preparation/warmup/final restore; loop time excludes initial loading. Existing data cache reused. Authoring/finalization excluded.',
        finalization_seconds=time.perf_counter()-start)
    atomic_json(STUDY/'timing.json',timing)
    lines=['# v41 PERM分布族对照完成','',
        '## 探索重点与条件','',
        '冻结当前独立PERM预测器，两个同初始化11824参数小头修正16分量权重、位置与尺度。只改变Gaussian或Student-t3密度及对应CDF；t3 scale=SD/sqrt(3)，初始分量均值/标准差相同但分布不相同。默认概率固定，原log10标签、Q50、原9曲线/206主特征、数据划分、支持网格和官方动作/评分保持。',
        '80轮上限/patience18/min10/batch64/64步，两头独立优化和早停。相同训练窗口，AdamW3e-4，原预测器权重全部冻结。',
        '', '## 已探索与验证','',
        '|候选|初始PERM|训练后最高PERM|训练后减初始|训练后减原模型|',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"|{row['mode']}|{row['report']['initial_candidate_score']:.12f}|{row['post_update_scores']['PERM']:.12f}|{row['training_best_minus_initial']:+.12f}|{row['training_best_minus_parent']:+.12f}|")
    lines += ['', '两组均第18轮早停，训练后最高均在第16轮；候选自身最佳均为第0轮，最终都回退原PERM。Gaussian初始全144665点逐点相同；两候选恢复、冻结权重和源哈希核对通过。输入/输出层均有实际梯度和参数变化。',
        '', '## 取得的收益与解释','',
        f"固定POR/PERM/SW实际三列评分TOTAL={scores['TOTAL']:.12f}，相对父配方+0。训练后最佳t3-Gaussian={comparison:+.12f} PERM，但t3初始已低于父模型，继续训练也低于自身初始。因此该局部差异不构成采用t3的正收益；单种子/重复开发折不作稳定性结论。",
        '', '## 不继续的点与证据','',
        '当前有界小头两种分布均未改善原PERM89.396724；不继续自由度、混合系数或解码网格扫描。结论限当前冻结预测器与小头/NLL配置，不能推广为所有重尾分布无效。',
        '', '## 耗时','',json.dumps(timing,ensure_ascii=False,indent=2),
        '', '## 目标与未做','',
        '第五折严格>80.5和完整原五折OOF>82仍未完成。原其他四折、旧队列均未启动，旧交付保留。当前配方沿用v31。后续若继续，应先提出不同机制的有界实验，避免再次扩展本次分布族小头扫参。',
        '结果见assembled_report.json、CURRENT_RECIPE.json；训练源/配置/逐轮记录与紧凑预测保留，负候选权重按删除清单清理。']
    text='\n'.join(lines)+'\n'
    for name in ('README.md','FINAL.md'):(STUDY/name).write_text(text,encoding='utf-8')
    note='## v41 PERM分布族完成\n'+f"Gaussian与Student-t3均18轮结束，均回退原PERM。训练后最佳89.367194812859/89.372730778423，父模型89.396724049366。实际三列TOTAL={scores['TOTAL']:.12f}不变，无其他折/旧队列；详细证据见results/deep_learning/perm_family_v41_20260929/FINAL.md。目标仍未完成，不延伸本设置分布族扫描。"
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md',STUDY/'PROJECTS.md',OUT/'PROJECTS.md',OUT/'README.md'):
        append(p,'## v41 PERM分布族完成',note)
    append(ROOT/'EXPLORATIONS.md','v41最终：','- v41最终：两种PERM分布均未超过父模型；实际TOTAL80.336513保持。[结果](results/deep_learning/perm_family_v41_20260929/FINAL.md)。')
    print(json.dumps(dict(scores=scores,comparison=comparison,timing=timing)),flush=True)

if __name__=='__main__':main()
