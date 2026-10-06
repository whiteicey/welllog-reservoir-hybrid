"""Publish compact final evidence; never rewrite predictions or training sources."""
from pathlib import Path
import hashlib
import json
import time
import psutil

S = Path(__file__).resolve().parent
R = S.parents[2]
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

final = read(S/'FINAL.json')
canonical = read(S/'FINAL_CANONICAL_OOF_AUDIT.json')
training = read(S/'FINAL_REMAINING_TRAINING_AUDIT.json')
state = read(S/'BATCH_STATE.json')
assert state['status']=='complete' and not state['active_jobs']
assert not psutil.pid_exists(state['controller_pid'])
assert canonical['coverage_once'] and canonical['canonical_population_and_group_split_exact']
assert canonical['scores']==final['scores']
assert training['remaining_fold_models']==8 and training['stages_per_model']==8
assert training['actual_checkpoint_loads_scoped_and_exact']
assert training['original_training_validation_raw_rows_exact']
assert training['dependency_source_hashes_exact'] and training['own_source_Q_before_best']
assert read(S/'FOLD4_PAIRED_NATIVE_QA.json')['two_complete8stage_models']
assert final['scores']['fixed']['TOTAL']>82
assert final['folds'][4]['fixed_scores']['TOTAL']>80.8
assert final['stages_per_model']==8 and final['historical_stages_per_pair']==16
checks = [
    ('experiments/pure_dl/train_pair_fullcv_source_v370.py', 'SOURCE constructs Network from initialization; fits input/target/depth statistics only on outer training wells; CUDA label tables contain only training rows; both expert and router parameters train.'),
    ('experiments/pure_dl/train_invariant.py', 'FeatureBuilder fits 64 training wells; feature medians/scales derive from training inputs; synthetic whitelist excludes DEVI/AZIM/BIT/CASE and derivatives.'),
    ('package/src/feature.py', 'FeatureBuilder uses declared input columns and per-well curve transforms, not output labels.'),
    ('experiments/pure_dl/fullcv_pair_context_v370.py', 'Shared cache has no learned weights or shared Q; 63 amplitude view statistics use training wells/rows; each model descriptor restores private source Q.'),
    ('experiments/pure_dl/fullcv_context_v97.py', 'Canonical grouped population and disjoint 64/16 wells; Q trace restricted to training raw rows and observation strictly before own source best; reference input bank averages training wells only.'),
    ('experiments/pure_dl/pair_fullcv_models_v370.py', 'Historical checkpoint paths redirect to own fold/seed parents; guarded actual loads forbid other seed/historical learned checkpoints; stage chain is SOURCE plus seven explicit stages.'),
    ('experiments/pure_dl/train_pair_fullcv_stage_v370.py', 'Loss receives only labeled training windows and own channel; forward receives input curves/depth/well inputs; fresh late branch optimizers include raw-feature backbone parameters; max80/pat10->14 preserved.'),
    ('experiments/pure_dl/por_conditional_rps_v339.py', 'POR conditional RPS/CE uses only supplied masked POR training labels, alongside learned residual representation.'),
    ('experiments/pure_dl/sw_direct_component_v275.py', 'SW state prior uses train_raw; categorical auxiliary targets use masked SW training labels; additional encoder/readout jointly train.'),
    ('experiments/pure_dl/sw_head_utility_v80r1.py', 'Action utility constructor uses fixed action/edge tables and train-fit metadata; forward receives masked training labels, not evaluator validation truth.'),
    ('experiments/pure_dl/evaluate_pair_fullcv_v370.py', 'Native prediction forward uses no labels; computes both joint distributions and fixed candidate weights uniformly for every row. Truth is not used in model forward or grid construction.'),
    ('experiments/pure_dl/record_pair_fullcv_v370.py', 'Fixed weights are read from predeclared plan, never reselected on folds1-4. Per-fold argmax ratios are saved in a separately labeled diagnostic. OOF scores use all723556 rows, not mean fold scores.'),
]
lineage = dict(observed_epoch=time.time(), inspection='Direct source inspection plus actual checkpoint/load-trace audits; CPU only, no training or new selection',
    files=[dict(path=p,sha256=hashlib.sha256((R/p).read_bytes()).hexdigest(),finding=f) for p,f in checks],
    fifth_source_and_head_evidence='FINAL_CANONICAL_OOF_AUDIT.json',
    remaining_fold_actual_evidence='FINAL_REMAINING_TRAINING_AUDIT.json',
    selection_boundary='The pair and fixed ratios were selected on fold5. Fold5 is screening/tuning data, not untouched validation. Foldwise optimal OOF additionally tunes each fold and is diagnostic only.',
    stage_boundary='Each selected model has8 actual stages. User-authorized two-model ensemble has16 historical stages; no claim of8 total ensemble stages.',
    speed_boundary='Batch elapsed wall includes four folds for both models, CPU work, overlap/serial operation and approval waiting; it is not one-model training time or a controlled concurrency speed benchmark.')
(S/'FINAL_SOURCE_LINEAGE_REVIEW.json').write_text(json.dumps(lineage,indent=2),encoding='utf-8')
old = [82.82307687736336,81.54716564017816,82.72091124192592,81.85756031202179,80.50419742447689]
old_oof = 81.89101929722543
lines = ['# 两种子补折最终结果','',
    '**固定比例完整 OOF：82.16741757218335；相对上次提交 +0.27639827495792。**','',
    '原模型占比 POR56% / PERM44% / SW68%，五折保持相同；每头所有行共用一个比例。配对和比例曾使用第五折选择，因此第五折不是独立验证。逐折最优82.19465144853783仅作调参诊断。','',
    '| 折 | 上次提交 | 固定融合 | Diff | 逐头最优诊断 |','|---|---:|---:|---:|---:|']
for i,f in enumerate(final['folds']):
    score=f['fixed_scores']['TOTAL']
    lines.append(f"| 第{i+1}折 | {old[i]:.6f} | {score:.6f} | {score-old[i]:+.6f} | {f['diagnostic_scores']['TOTAL']:.6f} |")
lines += [f"| 完整 OOF | {old_oof:.6f} | {final['scores']['fixed']['TOTAL']:.6f} | {final['scores']['fixed']['TOTAL']-old_oof:+.6f} | {final['scores']['diagnostic']['TOTAL']:.6f} |",'',
    '| 模型/融合 | POR | PERM | SW | TOTAL |','|---|---:|---:|---:|---:|']
for k in ('original','seed13','fixed','diagnostic'):
    q=final['scores'][k]
    lines.append('| '+k+' | '+' | '.join(f'{q[t]:.6f}' for t in ('POR','PERM','SW','TOTAL'))+' |')
lines += ['','## 已完成与验证','',
    '- 两种子原1–4折补齐，第五折复用已验证模型。最后SW351显式断点恢复：第12–17轮，最佳第3轮，恢复调用66.284154秒；包含失败调用合计206.750332秒，不含等待授权。',
    '- 原始723556行、GroupKFold、每折64训练井/16验证井、行序、官方评分和一次覆盖核验通过；四套OOF分数实际重算。',
    '- 原生分布融合端点、统一权重列和实际磁盘预测一致；未按行或真值类别选模型。',
    '- 补折实际64个阶段checkpoint/config/来源加载路径和SHA核验通过，来源/特征拟合/Q均限定本折训练井；第五折来源与六个终端头SHA/Q证据另行核验。',
    '- 每模型8阶段，两个模型合计16个历史阶段；三个头各新增一个联合骨干/读出修补阶段，融合扫描不增加训练阶段。历史第五折耐心配置差异保留并披露。',
    '- 固定OOF实际超过82；这是第五折选定配方的分组OOF成绩，不声称未知测试集或无偏外部泛化精度。旧交付保持不变。','',
    '## 尚未证实与可选后续','',
    '- 尚未制作本次新提交包，也未验证未知标签测试集精度。',
    '- 相邻阶段增益不能证明删阶段重训仍保持效果；8→5等压缩需独立消融，未启动。',
    '- 逐折调权比固定OOF只多0.027234，不能当成独立收益；不将诊断配比替换为无偏验证结果。',
    '- 已结束无收益尝试的清理沿用此前删除记录；当前有用模型、来源依赖和旧提交继续保留。','',
    '证据：FINAL.json、FINAL_CANONICAL_OOF_AUDIT.json、FINAL_REMAINING_TRAINING_AUDIT.json、FINAL_SOURCE_LINEAGE_REVIEW.json、FOLD4_PAIRED_NATIVE_QA.json。','']
(S/'FINAL_SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')
note = ('\n\n2026-10-02 V370 final verified result: fixed original grouped723556-row OOF '
        '82.16741757218335 (POR79.829021599653/PERM90.751217660625/SW75.587956888768), '
        'old submitted81.89101929722543 delta+0.27639827495792. Fold4 fixed82.152477252078, '
        'tuned82.220920308922; complete fold-tuned diagnostic82.194651448538. Canonical coverage/score/native uniform grid audit '
        'and all remaining actual checkpoints/configs/load traces/Q/source hashes passed; direct training-source inspection recorded. '
        'Final SW351 explicit epoch11 recovery completed next12 through17, best3, exit0; no current trainer/controller. '
        'Fifth-selected pair/fixed ratios remain disclosed; tuned OOF is diagnostic only. Each model8 stages; ensemble16 historical stages '
        '(accepted two-seed scope), no8-total claim. Old deliveries untouched, new submission package not prepared in this batch. '
        'Result and limitations: results/deep_learning/two_seed_fullcv_v370_20261002/FINAL_SUMMARY.md. '
        'Future optional independent pruning/generalization work remains unstarted.\n')
for p in (S/'README.md',S/'PROJECTS.md',R/'PROJECTS.md',R.parent/'PROJECTS.md',R/'docs/deep_learning/atlas/PROJECTS.md'):
    with p.open('a',encoding='utf-8') as out:out.write(note)
with (R/'EXPLORATIONS.md').open('a',encoding='utf-8') as out:
    out.write('\n[V370 最终结果](results/deep_learning/two_seed_fullcv_v370_20261002/FINAL_SUMMARY.md)：固定OOF82.167418，相对上次+0.276398；723556行完整核验通过，逐折最优82.194651仅调参诊断。第五折选择边界、8阶段/模型与16历史阶段/配对已披露。\n')
report=S/'FOLD_RESULTS.md'
text=report.read_text(encoding='utf-8').replace('完整五折OOF仍须等待全部723556行结果。','全部723556行OOF已完成并核验，固定82.167418；最终汇总见FINAL_SUMMARY.md。')
report.write_text(text,encoding='utf-8')
print(json.dumps(dict(fixed=final['scores']['fixed']['TOTAL'],delta=final['scores']['fixed']['TOTAL']-old_oof,diagnostic=final['scores']['diagnostic']['TOTAL'])))
