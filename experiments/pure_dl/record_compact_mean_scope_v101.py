"""Final fixed-fold update-scope comparison and shared compact recipe assembly."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from train_dual8222_heads80 import atomic_json
from compact_five_stage_v100 import ROOT,OUT as SHARED
from fold5_screening_808 import qualifies
import train_invariant as base

STUDY=ROOT/'results/deep_learning/compact_mean_scope_v101_20261001'
OWN=ROOT/'outputs/pure_dl_compact_mean_scope_v101_fold5/por'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    report=read(OWN/'report.json');control=read(SHARED/'por/report.json')
    if not report:return
    if not (STUDY/'ablation_report.json').exists():
        delta=report['target_score']-control['target_score']
        result=dict(original_fold=5,control_target_score=control['target_score'],candidate_target_score=report['target_score'],
          predefined_update_scope_delta=delta,absolute_vs_v97=report['against_v97_target'],same_initial_official_score=report['initial_score']==control['initial_score'],
          fixed_source_and_population=True,update_scope_ablation_not_equal_trainable_parameters=True,final_restoration_passed=report['predictions_reproduced_exactly'],
          best_epoch=report['best_epoch'],stopped_epoch=report['stopped_epoch'],seconds=report['seconds'],peak_MiB=report['peak_MiB'])
        atomic_json(STUDY/'ablation_report.json',result)
        with (STUDY/'README.md').open('a',encoding='utf-8') as f:f.write('\n\n## 单项最终结果\n'+f"POR={report['target_score']:.12f}，best/stop={report['best_epoch']}/{report['stopped_epoch']}；对预定义v100控制{delta:+.12f}，对旧绝对POR{report['against_v97_target']:+.12f}。初始官方POR完全一致，最佳恢复和冻结参数通过。该更新范围有实测正收益，但尚未完成本批三通道组合，不将它单独写成TOTAL收益。\n")
    paths=[OWN,SHARED/'perm',SHARED/'sw_density']
    if not all((p/'report.json').exists() for p in paths):
        print('V101_ABLATION_FINAL',json.dumps(read(STUDY/'ablation_report.json')),flush=True);return
    final=STUDY/'assembled_report.json'
    if final.exists():print('REUSE_V101_FINAL',json.dumps(read(final)['scores']),flush=True);return
    data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    pred=np.column_stack([np.load(p/'channel_pred.npy') for p in paths]);assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(data['truth'][data['validation']],pred)
    reports=[read(p/'report.json') for p in paths]
    assert all(abs(scores[base.TS[c]]-r['target_score'])<1e-9 for c,r in enumerate(reports))
    source=read(ROOT/'results/deep_learning/compact_five_stage_v100_20260930/TIMING_CONTRACT.json')
    joint=read(SHARED/'sw_joint/report.json')
    result=dict(scores=scores,original_fold=5,complete_five_folds=False,total_training_stages=5,
       dependency_stage_order=['original_source_pretrain','v101_por_mean_scope','v100_perm','v100_sw_joint','v100_sw_density'],
       reused_shared_stages_counted=True,old_18stage_downstream_weights_loaded=False,gain_vs_v96=scores['TOTAL']-80.50419742447689,
       strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,predefined_POR_scope_delta=read(STUDY/'ablation_report.json')['predefined_update_scope_delta'],
       source_pretrain_seconds=source['source_pretrain_seconds'],sum_dependency_stage_seconds=source['source_pretrain_seconds']+sum(r['seconds'] for r in reports)+joint['seconds'],
       timing_scope='dependency stage seconds including historical source and reused stages; not parallel wall or speedup proof',
       checkpoint_sha256=[hashlib.sha256((p/'model_best.pt').read_bytes()).hexdigest() for p in paths],selected_reports=reports)
    np.save(STUDY/'assembled_pred.npy',pred);atomic_json(final,result)
    text=f"# v101实际组合\n\n原第五TOTAL={scores['TOTAL']:.12f}，对旧80.504197={result['gain_vs_v96']:+.12f}；严格>80.8={result['strict_808_passed']}。\n\nPOR范围消融收益={result['predefined_POR_scope_delta']:+.12f}，本项POR和共享v100 PERM/SW实际组合；包含来源共5阶段，不借旧18阶段成品。完整OOF>82未完成，无其他折。\n\n固定源密度形状的局部收益保留；完整紧凑成绩若不足则优先改变SW表示，不重复扫描本项LR或容量。\n"
    (STUDY/'FINAL.md').write_text(text,encoding='utf-8')
    with (STUDY/'README.md').open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    for p in (STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write('\n\n'+text)
    print('V101_ASSEMBLED',json.dumps(result),flush=True)

if __name__=='__main__':main()
