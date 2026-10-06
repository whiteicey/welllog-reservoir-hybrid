"""Real restored PERM update-scope comparison and five-stage target assembly."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from compact_perm_mean_v111 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def main():
    reports={};snapshot={}
    for mode in STAGES:
        folder=OUT/mode;state=read(folder/'runtime_status.json');report=read(folder/'report.json');live=False
        try:live=any(Path(a).name=='train_compact_perm_mean_v111.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
        except (psutil.Error,ValueError):pass
        reports[mode]=report;snapshot[mode]=dict(live=live,state=state,report=report)
    atomic_json(STUDY/'progress.json',snapshot)
    finished=[m for m in STAGES if reports[m] and not (STUDY/(m+'_assembled.json')).exists()]
    if finished:
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        por=ROOT/'outputs/pure_dl_compact_mean_scope_v101_fold5/por';sw=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        parent=ROOT/'outputs/pure_dl_compact_v100_fold5/perm';parent_report=read(parent/'report.json')
        for mode in finished:
            report=reports[mode];folder=OUT/mode
            pred=np.column_stack((np.load(por/'channel_pred.npy'),np.load(folder/'channel_pred.npy'),np.load(sw/'channel_pred.npy')))
            assert pred.shape==(144665,3) and np.isfinite(pred).all()
            scores=base.metric(data['truth'][data['validation']],pred)
            expected=[read(por/'report.json')['target_score'],report['target_score'],read(sw/'report.json')['target_score']]
            assert all(abs(scores[c]-v)<1e-9 for c,v in zip(base.TS,expected))
            result=dict(scores=scores,original_fold=5,complete_five_folds=False,executed_recipe_training_stages=5,
                selected_effective_dependency_stages=4 if report['best_epoch']==0 else 5,source_stage_counted=True,
                dependency_stages=['original_source','v101_POR','v100_PERM_initial','v111_PERM_'+mode,'v104_SW'],
                gain_vs_parent_PERM=report['target_score']-parent_report['target_score'],strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,
                checkpoint_sha256=hashlib.sha256((folder/'model_best.pt').read_bytes()).hexdigest(),report=report,
                sum_executed_dependency_seconds=1289.1515023999964+sum(read(p/'report.json')['seconds'] for p in (por,parent,sw))+report['seconds'],
                timing_scope='historical shared stages/source plus executed refinement sum; pair wall separate, not parallel speedup')
            atomic_json(STUDY/(mode+'_assembled.json'),result);np.save(STUDY/(mode+'_assembled_pred.npy'),pred)
            text=f"\n{mode}最终：PERM={scores['PERM']:.12f}，对父={result['gain_vs_parent_PERM']:+.12f}；实际TOTAL={scores['TOTAL']:.12f}，strict808={result['strict_808_passed']}。执行5阶段、选定有效依赖{result['selected_effective_dependency_stages']}，含源，仅第五，未补其他折。best/stop={report['best_epoch']}/{report['stopped_epoch']}，{report['seconds']:.3f}秒，峰值{report['peak_MiB']:.1f}MiB。\n"
            for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with p.open('a',encoding='utf-8') as f:f.write(text)
    if all(reports.values()) and not (STUDY/'ablation_report.json').exists():
        control=reports['gate_route'];candidate=reports['mean_coadapt'];delta=candidate['target_score']-control['target_score']
        assert control['initial_score']==candidate['initial_score']==89.37839421114361
        configs=[read(OUT/m/'config.json') for m in STAGES]
        assert configs[1]['trainable_parameters']-configs[0]['trainable_parameters']==2064
        atomic_json(STUDY/'ablation_report.json',dict(original_fold=5,control=control,candidate=candidate,
            predefined_mean_minus_gate_route_PERM=delta,update_scope_ablation=True,equal_trainable_capacity=False,
            additional_active_mean_elements=2064,same_initial_function_score=True,complete_five_folds=False,remaining_folds_started=False))
        text=f"\n最终更新范围对照：mean-gate_route PERM={delta:+.12f}；候选多2064活跃均值变量，非同训练容量。两项从同一个已选PERM函数开始，仅原第五证据，非完整OOF收益。\n"
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(text)
    print('V111_STATUS',json.dumps({m:dict(live=s['live'],status=s['state'].get('status','prepared'),epoch=s['state'].get('epoch'),best=reports[m].get('target_score',s['state'].get('best'))) for m,s in snapshot.items()}),flush=True)

if __name__=='__main__':main()
