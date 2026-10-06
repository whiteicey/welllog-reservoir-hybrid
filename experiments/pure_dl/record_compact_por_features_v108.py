"""Real restored SW scores, fixed three-channel assembly, paired filter delta."""
import hashlib,json
import numpy as np
import psutil
import torch
from compact_por_features_v108 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def main():
    snapshot={};reports={}
    for mode in STAGES:
        state=read(OUT/mode/'runtime_status.json');report=read(OUT/mode/'report.json');live=False
        try:
            args=psutil.Process(state.get('pid',-1)).cmdline()
            live=any('train_compact_por_features_v108.py' in x for x in args) and mode in args
        except (psutil.Error,ValueError):pass
        snapshot[mode]=dict(live=live,state=state,report=report);reports[mode]=report
    atomic_json(STUDY/'progress.json',snapshot)
    finished=[m for m in STAGES if reports[m] and not (STUDY/(m+'_assembled.json')).exists()]
    if finished:
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        paths=[ROOT/'outputs/pure_dl_compact_mean_scope_v101_fold5/por',ROOT/'outputs/pure_dl_compact_v100_fold5/perm']
        fixed=[np.load(p/'channel_pred.npy') for p in paths]
        for mode in finished:
            pred=np.column_stack([*fixed,np.load(OUT/mode/'channel_pred.npy')])
            assert pred.shape==(144665,3) and np.isfinite(pred).all()
            scores=base.metric(data['truth'][data['validation']],pred)
            expected=[read(p/'report.json')['target_score'] for p in paths]+[reports[mode]['target_score']]
            assert all(abs(scores[c]-v)<1e-9 for c,v in zip(base.TS,expected))
            result=dict(scores=scores,original_fold=5,complete_five_folds=False,total_training_stages=4,optional_density_would_make_five=True,
                dependency_stages=['original_source','v101_por_mean_scope','v100_perm','v108_'+mode],source_stage_counted=True,
                old_SW_weights_or_source_hidden_features_used=False,frozen_compact_POR_posterior_used=(mode=="por_posterior"),strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,
                sum_dependency_seconds=1289.1515023999964+sum(read(p/'report.json')['seconds'] for p in paths)+reports[mode]['seconds'],
                feature_preparation_seconds=read(STUDY/'feature_manifest.json')['seconds'],
                timing_scope='historical source/shared training sum; input feature preparation separate; not parallel wall or speedup',
                gain_vs_old_best=scores['TOTAL']-80.50419742447689,
                checkpoint_sha256=hashlib.sha256((OUT/mode/'model_best.pt').read_bytes()).hexdigest(),report=reports[mode])
            atomic_json(STUDY/(mode+'_assembled.json'),result);np.save(STUDY/(mode+'_assembled_pred.npy'),pred)
            text=f"\n### {mode}最终实际组合\nSW={scores['SW']:.12f}，TOTAL={scores['TOTAL']:.12f}，对旧最好{result['gain_vs_old_best']:+.12f}；strict808={result['strict_808_passed']}。本项best/stop={reports[mode]['best_epoch']}/{reports[mode]['stopped_epoch']}，{reports[mode]['seconds']:.3f}秒，峰值{reports[mode]['peak_MiB']:.1f}MiB。仅原第五、4依赖训练阶段，来源算入，第5密度未执行；无其他折。\n"
            for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with p.open('a',encoding='utf-8') as f:f.write(text)
    if all(reports.values()) and not (STUDY/'ablation_report.json').exists():
        configs=[read(OUT/m/'config.json') for m in STAGES]
        assert all(c['trainable_parameters']==846297 for c in configs)
        control=reports['curve_control'];candidate=reports['por_posterior'];delta=candidate['target_score']-control['target_score']
        result=dict(original_fold=5,control=control,candidate=candidate,predefined_semantic_feature_delta=delta,same_trainable_parameters=True,
            same_initial_official_score=control['initial_score']==candidate['initial_score'],same_network_initialization=True,input_feature_semantics_differ=True,
            complete_five_folds=False,remaining_folds_started=False)
        atomic_json(STUDY/'ablation_report.json',result)
        text=f"\n### POR语义输入最终消融\nposterior-control SW={delta:+.12f}；same846297容量与其他参数初始化/原数据/Q50/预算，两项均实际恢复最佳。只有本4维POR语义输入设置的单折证据，非完整OOF收益。\n"
        if delta<=0:text+='关闭本4维POR语义统计设置；不自动扫描更多老师或统计，不泛化为所有语义特征无效。\n'
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(text)
    print('V108_STATUS',json.dumps({m:dict(live=v['live'],status=v['state'].get('status','prepared'),epoch=v['state'].get('epoch'),best=v['report'].get('target_score',v['state'].get('best'))) for m,v in snapshot.items()}),flush=True)

if __name__=='__main__':main()
