"""Record live handles and assemble real fixed three-channel predictions.

No queue or GPU launch. Fair encoder contrast requires both finished reports.
"""
from datetime import datetime
import hashlib,json
import numpy as np
import psutil
import torch
from compact_cached_features_v103 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base

def read(path):return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}

def main():
    stages={}
    for name in STAGES:
        out=OUT/name;state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
        try:
            process=psutil.Process(state.get('pid',-1));args=process.cmdline()
            live=process.is_running() and any('train_compact_cached_features_v103.py' in x for x in args) and name in args
        except (psutil.Error,ValueError):pass
        stages[name]=dict(state=state,report=report,live_process_verified=live)
    atomic_json(STUDY/'progress.json',dict(at=datetime.now().astimezone().isoformat(),stages=stages,remaining_folds_started=False))
    reports={name:item['report'] for name,item in stages.items()}
    finished=[name for name in STAGES if reports[name] and not (STUDY/(name+'_assembled.json')).exists()]
    if finished:
        cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        por=ROOT/'outputs/pure_dl_compact_mean_scope_v101_fold5/por'
        perm=ROOT/'outputs/pure_dl_compact_v100_fold5/perm'
        fixed=[np.load(p/'channel_pred.npy') for p in (por,perm)]
        costs=1289.1515023999964+sum(read(p/'report.json')['seconds'] for p in (por,perm))
        for name in finished:
            pred=np.column_stack([*fixed,np.load(OUT/name/'channel_pred.npy')])
            assert pred.shape==(144665,3) and np.isfinite(pred).all()
            scores=base.metric(cache['truth'][cache['validation']],pred)
            expected=[read(por/'report.json')['target_score'],read(perm/'report.json')['target_score'],reports[name]['target_score']]
            assert all(abs(scores[c]-v)<1e-9 for c,v in zip(base.TS,expected))
            result=dict(scores=scores,original_fold=5,complete_five_folds=False,
                total_training_stages=4,planned_density_stage_would_make_five=True,
                dependency_stages=['original_source','v101_por_mean_scope','v100_perm',name],
                source_and_shared_stage_costs_counted=True,old_18stage_weights_loaded=False,
                cache_preparation_seconds=read(STUDY/(name+'_cache.json'))['seconds'],
                sum_dependency_seconds=costs+reports[name]['seconds']+read(STUDY/(name+'_cache.json'))['seconds'],
                timing_scope='sum including historical source, reused stages and input cache; not parallel wall or speedup',
                gain_vs_v101=scores['TOTAL']-80.45661107908712,gain_vs_old_best=scores['TOTAL']-80.50419742447689,
                strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,
                checkpoint_sha256=hashlib.sha256((OUT/name/'model_best.pt').read_bytes()).hexdigest(),report=reports[name])
            np.save(STUDY/(name+'_assembled_pred.npy'),pred);atomic_json(STUDY/(name+'_assembled.json'),result)
            text=f"\n### {name}实际完成\nSW={scores['SW']:.12f}；固定三通道TOTAL={scores['TOTAL']:.12f}，对旧最好{result['gain_vs_old_best']:+.12f}，严格80.8={result['strict_808_passed']}。best/stop={reports[name]['best_epoch']}/{reports[name]['stopped_epoch']}，本项{reports[name]['seconds']:.3f}秒，峰值{reports[name]['peak_MiB']:.1f}MiB。仅4个依赖训练阶段，拟第五密度阶段尚未执行；完整OOF未完成。\n"
            for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with path.open('a',encoding='utf-8') as f:f.write(text)
    if all(reports.values()) and not (STUDY/'ablation_report.json').exists():
        control=reports['random'];candidate=reports['pretrained']
        delta=candidate['target_score']-control['target_score']
        configs=[read(OUT/name/'config.json') for name in STAGES]
        assert configs[0]['trainable_parameters']==configs[1]['trainable_parameters']
        result=dict(original_fold=5,control=control,candidate=candidate,predefined_source_feature_delta=delta,
            same_capacity=True,same_initial_official_score=control['initial_score']==candidate['initial_score'],
            source_preprocessing_buffers_shared=True,source_feature_window=256,source_stride=128,source_core=[64,192],SW_context_window=1024,
            predictions_and_frozen_parameters_verified=all(r['predictions_reproduced_exactly'] and r['frozen_parameters_exact'] for r in reports.values()),
            complete_five_folds=False,remaining_folds_started=False)
        atomic_json(STUDY/'ablation_report.json',result)
        text=f"\n### 预定义编码器来源消融\npretrained-random SW={delta:+.12f}。同容量、私有骨干/新增stem初始化、源预处理buffers、损失/Q50/原第五折/80-18；两项实际最佳恢复完毕。该差值只证明本设置的源隐藏特征收益；不能当完整OOF或总体架构收益。\n"
        if delta<=0:text+='本来源特征方案不继续扫描编码器种子、LR或层数；负结论只适用于本256源块/1024新SW骨干设置。\n'
        for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with path.open('a',encoding='utf-8') as f:f.write(text)
    print('V103_STATUS',json.dumps({name:dict(live=s['live_process_verified'],status=s['state'].get('status','prepared'),epoch=s['state'].get('epoch'),best=s['report'].get('target_score',s['state'].get('best'))) for name,s in stages.items()}),flush=True)

if __name__=='__main__':main()
