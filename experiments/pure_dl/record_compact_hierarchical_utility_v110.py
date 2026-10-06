"""Restored candidate, historical matched flat control, actual three-channel score."""
import json,hashlib
import numpy as np
import torch,psutil
from compact_hierarchical_utility_v110 import ROOT,STUDY,OUT
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def main():
    folder=OUT/'hierarchical';state=read(folder/'runtime_status.json');report=read(folder/'report.json');live=False
    try:live=any(__import__('pathlib').Path(a).name=='train_compact_hierarchical_utility_v110.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
    except (psutil.Error,ValueError):pass
    atomic_json(STUDY/'progress.json',dict(live=live,state=state,report=report))
    if report and not (STUDY/'assembled.json').exists():
        control=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        old=read(control/'report.json');cfg=read(folder/'config.json');previous=read(control/'config.json')
        assert cfg['trainable_parameters']==previous['trainable_parameters']==845913
        for key in ('seed','epochs','patience','minimum_epochs','batch_size','steps','cosine_T_max','eta_min','utility_weight','sam_rho','window_length','source_sha256'):
            assert cfg[key]==previous[key],key
        assert cfg['lr_groups']==previous['lr_groups'] and report['initial_score']==old['initial_score']
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        fixed=[ROOT/'outputs/pure_dl_compact_mean_scope_v101_fold5/por',ROOT/'outputs/pure_dl_compact_v100_fold5/perm']
        pred=np.column_stack([*[np.load(p/'channel_pred.npy') for p in fixed],np.load(folder/'channel_pred.npy')])
        assert pred.shape==(144665,3) and np.isfinite(pred).all()
        scores=base.metric(data['truth'][data['validation']],pred)
        expected=[read(p/'report.json')['target_score'] for p in fixed]+[report['target_score']]
        assert all(abs(scores[c]-v)<1e-9 for c,v in zip(base.TS,expected))
        delta=report['target_score']-old['target_score']
        result=dict(scores=scores,original_fold=5,complete_five_folds=False,total_training_stages=4,source_stage_counted=True,
            dependency_stages=['original_source','v101_por','v100_perm','v110_SW_hierarchical'],remaining_folds_started=False,
            strict_808_passed=qualifies(scores['TOTAL']),predefined_hierarchical_minus_flat_SW=delta,historical_flat_control_reused=True,
            same_architecture_parameters_initialization_and_budget=True,training_reward_surrogate_only_changed=True,validation_decoder_unchanged=True,
            gain_vs_old_best=scores['TOTAL']-80.50419742447689,checkpoint_sha256=hashlib.sha256((folder/'model_best.pt').read_bytes()).hexdigest(),
            sum_dependency_seconds=1289.1515023999964+sum(read(p/'report.json')['seconds'] for p in fixed)+report['seconds'],
            timing_scope='source/shared historical training and candidate sum; not parallel wall',report=report,control_report=old)
        atomic_json(STUDY/'assembled.json',result);np.save(STUDY/'assembled_pred.npy',pred)
        text=f"\n最终：SW={scores['SW']:.12f}，hierarchical-flat={delta:+.12f}，实际TOTAL={scores['TOTAL']:.12f}，strict808={result['strict_808_passed']}；含源4训练阶段，仅原第五，完整OOF未验证。best/stop={report['best_epoch']}/{report['stopped_epoch']}，{report['seconds']:.3f}秒，峰值{report['peak_MiB']:.1f}MiB。\n"
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(text)
    print('V110_STATUS',json.dumps(dict(live=live,status=state.get('status'),epoch=state.get('epoch'),best=report.get('target_score',state.get('best')))),flush=True)

if __name__=='__main__':main()
