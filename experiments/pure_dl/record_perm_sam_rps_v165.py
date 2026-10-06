"""Record actual three-channel assemblies and the completed matched optimizer pair."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch,psutil
from perm_sam_rps_v165 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def main():
    states={}
    for mode in STAGES:
        out=OUT/mode;state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
        try:live=any(Path(a).name=='train_perm_sam_rps_v165.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
        except (psutil.Error,ValueError):pass
        states[mode]=dict(live=live,state=state,report=report)
    atomic_json(STUDY/'progress.json',states)
    todo=[mode for mode in STAGES if states[mode]['report'] and not (STUDY/(mode+'_assembled.json')).exists()]
    if todo:
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        por=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps'
        sw=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        for mode in todo:
            out=OUT/mode;report=states[mode]['report']
            assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
            pred=np.column_stack((np.load(por/'channel_pred.npy'),np.load(out/'channel_pred.npy'),np.load(sw/'channel_pred.npy')))
            assert pred.shape==(144665,3) and np.isfinite(pred).all()
            scores=base.metric(data['truth'][data['validation']],pred)
            expected=(read(por/'report.json')['target_score'],report['target_score'],read(sw/'report.json')['target_score'])
            assert all(abs(scores[k]-v)<1e-9 for k,v in zip(base.TS,expected))
            result=dict(scores=scores,screening_fold=5,validation_rows=144665,total_training_stages=5,
                        source_stage_counted=True,complete_five_folds=False,remaining_folds_started=False,
                        strict_808_passed=qualifies(scores['TOTAL']),gain_vs_positive_RPS_PERM=scores['PERM']-89.40556485691894,
                        checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),report=report)
            atomic_json(STUDY/(mode+'_assembled.json'),result);np.save(STUDY/(mode+'_assembled_pred.npy'),pred)
            note=f"\nV165 {mode}: actual PERM={scores['PERM']:.12f}, TOTAL={scores['TOTAL']:.12f}; delta historicalRPS={result['gain_vs_positive_RPS_PERM']:+.12f}; best/stop={report['best_epoch']}/{report['stopped_epoch']}, seconds={report['seconds']:.3f}, peakMiB={report['peak_MiB']:.3f}; exact restored144665/frozen,5phases,no otherfolds. Pair gain not claimed until both finish.\n"
            for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with p.open('a',encoding='utf-8') as f:f.write(note)
    results={mode:read(STUDY/(mode+'_assembled.json')) for mode in STAGES}
    if all(results.values()) and not (STUDY/'control_delta.json').exists():
        configs={mode:read(OUT/mode/'config.json') for mode in STAGES}
        for key in ('seed','epochs','patience','minimum_epochs','batch_size','steps','cosine_T_max','eta_min','trainable_parameters','source_sha256','warm_parent_sha256','lr_groups','training_rows','validation_rows','quality'):
            assert configs['adamw'][key]==configs['sam'][key],key
        assert configs['adamw']['sam_rho']==0 and configs['sam']['sam_rho']==.05
        assert abs(results['sam']['report']['initial_score']-results['adamw']['report']['initial_score'])<1e-9
        delta=results['sam']['scores']['PERM']-results['adamw']['scores']['PERM']
        atomic_json(STUDY/'control_delta.json',dict(matched_pair_complete=True,sam_minus_adamw_PERM=delta,
                    equal_update_budget=True,equal_compute_budget=False,sam_passes_per_update=2,adamw_passes_per_update=1,
                    absolute_sam_gain_vs_positive_RPS=results['sam']['gain_vs_positive_RPS_PERM'],results=results))
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(f'\nV165 completed matched80/14 pair: SAM-AdamW PERM={delta:+.12f}. Equal updates, unequal compute; original fifth only, not fullOOF.\n')
    print(json.dumps({m:dict(live=s['live'],status=s['state'].get('status'),epoch=s['state'].get('epoch'),best=s['report'].get('target_score',s['state'].get('best'))) for m,s in states.items()}))

if __name__=='__main__':main()
