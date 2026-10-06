import json,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from compact_sw_readout_split_v121 import ROOT,STUDY,OUT,STAGES
import train_invariant as base
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    snap={}
    for mode in STAGES:
        o=OUT/mode;state=read(o/'runtime_status.json');report=read(o/'report.json');live=False
        try:live=any(Path(a).name=='train_compact_sw_readout_split_v121.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
        except (psutil.Error,ValueError):pass
        snap[mode]=dict(live=live,state=state,report=report)
    atomic_json(STUDY/'progress.json',snap)
    todo=[m for m in STAGES if snap[m]['report'] and not (STUDY/(m+'_assembled.json')).exists()]
    if todo:
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        por=ROOT/'outputs/pure_dl_compact_por_scale_v114_fold5/location_only';sw=ROOT/'outputs/pure_dl_compact_v100_fold5/perm';control=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        for mode in todo:
            report=snap[mode]['report'];assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
            pred=np.column_stack((np.load(por/'channel_pred.npy'),np.load(sw/'channel_pred.npy'),np.load(OUT/mode/'channel_pred.npy')))
            assert pred.shape==(144665,3) and np.isfinite(pred).all()
            scores=base.metric(data['truth'][data['validation']],pred)
            expected=[read(por/'report.json')['target_score'],read(sw/'report.json')['target_score'],report['target_score']]
            assert all(abs(scores[c]-v)<1e-9 for c,v in zip(base.TS,expected))
            result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,total_training_stages=5,source_stage_counted=True,
                gain_vs_local_SW=report['target_score']-read(control/'report.json')['target_score'],checkpoint_sha256=hashlib.sha256((OUT/mode/'model_best.pt').read_bytes()).hexdigest(),report=report)
            atomic_json(STUDY/(mode+'_assembled.json'),result);np.save(STUDY/(mode+'_assembled_pred.npy'),pred)
            text=f"\nV121 {mode}: restored SW={scores['SW']:.12f}; original local delta={result['gain_vs_local_SW']:+.12f}; actual fifth TOTAL={scores['TOTAL']:.12f};5 stages including source and2 POR phases; strict808={result['strict_808_passed']}; no other folds. best/stop={report['best_epoch']}/{report['stopped_epoch']}, seconds={report['seconds']:.3f}, peakMiB={report['peak_MiB']:.3f}.\n"
            for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with p.open('a',encoding='utf-8') as f:f.write(text)
    if all(x['report'] for x in snap.values()) and not (STUDY/'ablation_report.json').exists():
        a,b=[snap[m]['report'] for m in STAGES];cfg=[read(OUT/m/'config.json') for m in STAGES]
        assert cfg[0]['trainable_parameters']==cfg[1]['trainable_parameters']==879577
        atomic_json(STUDY/'ablation_report.json',dict(shared=a,split=b,split_minus_shared_SW=b['target_score']-a['target_score'],equal_active_capacity=True,initial_raw_identical=True,initial_predictive_functions_differ=False,original_fold=5,complete_five_folds=False))
    print(json.dumps({m:dict(live=v['live'],status=v['state'].get('status'),epoch=v['state'].get('epoch'),best=v['report'].get('target_score',v['state'].get('best'))) for m,v in snap.items()}))
if __name__=='__main__':main()
