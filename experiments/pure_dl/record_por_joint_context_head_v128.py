import json,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from por_joint_context_head_v128 import ROOT,STUDY,OUT,STAGES
import train_invariant as base
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    snap={}
    for mode in STAGES:
        out=OUT/mode;state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
        try:live=any(Path(a).name=='train_por_joint_context_head_v128.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
        except (psutil.Error,ValueError):pass
        snap[mode]=dict(live=live,state=state,report=report)
    atomic_json(STUDY/'progress.json',snap)
    todo=[m for m in STAGES if snap[m]['report'] and not (STUDY/(m+'_assembled.json')).exists()]
    if todo:
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        perm=ROOT/'outputs/pure_dl_compact_v100_fold5/perm';sw=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        for mode in todo:
            report=snap[mode]['report'];assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
            pred=np.column_stack((np.load(OUT/mode/'channel_pred.npy'),np.load(perm/'channel_pred.npy'),np.load(sw/'channel_pred.npy')))
            assert pred.shape==(144665,3) and np.isfinite(pred).all()
            scores=base.metric(data['truth'][data['validation']],pred)
            assert abs(scores['POR']-report['target_score'])<1e-9
            assert abs(scores['PERM']-read(perm/'report.json')['target_score'])<1e-9 and abs(scores['SW']-read(sw/'report.json')['target_score'])<1e-9
            result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,total_training_stages=5,source_stage_counted=True,
                gain_vs_v114_POR=scores['POR']-77.54806949489308,gain_vs_historical_POR=scores['POR']-77.551936801885,strict_channel_gain_025=bool(scores['POR']>77.801936801885),checkpoint_sha256=hashlib.sha256((OUT/mode/'model_best.pt').read_bytes()).hexdigest(),report=report)
            atomic_json(STUDY/(mode+'_assembled.json'),result);np.save(STUDY/(mode+'_assembled_pred.npy'),pred)
            text=f"\nV128 {mode}: restored POR={scores['POR']:.12f}; delta vs v114={result['gain_vs_v114_POR']:+.12f}, vs historical={result['gain_vs_historical_POR']:+.12f}; actual fifth TOTAL={scores['TOTAL']:.12f};5 phases incl source; channel gain>0.25={result['strict_channel_gain_025']}; strict808={result['strict_808_passed']}; no otherfolds. best/stop={report['best_epoch']}/{report['stopped_epoch']}, seconds={report['seconds']:.3f}, peakMiB={report['peak_MiB']:.3f}.\n"
            for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
                with p.open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps({m:dict(live=v['live'],status=v['state'].get('status'),epoch=v['state'].get('epoch'),best=v['report'].get('target_score',v['state'].get('best'))) for m,v in snap.items()}))
if __name__=='__main__':main()
