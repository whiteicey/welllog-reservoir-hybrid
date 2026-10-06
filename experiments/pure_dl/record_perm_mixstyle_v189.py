import json,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from perm_mixstyle_v189 import ROOT,STUDY,OUT
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    out=OUT/'perm_conditional';state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
    try:live=any(Path(a).name=='train_perm_mixstyle_v189.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
    except (psutil.Error,ValueError):pass
    atomic_json(STUDY/'progress.json',dict(live=live,state=state,report=report))
    if report and not (STUDY/'perm_conditional_assembled.json').exists():
        assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        por=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps';sw=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        pred=np.column_stack((np.load(por/'channel_pred.npy'),np.load(out/'channel_pred.npy'),np.load(sw/'channel_pred.npy')))
        assert pred.shape==(144665,3) and np.isfinite(pred).all()
        scores=base.metric(data['truth'][data['validation']],pred)
        expected=[read(por/'report.json')['target_score'],report['target_score'],read(sw/'report.json')['target_score']]
        assert all(abs(scores[k]-v)<1e-9 for k,v in zip(base.TS,expected))
        result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,total_training_stages=5,source_stage_counted=True,
            gain_vs_Gaussian_RPS_PERM=scores['PERM']-89.40556485691894,gain_vs_conditional_PERM=scores['PERM']-89.38793609578403,gain_vs_compact_PERM=scores['PERM']-89.37839421114361,gain_vs_source_PERM=scores['PERM']-89.3447601747529,gain_vs_historical_PERM=scores['PERM']-89.39672404936582,
            checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),report=report)
        atomic_json(STUDY/'perm_conditional_assembled.json',result);np.save(STUDY/'perm_conditional_assembled_pred.npy',pred)
        text=f"\nV189 final PERM={scores['PERM']:.12f}, compactcontrolgain={result['gain_vs_compact_PERM']:+.12f}; actualTOTAL={scores['TOTAL']:.12f}, fivephases inclsource;strict808={result['strict_808_passed']}, no otherfolds. best/stop={report['best_epoch']}/{report['stopped_epoch']}, seconds={report['seconds']:.3f}, peakMiB={report['peak_MiB']:.3f}; all144665replay/frozen exact.\n"
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps(dict(live=live,status=state.get('status'),epoch=state.get('epoch'),best=report.get('target_score',state.get('best')))))
if __name__=='__main__':main()
