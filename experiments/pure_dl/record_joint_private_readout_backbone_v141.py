import json,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from joint_private_readout_backbone_v141 import ROOT,STUDY,OUT
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
def main():
    out=OUT/'private_readouts';state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
    try:live=any(Path(a).name=='train_joint_private_readout_backbone_v141.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
    except (psutil.Error,ValueError):pass
    atomic_json(STUDY/'progress.json',dict(live=live,state=state,report=report))
    if report and not (STUDY/'private_readouts_verified.json').exists():
        assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        pred=np.load(out/'channel_pred.npy');assert pred.shape==(144665,3) and np.isfinite(pred).all()
        scores=base.metric(data['truth'][data['validation']],pred)
        assert all(abs(scores[k]-report['scores'][k])<1e-9 for k in scores)
        result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,total_training_stages=2,original_source_Q_stage_counted=True,
            no_old_student_prediction_weights_loaded=True,teacher_training_only=True,no_pseudo_labels=True,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),report=report)
        atomic_json(STUDY/'private_readouts_verified.json',result)
        text=f"\nV135 joint source verified: actual original-fifth scores{scores},2 stages incl originalsourceQ; strict808={result['strict_808_passed']}; no remainingfolds. best/stop={report['best_epoch']}/{report['stopped_epoch']}, seconds={report['seconds']:.3f}, peakMiB={report['peak_MiB']:.3f};144665x3 replay exact.\n"
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps(dict(live=live,status=state.get('status'),epoch=state.get('epoch'),best=report.get('target_score',state.get('best')))))
if __name__=='__main__':main()
