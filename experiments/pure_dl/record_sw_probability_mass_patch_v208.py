import json,hashlib
from pathlib import Path
import numpy as np
import torch,psutil
from sw_probability_mass_patch_v208 import ROOT,STUDY,OUT
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def main():
    out=OUT/'mass';state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
    try:live=any(Path(a).name=='train_sw_probability_mass_patch_v208.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
    except (psutil.Error,ValueError):pass
    atomic_json(STUDY/'mass_progress.json',dict(live=live,state=state,report=report))
    if report and not (STUDY/'mass_assembled.json').exists():
        assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        por=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps'
        perm=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional'
        pred=np.column_stack((np.load(por/'channel_pred.npy'),np.load(perm/'channel_pred.npy'),np.load(out/'channel_pred.npy')))
        assert pred.shape==(144665,3) and np.isfinite(pred).all()
        scores=base.metric(data['truth'][data['validation']],pred)
        expected=[read(por/'report.json')['target_score'],read(perm/'report.json')['target_score'],report['target_score']]
        assert all(abs(scores[k]-v)<1e-9 for k,v in zip(base.TS,expected))
        result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,
                    strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,
                    total_training_stages=6,stage_upper_bound=8,added_head_stages=dict(POR=0,PERM=0,SW=1),
                    source_stage_counted=True,gain_vs_original_SW=scores['SW']-74.12175282815758,
                    gain_vs_original_TOTAL=scores['TOTAL']-80.61247541799393,
                    checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),report=report)
        atomic_json(STUDY/'mass_assembled.json',result);np.save(STUDY/'mass_assembled_pred.npy',pred)
        t=f"\nV208 final SW{scores['SW']:.12f}, gain{result['gain_vs_original_SW']:+.12f}, actualTOTAL{scores['TOTAL']:.12f}; sixphases inclsource/one addedSWstage, strict808={result['strict_808_passed']}; all144665replay/frozenexact, best/stop{report['best_epoch']}/{report['stopped_epoch']}, trainingseconds{report['seconds']:.6f}/peakMiB{report['peak_MiB']:.6f}. No otherfolds.\n"
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(t)
    print(json.dumps(dict(live=live,status=state.get('status'),epoch=state.get('epoch'),best=report.get('target_score',state.get('best')))))

if __name__=='__main__':main()

