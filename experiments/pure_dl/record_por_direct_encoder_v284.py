import json,hashlib,argparse
from pathlib import Path
import numpy as np
import torch,psutil
from por_direct_encoder_v284 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json
from fold5_screening_808 import qualifies
import train_invariant as base
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=list(STAGES),required=True);stage=parser.parse_args().stage
    out=OUT/stage;state=read(out/'runtime_status.json');report=read(out/'report.json');live=False
    try:live=any(Path(a).name=='train_por_direct_encoder_v284.py' for a in psutil.Process(state.get('pid',-1)).cmdline())
    except (psutil.Error,ValueError):pass
    atomic_json(STUDY/f'{stage}_progress.json',dict(live=live,state=state,report=report))
    if report and not (STUDY/f'{stage}_assembled.json').exists():
        assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
        data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
        sw=ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local'
        perm=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional'
        pred=np.column_stack((np.load(out/'channel_pred.npy'),np.load(perm/'channel_pred.npy'),np.load(sw/'channel_pred.npy')))
        assert pred.shape==(144665,3) and np.isfinite(pred).all()
        scores=base.metric(data['truth'][data['validation']],pred)
        expected=[report['target_score'],read(perm/'report.json')['target_score'],read(sw/'report.json')['target_score']]
        assert all(abs(scores[k]-v)<1e-9 for k,v in zip(base.TS,expected))
        result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,
                    strict_808_passed=qualifies(scores['TOTAL']),remaining_folds_started=False,
                    total_training_stages=6,stage_upper_bound=8,added_head_stages=dict(POR=1,PERM=0,SW=0),
                    source_stage_counted=True,gain_vs_original_POR=scores['POR']-77.92638076072386,
                    gain_vs_original_TOTAL=scores['TOTAL']-80.61247541799393,
                    checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),report=report)
        atomic_json(STUDY/f'{stage}_assembled.json',result);np.save(STUDY/f'{stage}_assembled_pred.npy',pred)
        t=f"\nV284 final POR{scores['POR']:.12f}, gain{result['gain_vs_original_POR']:+.12f}, actualTOTAL{scores['TOTAL']:.12f}; sixphases inclsource/one addedPORstage, strict808={result['strict_808_passed']}; all144665replay/frozenexact, best/stop{report['best_epoch']}/{report['stopped_epoch']}, trainingseconds{report['seconds']:.6f}/peakMiB{report['peak_MiB']:.6f}. No otherfolds.\n"
        for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
            with p.open('a',encoding='utf-8') as f:f.write(t)
    print(json.dumps(dict(live=live,status=state.get('status'),epoch=state.get('epoch'),best=report.get('target_score',state.get('best')))))

if __name__=='__main__':main()



