import argparse,json,hashlib
import numpy as np
import torch
from por_late_vrex_v265 import ROOT,STUDY,OUT,STAGES
from train_dual8222_heads80 import atomic_json,atomic_array
import train_invariant as base
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=list(STAGES),required=True);args=parser.parse_args()
    out=OUT/args.stage
    if not (out/'report.json').exists():
        print(json.dumps(read(out/'runtime_status.json') if (out/'runtime_status.json').exists() else dict(status='prepared')));return
    resultfile=STUDY/f'{args.stage}_assembled.json'
    if resultfile.exists():print(json.dumps(read(resultfile)['scores']));return
    report=read(out/'report.json');assert report['predictions_reproduced_exactly'] and report['frozen_parameters_exact']
    data=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    por=ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps';perm=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional'
    pred=np.column_stack((np.load(out/'channel_pred.npy'),np.load(perm/'channel_pred.npy'),np.load('outputs/pure_dl_compact_frequency_v104_fold5/local/channel_pred.npy')))
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    scores=base.metric(data['truth'][data['validation']],pred);assert abs(scores['POR']-report['target_score'])<1e-9
    result=dict(scores=scores,screening_fold=5,validation_rows=144665,complete_five_folds=False,remaining_folds_started=False,
                total_training_stages=6,added_head_stages=dict(POR=1,PERM=0,SW=0),source_stage_counted=True,
                gain_vs151_POR=scores['POR']-77.92638076072386,strict_808_passed=scores['TOTAL']>80.8,
                learned_gain_vs_initial=report['against_initial'],checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest(),report=report)
    atomic_json(resultfile,result);atomic_array(STUDY/f'{args.stage}_assembled_pred.npy',pred)
    note=f"\nV265 {args.stage} finalPOR{scores['POR']:.12f}, actualTOTAL{scores['TOTAL']:.12f}; gainvs151{result['gain_vs151_POR']:+.12f}. Sixphases alone, onePORaddition/no otherfolds; restoredRAM replay, not independent diskreload.\n"
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(result['scores']))
if __name__=='__main__':main()
