"""CPU decomposition of observed phase and fusion gains; no new model fitting."""
from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
from train import metric,dmask
OUT=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5'
values=[]
def decompose(truth,before,after):
    n=len(truth);dm=dmask(truth);channels={};weights=(.3,.35,.35)
    s0=metric(truth,before);s1=metric(truth,after)
    for c,target in enumerate(('POR','PERM','SW')):
        part={}
        for label,mask in [('DEFAULT',dm[:,c]),('CONT',~dm[:,c])]:
            a=metric(truth[mask],before[mask])[target];b=metric(truth[mask],after[mask])[target]
            part[label]=dict(rows=int(mask.sum()),before=a,after=b,delta=b-a,
                             contribution_to_TOTAL=(b-a)*mask.sum()/n*weights[c])
        whole_delta=s1[target]-s0[target]
        assert abs(sum(v['contribution_to_TOTAL'] for v in part.values())-whole_delta*weights[c])<1e-9
        channels[target]=dict(before=s0[target],after=s1[target],delta=whole_delta,
                              contribution_to_TOTAL=whole_delta*weights[c],strata=part)
    total_delta=s1['TOTAL']-s0['TOTAL']
    assert abs(sum(v['contribution_to_TOTAL'] for v in channels.values())-total_delta)<1e-9
    return dict(before=s0,after=s1,TOTAL_delta=total_delta,channels=channels,
                interpretation='Observed complete predictions on fixed fifth, not independent generalization or a matched-capacity causal ablation.')

for path in sorted(OUT.glob('seed_*/COMPLETE_MODEL.json')):
    own=path.parent;dest=own/'PHASE_GAIN_SOURCES.json'
    if dest.exists():values.append(json.loads(dest.read_text()));continue
    data=torch.load(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=np.asarray(data['truth'][data['validation']]);del data
    prefix=np.column_stack([np.load(own/s/'channel_pred.npy') for s in ('POR151','PERM154','SW104')])
    after=np.load(own/'complete_model_pred.npy')
    result=dict(seed=json.loads(path.read_text())['seed'],added_three_phases=decompose(truth,prefix,after),
                complete_five_folds=False,remaining_folds_started=False)
    dest.write_text(json.dumps(result,indent=2),encoding='utf-8');values.append(result)

fusion_path=ROOT/'results/deep_learning/original_seed_weight_grid_v364_20261002/best_common_weight_pred.npy'
fusion_dest=STUDY/'WEIGHT81_19_GAIN_SOURCES.json'
if fusion_path.exists() and not fusion_dest.exists():
    own=OUT/'seed_20261010';data=torch.load(own/'SOURCE/prepared_data.pt',map_location='cpu',weights_only=False)
    truth=np.asarray(data['truth'][data['validation']]);del data
    before=np.load(ROOT/'results/deep_learning/sw_dynamic_cross_spectrum_v351_20261002/eight_stage_pred.npy')
    result=decompose(truth,before,np.load(fusion_path))
    result.update(original_weight=.81,new_seed_weight=.19,same_fifth_validation_tuned=True,
                  no_new_training=True,complete_five_folds=False,remaining_folds_started=False)
    fusion_dest.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(dict(phase_gains=values,fusion= json.loads(fusion_dest.read_text()) if fusion_dest.exists() else {}),indent=2))
