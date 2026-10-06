"""One migration check: the generic fold decoder reproduces the frozen fifth.

No parameter update or new model selection. Other-fold weights are never used.
"""
from pathlib import Path
import json
import numpy as np
import torch
from fullcv_context_v97 import ROOT, STUDY, add_well_bank
from fullcv_models_v97 import Chosen
from train_stage_fullcv_v97 import Evaluator
from champion_scaling_v21 import load_data


def main():
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    data,master=load_data()
    data['population_rows']=dict(train_wells=np.unique(data['wells'][data['fold']!=4]).tolist(),heldout_wells=np.unique(data['wells'][data['fold']==4]).tolist())
    metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')};metadata.update(add_well_bank(data))
    with np.load(ROOT/'results/deep_learning/feature_structure_v20_20260928/side_views.npz') as z:side=torch.as_tensor(z['amplitude'],device='cuda')
    full=data['x'];amplitude=torch.cat((full,side),-1);results=[]
    for c,target in enumerate(('POR','PERM','SW')):
        if c==0:
            from por_reparam_v31 import Network
            model=Network(target,master,modes=('anchored',));source=ROOT/'outputs/pure_dl_mechanism_v31_POR_fold5/anchored'
            cp=torch.load(source/'model_best.pt',map_location='cpu',weights_only=False);model.heads['anchored'].load_state_dict(cp['delta_state'],strict=True)
            wrapped=Chosen(model,c,mode='anchored');data['x']=amplitude;length=256
        elif c==1:
            from champion_split_route_v29 import Network
            model=Network(target,master,modes=('coupled',));source=ROOT/'outputs/pure_dl_split_route_v29_PERM_fold5/coupled'
            cp=torch.load(source/'model_best.pt',map_location='cpu',weights_only=False);model.heads['coupled'].load_state_dict(cp['delta_state'],strict=True)
            wrapped=Chosen(model,c,mode='coupled');data['x']=amplitude;length=256
        else:
            from sw_factor_sam_v96 import Network
            source=ROOT/'outputs/pure_dl_sw_factor_sam_v96_SW_likelihood_fold5';model=Network('likelihood',metadata)
            cp=torch.load(source/'model_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(cp['state_dict'],strict=True)
            wrapped=Chosen(model,c,bank=True);data['x']=full;length=1024
        wrapped.cuda();prediction,score=Evaluator(data,master,c,length)(wrapped)
        expected=np.load(source/'channel_pred.npy');assert np.array_equal(prediction,expected),(target,float(np.max(np.abs(prediction-expected))))
        results.append(dict(target=target,score=score,rows=len(prediction),frozen_fifth_predictions_exact=True))
        if hasattr(model,'close'):model.close()
        del model,wrapped,cp;torch.cuda.empty_cache()
    (STUDY/'decoder_migration_proof.json').write_text(json.dumps(dict(original_fold=5,no_training=True,no_new_selection=True,results=results),indent=2),encoding='utf-8')
    print('GENERIC_DECODER_EXACT_FIFTH',json.dumps(results))


if __name__=='__main__':main()
