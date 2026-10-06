"""Read-only original SOURCE replay and actual fixed-head stage ablations."""
import json,time,hashlib
import numpy as np
import torch
from torch import nn
from por_stage_ablation_v201 import ROOT,STUDY,SOURCE,load_data
from compact_five_stage_v100 import Original,retained_indices,calibration_indices
from train_stage_fullcv_v97 import Evaluator
from train_dual8222_heads80 import atomic_json,atomic_array
from gpu_job_guard_v2 import check_slot
import train_invariant as base

class SourcePOR(nn.Module):
    def __init__(self,master):
        super().__init__()
        numeric,condition=calibration_indices(master['selected_feature_names'])
        self.body=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
        self.body.load_state_dict(master['state_dict'],strict=True)
        self.requires_grad_(False)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        gate,raw=self.body(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
        return gate[:,:,:1],raw[:,:,:1]

def main():
    torch.set_num_threads(3)
    print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    started=time.perf_counter()
    data,master,metadata=load_data('skip_middle')
    pred,score=Evaluator(data,master,0,256)(SourcePOR(master).cuda())
    atomic_array(STUDY/'source_por_pred.npy',pred)
    perm=np.load(ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/channel_pred.npy')
    sw=np.load(ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/channel_pred.npy')
    results=[]
    for name,por,stages in [('source_only',pred,1),('source_113',np.load(ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/channel_pred.npy'),2),('source_113_151',np.load(ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/channel_pred.npy'),3)]:
        combined=np.column_stack((por,perm,sw))
        assert combined.shape==(144665,3) and np.isfinite(combined).all()
        scores=base.metric(data['truth'][data['validation']],combined)
        results.append(dict(name=name,POR_training_stages=stages,total_training_stages=stages+2,scores=scores,delta_vs_full_POR=scores['POR']-77.92638076072386,delta_vs_full_TOTAL=scores['TOTAL']-80.61247541799393))
    result=dict(screening_fold=5,validation_rows=144665,complete_five_folds=False,original_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_POR=score,results=results,seconds_including_loading=time.perf_counter()-started,read_only_no_training=True,source_family='original_all_T3',later_family='8_T3_8_Cauchy',controls_reused_verified_predictions=True)
    atomic_json(STUDY/'prefix_ablation.json',result)
    note='\nRead-only prefix ablation, fixed PERM154/SW104: '+json.dumps(results)+'\nOriginal SOURCE uses original T3 family; later stages use mixed T3/Cauchy. These are actual prefix recipes, not a pure family-matched mechanism ablation. Direct skip-middle training is separate.\n'
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
