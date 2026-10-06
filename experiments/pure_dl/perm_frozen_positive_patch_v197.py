"""Freeze verified154 learned readout; train only one small additive core patch."""
import hashlib
import torch
from perm_ranked_probability_v154 import ROOT,SOURCE,Network as Positive,load_data,Evaluator,channel_objective
from perm_local_core_patch_v196 import AugmentedHead,CURVES,Network as Additive
STUDY=ROOT/'results/deep_learning/perm_frozen_positive_patch_v197_20261001'
OUT=ROOT/'outputs/pure_dl_perm_frozen_positive_patch_v197_fold5'
PARENT=ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt'
STAGES={k:('PERM',256,False,0.,1e-4) for k in ('mean','weighted')}
PARENT_SHA='2b355bb8c3736990b05e2726373a4343c574a93ab6ab1a3eda0ef796ac70cf11'

class Network(Positive):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.study_mode=mode
        assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
        saved=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True)
        self.requires_grad_(False)
        names=list(master['selected_feature_names']);indices=[names.index(k) for k in CURVES]
        assert len(names)==206 and len(set(indices))==9
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.context_head=AugmentedHead(self.context_head,indices,mode)
    forward=Additive.forward
    def groups(self):
        return [dict(params=list(self.context_head.patch.parameters()),lr=1e-4,name='PERM_only_additive_core_patch')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(parent_checkpoint=str(PARENT),parent_sha256=PARENT_SHA,SOURCE_checkpoint=str(SOURCE),
                    trained154_head_loaded_and_frozen=True,only_new_core_patch_trainable=True,
                    retained_full_154_function=True,old_predictor_fully_frozen=True,
                    additive_core=self.study_mode,curve_names=list(CURVES),curve_indices=self.context_head.patch.indices.tolist(),
                    patch_length=16,series_width=16,core_width=8,residual_scale=.1,
                    extra_active_parameters=sum(p.numel() for p in self.context_head.patch.parameters()),
                    same_initial_predictive_function=True,zero_projection=True,
                    original_NLL_BCE_jointRPS_retained=True,executed_training_stages_including_source=6,
                    total_stage_upper_bound=8,no_stochastic_pooling=True)
        return info

def make_model(stage,master,metadata):return Network(master,stage)
