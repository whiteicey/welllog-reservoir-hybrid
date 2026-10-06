"""Frozen input features from already-shared SOURCE; no source predictions or labels used as targets."""
import torch
from sw_probability_mass_patch_v209 import ROOT,SOURCE,PARENT,PARENT_SHA,Network as Parent,Evaluator,load_data,probability
from sw_mass_likelihood_only_v211 import DecisionObjective
from sw_mass_posterior_features_v213 import ViewPatch
from compact_five_stage_v100 import Original,retained_indices,calibration_indices
from torch.nn import functional as F
STUDY=ROOT/'results/deep_learning/sw_shared_feature_mass_v216_20261001'
OUT=ROOT/'outputs/pure_dl_sw_shared_feature_mass_v216_fold5'
STAGES={mode:('SW',1024,True,0.,1e-4) for mode in ('repeat','source')}

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata);self.study_mode=mode
        numeric,condition=calibration_indices(master['selected_feature_names'])
        self.source=Original(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
        self.source.load_state_dict(master['state_dict'],strict=True);self.source.requires_grad_(False).eval()
        self.source_features={}
        self.source_handles=[member.main_head.register_forward_pre_hook(self.capture_source_feature(i)) for i,member in enumerate(self.source.members)]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=ViewPatch()
    def capture_source_feature(self,i):
        def hook(module,args):self.source_features[i]=args[0].detach().transpose(1,2)
        return hook
    def train(self,mode=True):
        super().train(mode);self.source.eval();return self
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(source_checkpoint=str(SOURCE),source_weights_same_as_shared_original_recipe=True,
                    additional_SOURCE_training_stages=0,source_stage_counted_once=True,
                    SOURCE_all_old_weights_frozen=True,source_predictions_not_used=True,
                    source_hidden_features_only=True,view=self.study_mode,
                    source_branch_computed_in_both_arms_for_matched_cost=True,
                    own_bin_cross_entropy_BCE_joint_utility1=False,actual_utility_coefficient=0.,
                    extra_active_parameters=21280,only_new_mass_patch_trainable=True,
                    no209_or211_or213_trained_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None;self.source_features.clear()
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            # Discard all source outputs: only input-hidden features enter the patch.
            start=x.shape[1]//2-128;stop=start+256
            assert x.shape[1]>=256
            depth=None if absolute_depth is None else absolute_depth[:,start:stop]
            for member in self.source.members:
                member(x[:,start:stop,:206].contiguous(),relative[:,start:stop],valid[:,start:stop],absolute_depth=depth)
            central=torch.cat((self.source_features[0],self.source_features[1]),dim=-1)
            assert central.shape==(len(x),256,256)
            extra=central.new_zeros(len(x),x.shape[1],256)
            extra[:,start:stop]=central
            self.source_features.clear()
            flat=raw[:,:,0].float().reshape(-1,3,8);parts=[]
            for chunk in flat.split(4096):
                parts.append(probability(2,chunk[:,0].softmax(-1),chunk[:,1],F.softplus(chunk[:,2])+.025,self.edges))
            prior=torch.cat(parts).reshape(*x.shape[:2],256)
            side=extra if self.study_mode=='source' else self.feature.repeat(1,1,2)
        correction=.5*torch.tanh(self.patch(self.feature,side)/.5);self.feature=None
        return gate,(prior.clamp_min(1e-12).log()+correction).log_softmax(-1).unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(*args):raise RuntimeError('Own SW bin likelihood+BCE only; no source prediction supervision')

