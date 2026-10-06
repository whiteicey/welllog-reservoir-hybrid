"""Same-capacity distribution-feature versus repeated-hidden view, frozen104 prior."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_probability_mass_patch_v209 import ROOT,SOURCE,PARENT,PARENT_SHA,Network as Parent,Evaluator,load_data,probability
from sw_mass_likelihood_only_v211 import DecisionObjective
STUDY=ROOT/'results/deep_learning/sw_mass_posterior_features_v213_20261001'
OUT=ROOT/'outputs/pure_dl_sw_mass_posterior_features_v213_fold5'
STAGES={mode:('SW',1024,True,0.,1e-4) for mode in ('repeat','posterior')}

class ViewPatch(nn.Module):
    def __init__(self):
        super().__init__();self.side_norm=nn.LayerNorm(256)
        self.input=nn.Linear(384,32);self.output=nn.Linear(32,256)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,h,side):
        x=torch.cat((h.float(),self.side_norm(side.float())),dim=-1)
        return self.output(F.gelu(self.input(x))).float()

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261010);self.patch=ViewPatch()
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(own_bin_cross_entropy_BCE_joint_utility1=False,actual_utility_coefficient=0.,
                    view=self.study_mode,view_dimension=256,hidden_dimension=128,
                    conditional_distribution_logmass_features=self.study_mode=='posterior',
                    logprior_feature_floor=-20,side_normalization='same learnable LayerNorm256 both modes',
                    same_21280_trainable_parameters=True,extra_active_parameters=21280,
                    matched_parameter_count_not_equal_effective_function_rank_claim=True,
                    no209_or211_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.feature=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8);parts=[]
            for chunk in flat.split(4096):
                parts.append(probability(2,chunk[:,0].softmax(-1),chunk[:,1],F.softplus(chunk[:,2])+.025,self.edges))
            prior=torch.cat(parts).reshape(*x.shape[:2],256)
            logprior=prior.clamp_min(1e-12).log()
            side=logprior.clamp_min(-20) if self.study_mode=='posterior' else self.feature.repeat(1,1,2)
        correction=.5*torch.tanh(self.patch(self.feature,side)/.5);self.feature=None
        return gate,(logprior+correction).log_softmax(-1).unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(*args):raise RuntimeError('Own SW categorical likelihood+BCE, utility coefficient0')
