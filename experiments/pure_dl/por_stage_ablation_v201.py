"""Skip the trained113 stage; same final151 head, only original SOURCE loaded."""
import torch
from torch.nn import functional as F
from compact_por_mixed_family_v113 import ROOT,SOURCE,Network as MixedSource,load_data
from por_joint_density_width_v140 import ContextHead,Evaluator
from por_ranked_probability_v151 import channel_objective
from src.utils import DEFAULTS
import math
STUDY=ROOT/'results/deep_learning/por_stage_ablation_v201_20261001'
OUT=ROOT/'outputs/pure_dl_por_stage_ablation_v201_fold5'
PARENT=SOURCE
STAGES={'skip_middle':('POR',256,False,0.,3e-4)}

class Network(MixedSource):
    def __init__(self,master,mode):
        super().__init__(master,'t3_cauchy');self.requires_grad_(False)
        self.context_head=ContextHead();self.local_captured={}
        self.new_handles=[m.main_head.register_forward_pre_hook(self.capture_new(i)) for i,m in enumerate(self.backbone.members)]
        self.register_buffer('rps_edges',torch.as_tensor(master['density_edges'][0][1:-1],dtype=torch.float32))
        self.register_buffer('default_z',torch.tensor((math.log1p(DEFAULTS['POR'])-float(master['regression_median'][0]))/float(master['regression_scale'][0]),dtype=torch.float32))
    def capture_new(self,i):
        def hook(module,args):self.local_captured[i]=args[0].detach()
        return hook
    def groups(self):return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='POR_direct_final_head_no_middle_training')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(SOURCE),SOURCE_only_weights_loaded=True,no113_or151_weights_loaded=True,
                    POR_training_stages=2,total_recipe_training_stages=4,sharedSOURCEcountedonce=True,
                    final_head_capacity=230944,same_final151_architecture=True,
                    family_mode='t3_cauchy',fixed_IQR_family_interpretation=True,
                    same_initial_as113_mixedSOURCE=True,old_predictor_fully_frozen=True,
                    own_new_locations_and_logweights=True,location_bound_reference_sd=1.,logweight_bound=1.,
                    NLL_BCE_jointRPS1=True,head_width=256,source206_amp63_hidden256=True,
                    no_finite_whole_mixture_mean_variance=True,maximum_extra_stages_per_head=1)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        with torch.no_grad():
            gate,raw=MixedSource.forward(self,x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();u=self.context_head(features,valid)
        own=raw[:,:,0];reference=F.softplus(own[:,:,2].detach())+.025
        return gate,torch.stack((own[:,:,0]+u[...,16:].tanh(),own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2).unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
