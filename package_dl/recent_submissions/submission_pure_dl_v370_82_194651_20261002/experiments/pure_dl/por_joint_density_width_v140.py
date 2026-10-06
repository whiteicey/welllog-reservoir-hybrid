"""POR route: new joint-view depth-local location head on frozen positive v113."""
import torch
from torch import nn
from torch.nn import functional as F
from compact_por_mixed_family_v113 import ROOT,SOURCE,Network as Parent,Evaluator as ParentEvaluator,channel_objective as parent_loss,load_data
STUDY=ROOT/'results/deep_learning/por_joint_density_width_v140_20261001'
OUT=ROOT/'outputs/pure_dl_por_joint_density_width_v140_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt'
STAGES={'wide256':('POR',256,False,0.,3e-4)}
class ContextHead(nn.Module):
    def __init__(self):
        super().__init__();self.stem=nn.Conv1d(319,256,1);self.norm=nn.LayerNorm(256)
        self.depth=nn.Conv1d(256,256,33,padding=16,groups=256)
        self.expand=nn.Conv1d(256,512,1);self.output=nn.Conv1d(256,32,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,x,valid):
        h=F.gelu(self.stem(x)).masked_fill(~valid[:,None],0)
        z=self.norm(h.transpose(1,2)).transpose(1,2).masked_fill(~valid[:,None],0)
        h=(h+F.gelu(F.conv1d(z,self.depth.weight.sum(-1,keepdim=True),self.depth.bias,groups=256))).masked_fill(~valid[:,None],0)
        a,b=self.expand(h).chunk(2,1)
        return self.output(F.gelu(a)*b.sigmoid()).transpose(1,2).float().masked_fill(~valid[...,None],0)
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,'t3_cauchy');self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict'])
        self.requires_grad_(False);self.context_head=ContextHead();self.local_captured={}
        self.new_handles=[m.main_head.register_forward_pre_hook(self.capture_new(i)) for i,m in enumerate(self.backbone.members)]
    def capture_new(self,i):
        def hook(module,args):self.local_captured[i]=args[0].detach()
        return hook
    def groups(self):return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='wide256_joint_location_weight_POR_head')]
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),same_initial_predictive_function=True,old_predictor_fully_frozen=True,
            own_new_locations_only=False,new_component_logweight_correction=True,logweight_bound=1.,branch_bound_reference_sd=1.,bound_only_new_branch=True,
            input_hidden=256,input_side=63,depth_kernel=1,stored_kernel=33,effective_point_kernel_is_sum=True,width=256,family_mode='t3_cauchy',
            no_finite_whole_mixture_mean_variance=True,executed_training_stages_including_source=5)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.local_captured.clear()
        with torch.no_grad():gate,raw=super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        features=torch.cat((self.local_captured[0],self.local_captured[1],x[...,206:].transpose(1,2)),1)
        self.local_captured.clear();unbounded=self.context_head(features,valid)
        reference=(F.softplus(raw[:,:,0,2].detach())+.025)
        delta=reference*torch.tanh(unbounded[...,:16]/reference)
        weight_delta=torch.tanh(unbounded[...,16:])
        own=raw[:,:,0];corrected=torch.stack((own[:,:,0]+weight_delta,own[:,:,1]+delta,own[:,:,2]),2).unsqueeze(2)
        return gate,corrected
def make_model(stage,master,metadata):return Network(master,stage)
def channel_objective(output,truth,default,mask,quality,target_index,mode):return parent_loss(output,truth,default,mask,quality,target_index,'t3_cauchy')
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length,'t3_cauchy')
