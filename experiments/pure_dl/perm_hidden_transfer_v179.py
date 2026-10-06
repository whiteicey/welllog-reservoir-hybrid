"""Train-only own-fold SOURCE hidden transfer into the private PERM backbone."""
import torch
from torch import nn
from perm_well_sam_v177 import ROOT,SOURCE,PARENT,STAGES,Network as Parent,load_data,Evaluator,prepare_sampling,sample_anchors,channel_objective as predictive_loss
from paired_local_context_router_sequence import Network as Teacher,retained_indices,calibration_indices
STUDY=ROOT/'results/deep_learning/perm_hidden_transfer_v179_20261001'
OUT=ROOT/'outputs/pure_dl_perm_hidden_transfer_v179_fold5'
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.align_project=nn.Linear(128,256)
        with torch.random.fork_rng(devices=[]):
            numeric,condition=calibration_indices(master['selected_feature_names'])
            self.teacher=Teacher(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
            self.teacher.load_state_dict(master['state_dict'],strict=True)
        self.teacher.eval().requires_grad_(False);self.teacher_hidden={};self.last_alignment=None
        self.handles=[member.main_head.register_forward_pre_hook(self.capture(i)) for i,member in enumerate(self.teacher.members)]
    def capture(self,i):
        def hook(module,args):self.teacher_hidden[i]=args[0].detach()
        return hook
    def train(self,mode=True):super().train(mode);self.teacher.eval();return self
    def groups(self):return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=6e-4,name='private_PERM_complete_model')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(no_source_predictor_weights_loaded=False,source_weights_in_student=False,source_weights_in_teacher_only=True,
                    train_only_hidden_transfer=True,alignment_weight=.1,aux_training_parameters=33024,
                    student_predictor_parameters=685873,no_teacher_prediction_loss=True,no_teacher_inference=True,
                    original_teacher_pretraining_counted_in_shared_source_phase=True,no_trained_v177_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.teacher_hidden.clear();self.last_alignment=None
        hidden,_,_=self.body(x[...,:206],relative,valid)
        gate=self.gate(hidden).float();raw=self.density(hidden).reshape(*hidden.shape[:2],1,3,16).float()
        if self.training:
            with torch.no_grad():
                for member in self.teacher.members:member(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
            target=torch.stack([self.teacher_hidden[i].transpose(1,2).float() for i in range(2)],2)
            proposed=self.align_project(hidden).reshape(*hidden.shape[:2],2,128).float()
            self.last_alignment=((proposed-target).square().mean(-1)/target.square().mean(-1).clamp_min(1e-4)).mean(-1)
        self.teacher_hidden.clear();return gate,raw
def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    original=predictive_loss(output,truth,default,mask,quality,target_index,mode,model)
    assert target_index==1 and model.last_alignment is not None
    alignment=(quality[mask].detach().float()*model.last_alignment[mask]).mean()/3
    return original[0]+.1*alignment,original[1],original[2]+.3*alignment
def make_model(stage,master,metadata):return Network(master,stage)
