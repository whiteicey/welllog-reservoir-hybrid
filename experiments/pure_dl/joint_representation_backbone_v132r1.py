"""New joint backbone with train-window hidden-feature alignment; no prediction distillation."""
import torch
from torch import nn
from torch.nn import functional as F
from joint_hierarchical_backbone_v130 import ROOT,SOURCE,Network as Student,Evaluator,load_data,channel_objective as own_loss
from paired_local_context_router_sequence import Network as Teacher,retained_indices,calibration_indices
STUDY=ROOT/'results/deep_learning/joint_representation_backbone_v132r1_20261001'
OUT=ROOT/'outputs/pure_dl_joint_representation_backbone_v132r1_fold5'
STAGES={'aligned':('TOTAL',512,False,0.,6e-4)}
class Network(Student):
    def __init__(self,master):
        super().__init__()
        self.align=nn.ModuleList([nn.Linear(96,128),nn.Linear(96,128)])
        numeric,condition=calibration_indices(master['selected_feature_names'])
        self.teacher=Teacher(retain_indices=retained_indices(master['selected_feature_names']),numeric_indices=numeric,conditioning_indices=condition)
        self.teacher.load_state_dict(master['state_dict'],strict=True);self.teacher.requires_grad_(False).eval()
        self.student_features=None;self.teacher_features={}
        self.hook=self.norm.register_forward_hook(self.capture_student)
        self.teacher_hooks=[m.main_head.register_forward_pre_hook(self.capture_teacher(i)) for i,m in enumerate(self.teacher.members)]
    def capture_student(self,module,args,output):self.student_features=output
    def capture_teacher(self,i):
        def hook(module,args):self.teacher_features[i]=args[0].detach().transpose(1,2)
        return hook
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def groups(self):return [dict(params=[p for p in self.parameters() if p.requires_grad],lr=6e-4,name='new_student_and_alignment_heads')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.pop('random_all_network_weights');info.pop('no_old_network_weights_loaded')
        info.update(student_all_weights_random=True,no_old_student_prediction_weights_loaded=True,
            frozen_teacher_own_training_fold_only=True,teacher_only_used_during_training=True,
            no_teacher_predictions_in_loss_or_inference=True,no_pseudo_labels=True,alignment_weight=.1,
            alignment_targets='two normalized128-dim hidden feature vectors',additional_auxiliary_parameters=24832,teacher_window_length=256,student_window_length=512,
            inference_prediction_capacity_same_as_v130=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.teacher_features.clear();output=super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        if self.training:
            with torch.no_grad():
                lo=x.shape[1]//2-128;hi=lo+256
                for m in self.teacher.members:m(x[:,lo:hi],relative[:,lo:hi],valid[:,lo:hi],absolute_depth=absolute_depth[:,lo:hi] if absolute_depth is not None else None)
        else:self.student_features=None
        return output
    def alignment_loss(self,mask,quality):
        assert self.training and len(self.teacher_features)==2 and self.student_features is not None
        lo=mask.shape[1]//2-128;hi=lo+256;use=mask[:,lo:hi]
        q=quality[:,lo:hi][use].detach().float();terms=[]
        for i,p in enumerate(self.align):
            student=F.layer_norm(p(self.student_features[:,lo:hi]).float(),(128,))
            teacher=F.layer_norm(self.teacher_features[i].float(),(128,)).detach()
            terms.append((q*(student[use]-teacher[use]).square().mean(-1)).mean())
        return .5*sum(terms)
def make_model(stage,master,metadata):return Network(master)
def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    own=own_loss(output,truth,default,mask,quality,target_index,mode)
    return own[0]+.1*model.alignment_loss(mask,quality),own[1],own[2]
