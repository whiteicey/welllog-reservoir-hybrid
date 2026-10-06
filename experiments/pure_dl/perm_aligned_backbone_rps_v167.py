"""Train-only own-fold hidden anchor on private PERM backbone coadaptation."""
import copy
import torch
from perm_private_backbone_rps_v166 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective as predictive_loss
STUDY=ROOT/'results/deep_learning/perm_aligned_backbone_rps_v167_20261001'
OUT=ROOT/'outputs/pure_dl_perm_aligned_backbone_rps_v167_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Network(Previous):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        self.teacher=copy.deepcopy(self.backbone).eval().requires_grad_(False)
        self.student_hidden={};self.teacher_hidden={};self.last_alignment=None
        # Deepcopy preserves function closures of the student's capture hooks.
        # Remove them from the teacher before adding teacher-only captures.
        for member in self.teacher.members:member.main_head._forward_pre_hooks.clear()
        self.teacher_handles=[m.main_head.register_forward_pre_hook(self.capture_teacher(i))
                              for i,m in enumerate(self.teacher.members)]
    def capture_new(self,e):
        def hook(module,args):
            self.local_captured[e]=args[0];self.student_hidden[e]=args[0]
        return hook
    def capture_teacher(self,e):
        def hook(module,args):self.teacher_hidden[e]=args[0].detach()
        return hook
    def train(self,mode=True):
        super().train(mode);self.teacher.eval();return self
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(train_only_hidden_alignment=True,alignment_weight=.1,
                    original_own_fold_source_teacher=True,no_teacher_prediction_loss=True,
                    no_teacher_inference=True,same_active_parameter_capacity_as_v166=True,
                    no_new_pretraining_phase=True,teacher_training_windows_only=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.student_hidden.clear();self.teacher_hidden.clear();self.last_alignment=None
        output=super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        if self.training:
            with torch.no_grad():
                for member in self.teacher.members:
                    member(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
            losses=[]
            for e in range(2):
                student=self.student_hidden[e].float();teacher=self.teacher_hidden[e].float()
                reference=teacher.square().mean(1).clamp_min(1e-4)
                losses.append((student-teacher).square().mean(1)/reference)
            self.last_alignment=torch.stack(losses).mean(0)
        self.student_hidden.clear();self.teacher_hidden.clear()
        return output

def channel_objective(output,truth,default,mask,quality,target_index,mode,model):
    original=predictive_loss(output,truth,default,mask,quality,target_index,mode,model)
    assert target_index==1 and model.last_alignment is not None
    alignment=(quality[mask].detach().float()*model.last_alignment[mask]).mean()/3
    return original[0]+.1*alignment,original[1],original[2]+.3*alignment

def make_model(stage,master,metadata):return Network(master,stage)
