"""Matched warm-start update scopes for compact PERM, five total recipe stages."""
import hashlib
import torch
from torch import nn
from compact_five_stage_v100 import ROOT,SOURCE,CompactReadout,PackedHead,load_data as original_data

STUDY=ROOT/'results/deep_learning/compact_perm_mean_v111_20261001'
OUT=ROOT/'outputs/pure_dl_compact_perm_mean_v111_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_v100_fold5/perm/model_best.pt'
STAGES={name:('PERM',256,False,0.,3e-5) for name in ('gate_route','mean_coadapt')}

class MeanHead(PackedHead):
    def packed(self,stem):
        if stem.startswith('density'):
            fixed=getattr(self,stem);mean=self.mean_weight if stem.endswith('weight') else self.mean_bias
            own=torch.cat((fixed[:8],mean,fixed[16:]),0)
            return torch.cat((getattr(self,stem+'_before'),own,getattr(self,stem+'_after')),0)
        return super().packed(stem)

class Network(CompactReadout):
    def __init__(self,master,mode):
        super().__init__(master,1);self.update_mode=mode
        parent=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(parent['state_dict'],strict=True)
        if mode=='mean_coadapt':
            for head in self.heads:
                head.__class__=MeanHead
                head.mean_weight=nn.Parameter(head.density_weight[8:16].detach().clone())
                head.mean_bias=nn.Parameter(head.density_bias[8:16].detach().clone())
        self.parent_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            parent_best_epoch=parent['best_epoch'],parent_target_score=parent['target_score'],update_scope=mode,
            source_pretraining_counted=True,total_dependency_training_stages=5,optimizer_reset=True,
            new_inference_branch=False,additional_active_mean_elements=2064 if mode=='mean_coadapt' else 0,
            mixture_weights_and_scales_frozen=True,original_backbones_frozen=True)

    def initialize_priors(self,data):return self.parent_info

    def groups(self):
        original=[];nonlinear=[]
        for head in self.heads:
            for name,p in head.named_parameters():
                if p.requires_grad:
                    (original if name in ('gate_weight','gate_bias','mean_weight','mean_bias') else nonlinear).append(p)
        nonlinear.extend(p for p in self.routes.parameters() if p.requires_grad)
        return [dict(params=original,lr=3e-5,name='original_gate_and_optional_means'),dict(params=nonlinear,lr=3e-5,name='selected_side_and_routes')]

def load_data(stage):return original_data('perm')
def make_model(stage,master,metadata):return Network(master,stage)
