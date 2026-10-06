"""Random joint student with masked convolutional encoder-decoder context."""
import torch
from torch import nn
from torch.nn import functional as F
from joint_representation_backbone_v132r1 import ROOT,SOURCE,Network as Aligned,Evaluator,load_data,channel_objective
from por_convgru_backbone_v127 import ConvBlock

STUDY=ROOT/'results/deep_learning/joint_pyramid_backbone_v137_20261001'
OUT=ROOT/'outputs/pure_dl_joint_pyramid_backbone_v137_fold5'
STAGES={'pyramid':('TOTAL',512,False,0.,6e-4)}

class PyramidContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.levels=nn.ModuleList([nn.ModuleList([ConvBlock(1),ConvBlock(2)]) for _ in range(2)])
        self.bottom=nn.ModuleList([ConvBlock(d) for d in (1,2,4)])
        self.merge=nn.ModuleList([nn.Linear(192,192) for _ in range(2)])
        self.refine=nn.ModuleList([ConvBlock(1),ConvBlock(1)])

    @staticmethod
    def pool(h,valid):
        b,l,w=h.shape
        assert l%2==0
        mask=valid.reshape(b,l//2,2)
        counts=mask.sum(2).clamp_min(1)
        h=(h.reshape(b,l//2,2,w)*mask[...,None]).sum(2)/counts[...,None]
        return h,mask.any(2)

    def forward(self,h,src_key_padding_mask):
        valid=~src_key_padding_mask;skips=[]
        for blocks in self.levels:
            for block in blocks:h=block(h,valid)
            skips.append((h,valid));h,valid=self.pool(h,valid)
        for block in self.bottom:h=block(h,valid)
        for i,(skip,valid) in enumerate(reversed(skips)):
            up=F.interpolate(h.transpose(1,2),size=skip.shape[1],mode='linear',align_corners=False).transpose(1,2)
            candidate,gate=self.merge[i](torch.cat((skip,up),-1)).chunk(2,-1)
            h=(skip+gate.sigmoid()*candidate.tanh()).masked_fill(~valid[...,None],0)
            h=self.refine[i](h,valid)
        return h

class Network(Aligned):
    def __init__(self,master):
        super().__init__(master)
        self.global_blocks=nn.ModuleList([PyramidContext()])

    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(inference_prediction_capacity_same_as_v130=False,
            context_architecture='masked convolutional pyramid 64/32/16 tokens with gated skip decoder',
            temporal_attention_removed=True,point_stem_original_dense207=True,
            architecture_and_capacity_change=True,no_old_student_weights_loaded=True)
        return info

def make_model(stage,master,metadata):return Network(master)
