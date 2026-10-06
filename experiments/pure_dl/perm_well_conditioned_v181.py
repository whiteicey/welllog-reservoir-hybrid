"""Input-only whole-well FiLM conditioning of the private PERM backbone."""
import torch
from torch.nn import functional as F
from perm_well_sam_v177 import ROOT,SOURCE,PARENT,STAGES,Network as Parent,load_data,Evaluator,channel_objective,prepare_sampling,sample_anchors
from well_conditioned_v46 import WellEncoder,modulate
STUDY=ROOT/'results/deep_learning/perm_well_conditioned_v181_20261001'
OUT=ROOT/'outputs/pure_dl_perm_well_conditioned_v181_fold5'
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,mode);self.condition=WellEncoder()
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(input_only_well_conditioning=True,well_bank_points=128,well_bank_features=206,
                    well_labels_or_ID_not_used=True,no_validation_fit=True,zero_FiLM_initialization=True,
                    no_old_SW_conditioned_weights_loaded=True,no_trained_v177_weights_loaded=True,
                    no_teacher=True,no_extra_training_phase=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        assert well_inputs is not None
        x=x[...,:206];b,l,_=x.shape;c=self.body;values=self.condition(well_inputs[...,:206])
        h=F.gelu(c.stem(torch.cat((x,relative[...,None]),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,values[0],valid)
        for block in c.local:h=block(h,valid)
        pv=valid.reshape(b,l//16,16);n=pv.sum(-1).clamp_min(1)
        token=(h.reshape(b,l//16,16,96)*pv[...,None]).sum(2)/n[...,None]
        pos=torch.linspace(-1,1,l,device=x.device).expand(b,-1)
        loc=torch.stack((relative,pos),-1).reshape(b,l//16,16,2)
        loc=(loc*pv[...,None]).sum(2)/n[...,None]
        tv=pv.any(-1);token=(token+c.token_pos(loc)).masked_fill(~tv[...,None],0)
        for j,block in enumerate(c.global_blocks):token=modulate(block(token,tv),values[j+1],tv)
        global_h=F.interpolate(token.transpose(1,2),size=l,mode='linear',align_corners=False).transpose(1,2)
        h=F.gelu(c.fuse(torch.cat((h,global_h),-1))).masked_fill(~valid[...,None],0)
        h=modulate(h,values[5],valid)
        for block in c.detail:h=block(h,valid)
        gate=self.gate(h).float();raw=self.density(h).reshape(b,l,1,3,16).float()
        return gate,raw
def make_model(stage,master,metadata):return Network(master,stage)
