"""Own PERM backbone coadaptation with positive RPS readout, five total phases."""
import torch
from torch.nn import functional as F
from perm_ranked_probability_v154 import ROOT,SOURCE,PARENT,Network as Previous,load_data,Evaluator,channel_objective
from compact_five_stage_v100 import shift
from paired_detached_router_sequence import routed_merge
STUDY=ROOT/'results/deep_learning/perm_private_backbone_rps_v166_20261001'
OUT=ROOT/'outputs/pure_dl_perm_private_backbone_rps_v166_fold5'
STAGES={'perm_conditional':('PERM',256,False,0.,3e-4)}

class Network(Previous):
    def __init__(self,master,mode):
        super().__init__(master,mode)
        for member in self.backbone.members:
            for name,param in member.named_parameters():
                if not name.startswith(('main_head.','aux_mid.','aux_coarse.')):
                    param.requires_grad_(True)
    def capture(self,e):
        def hook(module,args):self.captured[e]=args[0]
        return hook
    def capture_new(self,e):
        def hook(module,args):self.local_captured[e]=args[0]
        return hook
    def groups(self):
        body=[p for member in self.backbone.members for p in member.parameters() if p.requires_grad]
        return [dict(params=list(self.context_head.parameters()),lr=3e-4,name='PERM_location_weight_head'),
                dict(params=body,lr=1e-5,name='PERM_private_backbone')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(old_predictor_fully_frozen=False,own_PERM_body_coadaptation=True,body_lr=1e-5,
                    original_output_rows_router_frozen=True,original_scale_gate_functions_can_change_with_features=True,
                    backbone_eval_mode_dropout_disabled=True,trained_v154_head_not_loaded=True,
                    same_initial_function=True,no_other_target_training_labels=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.captured.clear();self.local_captured.clear()
        for member in self.backbone.members:
            member(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
        with torch.no_grad():logits=self.backbone.route_logits(x[...,:206].contiguous(),valid)[:,:,1:2]
        hidden=[self.captured[e] for e in range(2)];side=x[...,206:]
        outputs=[head(h,side,valid) for head,h in zip(self.heads,hidden)]
        common,a,b=[route(hidden,side,valid) for route in self.routes]
        routing=shift(logits,common+(a+b)*.5)
        gate,raw=routed_merge(outputs,routing)
        features=torch.cat((self.local_captured[0],self.local_captured[1],side.transpose(1,2)),1)
        self.captured.clear();self.local_captured.clear()
        u=self.context_head(features,valid);own=raw[:,:,0]
        reference=F.softplus(own[:,:,2].detach())+.025
        corrected=torch.stack((own[:,:,0]+torch.tanh(u[...,16:]),
                    own[:,:,1]+reference*torch.tanh(u[...,:16]/reference),own[:,:,2]),2)
        return gate,corrected.unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,stage)
