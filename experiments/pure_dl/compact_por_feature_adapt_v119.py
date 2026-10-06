"""Mixed-family POR warm feature adaptation, fixed input router and readout coefficients."""
import torch
from compact_por_scale_v114 import ROOT,SOURCE,PARENT,Network as Parent,Evaluator,load_data,channel_objective
from compact_five_stage_v100 import shift
from paired_detached_router_sequence import routed_merge
STUDY=ROOT/'results/deep_learning/compact_por_feature_adapt_v119_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_feature_adapt_v119_fold5'
STAGES={n:('POR',256,False,0.,3e-5) for n in ('calibration','features')}
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,'location_only');self.feature_scope=mode;self.feature_parameters=[]
        for m in self.backbone.members:
            for n,p in m.named_parameters():
                enabled=(n.startswith('calibration.') if mode=='calibration' else 'head' not in n)
                if enabled:p.requires_grad_(True);self.feature_parameters.append(p)
        assert self.feature_parameters
    def capture(self,e):
        def hook(module,args):self.captured[e]=args[0]
        return hook
    def groups(self):
        groups=super().groups()
        groups.append(dict(params=self.feature_parameters,lr=1e-5,name='expert_'+self.feature_scope))
        return groups
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),same_initial_function=True,feature_scope=self.feature_scope,
            fixed_source_readout_coefficients_and_input_router=True,gate_density_predictions_can_change_via_hidden_features=True,
            backbone_eval_mode_with_gradients=True,no_training_dropout_change=True,
            existing_head_lr=3e-5,feature_lr=1e-5,feature_cosine_equals_eta_min_constant=True,
            source_stage_and_initial_POR_counted=True,total_training_stages=5)
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.captured.clear()
        for m in self.backbone.members:m(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
        with torch.no_grad():logits=self.backbone.route_logits(x[...,:206].contiguous(),valid)[:,:,:1]
        hidden=[self.captured[e] for e in range(2)];side=x[...,206:]
        outputs=[head(h,side,valid) for head,h in zip(self.heads,hidden)]
        common,a,b=[route(hidden,side,valid) for route in self.routes]
        gate,_=routed_merge(outputs,shift(logits,a));_,raw=routed_merge(outputs,shift(logits,common+b))
        self.captured.clear();return gate,raw
def make_model(stage,master,metadata):return Network(master,stage)
