"""New raw backbone learns continuous distribution and default-state readout jointly."""
import torch
from torch import nn
from sw_direct_component_v275 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_joint_state_encoder_v282_20261002'
OUT=ROOT/'outputs/pure_dl_sw_joint_state_encoder_v282_fold5'
STAGES={k:('SW',1024,False,0.,3e-4) for k in ('ordinary','balanced')}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'mix');self.gate_mode=mode;self.state_feature=None
        self.gate_patch=nn.Linear(48,1);nn.init.zeros_(self.gate_patch.weight);nn.init.zeros_(self.gate_patch.bias)
        self.fuse.register_forward_hook(self.capture)
    def capture(self,module,args,result):self.state_feature=result
    def groups(self):return super().groups()+[dict(params=list(self.gate_patch.parameters()),lr=3e-4,name='SW_new_state_readout')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(gate_mode=self.gate_mode,new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),new_gate_readout_parameters=49,original104_prior_atom_weights_frozen_but_learned_gate_residual=True,new_gate_bound=2.,newencoder_density_state_joint=True,original_gate_predictions_not_fixed=True,training_BCE_mode=self.gate_mode,training_balanced_prior_from_train_Q_only=True,inference_BCE_shift_removed=True,no275_or269trained_weights_loaded=True)
        return info
    def forward(self,*args,**kwargs):
        self.state_feature=None;gate,mass=super().forward(*args,**kwargs);h=self.state_feature;self.state_feature=None
        return gate.float()+2*torch.tanh(self.gate_patch(h).float()/2),mass
def make_model(stage,master,metadata):return Network(master,metadata,stage)
