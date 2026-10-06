"""Protect original154 and jointly train new continuous and state representation."""
import torch
from torch import nn
from perm_adaptive_share_v298 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_joint_state_share_v300_20261002'
OUT=ROOT/'outputs/pure_dl_perm_joint_state_share_v300_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('fixed_gate','joint_gate')}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'conditional');self.state_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261300);self.gate_patch=nn.Linear(48,1)
            nn.init.zeros_(self.gate_patch.weight);nn.init.zeros_(self.gate_patch.bias)
        if mode=='fixed_gate':self.gate_patch.requires_grad_(False)
        self.current_state_feature=None;self.share.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.current_state_feature=args[0]
    def groups(self):
        groups=super().groups()
        if self.state_mode=='joint_gate':groups.append(dict(params=list(self.gate_patch.parameters()),lr=3e-4,name='PERM_new_state_readout'))
        return groups
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(state_mode=self.state_mode,new_state_bound=2.,new_state_active_parameters=49 if self.state_mode=='joint_gate' else 0,new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),original154_gate_weights_frozen=True,old_gate_predictions_fixed=self.state_mode=='fixed_gate',old_gate_frozen=self.state_mode=='fixed_gate',share_mode='conditional',primary_control='same registered statehead fixed0 versus jointlytrained49parameters',active_capacity_difference=49,no298repair_weights_loaded=True,own_BCE_jointly_coupled_with_continuous_NLL_RPS=True)
        return info
    def forward(self,*args,**kwargs):
        self.current_state_feature=None
        try:
            gate,mass=super().forward(*args,**kwargs)
            assert self.current_state_feature is not None
            return gate.float()+2*torch.tanh(self.gate_patch(self.current_state_feature).float()/2),mass
        finally:self.current_state_feature=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
