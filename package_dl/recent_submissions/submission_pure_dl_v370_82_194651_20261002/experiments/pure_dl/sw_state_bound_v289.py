"""Joint raw encoder/state/density with stronger bounded state correction."""
import torch
from sw_joint_share_v285 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
from sw_direct_component_v275 import Network as RawNetwork
STUDY=ROOT/'results/deep_learning/sw_state_bound_v289_20261002'
OUT=ROOT/'outputs/pure_dl_sw_state_bound_v289_fold5'
STAGES={'balanced':('SW',1024,False,0.,3e-4)}
class Network(Parent):
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(new_gate_bound=4.,reference_gate_bound=2.,reference='v285natural80-14',no285trained_weights_loaded=True,state_bound_only_changed=True)
        return info
    def forward(self,*args,**kwargs):
        self.state_feature=None;gate,mass=RawNetwork.forward(self,*args,**kwargs);h=self.state_feature;self.state_feature=None
        return gate.float()+4*torch.tanh(self.gate_patch(h).float()/4),mass
def make_model(stage,master,metadata):return Network(master,metadata,stage)
