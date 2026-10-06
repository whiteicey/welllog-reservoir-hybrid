"""State readout trains, but only continuous losses shape the new raw encoder."""
from sw_state_bound_v289 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective as ParentObjective,Network as Parent
from sw_direct_component_v275 import Network as RawNetwork
import torch
STUDY=ROOT/'results/deep_learning/sw_density_owned_encoder_v301_20261002'
OUT=ROOT/'outputs/pure_dl_sw_density_owned_encoder_v301_fold5'
STAGES={'density_owned':('SW',1024,False,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,metadata,mode):super().__init__(master,metadata,'balanced')
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(new_raw_backbone_gradient_owner='continuous mixedCE/conditionalRPS/directcomponentCE only',state_readout_input_detached=True,state_readout_still_trained=True,no289_or297_or299_weights_loaded=True,all_raw_density_parameters_jointly_trained=True,forward_function_and_capacity_same_as289=True,reference='original289 complete80-14 sameconfiguration historical control',comparison_scope='same capacity/initialization/loss/inference except gate-to-newbody gradient; historical control, not concurrently rerun',new_active_parameters=57617)
        return info
    def forward(self,*args,**kwargs):
        self.state_feature=None
        try:
            gate,mass=RawNetwork.forward(self,*args,**kwargs)
            h=self.state_feature
            return gate.float()+4*torch.tanh(self.gate_patch(h.detach()).float()/4),mass
        finally:self.state_feature=None
class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):super().__init__(evaluator,metadata,'balanced')
def make_model(stage,master,metadata):return Network(master,metadata,stage)
