"""Transfer a locally positive grouped response mechanism, no PERM weights."""
from por_curve_scope_v302 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from perm_group_shape_v312 import ShapeBank
STUDY=ROOT/'results/deep_learning/por_group_shape_v314_20261002'
OUT=ROOT/'outputs/pure_dl_por_group_shape_v314_fold5'
STAGES={s:('POR',256,False,0.,3e-4) for s in ('rational','polynomial')}

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'full');self.shape_mode=mode
        self.body_shape=ShapeBank(mode);self.readout_shape=ShapeBank(mode)
        for branch in self.branches:branch.net[0].register_forward_pre_hook(self.shape_body_input)
        self.readout.register_forward_pre_hook(self.shape_readout_input)
    def shape_body_input(self,module,args):return (self.body_shape(args[0]),)
    def shape_readout_input(self,module,args):return (self.readout_shape(args[0]),)
    def groups(self):
        return super().groups()+[dict(params=list(self.body_shape.parameters())+list(self.readout_shape.parameters()),lr=3e-4,name='POR_joint_body_readout_shape')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        active=sum(p.numel() for p in self.parameters() if p.requires_grad);assert active==76696
        info.update(shape_mode=self.shape_mode,new_active_parameters=active,new_shape_parameters=104,
                    fixed_anchor_buffers=True,initial_feature_function_identity=True,groups_per_bank=8,
                    body_bank_shared_across_three_MLP_inputs=True,readout_bank_independent=True,
                    full_curve_mechanism_preserved=True,original151_only=True,
                    no302_or312_trained_weights_loaded=True,one_POR_additional_phase=True,
                    cross_target_mechanism_transfer_only=True,shape_bound=.25,shape_input_scale=3.)
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
