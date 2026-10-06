"""Anchored grouped rational versus polynomial feature/readout response."""
import torch
from torch import nn
from perm_sparse_reacquisition_v291 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/perm_group_shape_v312_20261002'
OUT=ROOT/'outputs/pure_dl_perm_group_shape_v312_fold5'
STAGES={s:('PERM',256,False,0.,3e-4) for s in ('rational','polynomial')}

class ShapeBank(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        a=torch.zeros(8,6);a[:,1]=1.;b=torch.tensor([.1,0.,0.,0.])
        self.a=nn.Parameter(a);self.b=nn.Parameter(b)
        self.register_buffer('anchor_a',a.detach().clone());self.register_buffer('anchor_b',b.detach().clone())
    def evaluate(self,z,a,b):
        coefficients=a.repeat_interleave(6,dim=0)
        numerator=coefficients[:,5]
        for k in range(4,-1,-1):numerator=numerator*z+coefficients[:,k]
        denominator_poly=b[3]
        for k in range(2,-1,-1):denominator_poly=denominator_poly*z+b[k]
        if self.mode=='rational':return numerator/(1+(denominator_poly*z).abs())
        return numerator+denominator_poly*z.pow(6)
    def forward(self,v):
        z=torch.tanh(v.float()/3)
        delta=self.evaluate(z,self.a,self.b)-self.evaluate(z,self.anchor_a,self.anchor_b)
        return (v.float()+.25*torch.tanh(delta)).to(v.dtype)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'point');self.shape_mode=mode
        self.body_shape=ShapeBank(mode);self.readout_shape=ShapeBank(mode)
        for transform in self.transforms:transform.register_forward_hook(self.shape_value)
        self.readout.register_forward_pre_hook(self.shape_readout)
    def shape_value(self,module,args,value_gate):
        value,gate=value_gate.chunk(2,dim=-1)
        return torch.cat((self.body_shape(value),gate),-1)
    def shape_readout(self,module,args):return (self.readout_shape(args[0]),)
    def groups(self):
        return super().groups()+[dict(params=list(self.body_shape.parameters())+list(self.readout_shape.parameters()),lr=3e-4,name='PERM_joint_body_readout_shape')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        active=sum(p.numel() for p in self.parameters() if p.requires_grad);assert active==36570
        info.update(shape_mode=self.shape_mode,new_active_parameters=active,new_shape_parameters=104,
                    fixed_anchor_buffers=True,initial_feature_function_identity=True,groups_per_bank=8,
                    body_bank_shared_across_two_value_blocks=True,readout_bank_independent=True,
                    hidden_width_and_block_count_unchanged=True,no291_trained_weights_loaded=True,
                    original154_only=True,one_PERM_additional_phase=True,not_full_KAT_reproduction=True)
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
