"""Input-only whole-well conditioning of the fresh joint SW raw encoder."""
import torch
from torch import nn
from sw_state_bound_v289 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent,DecisionObjective as ParentObjective
STUDY=ROOT/'results/deep_learning/sw_raw_well_condition_v297_20261002'
OUT=ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('reference','well')}

class BankCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(9,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.code=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,96))
        nn.init.zeros_(self.code[-1].weight);nn.init.zeros_(self.code[-1].bias)
    def forward(self,x):
        h=self.point(x)
        stats=torch.cat((h.mean(1),h.var(1,unbiased=False).add(1e-6).sqrt(),h.amax(1)),-1)
        return (.25*torch.tanh(self.code(stats).float())).chunk(2,-1)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'balanced');self.conditioning_mode=mode
        names=list(master['selected_feature_names'])
        indices=[names.index(c) for c in ('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')]
        assert len(set(indices))==9 and max(indices)<206
        self.register_buffer('curve_indices',torch.tensor(indices,dtype=torch.long))
        self.register_buffer('raw_reference_inputs',metadata['reference_inputs'].detach().clone().cpu())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261297);self.condition=BankCondition()
        self.current_modulation=None
        self.stem.register_forward_hook(self.modulate_stem)
    def groups(self):
        return super().groups()+[dict(params=list(self.condition.parameters()),lr=3e-4,name='SW_new_raw_well_condition')]
    def modulate_stem(self,module,args,h):
        if self.current_modulation is None:raise RuntimeError('Explicit new-branch bank context required')
        gamma,beta=self.current_modulation
        return (h.float()*(1+gamma[:,None])+beta[:,None]).to(h.dtype)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(conditioning_mode=self.conditioning_mode,new_condition_parameters=sum(p.numel() for p in self.condition.parameters()),new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),new_branch_bank_curve_indices=self.curve_indices.tolist(),old104_always_uses_actual_well_bank=True,new_branch_reference_from64training_inputs_only=True,bank_labels_or_well_ID_used=False,FiLM_bound=.25,FiLM_zero_initial_output=True,no289_or277_or278trained_weights_loaded=True,not_first_whole_well_encoder_experiment=True,same_raw_branch_init_before_training=True,state_bound_only_changed=False,share_schedule_only_changed=False)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Input-only same-well bank required')
        context=well_inputs if self.conditioning_mode=='well' else self.raw_reference_inputs[None].expand(x.shape[0],-1,-1)
        self.current_modulation=self.condition(context.index_select(-1,self.curve_indices))
        try:
            # The frozen prior remains conditioned on the actual well in BOTH arms.
            return super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        finally:self.current_modulation=None

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):super().__init__(evaluator,metadata,'balanced')

def make_model(stage,master,metadata):return Network(master,metadata,stage)
