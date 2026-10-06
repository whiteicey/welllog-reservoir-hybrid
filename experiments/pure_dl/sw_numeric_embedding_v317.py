"""Per-curve learned numeric embeddings before fresh SW feature mixing."""
import torch
from torch import nn
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_numeric_embedding_v317_20261002'
OUT=ROOT/'outputs/pure_dl_sw_numeric_embedding_v317_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('linear','relu')}
class NumericEmbedding(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.empty(9,8));self.bias=nn.Parameter(torch.empty(9,8));self.out=nn.Linear(72,48,bias=False)
        nn.init.uniform_(self.weight,-1,1);nn.init.uniform_(self.bias,-1,1);nn.init.zeros_(self.out.weight)
    def forward(self,x,relu):
        z=x.float().unsqueeze(-1)*self.weight+self.bias
        if relu:z=z.relu()
        return self.out(z.flatten(-2))
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.embedding_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261317);self.numeric=NumericEmbedding()
        self.current_numeric=None;self.stem[0].register_forward_hook(self.add_numeric)
    def add_numeric(self,module,args,result):
        if self.current_numeric is None:raise RuntimeError('Explicit original curve embedding required')
        return result+self.current_numeric.to(result.dtype)
    def groups(self):
        return super().groups()+[dict(params=list(self.numeric.parameters()),lr=3e-4,name='SW_new_per_curve_numeric_embedding')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data);active=sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert active==68865 and self.scales==(1,1,1)
        info.update(embedding_mode=self.embedding_mode,new_active_parameters=active,new_numeric_parameters=3600,
            original9curves_each8dimensions=True,zero_output_linear72to48_biasFalse=True,
            injection_after_stem_linear_before_GELU_and_FiLM=True,registered_capacity_same=True,active_capacity_same=True,
            effective_function_class_differs=True,linear_control_foldable_into_original_stem=True,
            extra_statistics_fit=False,old206_and_relative_inputs_preserved=True,new_formula_features=False,
            first_step_embedding_zero_gradient_expected=True,individual_ReLU_units_may_be_inactive=True,
            only_original104_checkpoint_loaded=True,no297_or276_or316trained_weights_loaded=True,
            one_SW_additional_phase=True,conditioning_mode='well',share_schedule_only_changed=False,state_bound_only_changed=False)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.current_numeric=self.numeric(x.index_select(-1,self.curve_indices),self.embedding_mode=='relu')
        try:return super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        finally:self.current_numeric=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
