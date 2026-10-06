"""Input-bank conditioned low-rank feature kernels, fresh joint SW model."""
import math
import torch
from torch import nn
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_bank_kernel_v316_20261002'
OUT=ROOT/'outputs/pure_dl_sw_bank_kernel_v316_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('reference','actual')}
class KernelBank(nn.Module):
    def __init__(self):
        super().__init__();self.u=nn.Parameter(torch.zeros(4,96,8));self.v=nn.Parameter(torch.empty(4,8,48))
        for v in self.v:nn.init.kaiming_uniform_(v,a=math.sqrt(5))
    def forward(self,pi):
        return torch.einsum('be,eoi->boi',pi,self.u@self.v)
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.kernel_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261316);self.kernels=nn.ModuleList([KernelBank() for _ in range(3)])
            self.kernel_router=nn.Sequential(nn.Linear(96,16),nn.GELU(),nn.Linear(16,4))
        nn.init.zeros_(self.kernel_router[-1].weight);nn.init.zeros_(self.kernel_router[-1].bias)
        self.current_kernels=None;self.last_router_entropy=0.
        for index,branch in enumerate(self.branches):branch.net[0].register_forward_hook(self.kernel_hook(index))
    def kernel_hook(self,index):
        def hook(module,args,result):
            if self.current_kernels is None:raise RuntimeError('Explicit bank kernel required')
            correction=torch.bmm(args[0],self.current_kernels[index].transpose(1,2).to(args[0].dtype))
            return result+correction.to(result.dtype)
        return hook
    def groups(self):
        return super().groups()+[dict(params=list(self.kernels.parameters())+list(self.kernel_router.parameters()),lr=3e-4,name='SW_new_bank_matrix_kernels')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data);active=sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert active==80709 and self.scales==(1,1,1)
        info.update(kernel_mode=self.kernel_mode,new_active_parameters=active,new_kernel_parameters=15444,
            kernel_bases=4,kernel_rank=8,kernel_matrix_shape=[96,48],kernel_ROUTER_input96_hidden16=True,
            original_FiLM_actual_bank_in_both_arms=True,new_kernel_source_only_varies=True,
            initial_U_zero_V_nonzero=True,initial_router_uniform=True,zero_router_first_step_expected=True,
            reference_fitted_from64training_inputs_only=True,registered_capacity_same=True,active_capacity_same=True,
            effective_function_class_differs=True,only_original104_checkpoint_loaded=True,
            no297_or296_or315trained_weights_loaded=True,no_prediction_experts_or_voting=True,
            one_SW_additional_phase=True,conditioning_mode='well',share_schedule_only_changed=False,state_bound_only_changed=False)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Same-well input bank required')
        context=well_inputs if self.kernel_mode=='actual' else self.raw_reference_inputs[None].expand(x.shape[0],-1,-1)
        h=self.condition.point(context.index_select(-1,self.curve_indices))
        stats=torch.cat((h.mean(1),h.var(1,unbiased=False).add(1e-6).sqrt(),h.amax(1)),-1)
        pi=self.kernel_router(stats).float().softmax(-1)
        if self.training:self.last_router_entropy=float(-(pi.detach()*pi.detach().clamp_min(1e-12).log()).sum(-1).mean())
        self.current_kernels=[bank(pi) for bank in self.kernels]
        try:return super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        finally:self.current_kernels=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
