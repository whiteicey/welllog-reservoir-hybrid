"""Fresh SW encoder/state/density and bounded mixture allocation, one phase."""
import torch
from torch import nn
from torch.nn import functional as F
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/sw_adaptive_share_v315_20261002'
OUT=ROOT/'outputs/pure_dl_sw_adaptive_share_v315_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('scalar','conditional')}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.share_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261315);self.share=nn.Linear(48,1)
        nn.init.zeros_(self.share.weight);nn.init.zeros_(self.share.bias)
        if mode=='scalar':self.share.weight.requires_grad_(False)
        self.last_alpha_min=self.last_alpha_mean=self.last_alpha_max=0.
    def groups(self):
        return super().groups()+[dict(params=[p for p in self.share.parameters() if p.requires_grad],lr=3e-4,name='SW_new_adaptive_allocation')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        assert self.scales==(1,1,1)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==(65266 if self.share_mode=='scalar' else 65314)
        info.update(share_mode=self.share_mode,extra_share_active_parameters=1 if self.share_mode=='scalar' else 49,
            new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),registered_capacity_same=True,active_capacity_same=False,
            only_original104_checkpoint_loaded=True,no297_or298_or313trained_weights_loaded=True,
            nominal_alpha_max=.25,actual_alpha_bounds=[0.,.5],bounded_alpha_formula='nominal_ramp*(1+tanh(raw/.3))',
            all_new_encoder_state_density_condition_allocation_jointly_train=True,
            conditioning_mode='well',one_SW_additional_phase=True,share_schedule_only_changed=False,state_bound_only_changed=False)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Same-well input bank required')
        self.aux_features=None;self.state_feature=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8)
            prior=torch.cat([probability(2,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)
        self.current_modulation=self.condition(well_inputs.index_select(-1,self.curve_indices))
        try:
            features=[branch(self.stem(inp)).masked_fill(~valid.unsqueeze(-1),0) for branch in self.branches]
            fused=self.fuse(torch.cat(features,-1)).masked_fill(~valid.unsqueeze(-1),0)
        finally:self.current_modulation=None;self.state_feature=None
        logits=self.readout(fused).float()
        if self.training:self.aux_features=features+[fused]
        prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        alpha=self.mix_alpha*(1+torch.tanh(self.share(fused).float()/.3))
        if self.training:
            active=alpha[valid].detach();self.last_alpha_min=float(active.min());self.last_alpha_mean=float(active.mean());self.last_alpha_max=float(active.max())
        mass=((1-alpha)*prior+alpha*logits.softmax(-1)).clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate.float()+4*torch.tanh(self.gate_patch(fused).float()/4),mass.log().unsqueeze(2)
def make_model(stage,master,metadata):return Network(master,metadata,stage)
