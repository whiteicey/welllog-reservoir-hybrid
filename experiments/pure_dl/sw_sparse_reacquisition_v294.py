"""Joint SW state/density sparse raw-curve encoder with fixed original104 prior."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from sw_state_bound_v289 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent,DecisionObjective as ParentObjective
from perm_sparse_reacquisition_v291 import sparsemax
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/sw_sparse_reacquisition_v294_20261002'
OUT=ROOT/'outputs/pure_dl_sw_sparse_reacquisition_v294_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('point','window')}

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'balanced');self.selection_mode=mode
        names=list(master['selected_feature_names'])
        indices=[names.index(c) for c in ('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')]
        assert max(indices)<206 and len(set(indices))==9
        self.register_buffer('curve_indices',torch.tensor(indices,dtype=torch.long))
        del self.branches,self.fuse
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261294)
            self.stem=nn.Sequential(nn.Linear(207,48),nn.GELU())
            self.selectors=nn.ModuleList([nn.Sequential(nn.Linear(48,24),nn.GELU(),nn.Linear(24,9)) for _ in range(2)])
            self.transforms=nn.ModuleList([nn.Linear(57,96) for _ in range(2)])
            self.readout=nn.Linear(48,256);self.gate_patch=nn.Linear(48,1)
            for selector in self.selectors:
                nn.init.zeros_(selector[-1].weight);nn.init.zeros_(selector[-1].bias)
            nn.init.zeros_(self.gate_patch.weight);nn.init.zeros_(self.gate_patch.bias)
        self.last_support=0.;self.last_mask_variation=0.
    def groups(self):
        return [dict(params=list(self.stem.parameters())+list(self.selectors.parameters())+list(self.transforms.parameters()),lr=3e-4,name='SW_sparse_reacquisition_body'),dict(params=list(self.readout.parameters()),lr=3e-4,name='SW_direct_categorical_readout'),dict(params=list(self.gate_patch.parameters()),lr=3e-4,name='SW_new_state_readout')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),raw_curve_indices=self.curve_indices.tolist(),gamma=1.3,selection_blocks=2,selection_mode=self.selection_mode,dense_bypass207=True,aux_CE_final_feature_only=True,no289_or291trained_weights_loaded=True,paired_initialization=True,historical289_not_same_capacity_or_auxiliary_count=True,only_state_bound_changed=False,share_schedule_only_changed=False)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_features=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8)
            prior=torch.cat([probability(2,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)
        h=self.stem(inp).masked_fill(~valid.unsqueeze(-1),0)
        curves=x.index_select(-1,self.curve_indices);budget=torch.ones_like(curves,dtype=torch.float32)
        vf=valid.unsqueeze(-1).float();supports=[];variations=[]
        for selector,transform in zip(self.selectors,self.transforms):
            logits=selector(h).float()
            if self.selection_mode=='window':logits=((logits*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)).expand_as(logits)
            mask=sparsemax(budget*logits);budget=budget*(1.3-mask)
            selected=(9*mask*curves.float()).to(h.dtype)
            h=((h+F.glu(transform(torch.cat((h,selected),-1)),-1))/math.sqrt(2)).masked_fill(~valid.unsqueeze(-1),0)
            if self.training:
                supports.append(((mask>0).float()*vf).sum()/vf.sum().clamp_min(1))
                mean=(mask*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)
                variations.append(((mask-mean).square()*vf).sum()/vf.sum().clamp_min(1))
        logits=self.readout(h).float()
        if self.training:
            self.aux_features=[h]
            self.last_support=float(torch.stack(supports).mean().detach());self.last_mask_variation=float(torch.stack(variations).mean().detach())
        prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        mass=((1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)).clamp_min(1e-12)
        mass=mass/mass.sum(-1,keepdim=True)
        return gate.float()+4*torch.tanh(self.gate_patch(h).float()/4),mass.log().unsqueeze(2)

class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):
        # Selection arm must not accidentally disable the proven balanced BCE.
        super().__init__(evaluator,metadata,'balanced')

def make_model(stage,master,metadata):return Network(master,metadata,stage)
