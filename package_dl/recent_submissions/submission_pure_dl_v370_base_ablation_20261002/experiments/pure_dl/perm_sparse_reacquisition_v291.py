"""Two sparse raw-curve reacquisition blocks; jointly trained PERM body/readout."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from perm_direct_share_v283 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from por_t3_patch_loss import probability

STUDY=ROOT/'results/deep_learning/perm_sparse_reacquisition_v291_20261002'
OUT=ROOT/'outputs/pure_dl_perm_sparse_reacquisition_v291_fold5'
STAGES={s:('PERM',256,False,0.,3e-4) for s in ('point','window')}

def sparsemax(logits):
    z=logits.float()-logits.float().amax(-1,keepdim=True)
    ordered=z.sort(-1,descending=True).values
    rank=torch.arange(1,z.shape[-1]+1,device=z.device,dtype=z.dtype)
    cumulative=ordered.cumsum(-1)
    support=1+rank*ordered>cumulative
    count=support.sum(-1,keepdim=True).clamp_min(1)
    threshold=(cumulative.gather(-1,count-1)-1)/count
    return (z-threshold).clamp_min(0)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,mode)
        names=list(master['selected_feature_names'])
        curves=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
        indices=[names.index(c) for c in curves]
        assert len(set(indices))==9 and max(indices)<206
        self.register_buffer('curve_indices',torch.tensor(indices,dtype=torch.long))
        del self.branches,self.fuse
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261291)
            self.stem=nn.Sequential(nn.Linear(207,48),nn.GELU())
            self.selectors=nn.ModuleList([nn.Sequential(nn.Linear(48,24),nn.GELU(),nn.Linear(24,9)) for _ in range(2)])
            self.transforms=nn.ModuleList([nn.Linear(57,96) for _ in range(2)])
            self.readout=nn.Linear(48,256)
            for selector in self.selectors:
                nn.init.zeros_(selector[-1].weight);nn.init.zeros_(selector[-1].bias)
        self.last_support=0.;self.last_mask_variation=0.
    def groups(self):
        return [dict(params=list(self.stem.parameters())+list(self.selectors.parameters())+list(self.transforms.parameters()),lr=3e-4,name='PERM_sparse_reacquisition_body'),dict(params=list(self.readout.parameters()),lr=3e-4,name='PERM_direct_categorical_readout')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(raw_curve_indices=self.curve_indices.tolist(),selection_blocks=2,gamma=1.3,selection_granularity=self.mode,dense_bypass_207=True,matched_point_window_initialization=True,primary_control='same architecture window vs point',historical283_not_same_capacity=True,aux_CE_uses_final_component_only=True,no_previous_repair_weights_loaded=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_logits=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,16)
            prior=torch.cat([probability(1,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)
        h=self.stem(inp).masked_fill(~valid.unsqueeze(-1),0)
        curves=x.index_select(-1,self.curve_indices);budget=torch.ones_like(curves,dtype=torch.float32)
        vf=valid.unsqueeze(-1).float();supports=[];variations=[]
        for selector,transform in zip(self.selectors,self.transforms):
            logits=selector(h).float()
            if self.mode=='window':logits=((logits*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)).expand_as(logits)
            mask=sparsemax(budget*logits);budget=budget*(1.3-mask)
            selected=(9*mask*curves.float()).to(h.dtype)
            h=((h+F.glu(transform(torch.cat((h,selected),-1)),-1))/math.sqrt(2)).masked_fill(~valid.unsqueeze(-1),0)
            if self.training:
                supports.append(((mask>0).float()*vf).sum()/vf.sum().clamp_min(1))
                mean=(mask*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)
                variations.append(((mask-mean).square()*vf).sum()/vf.sum().clamp_min(1))
        logits=self.readout(h).float()
        if self.training:
            self.aux_logits=[logits]
            self.last_support=float(torch.stack(supports).mean().detach())
            self.last_mask_variation=float(torch.stack(variations).mean().detach())
        mass=((1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1)).clamp_min(1e-12)
        mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)

def make_model(stage,master,metadata):return Network(master,metadata,stage)
