"""Joint input neighborhood encoder and continuous value-coordinate energy basis."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from sw_direct_component_v275 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from sw_calibrated_state_v269 import DecisionObjective as OriginalObjective
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/sw_value_coordinate_v280_20261002'
OUT=ROOT/'outputs/pure_dl_sw_value_coordinate_v280_fold5'
STAGES={k:('SW',1024,False,0.,3e-4) for k in ('ordered','shuffled')}
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'mix');self.mode=mode
        del self.stem,self.branches,self.fuse,self.readout
        self.register_buffer('curve_indices',torch.tensor([master['selected_feature_names'].index(n) for n in CURVES]))
        physical=torch.as_tensor(metadata['centers'][2],dtype=torch.float32);centers=(physical.log()-metadata['sw_target_median'])/metadata['sw_target_scale'];coords=(centers-centers.mean())/centers.std().clamp_min(1e-6)
        assert bool(torch.isfinite(coords).all()),'Finite original physical-grid center coordinates required'
        if mode=='shuffled':coords=coords[torch.randperm(256,generator=torch.Generator().manual_seed(20261279))]
        self.register_buffer('value_coordinates',coords[:,None])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261279)
            self.input_encoder=nn.Sequential(nn.Linear(377,48),nn.GELU(),nn.Linear(48,96),nn.GELU(),nn.Linear(96,48),nn.GELU())
            self.coefficients=nn.Linear(48,32);nn.init.zeros_(self.coefficients.weight);nn.init.zeros_(self.coefficients.bias)
            self.value_encoder=nn.Sequential(nn.Linear(1,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,32))
        self.aux_logits=None
    def groups(self):return [dict(params=list(self.input_encoder.parameters())+list(self.coefficients.parameters()),lr=3e-4,name='SW_ordered_neighborhood_input_backbone'),dict(params=list(self.value_encoder.parameters()),lr=3e-4,name='SW_continuous_value_basis_readout')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data);info.update(new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),input_encoder_width=48,neighborhood_points=17,value_basis=32,value_coordinate_mode=self.mode,coordinate_shuffle_seed=20261279,branch_coefficients_zero_initial_uniform_component=True,first_optimizer_step_value_basis_gradient_zero_expected=True,gradient_observation_second_optimizer_step=True,no275_or_other_repair_weights_loaded=True,all_new_encoders_jointly_trainable=True,bin_output_order_unchanged=True,direct_CE_feature_count=1)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.aux_logits=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8)
            prior=torch.cat([probability(2,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        rawcurve=x.index_select(-1,self.curve_indices).masked_fill(~valid[...,None],0)
        neighborhood=F.pad(rawcurve.transpose(1,2),(8,8)).unfold(-1,17,1).permute(0,2,1,3).reshape(*x.shape[:2],153)
        flags=F.pad(valid.float(),(8,8)).unfold(-1,17,1)
        inp=torch.cat((x[...,:206],relative[...,None],neighborhood,flags.to(x.dtype)),dim=-1)
        h=self.input_encoder(inp).masked_fill(~valid[...,None],0);coef=self.coefficients(h).float()
        basis=self.value_encoder(self.value_coordinates).float();logits=coef@basis.T/math.sqrt(32)
        if self.training:self.aux_logits=[logits]
        prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        mass=(1-self.mix_alpha)*prior+self.mix_alpha*logits.softmax(-1);mass=mass.clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate,mass.log().unsqueeze(2)
class DecisionObjective(OriginalObjective):
    def forward(self,output,labels,default,mask,quality,model):
        original=super().forward(output,labels,default,mask,quality)
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),self.edges[1:-1]);truth=torch.exp(z*self.scale+self.median)
        soft=(1-(self.physical_centers[None]-truth[:,None]).abs()/(.05*(truth.abs()[:,None]+1e-5))).clamp_min(0);den=soft.sum(-1,keepdim=True);soft=torch.where(den>0,soft/den.clamp_min(1e-12),F.one_hot(bins,256).float()).detach()
        weight=q*(~d).float();continuous=weight.sum().clamp_min(1e-12);terms=[]
        for logits in model.aux_logits:
            lp=logits[mask].float().log_softmax(-1);hard=-lp.gather(1,bins[:,None]).squeeze(1);ce=.5*hard-.5*(soft*lp).sum(-1);terms.append((weight*ce).sum()/continuous)
        aux=sum(terms)/len(terms);model.aux_logits=None;model.last_aux_ce=float(aux.detach())
        return original[0]+aux*q.mean()/3,original[1],original[2]
def make_model(stage,master,metadata):return Network(master,metadata,stage)
