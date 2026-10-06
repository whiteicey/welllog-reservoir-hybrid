"""Migrate historically useful rank-one shared encoding to protected SW distributions.

Four internal members, one jointly trained model and one added SW stage.
Matched independent versus pooled continuous component supervision.
"""
import torch
from torch import nn
from torch.nn import functional as F
from sw_sparse_reacquisition_v294 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent,sparsemax,probability
from sw_calibrated_state_v269 import DecisionObjective as OriginalObjective
STUDY=ROOT/'results/deep_learning/sw_rankone_continuous_v296_20261002'
OUT=ROOT/'outputs/pure_dl_sw_rankone_continuous_v296_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('independent','pooled')}

class RankOneStem(nn.Module):
    def __init__(self,linear):
        super().__init__()
        self.weight=nn.Parameter(linear.weight.detach().clone())
        self.bias=nn.Parameter(linear.bias.detach().clone())
        self.input_gain=nn.Parameter(1+.1*torch.randn(4,207))
        self.output_gain=nn.Parameter(1+.1*torch.randn(4,48))
    def forward(self,x):
        return F.gelu(F.linear(x.unsqueeze(-2)*self.input_gain,self.weight)*self.output_gain+self.bias)

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'point');self.supervision_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261296);self.stem=RankOneStem(self.stem[0])
        self.member_logits=None
    def groups(self):
        return [dict(params=[self.stem.weight,self.stem.bias]+list(self.selectors.parameters())+list(self.transforms.parameters()),lr=3e-4,name='SW_shared_sparse_representation'),dict(params=[self.stem.input_gain,self.stem.output_gain],lr=3e-4,name='SW_rankone_input_modulation'),dict(params=list(self.readout.parameters()),lr=3e-4,name='SW_direct_categorical_readout'),dict(params=list(self.gate_patch.parameters()),lr=3e-4,name='SW_new_state_readout')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(new_active_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),members=4,rankone_first_linear_only=True,rankone_initial_gain='1+0.1normal seed20261296',shared_density_readout=True,shared_state_uses_mean_hidden=True,probability_mean_not_logit_mean=True,direct_continuous_supervision=self.supervision_mode,no294trained_weights_loaded=True,no_historical_batchensemble_or_tabm_weights_loaded=True,historical_rankone_is_not_new_algorithm=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.member_logits=None;self.aux_features=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8)
            prior=torch.cat([probability(2,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)
        h=self.stem(inp).masked_fill(~valid[:,:,None,None],0)
        curves=x.index_select(-1,self.curve_indices).unsqueeze(2)
        budget=torch.ones((*x.shape[:2],4,9),dtype=torch.float32,device=x.device)
        vf=valid[:,:,None,None].float();supports=[];variations=[]
        for selector,transform in zip(self.selectors,self.transforms):
            logits=selector(h).float();mask=sparsemax(budget*logits);budget=budget*(1.3-mask)
            selected=(9*mask*curves.float()).to(h.dtype)
            h=((h+F.glu(transform(torch.cat((h,selected),-1)),-1))/(2**.5)).masked_fill(~valid[:,:,None,None],0)
            if self.training:
                supports.append(((mask>0).float()*vf).sum()/(4*vf.sum().clamp_min(1)))
                mean=(mask*vf).sum(1,keepdim=True)/vf.sum(1,keepdim=True).clamp_min(1)
                variations.append(((mask-mean).square()*vf).sum()/(4*vf.sum().clamp_min(1)))
        logits=self.readout(h).float()
        if self.training:
            self.member_logits=logits
            self.last_support=float(torch.stack(supports).mean().detach());self.last_mask_variation=float(torch.stack(variations).mean().detach())
        new=logits.softmax(-1).mean(2)
        prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True)
        mass=((1-self.mix_alpha)*prior+self.mix_alpha*new).clamp_min(1e-12);mass=mass/mass.sum(-1,keepdim=True)
        return gate.float()+4*torch.tanh(self.gate_patch(h.mean(2)).float()/4),mass.log().unsqueeze(2)

class DecisionObjective(OriginalObjective):
    def __init__(self,evaluator,metadata,mode):
        super().__init__(evaluator,metadata,'balanced');self.component_mode=mode
    def forward(self,output,labels,default,mask,quality,model):
        original=super().forward(output,labels,default,mask,quality)
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float()
        bins=torch.bucketize(z.contiguous(),self.edges[1:-1]);truth=torch.exp(z*self.scale+self.median)
        soft=(1-(self.physical_centers[None]-truth[:,None]).abs()/(.05*(truth.abs()[:,None]+1e-5))).clamp_min(0)
        den=soft.sum(-1,keepdim=True);soft=torch.where(den>0,soft/den.clamp_min(1e-12),F.one_hot(bins,256).float()).detach()
        lp=model.member_logits[mask].float().log_softmax(-1)
        if self.component_mode=='independent':
            hard=-lp.gather(-1,bins[:,None,None].expand(-1,4,1)).squeeze(-1)
            ce=(.5*hard-.5*(soft[:,None]*lp).sum(-1)).mean(-1)
        else:
            pooled=torch.logsumexp(lp,dim=1)-__import__('math').log(4)
            hard=-pooled.gather(-1,bins[:,None]).squeeze(-1)
            ce=.5*hard-.5*(soft*pooled).sum(-1)
        weights=q*(~d).float();aux=(weights*ce).sum()/weights.sum().clamp_min(1e-12)
        model.member_logits=None;model.last_aux_ce=float(aux.detach())
        return original[0]+aux*q.mean()/3,original[1],original[2]

def make_model(stage,master,metadata):return Network(master,metadata,stage)
