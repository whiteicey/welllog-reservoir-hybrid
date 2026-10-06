"""Prior-offset only ablation for auxiliary residual-density supervision."""
import torch
from torch.nn import functional as F
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from sw_calibrated_state_v269 import DecisionObjective as MainObjective
from por_t3_patch_loss import probability
STUDY=ROOT/'results/deep_learning/sw_prior_offset_aux_v318_20261002'
OUT=ROOT/'outputs/pure_dl_sw_prior_offset_aux_v318_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('neutral','offset')}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.aux_mode=mode;self.aux_prior_log=None
    def delta(self,raw):return (self.mix_alpha/.25)*3*torch.tanh(raw.float()/3)
    def initialize_priors(self,data):
        info=super().initialize_priors(data);assert sum(p.numel() for p in self.parameters() if p.requires_grad)==65265
        info.update(auxiliary_mode=self.aux_mode,new_active_parameters=65265,only_original104_checkpoint_loaded=True,
            no275_or297_or317trained_weights_loaded=True,no317_numeric_module=True,
            actual_fusion='normalized logprior plus bounded delta in both arms',delta_bound=3.,gamma_ramp_epochs=5,final_gamma=1.,
            nominal_mix_alpha_buffer_is_only_gamma_schedule=True,auxiliary_offset_only_varies=True,
            auxiliary_delta_same_tanh_and_ramp_in_both=True,not_old275_rawaux_replication=True,
            labels_supervise_auxiliary_no_teacher_KL=True,one_SW_additional_phase=True,registered_capacity_same=True,active_capacity_same=True,
            new_encoder_state_condition_density_jointly_train=True,share_schedule_only_changed=False,state_bound_only_changed=False)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Same-well input bank required')
        self.aux_features=None;self.aux_prior_log=None;self.state_feature=None
        with torch.no_grad():
            gate,raw=self.base(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
            flat=raw[:,:,0].float().reshape(-1,3,8)
            prior=torch.cat([probability(2,c[:,0].softmax(-1),c[:,1],F.softplus(c[:,2])+.025,self.edges) for c in flat.split(4096)]).reshape(*x.shape[:2],256)
            prior=prior.clamp_min(1e-12);prior=prior/prior.sum(-1,keepdim=True);logprior=prior.log()
        inp=torch.cat((x[...,:206],relative.unsqueeze(-1)),-1)
        self.current_modulation=self.condition(well_inputs.index_select(-1,self.curve_indices))
        try:
            features=[branch(self.stem(inp)).masked_fill(~valid.unsqueeze(-1),0) for branch in self.branches]
            fused=self.fuse(torch.cat(features,-1)).masked_fill(~valid.unsqueeze(-1),0)
        finally:self.current_modulation=None;self.state_feature=None
        logits=self.readout(fused).float();logmass=(logprior+self.delta(logits)).log_softmax(-1)
        if self.training:self.aux_features=features+[fused];self.aux_prior_log=logprior.detach()
        return gate.float()+4*torch.tanh(self.gate_patch(fused).float()/4),logmass.unsqueeze(2)
class DecisionObjective(MainObjective):
    def __init__(self,evaluator,metadata,mode):super().__init__(evaluator,metadata,'balanced')
    def forward(self,output,labels,default,mask,quality,model):
        original=super().forward(output,labels,default,mask,quality)
        d=default[mask][:,2];z=labels[mask][:,2].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),self.edges[1:-1]);truth=torch.exp(z*self.scale+self.median)
        soft=(1-(self.physical_centers[None]-truth[:,None]).abs()/(.05*(truth.abs()[:,None]+1e-5))).clamp_min(0);den=soft.sum(-1,keepdim=True);soft=torch.where(den>0,soft/den.clamp_min(1e-12),F.one_hot(bins,256).float()).detach()
        weight=q*(~d).float();continuous=weight.sum().clamp_min(1e-12);terms=[]
        prior=model.aux_prior_log[mask]
        for h in model.aux_features:
            delta=model.delta(model.readout(h[mask].float()))
            logits=prior+delta if model.aux_mode=='offset' else delta
            lp=logits.log_softmax(-1);hard=-lp.gather(1,bins[:,None]).squeeze(1);ce=.5*hard-.5*(soft*lp).sum(-1);terms.append((weight*ce).sum()/continuous)
        aux=sum(terms)/len(terms);model.aux_features=None;model.aux_prior_log=None;model.last_aux_ce=float(aux.detach())
        return original[0]+aux*q.mean()/3,original[1],original[2]
def make_model(stage,master,metadata):return Network(master,metadata,stage)
