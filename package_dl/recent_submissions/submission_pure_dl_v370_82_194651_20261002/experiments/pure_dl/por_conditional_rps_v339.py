"""Conditional-only finite-edge RPS; fresh positive encoder/readout jointly trained."""
from torch.nn import functional as F
import torch
from por_curve_token_width_v335 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,Network as Parent
STUDY=ROOT/'results/deep_learning/por_conditional_rps_v339_20261002'
OUT=ROOT/'outputs/pure_dl_por_conditional_rps_v339_fold5'
STAGES={'conditional':('POR',256,False,0.,3e-4)}
class Network(Parent):
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(joint_ranked_probability_loss=False,conditional_only_RPS=True,conditional_RPS_all_rows_mean_normalization=True,default_RPS_gradient_removed=True,gate_frozen_original151=True,no_trained335_weights=True,not_same_effective_gradient_weight_as_joint=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

def channel_objective(output,labels,default,mask,quality,target_index,mode,model,well_ids=None):
    assert target_index==0
    gate,logmass=output;g=gate[mask][:,0].float();lp=logmass[mask][:,0].float()
    d=default[mask][:,0];z=labels[mask][:,0].float();q=quality[mask].detach().float();bins=torch.bucketize(z.contiguous(),model.edges[1:-1]);truth=torch.expm1(z*model.target_scale+model.target_median)
    soft=(1-(model.physical_centers[None]-truth[:,None]).abs()/(.08*(truth.abs()[:,None]+1e-5))).clamp_min(0);den=soft.sum(-1,keepdim=True);soft=torch.where(den>0,soft/den.clamp_min(1e-12),F.one_hot(bins,256).float()).detach()
    hard=-lp.gather(1,bins[:,None]).squeeze(1);ce=.5*hard-.5*(soft*lp).sum(-1)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none');likelihood=(q*(bce+ce.masked_fill(d,0))).mean()/3
    conditional=lp.exp().cumsum(-1)[:,:-1]
    target_cdf=(z[:,None]<=model.edges[None,1:-1]).float()
    # Keep global-row normalization; no extra continuous-population weight sweep.
    row_rps=(conditional-target_cdf).square().mean(-1).masked_fill(d,0)
    rps=(q*row_rps).mean()/3
    weight=q*(~d).float();continuous=weight.sum().clamp_min(1e-12);terms=[]
    for h in model.aux_features:
        own=model.readout(h[mask].float()).log_softmax(-1);hard=-own.gather(1,bins[:,None]).squeeze(1);ce=.5*hard-.5*(soft*own).sum(-1);terms.append((weight*ce).sum()/continuous)
    aux=sum(terms)/len(terms);model.aux_features=None;model.last_aux_ce=float(aux.detach())
    return likelihood+rps+aux*q.mean()/3,likelihood,rps
