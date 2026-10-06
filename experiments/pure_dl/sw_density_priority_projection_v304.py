"""Asymmetric train-only gradient conflict projection, original289 forward."""
import torch
from torch.nn import functional as F
from sw_state_bound_v289 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective as ParentObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_density_priority_projection_v304_20261002'
OUT=ROOT/'outputs/pure_dl_sw_density_priority_projection_v304_fold5'
STAGES={'density_priority':('SW',1024,False,0.,3e-4)}
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'balanced');self.last_projection_cosine=0.;self.last_projection_conflict=0.
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(shared_body_gradient_rule='density + state, projectstate alongdensity only if dot<0',private_head_gradients_unchanged=True,state_gradient_not_unconditionally_detached=True,new_active_parameters=57617,no289_or301_trained_weights=True,original_forward_loss_values_and_inference_unchanged=True,reference='complete historical289, same capacity/init/config; not rerun',not_first_PCGrad=True,not_standard_random_symmetric_PCGrad=True)
        return info
class DecisionObjective(ParentObjective):
    def __init__(self,evaluator,metadata,mode):super().__init__(evaluator,metadata,'balanced')

def backward_with_density_priority(loss,output,labels,default,mask,quality,shared,criterion):
    assert criterion.mode=='balanced'
    g=output[0][mask][:,0].float();d=default[mask][:,2];q=quality[mask].detach().float()
    w1=.5/criterion.state_prior;w0=.5/(1-criterion.state_prior);shift=torch.log(w1/w0)
    bce=torch.where(d,w1,w0)*F.binary_cross_entropy_with_logits(g+shift,d.float(),reduction='none')
    state_loss=(q*bce).mean()/3
    gs=torch.autograd.grad(state_loss,shared,retain_graph=True,allow_unused=True)
    gd=torch.autograd.grad(loss-state_loss,shared,retain_graph=True,allow_unused=True)
    gs=[torch.zeros_like(p) if a is None else a for p,a in zip(shared,gs)]
    gd=[torch.zeros_like(p) if b is None else b for p,b in zip(shared,gd)]
    dot=sum((a.float()*b.float()).sum() for a,b in zip(gs,gd))
    nd=sum(b.float().square().sum() for b in gd);ns=sum(a.float().square().sum() for a in gs)
    assert bool(torch.isfinite(dot)&torch.isfinite(nd)&torch.isfinite(ns))
    coefficient=torch.where((dot<0)&(nd>1e-12),dot/nd.clamp_min(1e-12),torch.zeros_like(dot))
    # Private state/density heads keep the original summed-loss gradients.
    loss.backward()
    for p,a,b in zip(shared,gs,gd):p.grad=a-coefficient*b+b
    return float(dot/torch.sqrt((ns*nd).clamp_min(1e-24))),float(coefficient<0)

def make_model(stage,master,metadata):return Network(master,metadata,stage)
