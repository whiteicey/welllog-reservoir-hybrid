"""Small true-label default-bin auxiliary regularizer in fresh PERM encoder."""
import torch
from torch.nn import functional as F
from perm_curve_scope_v307 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,Network as Parent,channel_objective as original_objective
STUDY=ROOT/'results/deep_learning/perm_default_anchor_v309_20261002'
OUT=ROOT/'outputs/pure_dl_perm_default_anchor_v309_fold5'
STAGES={'anchor':('PERM',256,False,0.,3e-4)}

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'full');self.last_default_aux_ce=0.
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        ix=data['train_raw'];z=data['labels'][ix,1];d=data['default'][ix,1]
        values=z[d]
        bins,counts=torch.bucketize(values.contiguous(),self.edges[1:-1]).unique(return_counts=True)
        canonical_bin=torch.bucketize(torch.tensor(float(self.base.default_z),device=z.device),self.edges[1:-1])
        info.update(default_auxiliary_CE_coefficient=.05,default_target_source='actual training DEFAULT labels, not prior/teacher',default_train_rows=int(d.sum()),default_labels_outside_density_edges=float(((values<self.edges[0])|(values>self.edges[-1])).float().mean()),default_train_bins=bins.cpu().tolist(),default_train_bin_counts=counts.cpu().tolist(),first_finite_density_boundary=float(self.edges[1]),canonical_default_bin=int(canonical_bin),outer_density_edges_infinite=bool(torch.isinf(self.edges[[0,-1]]).all()),default_target_bins_use_actual_z=True,conditional_NLL_unchanged=True,joint_RPS_already_supervises_DEFAULT=True,not_proper_conditional_likelihood_correction=True,new_active_parameters=55490,no307_trained_weights=True,only_original154_checkpoint=True,reference='completed307full lambda0, same init/forward/config; no concurrent replay')
        return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)

def channel_objective(output,labels,default,mask,quality,c,mode,model,well_ids):
    assert c==1 and model.training and model.aux_logits is not None
    # Keep the live final-q graph before original objective clears its pointer.
    logits=model.aux_logits[-1][mask].float()
    original=original_objective(output,labels,default,mask,quality,c,mode,model,well_ids)
    d=default[mask][:,1];z=labels[mask][:,1].float();q=quality[mask].detach().float()
    bins=torch.bucketize(z.contiguous(),model.edges[1:-1]);weights=q*d.float()
    aux=(weights*F.cross_entropy(logits,bins,reduction='none')).sum()/weights.sum().clamp_min(1e-12)
    model.last_default_aux_ce=float(aux.detach())
    return original[0]+.05*aux*q.mean()/3,original[1],original[2]
