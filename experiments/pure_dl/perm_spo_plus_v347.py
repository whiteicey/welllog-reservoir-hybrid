"""One final-mixture SPO+ loss on fresh positive joint PERM encoder/readout."""
import numpy as np
import torch
from perm_low_rank_cross_v336 import ROOT,SOURCE,PARENT,PARENT_SHA,Evaluator,Network as Parent,channel_objective as original_objective,load_data as original_data
STUDY=ROOT/'results/deep_learning/perm_spo_plus_v347_20261002'
OUT=ROOT/'outputs/pure_dl_perm_spo_plus_v347_fold5'
STAGES={s:('PERM',256,False,0.,3e-4) for s in ('reference','spo')}
def load_data(stage):
 data,master,metadata=original_data(stage);ix=data['train_raw'];rows=ix.detach().cpu().numpy();truth=np.asarray(data['truth'])[rows,1]
 # Never transfer heldout truth to this training-only supervision buffer.
 data['spo_truth']=torch.full((len(data['truth']),),float('nan'),device=ix.device,dtype=torch.float32)
 data['spo_truth'][ix]=torch.as_tensor(truth,device=ix.device,dtype=torch.float32)
 metadata.update(spo_train_truth_rows=len(rows),spo_actual_physical_training_truth_only=True,spo_other_rows_NaN=True)
 return data,master,metadata
class Network(Parent):
 def __init__(self,master,metadata,mode):
  super().__init__(master,metadata,'cross');self.spo_mode=mode;self.spo_coefficient=.1 if mode=='spo' else 0.;self.last_spo_loss=0.;self.last_loss_augmented_disagreement=0.
 def configure_decision(self,evaluator):
  self.register_buffer('decision_utilities',evaluator.utilities.detach().clone())
  self.register_buffer('decision_log_actions',torch.as_tensor(np.log10(np.asarray(evaluator.actions).clip(1e-5)),device=self.edges.device,dtype=torch.float32))
  return dict(spo_coefficient=self.spo_coefficient,decision_actions=len(evaluator.actions),decision_support=self.decision_utilities.shape[0],decision_kernel_from_original_evaluator=True,no_decoder_changes=True,actual_own_train_physical_truth=True,DEFAULT_not_canonicalized_for_utility=True,chunk_size=1024,nonproper_surrogate_not_theoretical_gain_guarantee=True)
 def initialize_priors(self,data):
  info=super().initialize_priors(data)
  info.update(original154_only=True,no336_or346trained_weights=True,all_new_encoder_shape_cross_readout_joint=True,one_PERM_added_phase=True,no_new_parameters=True,active_parameters=37386,primary_control='fresh336 reference loss vs samecapacity/init loss plus.1 SPO+ finalmixture',actual_train_truth_rows=len(data['train_raw']),training_other_truth_rows_NaN=True)
  return info

def make_model(stage,master,metadata):return Network(master,metadata,stage)
def channel_objective(output,labels,default,mask,quality,target_index,mode,model,well_ids=None,physical_truth=None):
 original=original_objective(output,labels,default,mask,quality,target_index,mode,model,well_ids=well_ids);assert target_index==1 and physical_truth is not None
 with torch.autocast(device_type=output[0].device.type,enabled=False):
  g=output[0][mask][:,0].float().sigmoid();lp=output[1][mask][:,0].float();q=quality[mask].detach().float();y=physical_truth[mask].detach().float();assert bool(torch.isfinite(y).all()),'Training utility accessed nontraining/unlabeled row'
  log_y=y.clamp_min(1e-5).log10();parts=[];disagreements=[]
  for begin in range(0,len(g),1024):
   take=slice(begin,begin+1024);gate=g[take,None];P=torch.cat((gate,(1-gate)*lp[take].exp()),-1);scores=P@model.decision_utilities
   utility=(1-(log_y[take,None]-model.decision_log_actions[None]).abs()).clamp_min(0);astar=utility.argmax(-1)
   augmented=2*scores-utility;worst=augmented.max(-1);ell=worst.values-2*scores.gather(1,astar[:,None]).squeeze(1)+utility.gather(1,astar[:,None]).squeeze(1)
   parts.append(q[take]*ell);disagreements.append((worst.indices!=scores.argmax(-1)).float())
  loss=torch.cat(parts).mean()/3;model.last_spo_loss=float(loss.detach());model.last_loss_augmented_disagreement=float(torch.cat(disagreements).mean())
 return original[0]+model.spo_coefficient*loss,original[1],original[2]
