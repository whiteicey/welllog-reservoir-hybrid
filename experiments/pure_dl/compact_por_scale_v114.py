"""Warm refinement of fixed mixed-family compact POR: position versus width scope."""
import torch
from torch import nn
from compact_por_mixed_family_v113 import ROOT,SOURCE,Network as Parent,Evaluator as ParentEvaluator,channel_objective as parent_loss,load_data
from compact_mean_scope_v101 import LocationHead
STUDY=ROOT/'results/deep_learning/compact_por_scale_v114_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_scale_v114_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt'
STAGES={n:('POR',256,False,0.,3e-5) for n in ('location_only','location_scale')}
class ScaleHead(LocationHead):
    def packed(self,stem):
        if stem.startswith('density'):
            fixed=getattr(self,stem);mean=self.mean_weight if stem.endswith('weight') else self.mean_bias
            scale=self.scale_weight if stem.endswith('weight') else self.scale_bias
            own=torch.cat((fixed[:8],mean,scale),0)
            return torch.cat((getattr(self,stem+'_before'),own,getattr(self,stem+'_after')),0)
        return super().packed(stem)
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,'t3_cauchy');self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict']);self.scope=mode
        if mode=='location_scale':
            for h in self.heads:
                h.__class__=ScaleHead
                h.scale_weight=nn.Parameter(h.density_weight[16:24].detach().clone())
                h.scale_bias=nn.Parameter(h.density_bias[16:24].detach().clone())
    def groups(self):
        groups=super().groups()
        for g in groups:g['lr']=3e-5
        if self.scope=='location_scale':
            # Exclude new scale rows from the inherited generic nonlinear group.
            scales=[p for h in self.heads for p in (h.scale_weight,h.scale_bias)]
            ids={id(p) for p in scales}
            for g in groups:g['params']=[p for p in g['params'] if id(p) not in ids]
            groups.append(dict(params=scales,lr=3e-5,name='original_reference_widths'))
        return groups
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),same_initial_predictive_function=True,family_mode='t3_cauchy',scope=self.scope,
            original_density_weights_gate_backbone_frozen=True,width_changes_detached_location_bound_numerically=self.scope=='location_scale',
            no_finite_whole_mixture_mean_variance=True,executed_training_stages_including_source=5)
def make_model(stage,master,metadata):return Network(master,stage)
def channel_objective(output,truth,default,mask,quality,target_index,mode):
    return parent_loss(output,truth,default,mask,quality,target_index,'t3_cauchy')
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length,'t3_cauchy')
