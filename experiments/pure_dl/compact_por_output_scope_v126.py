"""POR default-gate versus component-weight update scope from positive v113."""
import torch
from torch import nn
from compact_por_mixed_family_v113 import ROOT,SOURCE,Network as Parent,Evaluator as ParentEvaluator,channel_objective as parent_loss,load_data
from compact_mean_scope_v101 import LocationHead
STUDY=ROOT/'results/deep_learning/compact_por_output_scope_v126_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_output_scope_v126_fold5'
PARENT=ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt'
STAGES={n:('POR',256,False,0.,3e-5) for n in ('gate','weight')}
class WeightHead(LocationHead):
    def packed(self,stem):
        if stem.startswith('density'):
            fixed=getattr(self,stem);mean=self.mean_weight if stem.endswith('weight') else self.mean_bias
            weights=self.mixture_weight if stem.endswith('weight') else self.mixture_bias
            own=torch.cat((weights,mean,fixed[16:]),0)
            return torch.cat((getattr(self,stem+'_before'),own,getattr(self,stem+'_after')),0)
        return super().packed(stem)
class Network(Parent):
    def __init__(self,master,mode):
        super().__init__(master,'t3_cauchy');self.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=False)['state_dict']);self.scope=mode
        for h in self.heads:
            if mode=='gate':
                h.gate_weight.requires_grad_(True);h.gate_bias.requires_grad_(True)
            elif mode=='weight':
                h.__class__=WeightHead
                h.mixture_weight=nn.Parameter(h.density_weight[:8].detach().clone())
                h.mixture_bias=nn.Parameter(h.density_bias[:8].detach().clone())
            else:raise ValueError(mode)
    def groups(self):
        groups=super().groups()
        for g in groups:g['lr']=3e-5
        if self.scope=='gate':extra=[p for h in self.heads for p in (h.gate_weight,h.gate_bias)]
        else:extra=[p for h in self.heads for p in (h.mixture_weight,h.mixture_bias)]
        ids={id(p) for p in extra}
        for g in groups:g['params']=[p for p in g['params'] if id(p) not in ids]
        groups.append(dict(params=extra,lr=3e-5,name='POR_default_gate' if self.scope=='gate' else 'POR_component_weights'))
        return groups
    def initialize_priors(self,data):
        return dict(parent_checkpoint=str(PARENT),same_initial_predictive_function=True,family_mode='t3_cauchy',scope=self.scope,
            original_backbones_reference_widths_frozen=True,no_finite_whole_mixture_mean_variance=True,
            original_coefficient_scope_changed=True,executed_training_stages_including_source=5)
def make_model(stage,master,metadata):return Network(master,stage)
def channel_objective(output,truth,default,mask,quality,target_index,mode):
    return parent_loss(output,truth,default,mask,quality,target_index,'t3_cauchy')
class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,length,mode):super().__init__(data,master,c,length,'t3_cauchy')
