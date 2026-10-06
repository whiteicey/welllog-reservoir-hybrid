"""Existing joint proxy, own-well versus train-reference background."""
import torch
from torch import nn
from sw_raw_well_condition_v297 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,DecisionObjective,Network as Parent
STUDY=ROOT/'results/deep_learning/sw_proxy_background_v313_20261002'
OUT=ROOT/'outputs/pure_dl_sw_proxy_background_v313_fold5'
STAGES={s:('SW',1024,False,0.,3e-4) for s in ('well','reference')}

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'well');self.background_mode=mode
        self.proxy_index=list(master['selected_feature_names']).index('ARCHIE_PROXY')
        assert self.proxy_index<206
        old=self.stem[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261313);expanded=nn.Linear(208,48)
        with torch.no_grad():
            expanded.weight[:,:207].copy_(old.weight);expanded.weight[:,207:].zero_();expanded.bias.copy_(old.bias)
        self.stem[0]=expanded
        self.register_buffer('proxy_reference',torch.tensor(0.))
        self.current_background=None;self.stem.register_forward_pre_hook(self.append_proxy)
    def append_proxy(self,module,args):
        if self.current_background is None:raise RuntimeError('Explicit input-only proxy background required')
        z=args[0];coordinate=(z[...,self.proxy_index:self.proxy_index+1].float()-self.current_background).clamp(-8,8)
        return (torch.cat((z,coordinate.to(z.dtype)),-1),)
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        ids=data['row_well'][data['train_raw']].unique();assert len(ids)==64
        backgrounds=torch.quantile(data['well_bank'][ids,:,self.proxy_index].float(),.2,dim=1,interpolation='linear')
        self.proxy_reference.copy_(backgrounds.mean())
        active=sum(p.numel() for p in self.parameters() if p.requires_grad);assert active==65313
        info.update(background_mode=self.background_mode,proxy_index=self.proxy_index,proxy_quantile=.2,
                    proxy_reference=float(self.proxy_reference),reference_fitted_from64training_input_banks_only=True,
                    background_uses_labels=False,new_raw_stem_input_dimensions=208,original206_and_relative_preserved=True,
                    added_derived_coordinates=1,new_column_parameters=48,new_active_parameters=active,
                    extra_column_zero_initialized=True,no297_or310_or311_weights_loaded=True,
                    only_original104_trained_prior=True,one_SW_additional_phase=True,
                    not_Rw_or_waterline_estimate=True,not_new_Archie_formula=True)
        return info
    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        if well_inputs is None:raise ValueError('Same-well input bank required')
        if self.background_mode=='well':
            self.current_background=torch.quantile(well_inputs[...,self.proxy_index].float(),.2,dim=1,interpolation='linear')[:,None,None]
        else:self.current_background=self.proxy_reference
        try:return super().forward(x,relative,valid,absolute_depth=absolute_depth,well_inputs=well_inputs)
        finally:self.current_background=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
