"""Transfer a locally positive numeric curve interaction into PERM sparse body."""
import torch
from perm_sparse_reacquisition_v291 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
from por_curve_scope_v302 import CurveInteraction
STUDY=ROOT/'results/deep_learning/perm_curve_scope_v307_20261002'
OUT=ROOT/'outputs/pure_dl_perm_curve_scope_v307_fold5'
STAGES={a:('PERM',256,False,0.,3e-4) for a in ('grouped','full')}

class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'point');self.interaction_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261302);self.curve_interaction=CurveInteraction(mode)
        self.current_curve_correction=None
        self.stem.register_forward_hook(self.add_curve_correction)
    def add_curve_correction(self,module,args,result):
        assert self.current_curve_correction is not None
        return result+self.current_curve_correction.to(result.dtype)
    def groups(self):
        return super().groups()+[dict(params=list(self.curve_interaction.parameters()),lr=3e-4,name='PERM_curve_interaction_backbone')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(primary_control='grouped/full nine-token connectivity; both sparse selectors remain pointwise',curve_interaction_scope=self.interaction_mode,curve_interaction_parameters=19024,new_active_parameters=55490,only_original154_prior=True,no291_or302_trained_weights=True,one_PERM_added_phase=True,numeric_token_width=32,attention_blocks=2,attention_heads=4,zero_stem_bridge=True,registered_and_active_capacity_same_pair=True,curve_groups=[['GR','PE','SP'],['AC','DEN','CNL'],['CAL','RXO','RT']],matched_point_window_initialization=False)
        return info
    def forward(self,x,*args,**kwargs):
        self.current_curve_correction=self.curve_interaction(x.index_select(-1,self.curve_indices))
        try:return super().forward(x,*args,**kwargs)
        finally:self.current_curve_correction=None

def make_model(stage,master,metadata):return Network(master,metadata,stage)
