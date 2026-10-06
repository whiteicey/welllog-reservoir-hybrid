"""Fresh joint POR raw backbone/readout plus grouped/full curve interaction."""
import torch
from torch import nn
from por_direct_share_v290 import ROOT,SOURCE,PARENT,PARENT_SHA,load_data,Evaluator,channel_objective,Network as Parent
STUDY=ROOT/'results/deep_learning/por_curve_scope_v302_20261002'
OUT=ROOT/'outputs/pure_dl_por_curve_scope_v302_fold5'
STAGES={a:('POR',256,False,0.,3e-4) for a in ('grouped','full')}
class Block(nn.Module):
    def __init__(self):
        super().__init__();self.n1=nn.LayerNorm(32);self.attention=nn.MultiheadAttention(32,4,batch_first=True,dropout=0.)
        self.n2=nn.LayerNorm(32);self.ff=nn.Sequential(nn.Linear(32,64),nn.GELU(),nn.Linear(64,32))
    def forward(self,h,mask):
        z=self.n1(h);h=h+self.attention(z,z,z,attn_mask=mask,need_weights=False)[0]
        return h+self.ff(self.n2(h))
class CurveInteraction(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.numeric=nn.Linear(1,32);self.identity=nn.Parameter(torch.randn(9,32)*.02)
        self.blocks=nn.ModuleList([Block(),Block()]);self.bridge=nn.Linear(32,48)
        nn.init.zeros_(self.bridge.weight);nn.init.zeros_(self.bridge.bias)
        # [GR,PE,SP], [AC,DEN,CNL], [CAL,RXO,RT]; input-defined, never label-fitted.
        groups=torch.tensor([0,0,0,2,1,1,1,2,2])
        self.register_buffer('group_mask',groups[:,None]!=groups[None,:])
    def forward(self,curves):
        shape=curves.shape[:2];tokens=self.numeric(curves.reshape(-1,9,1))+self.identity[None]
        mask=self.group_mask if self.mode=='grouped' else None
        for block in self.blocks:tokens=block(tokens,mask)
        return self.bridge(tokens.mean(1)).reshape(*shape,48)
class Network(Parent):
    def __init__(self,master,metadata,mode):
        super().__init__(master,metadata,'raw');self.interaction_mode=mode
        names=list(master['selected_feature_names']);curves=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')
        indices=[names.index(c) for c in curves];assert len(set(indices))==9 and max(indices)<206
        self.register_buffer('curve_indices',torch.tensor(indices,dtype=torch.long))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261302);self.curve_interaction=CurveInteraction(mode)
        self.current_curve_correction=None;self.stem.register_forward_hook(self.add_curve_correction)
    def add_curve_correction(self,module,args,result):
        assert self.current_curve_correction is not None
        return result+self.current_curve_correction.to(result.dtype)
    def groups(self):return super().groups()+[dict(params=list(self.curve_interaction.parameters()),lr=3e-4,name='POR_curve_interaction_backbone')]
    def initialize_priors(self,data):
        info=super().initialize_priors(data)
        info.update(curve_interaction_scope=self.interaction_mode,curve_indices=self.curve_indices.tolist(),curve_groups=[['GR','PE','SP'],['AC','DEN','CNL'],['CAL','RXO','RT']],curve_tokens=9,token_width=32,attention_blocks=2,heads=4,FFN=64,zero_output_bridge=True,new_interaction_parameters=19024,new_active_parameters=76592,registered_and_active_capacity_same=True,diagonal_only_control_rejected_due_zero_QK_gradients=True,no290_or_old_curve_attention_trained_weights_loaded=True,not_first_curve_token_attention=True,only_scope_differs_in_matched_pair=True)
        return info
    def forward(self,x,*args,**kwargs):
        self.current_curve_correction=self.curve_interaction(x.index_select(-1,self.curve_indices))
        try:return super().forward(x,*args,**kwargs)
        finally:self.current_curve_correction=None
def make_model(stage,master,metadata):return Network(master,metadata,stage)
