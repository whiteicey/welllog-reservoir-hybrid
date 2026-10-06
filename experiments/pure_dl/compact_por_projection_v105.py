"""POR mean-scope readout plus last source feature projection adaptation.

Starts from original source, not a trained v101 head. This is one replacement
stage two, not an extra sequential stage. Only own POR loss is used.
"""
import torch
from compact_mean_scope_v101 import Network as Parent
from compact_five_stage_v100 import ROOT,SOURCE,load_data as original_data,shift
from paired_detached_router_sequence import routed_merge

STUDY=ROOT/'results/deep_learning/compact_por_projection_v105_20261001'
OUT=ROOT/'outputs/pure_dl_compact_por_projection_v105_fold5'
STAGES={'por':('POR',256,False,0.,3e-5)}

class Network(Parent):
    def __init__(self,master):
        super().__init__(master)
        for member in self.backbone.members:member.combine[0].requires_grad_(True)

    def capture(self,e):
        def hook(module,args):self.captured[e]=args[0]
        return hook

    def forward(self,x,relative,valid,absolute_depth=None,well_inputs=None):
        self.captured.clear()
        # Frozen upstream operations create no gradient tape. The final
        # 64->128 projection is trainable; its captured features keep gradients.
        for member in self.backbone.members:member(x[...,:206].contiguous(),relative,valid,absolute_depth=absolute_depth)
        with torch.no_grad():logits=self.backbone.route_logits(x[...,:206].contiguous(),valid)[:,:,:1]
        hidden=[self.captured[e] for e in range(2)];side=x[...,206:]
        outputs=[head(h,side,valid) for head,h in zip(self.heads,hidden)]
        common,a,b=[route(hidden,side,valid) for route in self.routes]
        gate,_=routed_merge(outputs,shift(logits,a));_,raw=routed_merge(outputs,shift(logits,common+b))
        self.captured.clear();return gate,raw

    def groups(self):
        return super().groups()+[dict(params=[p for m in self.backbone.members for p in m.combine[0].parameters()],lr=3e-6,name='source_final_projection')]

def load_data(stage):return original_data('por')
def make_model(stage,master,metadata):return Network(master)
