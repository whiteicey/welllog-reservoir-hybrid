"""Complete POR private warm start, fixed anchors; likelihood versus action utility."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from torch import nn
from por_reparam_v31 import Network as Parent
from champion_all_heads_v26 import Evaluator as ParentEvaluator,load_data,TARGETS
from por_joint_utility_v86 import DecisionObjective
from train_dual8222_heads80 import SOURCE

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_full_utility_v92_20260930'
PREVIOUS=ROOT/'results/deep_learning/sw_frozen_direct_v91_20260930'
MODES={'likelihood':0.,'joint':1.}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,master):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('POR',master,modes=('anchored',))
        path=ROOT/'outputs/pure_dl_mechanism_v31_POR_fold5/anchored/model_best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        head=self.heads['anchored'];head.load_state_dict(saved['delta_state'],strict=True)
        self.heads=nn.ModuleDict({mode:head});self.study_mode=mode
        for handle in self.handles:handle.remove()
        self.handles=[];self.captured.clear()
        for e,member in enumerate(self.source.members):
            def capture(_module,args,output,e=e):self.captured['h'+str(e)]=args[0]
            self.handles.append(member.main_head.register_forward_hook(capture))
        self.handles.append(self.source.router[2].register_forward_hook(lambda _m,_a,v:self.captured.update(logits=v)))
        self.source.requires_grad_(True);head.requires_grad_(True)
        # Source main-head outputs are superseded by the actual bundle readouts.
        # They still capture hidden features, but are excluded from learning.
        for member in self.source.members:member.main_head.requires_grad_(False)
        for name,p in head.named_parameters():
            if '.anchor_input.' in name or '.anchor_output.' in name:p.requires_grad_(False)
        self.metadata=dict(source_checkpoint=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_epoch=saved['best_epoch'],source_score=saved['target_score'],
            actual_source_checkpoint=self.source_path,actual_source_sha256=hashlib.sha256((ROOT/self.source_path).read_bytes()).hexdigest(),
            added_parameters=0,full_private_backbones_and_input_router=True,actual_bundle_readouts_trainable=True,
            unused_source_readouts_frozen=True,anchor_reference_parameters_frozen=True,old_capture_detach_removed=True,old_no_grad_removed=True,
            other_target_packed_rows='immutable buffers',optimizer_reset=True,initialization='actual v31 anchored best7/v30 bundle/v25-v8 private POR; no prior reset',
            body_lr=1e-5,head_lr=3e-5,eta_min=1e-5,schedule='body constant eta_min; active target readouts/corrections cosine3e-5 to1e-5',
            utility_coefficient=MODES[mode],temperature=.05,utility_chunk=512,gradient_checkpointing=True,
            original_decoder_unchanged=True,target_transform='log1p/inverse expm1',density='Student-t3 std parameterization')
        self.train(False)
    def train(self,mode=True):
        nn.Module.train(self,mode)
        # Bypass v25's historical frozen-body training override in this new route.
        nn.Module.train(self.source,mode)
        return self
    def extract(self,x,relative,valid,absolute):
        self.captured.clear()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            self.source(x,relative,valid,absolute_depth=absolute)
            h=[self.captured['h'+str(e)] for e in range(2)]
            logits=self.captured['logits'].reshape(*x.shape[:2],3,2)[:,:,:1]
        self.captured.clear();return h,None,logits,None
    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        h,o,logits,_=self.extract(x,relative,valid,absolute_depth)
        output=self.heads[self.study_mode](h,o,logits,x[...,206:],valid)
        if factors is not None:
            assert tuple(factors)==(self.study_mode,);return {self.study_mode:output}
        return output
    def parameter_groups(self):
        body=[];heads=[]
        for n,p in self.named_parameters():
            if p.requires_grad:(body if n.startswith('source.') else heads).append(p)
        return [dict(params=body,lr=1e-5,name='private_backbones_and_input_router'),dict(params=heads,lr=3e-5,name='active_target_readouts_and_corrections')]

class Evaluator(ParentEvaluator):
    def __init__(self,data,master,c,reference):
        super().__init__(data,master,c,reference)
        import train_invariant as base
        self.reference_scores=base.metric(data['truth'][data['validation']],reference)
    @torch.no_grad()
    def __call__(self,model):
        pred,score,total=self.many(model,(model.study_mode,))[model.study_mode]
        return pred,score,dict(POR=score,PERM=float(self.reference_scores['PERM']),SW=float(self.reference_scores['SW']),TOTAL=total)
