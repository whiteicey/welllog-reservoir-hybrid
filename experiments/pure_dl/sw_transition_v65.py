"""Matched SW transition supervision with detached versus live representation gradients."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sw_interaction_v56 import ROOT,Network as Parent,Evaluator,load_data
STUDY=ROOT/'results/deep_learning/sw_transition_v65_20260929'
PREVIOUS=ROOT/'results/deep_learning/por_interval_v64_20260929'
SOURCE=ROOT/'outputs/pure_dl_sw_interaction_v56_SW_additive_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('detached','joint')}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('additive',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261165);self.transition=nn.Sequential(nn.Linear(256,32),nn.GELU(),nn.Linear(32,1))
        self._capture=False;self._features=None
        self._handle=self.norm.register_forward_hook(self.capture)
        self.initialization_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact v56 additive predictor; training-only transition head, same random tensors',
            new_parameters=sum(p.numel() for p in self.transition.parameters()),auxiliary_backbone_gradient=mode=='joint')
    def capture(self,module,args,value):
        if self._capture:self._features=value
    def initialize_priors(self,data):return self.initialization_info
    def core_hash(self):
        return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for n,v in self.state_dict().items() if not n.startswith('transition.'))).hexdigest()
    def forward(self,x,relative,valid,absolute_depth=None,return_aux=False,generator=None,well_inputs=None):
        self._capture=return_aux;self._features=None
        result=super().forward(x,relative,valid,absolute_depth=absolute_depth,return_aux=False,well_inputs=well_inputs)
        h=self._features;self._features=None;self._capture=False
        return (result,h,None) if return_aux else result
    def transition_loss(self,h,default,mask,quality,indices):
        middle=h.shape[1]//2;sl=slice(middle-64,middle+64)
        z=h[:,sl];d=default[:,sl,2];m=mask[:,sl];q=quality[:,sl];ix=indices[:,sl]
        if self.study_mode=='detached':z=z.detach()
        keep=m[:,1:]&m[:,:-1]&(ix[:,1:]==ix[:,:-1]+1)
        label=(d[:,1:]!=d[:,:-1])[keep].float()
        pair=torch.cat(((z[:,1:]+z[:,:-1])*.5,(z[:,1:]-z[:,:-1]).abs()),-1)[keep]
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=self.transition(pair).squeeze(-1).float()
        losses=F.binary_cross_entropy_with_logits(logits,label,reduction='none')
        losses=losses*((q[:,1:]+q[:,:-1])*.5)[keep].detach()
        positive=label.bool();terms=[losses[sel].mean() for sel in (positive,~positive) if bool(sel.any())]
        result=torch.stack(terms).mean()/3 if terms else logits.sum()*0
        return result,dict(pairs=int(keep.sum()),transitions=int(positive.sum()),balanced_classes=len(terms),representation_gradient=self.study_mode=='joint')
