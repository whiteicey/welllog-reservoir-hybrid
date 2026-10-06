"""PERM gate/density gradient interaction on a retained smooth-CDF predictor."""
import hashlib
import numpy as np
import torch
from perm_quantile_v67 import ROOT,Network as Parent,Evaluator,load_data
from quantile_density_v48 import channel_objective
STUDY=ROOT/'results/deep_learning/perm_gradient_v68_20260929'
PREVIOUS=ROOT/'results/deep_learning/perm_quantile_v67_20260929'
SOURCE=ROOT/'outputs/pure_dl_perm_quantile_v67_PERM_rqs_fold5/model_best.pt'
PARENT_PRED=SOURCE.with_name('channel_pred.npy')
MODES={m:dict(target='PERM',c=1,length=1024) for m in ('sum','project')}
def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')
class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('rqs',metadata)
        saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.load_state_dict(saved['state_dict'],strict=True)
        self.study_mode=mode;self.mode='rqs'
        self.parent_info=dict(checkpoint=str(SOURCE.relative_to(ROOT)),sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact retained v67 RQS epoch24 function; no density reset, same complete network')
    def initialize_priors(self,data):return self.parent_info

def assign_gradients(model,bce,nll,mode):
    named=list(model.named_parameters());params=[p for _,p in named]
    ga=torch.autograd.grad(bce/3,params,retain_graph=True,allow_unused=True)
    gb=torch.autograd.grad(nll/3,params,allow_unused=True)
    shared=[i for i,(n,_) in enumerate(named) if not n.startswith(('gate.','output.'))]
    active=[i for i in shared if ga[i] is not None and gb[i] is not None]
    dot=sum((ga[i].float()*gb[i].float()).sum() for i in active)
    na=sum(ga[i].float().square().sum() for i in active);nb=sum(gb[i].float().square().sum() for i in active)
    conflict=bool(dot<0);project=mode=='project' and conflict;epsilon=1e-12
    alpha=dot/(nb+epsilon) if project else dot.new_zeros(())
    beta=dot/(na+epsilon) if project else dot.new_zeros(())
    sharedset=set(active)
    for i,(_,param) in enumerate(named):
        a,b=ga[i],gb[i]
        if a is None and b is None:param.grad=None;continue
        if a is None:g=b
        elif b is None:g=a
        elif i in sharedset:g=(a-alpha*b)+(b-beta*a)
        else:g=a+b
        param.grad=g.detach()
    return dict(conflict=conflict,projected=project,shared_gradient_cosine=float(dot/(na*nb).clamp_min(epsilon).sqrt()),
        classification_gradient_norm=float(na.sqrt()),density_gradient_norm=float(nb.sqrt()),projection_alpha=float(alpha),projection_beta=float(beta))
