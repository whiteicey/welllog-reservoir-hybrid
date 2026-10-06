"""Keep the RQS family; match likelihood versus likelihood plus quantile scores."""
import hashlib
import torch
from torch.nn import functional as F
from quantile_density_v48 import ROOT,Network as Parent,Evaluator,load_data,reference_prediction,knots,cdf_logpdf,TAIL,K

STUDY=ROOT/'results/deep_learning/quantile_supervision_v49_20260929'
PREVIOUS=ROOT/'results/deep_learning/quantile_density_v48_20260929'
PARENT=ROOT/'outputs/pure_dl_quantile_density_v48_SW_rqs_fold5/model_best.pt'
PARENT_PRED=PARENT.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('likelihood','quantile_aux')}

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('rqs',metadata)
        saved=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True)
        # Inference interpolation is RQS in both arms; study_mode only selects loss.
        self.mode='rqs';self.study_mode=mode
        self.parent_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            best_epoch=saved['best_epoch'],target_score=saved['target_score'],optimizer_reset=True,
            initialization='v48 trained RQS epoch4 exact full model; no empirical reinitialization or new module')

    def initialize_priors(self,data):return self.parent_info

def channel_objective(output,truth,default,mask,quality,target_index,mode):
    gate,raw=output;d=default[mask][:,target_index];g=gate[mask][:,0].float()
    r=raw[mask].float();y=truth[mask][:,target_index].masked_fill(d,0)
    _,logpdf=cdf_logpdf(r,y[:,None],'rqs')
    nll=-logpdf[:,0].masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    quantiles=knots(r);prob=torch.linspace(TAIL,1-TAIL,K+1,device=r.device)
    error=y[:,None]-quantiles
    # Twice the equally weighted finite-quantile pinball mean; NOT exact CRPS.
    pinball=(2*torch.maximum(prob*error,(prob-1)*error).mean(-1)).masked_fill(d,0)
    weight=quality[mask].detach().float();coef=.5 if mode=='quantile_aux' else 0.
    return (weight*(bce+nll+coef*pinball)).mean()/3,(weight*bce).mean(),(weight*nll).mean(),(weight*pinball).mean()
