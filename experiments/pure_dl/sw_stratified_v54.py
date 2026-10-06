"""Matched window-risk training: uniform versus state-stratified anchor draws."""
import hashlib,math
import numpy as np
import torch
from torch.nn import functional as F
from well_conditioned_v46 import ROOT,Network as Parent,Evaluator,load_data

STUDY=ROOT/'results/deep_learning/sw_stratified_v54_20260929'
PREVIOUS=ROOT/'results/deep_learning/por_transfer_v53_20260929'
PARENT=ROOT/'outputs/pure_dl_well_conditioned_v46_SW_well_fold5/model_best.pt'
PARENT_PRED=PARENT.with_name('channel_pred.npy')
MODES={m:dict(target='SW',c=2,length=1024) for m in ('uniform','stratified')}

def reference_prediction():return np.load(PREVIOUS/'assembled_pred.npy')

class Network(Parent):
    def __init__(self,mode,metadata):
        if mode not in MODES:raise ValueError(mode)
        super().__init__('well',metadata)
        saved=torch.load(PARENT,map_location='cpu',weights_only=False)
        self.load_state_dict(saved['state_dict'],strict=True);self.study_mode=mode
        self.initialization_info=dict(checkpoint=str(PARENT.relative_to(ROOT)),
            sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),best_epoch=saved['best_epoch'],
            target_score=saved['target_score'],optimizer_reset=True,
            initialization='exact v46 SW private predictor, all parameters trainable; no output reset')
    def initialize_priors(self,data):return self.initialization_info

class AnchorSampler:
    def __init__(self,data):
        self.all=data['train_raw'];state=data['default'][self.all,2].bool()
        self.pools=(self.all[~state],self.all[state]);n=len(self.all)
        assert all(len(p)>0 for p in self.pools)
        self.weights=torch.tensor([2*len(p)/n for p in self.pools],device='cuda')
        self.manifest=dict(training_anchors=n,nondefault_anchors=len(self.pools[0]),default_anchors=len(self.pools[1]),
            stratum_importance_weights=self.weights.cpu().tolist(),state_scope='own SW training labels only',
            pool_row_sha256=[hashlib.sha256(p.cpu().numpy().tobytes()).hexdigest() for p in self.pools],
            risk='mean over uniform training anchors of within-window mean Q-weighted SW BCE+Gaussian NLL divided by3',
            difference_from_historical_loss='both arms use mean of window means instead of batch-wide valid-point ratio',
            correction='32 draws per state; per-anchor multiplier 2*N_state/N, no batch renormalization')
    def sample(self,mode,generator):
        if mode=='uniform':
            anchors=self.all[torch.randint(len(self.all),(64,),device='cuda',generator=generator)]
            weights=torch.ones(64,device='cuda')
        else:
            anchors=torch.cat([p[torch.randint(len(p),(32,),device='cuda',generator=generator)] for p in self.pools])
            weights=self.weights.repeat_interleave(32)
        return anchors,weights

def window_objective(output,truth,default,mask,quality,c,importance):
    assert c==2
    gate,raw=output;g=gate[mask][:,0].float();r=raw[mask][:,0].float()
    d=default[mask][:,c];y=truth[mask][:,c].masked_fill(d,0)
    sd=F.softplus(r[:,2])+.025;mu=r[:,1]
    lp=-.5*((y[:,None]-mu)/sd).square()-sd.log()-.5*math.log(2*math.pi)
    nll=-(r[:,0].log_softmax(-1)+lp).logsumexp(-1).masked_fill(d,0)
    bce=F.binary_cross_entropy_with_logits(g,d.float(),reduction='none')
    point=quality[mask].detach().float()*(bce+nll)/3
    row=mask.nonzero(as_tuple=True)[0];counts=mask.sum(1)
    assert bool((counts>0).all())
    per_window=point.new_zeros(mask.shape[0]).scatter_add_(0,row,point)/counts
    return (per_window*importance).mean(),per_window.detach()
