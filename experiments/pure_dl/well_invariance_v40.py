"""Matched train-well discriminator: blocked versus reversed backbone gradients.

Only existing SW backbone normalization affine parameters change at inference.
The well classifier is training-only and never receives held-out wells.
"""
import numpy as np
import torch
from torch import nn
from backbone_norm_v39 import Network as Parent,ROOT,PREVIOUS,load_data,reference_prediction,Evaluator
from well_adversarial_sequence import GradientReverse,pool_core,make_domain_mapping

STUDY=ROOT/'results/deep_learning/well_invariance_v40_20260929'
MODES=('blocked','reversed')

class Network(Parent):
    def __init__(self,target,master,mode):
        if mode not in MODES:raise ValueError(mode)
        super().__init__(target,master,'all_norm')
        self.mode=mode;self.normalization_names=set(self.trainable_names)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261140)
            self.domain_head=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,64))
        self.trainable_names.update('domain_head.'+n for n,_ in self.domain_head.named_parameters())
        self.domain_features=[];self.domain_valid=None
        for member in self.current.source.members:
            def capture(_module,args,_output):
                if self.training:
                    pooled,usable=pool_core(args[0],self.domain_valid)
                    assert bool(usable.all())
                    self.domain_features.append(pooled)
            self.handles.append(member.main_head.register_forward_hook(capture))

    def forward(self,x,relative,valid,absolute_depth=None,factors=None):
        self.domain_features=[];self.domain_valid=valid
        try:return super().forward(x,relative,valid,absolute_depth,factors)
        finally:self.domain_valid=None

    def domain_logits(self,coefficient):
        assert self.training and len(self.domain_features)==2
        features=torch.cat(self.domain_features,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.domain_head(GradientReverse.apply(features,coefficient)).float()

    def optimizer_groups(self):
        normal=[p for n,p in self.named_parameters() if n in self.normalization_names]
        return [dict(params=normal,lr=1e-4,weight_decay=0.,name='backbone_norm'),
                dict(params=list(self.domain_head.parameters()),lr=3e-4,weight_decay=.01,name='training_only_discriminator')]

def coefficient(mode,epoch):
    # Same first three epochs; fixed ramp to full reversal at epoch8.
    return 0. if mode=='blocked' else max(0.,min(1.,(epoch-3)/5.))

def prepare_domains(data):
    raw_well=np.repeat(np.arange(len(data['starts'])-1),np.diff(data['starts']))
    train_raw=data['train_raw'].cpu().numpy()
    train_wells=np.unique(raw_well[train_raw]);mapping=make_domain_mapping(train_wells)
    labels=mapping[torch.as_tensor(raw_well)]
    counts=torch.bincount(labels[torch.as_tensor(train_raw)],minlength=64)
    validation_raw=np.flatnonzero(np.isin(data['raw_to_valid'],data['validation']))
    assert len(train_wells)==64 and bool((labels[validation_raw]==-1).all())
    return labels.cuda(),dict(training_wells=train_wells.tolist(),class_count=64,
        class_counts=counts.tolist(),majority_prior_accuracy=float(counts.max()/counts.sum()),
        validation_wells_not_mapped=True,validation_raw_points_checked=len(validation_raw),
        well_id_is_prediction_input=False,domain_loss_training_anchors_only=True)
