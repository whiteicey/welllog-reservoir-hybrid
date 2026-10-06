"""Matched last-residual-block adaptation, with or without original-weight anchor."""
import torch
from backbone_norm_v39 import Network as DifferentiableParent,ROOT,PREVIOUS,load_data,reference_prediction,Evaluator

STUDY=ROOT/'results/deep_learning/anchored_backbone_v42_20260929'
MODES=('free','anchored')

class Network(DifferentiableParent):
    def __init__(self,target,master,mode):
        if mode not in MODES:raise ValueError(mode)
        super().__init__(target,master,'all_norm')
        self.requires_grad_(False)
        self.mode=mode
        selected=[]
        for e,member in enumerate(self.current.source.members):
            for name,p in member.dec0.named_parameters():
                p.requires_grad_(True);selected.append(f'current.source.members.{e}.dec0.{name}')
        self.trainable_names=set(selected)
        assert self.trainable_names=={n for n,p in self.named_parameters() if p.requires_grad}

def original_anchor(model,origin):
    # Each tensor contributes equally after normalization by its initial RMS.
    # Floor .01 prevents near-zero biases/gains creating unbounded penalties.
    terms=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        start=origin[name]
        terms.append((p.float()-start).square().mean()/start.square().mean().clamp_min(1e-4))
    return torch.stack(terms).mean()
