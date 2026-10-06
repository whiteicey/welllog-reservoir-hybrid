"""Whole private SW feature-body adaptation with matched original-weight anchor."""
from backbone_norm_v39 import Network as DifferentiableParent,ROOT,PREVIOUS,load_data,reference_prediction,Evaluator
from anchored_backbone_v42 import original_anchor

STUDY=ROOT/'results/deep_learning/anchored_features_v43_20260929'
MODES=('free','anchored')

class Network(DifferentiableParent):
    def __init__(self,target,master,mode):
        if mode not in MODES:raise ValueError(mode)
        super().__init__(target,master,'all_norm')
        self.requires_grad_(False);self.mode=mode
        selected=[]
        for e,member in enumerate(self.current.source.members):
            for name,p in member.named_parameters():
                if name.startswith(('main_head.','aux_mid.','aux_coarse.')):continue
                p.requires_grad_(True);selected.append(f'current.source.members.{e}.{name}')
        self.trainable_names=set(selected)
        assert self.trainable_names=={n for n,p in self.named_parameters() if p.requires_grad}
