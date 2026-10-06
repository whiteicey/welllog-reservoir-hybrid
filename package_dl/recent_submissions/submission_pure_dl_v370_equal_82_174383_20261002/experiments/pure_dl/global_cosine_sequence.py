"""Local dot-product attention with global cosine patch attention."""
import torch
from patch_hierarchy_sequence import Network as Baseline
from cosine_patch_sequence import SmallAttention as CosineAttention

class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        old=self.stem.hierarchy.global_patches
        with torch.random.fork_rng(devices=[]):new=CosineAttention()
        result=new.load_state_dict(old.state_dict(),strict=False)
        assert result.missing_keys==['log_scale'] and not result.unexpected_keys
        self.stem.hierarchy.global_patches=new
