"""Allow 32 patch tokens for 512-point inputs; keep all other attention widths."""
from torch import nn
from torch.nn import functional as F
from patch_hierarchy_sequence import Network as Baseline


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        attention=self.stem.hierarchy.global_patches
        attention.maximum_tokens=32
        attention.relative_bias=nn.Parameter(F.pad(attention.relative_bias.detach(),(16,16)))
