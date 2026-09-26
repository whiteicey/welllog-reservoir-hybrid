"""Fixed equal pair with different local attention metrics; no pretrained weights."""
from torch import nn
from paired_backbone_sequence import Network as PairedNetwork
from global_cosine_sequence import Network as DotLocalMember
from cosine_patch_sequence import Network as CosineLocalMember


class Network(PairedNetwork):
    def __init__(self, features=206, initial_means=None):
        nn.Module.__init__(self)
        self.members = nn.ModuleList([DotLocalMember(features, initial_means),
                                      CosineLocalMember(features, initial_means)])
