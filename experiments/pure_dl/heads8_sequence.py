"""Eight attention heads at fixed width32 on the original global-cosine backbone."""
import math
import torch
from torch import nn
from global_cosine_sequence import Network as Baseline


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        hierarchy = self.stem.hierarchy
        for attention in (hierarchy.local, hierarchy.global_patches):
            attention.heads = 8
            attention.relative_bias = nn.Parameter(
                attention.relative_bias.new_zeros((8, 2 * attention.maximum_tokens - 1)))
        hierarchy.global_patches.log_scale = nn.Parameter(
            hierarchy.global_patches.log_scale.new_full((8,), math.log(math.sqrt(32 / 8))))
        # Existing qkv/project/norm tensors are untouched. Constant replacements
        # consume no random numbers; only the head partition and associated bias/
        # cosine-temperature shapes change.
