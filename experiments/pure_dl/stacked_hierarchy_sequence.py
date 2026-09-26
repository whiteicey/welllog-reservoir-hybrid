"""Two sequential local-dot/global-cosine patch hierarchies at the stem."""
import torch
from global_cosine_sequence import Network as Baseline
from patch_hierarchy_sequence import PatchStem, PatchHierarchy
from cosine_patch_sequence import SmallAttention as CosineAttention


class StackedStem(PatchStem):
    def __init__(self, old):
        super().__init__(old)
        # Keep the existing first hierarchy, including all initialized tensors.
        self.hierarchy = old.hierarchy
        self.hierarchy_second = PatchHierarchy()
        old_global = self.hierarchy_second.global_patches
        new_global = CosineAttention()
        result = new_global.load_state_dict(old_global.state_dict(), strict=False)
        assert result.missing_keys == ['log_scale'] and not result.unexpected_keys
        self.hierarchy_second.global_patches = new_global
        # PatchHierarchy initializes its output projection to zero.

    def forward(self, x):
        first = super().forward(x)
        return first + self.hierarchy_second(first, x[:, -2, :] > 0)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        with torch.random.fork_rng(devices=[]):
            self.stem = StackedStem(self.stem)
