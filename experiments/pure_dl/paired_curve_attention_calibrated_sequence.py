"""Same curve-attention residual, injected into the calibrated expert."""
import torch

from paired_curve_attention_residual_sequence import Network as FullAdapter
from paired_curve_attention_residual_sequence import retained_indices, calibration_indices, VIEW_NAMES, RAW_CURVES


class Network(FullAdapter):
    def _members(self, features, relative_depth, valid, auxiliary, absolute_depth):
        original = features.to(torch.float16 if features.device.type == 'cuda' else features.dtype)
        correction = self.curve_adapter(original, valid)
        calibrated = (original.float() + correction.float()).to(original.dtype)
        return [self.members[0](original, relative_depth, valid, auxiliary, absolute_depth),
                self.members[1](calibrated, relative_depth, valid, auxiliary, absolute_depth)], original
