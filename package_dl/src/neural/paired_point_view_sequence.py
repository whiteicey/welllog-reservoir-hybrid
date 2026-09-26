"""Full engineered-feature member plus a instantaneous raw/log/missing/relative/physics feature view."""
import torch
from torch import nn
from .global_cosine_sequence import Network as Member
from .paired_backbone_sequence import Network as Baseline, merge_predictions

RAW_CURVES = ('GR', 'PE', 'SP', 'CAL', 'AC', 'DEN', 'CNL', 'RXO', 'RT')
VIEW_NAMES = RAW_CURVES + ('GR_LOG', 'RXO_LOG', 'RT_LOG', 'DEPTH_REL') + tuple(
    name + '_MISSING' for name in RAW_CURVES) + (
    'RT_RXO_RATIO', 'RT_RXO_LOG_RATIO', 'AC_DEN_RATIO', 'CNL_DEN_RATIO',
    'GR_CNL_PRODUCT', 'AC_CNL_PRODUCT', 'DEN_CNL_PRODUCT', 'GR_AC_PRODUCT',
    'GR_DEN_PRODUCT', 'GR_RTV_PRODUCT', 'SP_GR_DIFF', 'DEN_AC_DIFF',
    'DENSITY_POROSITY', 'SONIC_POROSITY', 'CNL_DENSITY_POROSITY_DIFF',
    'CNL_DENSITY_POROSITY_MEAN', 'SHALE_INDEX_GR', 'ARCHIE_PROXY', 'INVASION_LOG_DIFF')


def retained_indices(whitelist):
    assert len(whitelist) == 206 and len(set(whitelist)) == 206
    assert len(VIEW_NAMES) == len(set(VIEW_NAMES)) == 41
    assert all(whitelist.count(name) == 1 for name in VIEW_NAMES)
    indices = sorted(whitelist.index(name) for name in VIEW_NAMES)
    assert set(whitelist[i] for i in indices) == set(VIEW_NAMES)
    return indices


class PointViewMember(Member):
    def __init__(self, original, indices, features):
        # Register the already initialized member objects unchanged: no second
        # random initialization or checkpoint loading.
        nn.Module.__init__(self)
        for name, child in original.named_children():
            self.add_module(name, child)
        for name, parameter in original.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in original.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        assert len(indices) == len(set(indices)) == 41
        assert all(0 <= i < features for i in indices)
        mask = torch.zeros(features, dtype=torch.bool)
        mask[indices] = True
        self.register_buffer('feature_view_mask', mask)

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        view = features.masked_fill(~self.feature_view_mask[None, None, :], 0)
        return super().forward(view, relative_depth, valid, auxiliary, absolute_depth)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices):
        super().__init__(features, initial_means)
        self.members[1] = PointViewMember(self.members[1], retain_indices, features)
