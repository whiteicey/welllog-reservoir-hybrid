"""Identity input calibration: freeze its original zero-output initialization."""
from paired_posterior_router_sequence import Network as Baseline, retained_indices, calibration_indices, VIEW_NAMES


class Network(Baseline):
    def __init__(self, features=206, initial_means=None, *, retain_indices, numeric_indices, conditioning_indices):
        super().__init__(features, initial_means, retain_indices=retain_indices,
                         numeric_indices=numeric_indices, conditioning_indices=conditioning_indices)
        # The original final layer is exactly zero, so gamma=beta=0 for every
        # context. Frozen parameters receive no optimizer update or weight decay.
        self.members[1].calibration.requires_grad_(False)
