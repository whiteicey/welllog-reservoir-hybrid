"""Eight current-row default states; marginal logits at the inference interface."""
import torch
from torch import nn
from global_cosine_sequence import Network as Baseline


def state_bits(device=None):
    return ((torch.arange(8, device=device)[:, None] >> torch.arange(3, device=device)) & 1).bool()


def marginal_logits(joint):
    logits = joint.float()
    bits = state_bits(logits.device)
    return torch.stack([logits[..., bits[:, j]].logsumexp(-1) -
                        logits[..., ~bits[:, j]].logsumexp(-1) for j in range(3)], -1)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        for head in [self.main_head, self.aux_mid, self.aux_coarse]:
            old = head.gate
            with torch.random.fork_rng(devices=[]):
                new = nn.Conv1d(old.in_channels, 8, 1)
            bits = state_bits().to(old.weight.dtype)
            with torch.no_grad():
                new.weight.copy_(torch.einsum('sj,jck->sck', bits, old.weight))
                new.bias.copy_(bits @ old.bias)
            head.gate = new

    def forward(self, features, relative_depth, valid, auxiliary=False,
                absolute_depth=None, joint_gate=False):
        result = super().forward(features, relative_depth, valid, auxiliary=auxiliary,
                                 absolute_depth=absolute_depth)
        # Auxiliary mode is the trainer's explicit loss interface (8 joint logits).
        # Validation and the full training probe use ordinary marginal logits (3).
        if auxiliary or joint_gate:
            return result
        joint, raw = result
        return marginal_logits(joint), raw
