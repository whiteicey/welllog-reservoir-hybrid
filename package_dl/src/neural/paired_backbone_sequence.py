"""Two independently initialized backbones; fixed equal full predictive mixture."""
import torch
from torch import nn
from torch.nn import functional as F
from .global_cosine_sequence import Network as Member


def merge_predictions(outputs):
    assert len(outputs) == 2
    gates = torch.stack([g.float() for g, _ in outputs])
    log_default = F.logsigmoid(gates).logsumexp(0)
    log_survival_each = F.logsigmoid(-gates)
    log_survival = log_survival_each.logsumexp(0)
    # log(2) cancels in the marginal log odds. Conditional component weights
    # depend on each member's nondefault probability; they are not simply1/16.
    gate = log_default - log_survival
    log_member = log_survival_each - log_survival[None]
    log_weights = torch.cat([r[..., 0, :].float().log_softmax(-1) + log_member[i, ..., None]
                             for i, (_, r) in enumerate(outputs)], -1)
    means = torch.cat([r[..., 1, :].float() for _, r in outputs], -1)
    raw_scales = torch.cat([r[..., 2, :].float() for _, r in outputs], -1)
    return gate, torch.stack([log_weights, means, raw_scales], -2)


class Network(nn.Module):
    def __init__(self, features=206, initial_means=None):
        super().__init__()
        # One fixed run seed, ordinary sequential independent initialization.
        # No checkpoint imports, seed search, bootstrap, or OOF member selection.
        self.members = nn.ModuleList([Member(features, initial_means) for _ in range(2)])

    def forward(self, features, relative_depth, valid, auxiliary=False, absolute_depth=None):
        outputs = [m(features, relative_depth, valid, auxiliary=auxiliary,
                     absolute_depth=absolute_depth) for m in self.members]
        return outputs if auxiliary else merge_predictions(outputs)
