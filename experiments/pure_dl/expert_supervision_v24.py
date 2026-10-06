"""Same private POR network; only expert gradient weighting changes.

Both modes train the input router from detached expert predictions. Mixed-mode
expert gradients match the final-mixture likelihood with routing detached;
independent mode averages the two experts' complete single-target likelihoods.
No parameter, input, decoder, auxiliary task or architecture is added.
"""
from pathlib import Path
import torch
from scratch_scaling_v23 import Network as RandomPrivateNetwork
from scratch_scaling_v23 import metadata_from_training, load_data, REFERENCE
from paired_detached_router_sequence import routed_merge
from train_scaling_v21 import channel_objective

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT/'results/deep_learning/expert_supervision_v24_20260928'
MODES = ('mixture', 'independent')


class Network(RandomPrivateNetwork):
    def __init__(self, metadata, mode):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(metadata, 'base1')
        self.mode = mode
        assert sum(p.numel() for p in self.parameters()) == 1290490

    def forward(self, x, relative, valid, absolute_depth=None, return_parts=False):
        outputs = []
        for member in self.members:
            member.main_head.valid = valid
            try:
                outputs.append(member(x, relative, valid, auxiliary=False, absolute_depth=absolute_depth))
            finally:
                member.main_head.valid = None
        clean = x.index_select(-1, self.router_indices).masked_fill(~valid[..., None], 0)
        h = self.router[1](self.router[0](clean)).masked_fill(~valid[..., None], 0)
        context = self.router.local_context(h.transpose(1, 2)).transpose(1, 2).masked_fill(~valid[..., None], 0)
        logits = self.router[2]((h+context).masked_fill(~valid[..., None], 0))
        logits = logits.reshape(*x.shape[:2], 3, 2)[:, :, 0:1]
        if return_parts:
            return outputs, logits
        return routed_merge(outputs, logits)


def training_losses(outputs, logits, truth, default, mask, quality, mode):
    """Return total, expert and router scalar losses; target axis is POR only."""
    if mode not in MODES or len(outputs) != 2:
        raise ValueError('Invalid supervision mode or expert count')
    args = (truth, default, mask, quality, 0)
    if mode == 'mixture':
        expert_loss = channel_objective(routed_merge(outputs, logits.detach()), *args)[0]
    else:
        expert_loss = sum(channel_objective(o, *args)[0] for o in outputs)/2
    stopped = [tuple(v.detach() for v in o) for o in outputs]
    router_loss = channel_objective(routed_merge(stopped, logits), *args)[0]
    return expert_loss+router_loss, expert_loss, router_loss
