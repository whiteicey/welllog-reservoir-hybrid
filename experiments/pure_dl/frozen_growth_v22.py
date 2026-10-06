"""Same v21 expansion and initialization, train only newly introduced weights.

The old parameters stay fixed. Their activations and downstream distributions
can still change through the learned extra paths. No no_grad/detach barrier is
inserted through the frozen backbone; gradients reach early new modules.
"""
from champion_scaling_v21 import Network as ExpandedNetwork
from champion_scaling_v21 import HEADS, PRIVATE, load_data
import torch

MODES = {'ffn2': 2, 'ffn4': 4}


class Network(ExpandedNetwork):
    def __init__(self, master, previous, champion, mode):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(master, previous, champion, mode)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_('.extra_' in name)
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        assert trainable and all('.extra_' in n for n in trainable)
        self.scaling_manifest.update(
            update_scope='newly_added_parameters_only',
            frozen_original_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            original_train_eval_module_behavior_unchanged=True)
        assert self.scaling_manifest['frozen_original_parameters'] == 1290490


def make_model(mode, master=None):
    from train_dual8222_heads80 import SOURCE
    master = master if master is not None else torch.load(SOURCE, map_location='cpu', weights_only=False)
    previous = torch.load(HEADS, map_location='cpu', weights_only=False)
    champion = torch.load(PRIVATE, map_location='cpu', weights_only=False)
    return Network(master, previous, champion, mode)
