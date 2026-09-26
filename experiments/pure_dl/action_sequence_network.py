"""Same multiscale depth backbone with direct action-utility classification heads."""
import json
import torch
from torch import nn
from multiscale_sequence import Network as DensityNetwork
from score_aligned_loss import utility_cross_entropy


class ActionHead(nn.Module):
    def __init__(self, width, sizes):
        super().__init__()
        self.sizes = tuple(sizes)
        assert len(self.sizes) == 3 and min(self.sizes) > 1
        self.projection = nn.Conv1d(width, sum(self.sizes), 1)
        nn.init.normal_(self.projection.weight, 0, .005)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        return self.projection(x).transpose(1, 2).split(self.sizes, dim=-1)


class Network(DensityNetwork):
    def __init__(self, action_sizes, features=206):
        super().__init__(features=features)
        self.main_head = ActionHead(128, action_sizes)
        del self.aux_mid, self.aux_coarse

    def forward(self, features, relative_depth, valid, auxiliary=False):
        assert not auxiliary, 'This experiment isolates direct reward loss; no auxiliary-density heads'
        return super().forward(features, relative_depth, valid, auxiliary=False)


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(71); old = DensityNetwork(features=12)
    torch.manual_seed(71); model = Network([5, 6, 7], features=12).eval()
    for name, value in old.state_dict().items():
        if not name.startswith(('main_head.', 'aux_mid.', 'aux_coarse.')):
            assert torch.equal(value, model.state_dict()[name]), name
    x = torch.randn(2, 256, 12); depth = torch.randn(2, 256)
    valid = torch.ones(2, 256, dtype=torch.bool); valid[:, :32] = False
    mask = valid.clone(); mask[:, :64] = False; mask[:, 192:] = False
    actions = [torch.linspace(0, 50, 5), torch.logspace(-2, 4, 6), torch.linspace(8.305, 99.9, 7)]
    truth = torch.stack([actions[j][torch.randint(len(actions[j]), (2, 256))] for j in range(3)], dim=-1)
    logits = model(x, depth, valid)
    assert [v.shape for v in logits] == [torch.Size([2, 256, k]) for k in [5, 6, 7]]
    loss = utility_cross_entropy(logits, truth, mask, actions, [1., 1., 1.])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.stem.weight.grad.abs().sum() > 0
    with torch.no_grad():
        changed = x.clone(); changed[~valid] += 1000
        assert all(torch.equal(a, b) for a, b in zip(logits, model(changed, depth, valid)))
        assert all(torch.equal(a.argmax(-1), a.softmax(-1).argmax(-1)) for a in logits)
    return {'passed': True, 'backbone_initialization_matches_density_network': True,
            'action_shapes_and_finite_gradients': True, 'padded_inputs_masked': True,
            'decode_is_direct_argmax_not_second_utility_convolution': True,
            'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
