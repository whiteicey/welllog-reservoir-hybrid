"""Target-private nonlinear adapters on the fixed dense sequence backbone."""
import json
import torch
from torch import nn
from multiscale_sequence import Network as Baseline, objective


class PrivateHead(nn.Module):
    def __init__(self, original):
        super().__init__()
        width = original.gate.in_channels
        self.width = width
        self.first = nn.Conv1d(3*width, 3*64, 1, groups=3)
        self.second = nn.Conv1d(3*64, 3*width, 1, groups=3)
        self.gain = nn.Parameter(torch.full((1, 3*width, 1), .1))
        self.gate = nn.Conv1d(3*width, 3, 1, groups=3)
        self.density = nn.Conv1d(3*width, 72, 1, groups=3)
        with torch.no_grad():
            self.gate.weight.copy_(original.gate.weight)
            self.gate.bias.copy_(original.gate.bias)
            self.density.weight.copy_(original.density.weight)
            self.density.bias.copy_(original.density.bias)

    def forward(self, x):
        shared = x.repeat(1, 3, 1)
        private = shared+self.gain*self.second(torch.nn.functional.gelu(self.first(shared)))
        return (self.gate(private).transpose(1, 2),
                self.density(private).transpose(1, 2).reshape(len(x), -1, 3, 3, 8))


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        self.main_head = PrivateHead(self.main_head)
        self.aux_mid = PrivateHead(self.aux_mid)
        self.aux_coarse = PrivateHead(self.aux_coarse)


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(109); base = Baseline(features=12).eval()
    torch.manual_seed(109); model = Network(features=12).eval()
    for key, value in base.state_dict().items():
        if not key.startswith(('main_head.', 'aux_mid.', 'aux_coarse.')):
            assert torch.equal(value, model.state_dict()[key])
    x = torch.randn(2, 256, 12); depth = torch.randn(2, 256)
    valid = torch.ones(2, 256, dtype=torch.bool); valid[0, :32] = False
    with torch.no_grad():
        for head in (model.main_head, model.aux_mid, model.aux_coarse): head.gain.zero_()
        for a, b in zip(base(x, depth, valid, True), model(x, depth, valid, True)):
            for u, v in zip(a, b): assert torch.allclose(u, v, atol=1e-6)
        for head in (model.main_head, model.aux_mid, model.aux_coarse): head.gain.fill_(.1)
        before = model(x, depth, valid)
        altered = x.clone(); altered[~valid] += 1e4
        for a, b in zip(before, model(altered, depth, valid)): assert torch.equal(a, b)
        model.main_head.second.bias[:128].add_(1)
        after = model(x, depth, valid)
        for a, b in zip(before, after): assert torch.equal(a[:, :, 1:], b[:, :, 1:])
    outputs = model(x, depth, valid, True)
    loss = objective(outputs, torch.randn(2, 256, 3), torch.rand(2, 256, 3) > .4, valid, True)
    loss.backward()
    for head in (model.main_head, model.aux_mid, model.aux_coarse):
        assert all(torch.isfinite(p.grad).all() for p in head.parameters())
        assert head.first.weight.grad.abs().sum() > 0
    return {'passed': True, 'same_backbone_initialization': True, 'zero_gain_recovers_baseline': True,
            'target_private_adapter_isolation': True, 'padded_input_invariance': True,
            'finite_nonzero_adapter_gradients': True,
            'parameters': sum(p.numel() for p in Network().parameters()), 'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
