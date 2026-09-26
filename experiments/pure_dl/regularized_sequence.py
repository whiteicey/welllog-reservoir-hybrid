"""Single-factor dropout ablation of the original dense sequence network."""
import json
import torch
from torch import nn
from multiscale_sequence import Network as Baseline, objective


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == .1
                module.p = .25


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(109); base = Baseline(features=12).eval()
    torch.manual_seed(109); model = Network(features=12).eval()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items())
    x = torch.randn(2, 256, 12); d = torch.randn(2, 256)
    mask = torch.ones(2, 256, dtype=torch.bool)
    with torch.no_grad():
        for a, b in zip(base(x, d, mask, True), model(x, d, mask, True)):
            assert all(torch.equal(u, v) for u, v in zip(a, b))
    model.train()
    outputs = model(x, d, mask, True)
    loss = objective(outputs, torch.randn(2, 256, 3), torch.rand(2, 256, 3) > .4, mask, True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    dropout = [module.p for module in model.modules() if isinstance(module, nn.Dropout)]
    assert dropout and all(p == .25 for p in dropout)
    return {'passed': True, 'initial_weights_and_eval_outputs_identical': True,
            'training_gradients_finite': True, 'dropout_probability': .25,
            'dropout_modules': len(dropout), 'parameters': sum(p.numel() for p in Network().parameters()),
            'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
