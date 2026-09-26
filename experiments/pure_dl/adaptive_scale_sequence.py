"""Input-conditioned shallow receptive fields on the target-private baseline."""
import json
import torch
from torch import nn
from private_head_sequence import Network as Baseline
from multiscale_sequence import objective


class AdaptiveDepthwise(nn.Module):
    def __init__(self, original, factors):
        super().__init__()
        self.local = original
        channels = original.in_channels
        kernel = original.kernel_size[0]
        self.wide = nn.ModuleList()
        for factor in factors:
            dilation = original.dilation[0]*factor
            layer = nn.Conv1d(channels, channels, kernel, padding=(kernel//2)*dilation,
                              dilation=dilation, groups=channels)
            with torch.no_grad():
                layer.weight.copy_(original.weight); layer.bias.copy_(original.bias)
            self.wide.append(layer)
        self.router = nn.Conv1d(channels, 2, 1)
        nn.init.zeros_(self.router.weight); nn.init.zeros_(self.router.bias)
        self.gain = nn.Parameter(torch.full((1, channels, 1), .1))

    def forward(self, x):
        local = self.local(x)
        weights = self.router(x).softmax(dim=1)
        alternative = sum(weights[:, i:i+1]*layer(x) for i, layer in enumerate(self.wide))
        return local+self.gain*(alternative-local)


class Network(Baseline):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features, initial_means)
        self.enc0.depthwise = AdaptiveDepthwise(self.enc0.depthwise, (3, 9))
        self.enc1.depthwise = AdaptiveDepthwise(self.enc1.depthwise, (2, 4))


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(111); base = Baseline(features=12).eval()
    torch.manual_seed(111); model = Network(features=12).eval()
    for key, value in base.state_dict().items():
        if not key.startswith(('enc0.depthwise.', 'enc1.depthwise.')):
            assert torch.equal(value, model.state_dict()[key])
    x = torch.randn(2, 256, 12); d = torch.randn(2, 256)
    valid = torch.ones(2, 256, dtype=torch.bool); valid[0, :32] = False
    with torch.no_grad():
        for block in (model.enc0, model.enc1): block.depthwise.gain.zero_()
        for a, b in zip(base(x, d, valid, True), model(x, d, valid, True)):
            assert all(torch.equal(u, v) for u, v in zip(a, b))
        for block in (model.enc0, model.enc1): block.depthwise.gain.fill_(.1)
        original = model(x, d, valid)
        changed = x.clone(); changed[~valid] += 1000
        assert all(torch.equal(a,b) for a,b in zip(original, model(changed,d,valid)))
        changed = x.clone(); changed[1] += 100
        assert all(torch.equal(a[0],b[0]) for a,b in zip(original, model(changed,d,valid)))
    outputs = model(x, d, valid, True)
    loss = objective(outputs, torch.randn(2,256,3), torch.rand(2,256,3)>.4, valid, True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for block in (model.enc0, model.enc1):
        assert block.depthwise.router.weight.grad.abs().sum()>0
        assert all(layer.weight.grad.abs().sum()>0 for layer in block.depthwise.wide)
    return {'passed':True, 'unchanged_backbone_initialization':True,
            'zero_gains_recover_baseline_exactly':True, 'padded_inputs_and_wells_isolated':True,
            'router_and_wide_branch_gradients_nonzero':True,
            'parameters':sum(p.numel() for p in Network().parameters()), 'gpu_training_started':False}


if __name__ == '__main__':
    print(json.dumps(self_test(),indent=2))
