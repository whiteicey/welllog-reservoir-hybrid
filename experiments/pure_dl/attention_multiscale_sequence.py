"""Small masked bottleneck Transformer for the dense multiscale depth backbone."""
import json
import torch
from torch import nn
from torch.nn import functional as F
from multiscale_sequence import Network as BaseNetwork, objective


class ContextBlock(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, 4, dropout=0., batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width*2), nn.GELU(), nn.Linear(width*2, width))
        self.attention_gain = nn.Parameter(torch.full((width,), .1))
        self.ff_gain = nn.Parameter(torch.full((width,), .1))

    def forward(self, tokens, valid, position):
        q = self.attention_norm(tokens)+position
        delta = self.attention(q, q, q, key_padding_mask=~valid, need_weights=False)[0]
        mask = valid[:, :, None].to(delta.dtype)
        tokens = tokens+self.attention_gain*delta*mask
        return tokens+self.ff_gain*self.ff(self.ff_norm(tokens))*mask


class Network(BaseNetwork):
    def __init__(self, features=206, initial_means=None):
        super().__init__(features=features, initial_means=initial_means)
        self.context_position = nn.Parameter(torch.randn(1, 32, 192)*.02)
        self.context_layers = nn.ModuleList([ContextBlock(), ContextBlock()])

    def context(self, h, valid):
        tokens = h.transpose(1, 2)
        assert tokens.shape[1] == 32 and valid.shape[1] == 256
        pooled = F.max_pool1d(valid[:, None].float(), 8, 8).squeeze(1).bool()
        assert bool(pooled.any(-1).all()), 'Each window requires at least one real depth point'
        position = self.context_position.to(tokens.dtype)
        for layer in self.context_layers:
            tokens = layer(tokens, pooled, position)
        return tokens.transpose(1, 2)

    def forward(self, features, relative_depth, valid, auxiliary=False):
        mask = valid[:, None].to(features.dtype)
        point = features.transpose(1, 2)*mask
        x = torch.cat([point, relative_depth[:, None]*mask, mask], dim=1)
        e0 = self.enc0(self.stem(x))
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        h = self.bottleneck(self.down3(e2))
        h = self.context(h, valid)
        h = self.dec2(self.merge2(torch.cat([F.interpolate(h, size=e2.shape[-1], mode='linear', align_corners=False), e2], 1)))
        coarse = h
        h = self.dec1(self.merge1(torch.cat([F.interpolate(h, size=e1.shape[-1], mode='linear', align_corners=False), e1], 1)))
        mid = h
        h = self.dec0(self.merge0(torch.cat([F.interpolate(h, size=e0.shape[-1], mode='linear', align_corners=False), e0], 1)))
        main = self.main_head(self.combine(torch.cat([h, self.point(point)], 1)))
        if not auxiliary:
            return main
        return main, self.aux_mid(mid), self.aux_coarse(coarse)


def self_test():
    torch.set_num_threads(2)
    torch.manual_seed(101); base = BaseNetwork(features=12).eval()
    torch.manual_seed(101); model = Network(features=12).eval()
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in base.state_dict().items())
    x = torch.randn(2, 256, 12); depth = torch.randn(2, 256)
    valid = torch.ones(2, 256, dtype=torch.bool); valid[0, :112] = False; valid[0, 144:] = False
    with torch.no_grad():
        for layer in model.context_layers:
            layer.attention_gain.zero_(); layer.ff_gain.zero_()
        original = base(x, depth, valid); identity = model(x, depth, valid)
        assert all(torch.equal(u, v) for u, v in zip(original, identity))
        for layer in model.context_layers:
            layer.attention_gain.fill_(.1); layer.ff_gain.fill_(.1)
    masked = x.clone(); masked[~valid] += 1000
    with torch.no_grad():
        a = model(x, depth, valid); b = model(masked, depth, valid)
        assert all(torch.equal(u, v) for u, v in zip(a, b))
        other = x.clone(); other[1] += 100
        c = model(other, depth, valid)
        assert all(torch.equal(u[0], v[0]) for u, v in zip(a, c))
        # Padded keys cannot affect valid token outputs at the attention block itself.
        tokens = torch.randn(2, 32, 192); mask = torch.zeros(2, 32, dtype=torch.bool); mask[:, 5:11] = True
        changed = tokens.clone(); changed[~mask] += 1000
        u = model.context_layers[0](tokens, mask, model.context_position)
        v = model.context_layers[0](changed, mask, model.context_position)
        assert torch.equal(u[mask], v[mask])
    target = torch.randn(2, 256, 3); default = torch.rand(2, 256, 3) > .3
    outputs = model(x, depth, valid, auxiliary=True)
    loss = objective(outputs, target, default, valid, deep_supervision=True)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.context_layers[0].attention.in_proj_weight.grad.abs().sum() > 0
    return {'passed': True, 'existing_backbone_initialization_unchanged': True,
            'zero_attention_gains_exactly_recover_original_network': True,
            'padded_inputs_and_attention_keys_masked': True, 'different_batch_wells_isolated': True,
            'attention_receives_finite_gradients': True, 'attention_tokens': 32,
            'gpu_training_started': False}


if __name__ == '__main__':
    print(json.dumps(self_test(), indent=2))
