"""Train the two factors of the copied dual head separately.

This is a parameter-update-scope ablation, not a new density family or decoder.
Gate-only still uses the complete dual likelihood: gate probabilities also affect
conditional expert shares. Density-only keeps gate logits, but may change the
utility-optimal final default/nondefault decision through the density.
"""
from __future__ import annotations
import torch
from dual8222_head_structure_v2 import Network as LinearNetwork


class Network(LinearNetwork):
    @staticmethod
    def parameter_count(architecture: str) -> int:
        return {'linear':19350, 'gate_only':774, 'density_only':18576}[architecture]

    def __init__(self, original, architecture: str):
        if architecture not in {'gate_only','density_only'}:
            raise ValueError(f'Unknown factor tuning mode: {architecture}')
        super().__init__(original,'linear')
        self.tuning_mode=architecture
        self.set_trainable_targets([True]*3)
        assert sum(p.numel() for p in self.parameters() if p.requires_grad)==self.parameter_count(architecture)

    def set_trainable_targets(self, active):
        """Also used after resume; never reactivate the fixed factor accidentally."""
        if len(active)!=3:raise ValueError('Exactly three target states required')
        for c,group in enumerate(self.heads):
            for head in group.branches:
                for name,param in head.named_parameters():
                    train_factor=(name.startswith('gate_') if self.tuning_mode=='gate_only' else name.startswith('density_'))
                    param.requires_grad_(bool(active[c]) and train_factor)

    def verify_fixed_factor(self):
        """Check copied frozen rows as well as the already-frozen source backbone."""
        for c,group in enumerate(self.heads):
            for e,head in enumerate(group.branches):
                original=self.backbone.members[e].main_head
                if self.tuning_mode=='gate_only':
                    assert torch.equal(head.density_weight,original.density.weight[24*c:24*(c+1)])
                    assert torch.equal(head.density_bias,original.density.bias[24*c:24*(c+1)])
                else:
                    assert torch.equal(head.gate_weight,original.gate.weight[c:c+1])
                    assert torch.equal(head.gate_bias,original.gate.bias[c:c+1])
