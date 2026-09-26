"""Gated near/far external context with independently seeded context dropout."""
import torch
from torch import nn
from external_context_sequence import Network as Baseline, ExternalContext


class GatedExternalContext(ExternalContext):
    def __init__(self):
        super().__init__()
        self.token_encoder[0]=nn.Linear(32,32)
        self.gate=nn.Conv1d(64,1,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,-2.)
        self.dropout_seed=20261010;self._generator=None

    def forward(self,x,valid,tokens,token_valid):
        if self.training:
            if self._generator is None:
                self._generator=torch.Generator(device=x.device).manual_seed(self.dropout_seed)
            keep=torch.rand((len(x),1),device=x.device,generator=self._generator)>=.5
            token_valid=token_valid&keep
        context=super().forward(x,valid,tokens,token_valid)
        return self.gate(x).sigmoid()*context


class Network(Baseline):
    def __init__(self,features=206,initial_means=None):
        super().__init__(features,initial_means)
        with torch.random.fork_rng(devices=[]):self.external=GatedExternalContext()
