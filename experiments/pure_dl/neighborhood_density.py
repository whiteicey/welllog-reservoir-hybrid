"""Untrained research module: learned cross-well metric and label-kernel posterior.

Unlike the earlier frozen-PLE residual retrieval, query and candidate embeddings
are trained jointly. No training runner or GPU job is launched by this module.
"""
import json
import math
import torch
from torch import nn
from retrieval_network import validate_memory_split


class NeighborhoodDensity(nn.Module):
    def __init__(self, inputs=206, hidden=128, embedding=32):
        super().__init__()
        self.embedding_size = embedding
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.BatchNorm1d(hidden),
                                     nn.ReLU(), nn.Linear(hidden, 3*embedding))
        self.log_temperature = nn.Parameter(torch.zeros(3))
        self.log_bandwidth = nn.Parameter(torch.full((3,), math.log(.15)))

    def encode(self, features):
        return self.encoder(features).reshape(-1, 3, self.embedding_size)

    def log_attention(self, query, memory, query_wells, memory_wells):
        embedded = self.encode(torch.cat([query, memory]))
        return self.log_attention_embeddings(embedded[:len(query)], embedded[len(query):], query_wells, memory_wells)

    def log_attention_embeddings(self, query, memory, query_wells, memory_wells):
        logits = self.attention_logits(query, memory, query_wells, memory_wells)
        if not bool(torch.isfinite(logits).any(-1).all()):
            raise ValueError('Every query requires memory from a different training well')
        return logits.log_softmax(-1)

    def attention_logits(self, query, memory, query_wells, memory_wells):
        if query_wells.shape != (len(query),) or memory_wells.shape != (len(memory),):
            raise ValueError('Missing query or memory well identity')
        allowed = query_wells[:, None] != memory_wells[None, :]
        q, k = query.permute(1, 0, 2), memory.permute(1, 0, 2)
        distance = torch.cdist(q.float(), k.float(), p=2)
        temperature = self.log_temperature.exp().clamp(.05, 20)[:, None, None]
        logits = (-distance/temperature).permute(1, 0, 2)
        return logits.masked_fill(~allowed[:, None, :], -torch.inf)

    @torch.no_grad()
    def streaming_posterior(self, query, memory, query_wells, memory_wells,
                            memory_default, kernels, block_size=4096):
        """Exact online softmax aggregation; no full query-by-memory matrix.

        query/memory are eval-mode embeddings, kernels is (memory,3,bins).
        Blocks may be completely masked for some query wells.
        """
        if self.training:
            raise ValueError('Streaming cached embeddings require eval mode')
        if block_size < 1 or kernels.shape[:2] != (len(memory), 3):
            raise ValueError('Invalid block size or conditional kernel shape')
        shape = (len(query), 3)
        maximum = torch.full(shape, -torch.inf, device=query.device)
        total = torch.zeros(shape, device=query.device)
        atom = torch.zeros_like(total); active_total = torch.zeros_like(total)
        mass = torch.zeros(*shape, kernels.shape[-1], device=query.device)
        for start in range(0, len(memory), block_size):
            stop = min(start+block_size, len(memory))
            logits = self.attention_logits(query, memory[start:stop], query_wells, memory_wells[start:stop])
            new_maximum = torch.maximum(maximum, logits.max(-1).values)
            safe_maximum = torch.where(torch.isfinite(new_maximum), new_maximum, torch.zeros_like(new_maximum))
            old_scale = torch.where(torch.isfinite(maximum), (maximum-safe_maximum).exp(), torch.zeros_like(maximum))
            weight = (logits-safe_maximum[:, :, None]).exp()
            default = memory_default[start:stop].T[None].float()
            active = weight*(1-default)
            total = total*old_scale+weight.sum(-1)
            atom = atom*old_scale+(weight*default).sum(-1)
            active_total = active_total*old_scale+active.sum(-1)
            block_mass = torch.bmm(active.permute(1, 0, 2), kernels[start:stop].permute(1, 0, 2)).permute(1, 0, 2)
            mass = mass*old_scale[:, :, None]+block_mass
            maximum = new_maximum
        if not bool(((total > 0) & (active_total > 0)).all()):
            raise ValueError('Insufficient allowed memory for the posterior')
        conditional = mass/active_total[:, :, None]
        return atom/total, conditional/conditional.sum(-1, keepdim=True)

    def conditional(self, log_attention, memory_default):
        default = memory_default.T[None].bool()
        log_nondefault = log_attention.masked_fill(default, -torch.inf)
        normalizer = log_nondefault.logsumexp(-1, keepdim=True)
        if not bool(torch.isfinite(normalizer).all()):
            raise ValueError('Each target requires nondefault memory outside the query well')
        gate = (log_attention.exp()*default).sum(-1)
        return gate, log_nondefault-normalizer

    def conditional_nll(self, log_weights, truth_z, memory_z):
        sd = self.log_bandwidth.exp().clamp(.02, 1.)[None, :, None]
        residual = (truth_z[:, :, None]-memory_z.T[None])/sd
        component_logp = -.5*residual.square()-sd.log()-.5*math.log(2*math.pi)
        return -(log_weights+component_logp).logsumexp(-1)


def self_test():
    torch.manual_seed(41)
    model = NeighborhoodDensity(inputs=12, hidden=24, embedding=8).eval()
    query = torch.randn(4, 12); memory = torch.randn(16, 12)
    qw = torch.arange(4); mw = torch.arange(4).repeat_interleave(4)
    defaults = torch.zeros(16, 3); defaults[::2] = 1
    values = torch.randn(16, 3); truth = torch.randn(4, 3)
    loga = model.log_attention(query, memory, qw, mw)
    cached = model.log_attention_embeddings(model.encode(query), model.encode(memory), qw, mw)
    assert torch.allclose(loga, cached, atol=1e-6), 'Cached evaluation changed the attention'
    gate, conditional = model.conditional(loga, defaults)
    assert torch.allclose(loga.exp().sum(-1), torch.ones(4, 3))
    assert torch.allclose(conditional.exp().sum(-1), torch.ones(4, 3))
    assert torch.count_nonzero(loga.exp().permute(0, 2, 1)[qw[:, None] == mw]) == 0
    before = model.conditional_nll(conditional, truth, values)
    changed = values.clone(); changed[mw == 0] += 100
    after = model.conditional_nll(conditional, truth, changed)
    assert torch.equal(before[0], after[0])
    flip = defaults.clone(); flip[mw == 0] = 1-flip[mw == 0]
    gate2, conditional2 = model.conditional(loga, flip)
    assert torch.equal(gate[0], gate2[0]) and torch.equal(conditional[0], conditional2[0])
    kernels = torch.rand(16, 3, 11); kernels /= kernels.sum(-1, keepdim=True)
    expected = torch.bmm(conditional.exp().permute(1, 0, 2), kernels.permute(1, 0, 2)).permute(1, 0, 2)
    stream_gate, stream_prob = model.streaming_posterior(model.encode(query), model.encode(memory), qw, mw, defaults, kernels, 3)
    assert torch.allclose(stream_gate, gate, atol=1e-6)
    assert torch.allclose(stream_prob, expected, atol=1e-6)
    (before.mean()+gate.mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    validate_memory_split([0, 1], [0, 1, 2], [3, 4])
    try:
        validate_memory_split([0, 3], [0, 1, 2], [3, 4])
    except ValueError:
        pass
    else:
        raise AssertionError('Held-out memory was not rejected')
    return {'passed': True, 'normalized_probabilities': True,
            'same_well_labels_cannot_change_query_posterior': True,
            'heldout_memory_rejected': True, 'encoder_and_bandwidth_gradients_finite': True,
            'cached_eval_matches_full_encoding': True,
            'streaming_matches_full_posterior_with_fully_masked_blocks': True,
            'gpu_job_started': False}


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(json.dumps(self_test(), indent=2))
