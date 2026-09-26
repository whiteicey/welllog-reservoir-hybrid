"""Synthetic resource benchmark only; never reads labels or trains CV models."""
import json
import time
from pathlib import Path
import torch
from neighborhood_ple import NeighborhoodDensity


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20261005)
    free, _ = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError('Requires 8 GiB free GPU memory')
    result = {'synthetic_only': True, 'device': torch.cuda.get_device_name(),
              'training': [], 'inference': []}
    for count in (2048, 8192):
        model = NeighborhoodDensity().cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        q = torch.randn(512, 206, device='cuda')
        m = torch.randn(count, 206, device='cuda')
        qw = torch.arange(512, device='cuda') % 64
        mw = torch.arange(count, device='cuda') % 64
        defaults = (torch.rand(count, 3, device='cuda') < .3)
        target_default = torch.rand(512, 3, device='cuda') < .3
        y = torch.randn(512, 3, device='cuda')
        my = torch.randn(count, 3, device='cuda')
        def step():
            optimizer.zero_grad(set_to_none=True)
            loga = model.log_attention(q, m, qw, mw)
            log_default = loga.masked_fill(~defaults.T[None], -torch.inf).logsumexp(-1)
            log_active = loga.masked_fill(defaults.T[None], -torch.inf).logsumexp(-1)
            logweights = loga.masked_fill(defaults.T[None], -torch.inf)-log_active[..., None]
            loss = -torch.where(target_default, log_default, log_active).mean()
            loss = loss + model.conditional_nll(logweights, y, my)[~target_default].mean()
            loss.backward()
            optimizer.step()
            return loss
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(10):
            loss = step()
        torch.cuda.synchronize()
        assert torch.isfinite(loss)
        result['training'].append({'query_batch': 512, 'memory': count,
            'seconds_per_step': (time.perf_counter()-started)/10,
            'allocated_mib': torch.cuda.max_memory_allocated()/1024**2})
        del optimizer, model, q, m, qw, mw, defaults, target_default, y, my, loss
        torch.cuda.empty_cache()
    model = NeighborhoodDensity().cuda().eval()
    for count in ():  # Inference on cached embeddings is unchanged and already measured.
        memory = torch.randn(count, 3, 32, device='cuda')
        query = torch.randn(512, 3, 32, device='cuda')
        mw = torch.arange(count, device='cuda') % 64
        qw = torch.full((512,), 79, device='cuda')
        defaults = torch.rand(count, 3, device='cuda') < .3
        kernels = torch.empty(count, 3, 256, device='cuda')
        for start in range(0, count, 4096):
            block = kernels[start:start+4096]
            block.uniform_()
            block /= block.sum(-1, keepdim=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        gate, prob = model.streaming_posterior(query, memory, qw, mw, defaults, kernels)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        assert torch.isfinite(prob).all() and torch.allclose(prob.sum(-1), torch.ones_like(gate), atol=1e-5)
        result['inference'].append({'query_batch': 512, 'memory': count,
            'seconds_per_batch': elapsed, 'projected_723556_rows_seconds': elapsed*723556/512,
            'allocated_mib': torch.cuda.max_memory_allocated()/1024**2})
        del memory, query, mw, qw, defaults, kernels, block, gate, prob
        torch.cuda.empty_cache()
    out = Path('outputs/pure_dl_neighborhood_ple')
    out.mkdir(parents=True, exist_ok=True)
    (out/'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
