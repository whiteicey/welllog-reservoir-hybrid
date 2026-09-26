"""Synthetic GPU resource measurement for dense sequence architecture/loss pair."""
import json
import time
from pathlib import Path
import torch
from multiscale_sequence import Network, objective, self_test

torch.set_num_threads(3)
checks = self_test()
assert torch.cuda.mem_get_info()[0] > 8*1024**3, 'Requires8 GiB free'
results = []
for deep in (False, True):
    torch.manual_seed(20261006)
    model = Network().cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x = torch.randn(64, 256, 206, device='cuda', dtype=torch.float16)
    depth = torch.linspace(-1, 1, 256, device='cuda')[None].expand(64, -1)
    valid = torch.ones(64, 256, device='cuda', dtype=torch.bool)
    supervised = valid.clone(); supervised[:, :64] = False; supervised[:, 192:] = False
    y = torch.randn(64, 256, 3, device='cuda')
    defaults = torch.rand(64, 256, 3, device='cuda') > .3
    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(x, depth, valid, auxiliary=deep)
        loss = objective(outputs, y, defaults, supervised, deep)
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
    for _ in range(4):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(20):
        step()
    torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)/20
    results.append({'deep_supervision': deep, 'parameters': sum(p.numel() for p in model.parameters()),
        'windows_per_step': 64, 'context_points': 256, 'supervised_core_points': 128,
        'mean_step_ms': elapsed*1000, 'projected_five_fold_updates_seconds': elapsed*40*64*5,
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
        'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2})
    del model, opt, x, depth, valid, supervised, y, defaults
    torch.cuda.empty_cache()
result = {'synthetic_only': True, 'cpu_checks': checks, 'device': torch.cuda.get_device_name(),
          'measurements': results, 'projection_excludes': 'real fold preprocessing, input table, inference and artifact writes'}
out = Path('outputs/pure_dl_multiscale_sequence');out.mkdir(parents=True, exist_ok=True)
(out/'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
