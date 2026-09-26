"""Synthetic resource test for the gated conditional official-utility sequence loss."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from hurdle_utility_sequence import Network, objective, fit_priors

torch.set_num_threads(3); torch.manual_seed(20261006)
assert torch.cuda.mem_get_info()[0] >= 8*1024**3
grids = [np.unique(np.r_[0., .1, np.geomspace(.001, 50, 800)]),
         np.unique(np.r_[.01, np.geomspace(.01, 10000, 800)]),
         np.unique(np.r_[99.9, np.linspace(8.305, 99.9, 800)])]
actions = [torch.tensor(a, device='cuda', dtype=torch.float32) for a in grids]
x = torch.randn(64, 256, 206, device='cuda', dtype=torch.float16)
depth = torch.linspace(-1, 1, 256, device='cuda')[None].expand(64, -1)
valid = torch.ones(64, 256, device='cuda', dtype=torch.bool)
mask = valid.clone(); mask[:, :64] = False; mask[:, 192:] = False
truth = torch.stack([a[torch.randint(len(a), (64, 256), device='cuda')] for a in actions], dim=-1)
defaults = torch.rand(64, 256, 3, device='cuda') < .7
truth = torch.where(defaults, torch.tensor([.1, .01, 99.9], device='cuda'), truth)
gp, cp, normalizers = fit_priors(truth[mask], defaults[mask], actions)
model = Network([len(a) for a in grids], cp, gp).cuda().train()
opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
def step():
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x, depth, valid)
    loss = objective(logits, truth, defaults, mask, actions, normalizers)
    assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2); opt.step()
for _ in range(4):
    step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
for _ in range(20):
    step()
torch.cuda.synchronize(); elapsed = (time.perf_counter()-start)/20
result = {'synthetic_only': True, 'device': torch.cuda.get_device_name(),
          'parameters': sum(p.numel() for p in model.parameters()),
          'action_counts': [len(a) for a in grids], 'batch_windows': 64, 'length': 256,
          'supervised_points_per_window': 128, 'mean_step_ms': elapsed*1000,
          'projected_five_fold_updates_seconds': elapsed*40*64*5,
          'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
          'peak_reserved_mib': torch.cuda.max_memory_reserved()/1024**2,
          'projection_excludes': 'real fold feature table/preprocessing, inference and file writes'}
out = Path('outputs/pure_dl_hurdle_utility_sequence'); out.mkdir(exist_ok=True)
(out/'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
