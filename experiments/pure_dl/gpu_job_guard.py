"""Project GPU admission check; latest user limit is two training/benchmark jobs."""
import os
from pathlib import Path
import psutil
import torch


def check_slot(min_free_gib=6.):
    active = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        info = process.info
        if info['pid'] == os.getpid() or (info['name'] or '').lower() != 'python.exe':
            continue
        for argument in info['cmdline'] or []:
            normalized = argument.replace('\\', '/')
            name = Path(normalized).name
            if 'pure_dl/' in normalized and name.startswith(('train_', 'benchmark_')) and name.endswith('.py'):
                active.append({'pid': info['pid'], 'script': name}); break
    if len(active) >= 2:
        raise RuntimeError(f'Two other training/benchmark jobs active: {active}; wait for a slot')
    free, total = torch.cuda.mem_get_info()
    if free < min_free_gib*1024**3:
        raise RuntimeError(f'Only {free/1024**3:.2f} GiB free; require {min_free_gib:.1f}')
    return {'maximum_concurrent_jobs': 2, 'other_jobs': active, 'free_gib': free/1024**3,
            'total_gib': total/1024**3}
