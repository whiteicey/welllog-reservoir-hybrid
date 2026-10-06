import sys,json,time
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compact_five_stage_v100 import ROOT,SOURCE
from gpu_job_guard_v2 import check_slot
import train_invariant as base
study=ROOT/'results/deep_learning/training_matmul_timing_v122_20261001'
torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
started=time.perf_counter();master=torch.load(SOURCE,map_location='cpu',weights_only=False);ct=np.asarray(master['centers'][2])
actions=np.unique(np.r_[99.9,np.clip(np.linspace(ct[0],ct[-1],800),*base.BOUNDS['SW'])]);utility=torch.tensor(base.kernel_matrix(actions,np.r_[99.9,ct],'SW').T,device='cuda',dtype=torch.float32)
torch.manual_seed(20261122);logits=torch.randn(8192,utility.shape[0],device='cuda',requires_grad=True);reward=torch.rand(8192,len(actions),device='cuda')
results={};torch.cuda.reset_peak_memory_stats()
def step():
    logits.grad=None;prob=logits.softmax(-1);policy=((prob@utility)/.05).softmax(-1);loss=-(policy*reward).sum(-1).mean();loss.backward()
for tf32 in (False,True):
    torch.backends.cuda.matmul.allow_tf32=tf32
    for _ in range(8):step()
    torch.cuda.synchronize();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(32):step()
    end.record();torch.cuda.synchronize()
    results['TF32' if tf32 else 'IEEE']=dict(forward_backward_ms=begin.elapsed_time(end)/32,warmup_repetitions=8,measured_repetitions=32,tf32_allowed=tf32)
results.update(operation_ratio_IEEE_over_TF32=results['IEEE']['forward_backward_ms']/results['TF32']['forward_backward_ms'],wall_seconds_including_load_warmup=time.perf_counter()-started,peak_MiB=torch.cuda.max_memory_allocated()/2**20,device=torch.cuda.get_device_name(),torch_version=torch.__version__,utility_shape=list(utility.shape),batch_points=8192,actual_other_task='active v121 splitSW',scope='Synthetic single training utility graph; no model update, labels, accuracy score, epoch/fullCV speedup or end-to-end benchmark')
(study/'timing.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results),flush=True)
