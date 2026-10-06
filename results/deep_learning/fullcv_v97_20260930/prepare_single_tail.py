"""Prepare a new CPU controller without mutating any historical entrypoint."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[3];study=Path(__file__).resolve().parent
p=root/'experiments/pure_dl/run_overlap_fullcv_v97.py'
s=p.read_text(encoding='utf-8')
s=s.replace('Keep the fixed recipe; overlap only future root pretraining with main stages.',
            'Keep the fixed recipe; admit the remaining single SW tail using measured memory.')
s=s.replace("any(Path(a).name=='run_fullcv_v97.py' for a in cmd)",
            "any(Path(a).name=='run_overlap_fullcv_v97.py' for a in cmd)")
s=s.replace('controller_takeover_plan.json','tail_controller_takeover_plan.json')
s=s.replace('controller_takeover_done.json','tail_controller_takeover_done.json')
s=s.replace('overlap_preview.json','tail_admission_preview.json')
needle="                    minimum=6.8 if next_task['role']=='root' else 4.\n                    if not next_task['gpu'] or (sum(j['gpu'] for j in jobs.values())<2 and free_gib()>=minimum):start(next_task)"
assert s.count(needle)==1
s=s.replace(needle,"""                    minimum=6.8 if next_task['role']=='root' else 4.
                    single_tail=(not jobs and next_task['fold']==4 and next_task['role']=='stage'
                                 and next_task['stage'] in ('sw_additive','sw_utility','sw_sam','sw_density'))
                    if single_tail:minimum=3.
                    if not next_task['gpu'] or (sum(j['gpu'] for j in jobs.values())<2 and free_gib()>=minimum):
                        if single_tail:
                            # This read-only CUDA process exits before training.
                            value=subprocess.check_output([sys.executable,'-c','import torch;print(torch.cuda.mem_get_info()[0]/1024**3)'],text=True)
                            cuda_free=float(value.strip().splitlines()[-1])
                            event('tail_admission',label=next_task['label'],physical_free_gib=free_gib(),cuda_free_gib=cuda_free,
                                  physical_minimum_gib=3.,cuda_minimum_gib=4.,single_worker_only=True)
                            if cuda_free>=4.:start(next_task)
                        else:start(next_task)""")
s=s.replace("execution='main stages serial; next-fold root pretraining overlaps'",
            "execution='single final-fold SW tail; prior root overlap preserved; physical3/CUDA4 GiB admission'")
s=s.replace("execution='main stages serial; root pretraining overlap'",
            "execution='fixed stages; prior root overlap and measured single-tail admission'")
new=p.with_name('run_single_tail_fullcv_v97.py');assert not new.exists()
new.write_text(s,encoding='utf-8')
plan=dict(status='prepared',old_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
          new_source_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),old_controller_pid=79892,
          scope='only fold4 sw_additive/sw_utility/sw_sam/sw_density with no GPU worker live',
          physical_minimum_gib=3.,cuda_minimum_gib=4.,measured_SW_peak_MiB=1693.316,
          observed_physical_free_gib=3525/1024,observed_cuda_free_gib=10.8037109375,
          training_source_manifest_unchanged=True,termination='CPU controller only; no training process active',
          remaining_folds_authorization='authorization.json')
(study/'tail_admission_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
print('Prepared single-tail controller; old source unchanged',flush=True)
