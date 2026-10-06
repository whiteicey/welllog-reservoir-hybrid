"""CPU-only observational timings, not a controlled speed benchmark."""
from pathlib import Path
import json,re,time,statistics,psutil,subprocess

ROOT=Path(__file__).resolve().parents[3]
STUDY=Path(__file__).resolve().parent
OUT=ROOT/'outputs/pure_dl_two_seed_fullcv_v370'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def source_times(p):
    return {int(e):float(t) for e,t in re.findall(r'fold=2 epoch=(\d+)/80 .*?seconds=([\d.]+)',p.read_text(encoding='utf-8'))}
serial=source_times(OUT/'fold_2/original/SOURCE/SOURCE_serial_stdout.log')
parallel=source_times(OUT/'fold_2/seed_20261013/SOURCE/SOURCE_safe_parallel_stdout.log')
intervals=[]
for first,last in [(5,10),(10,15),(5,15)]:
    assert first in serial and last in serial and first in parallel and last in parallel
    a=serial[last]-serial[first];b=parallel[last]-parallel[first]
    intervals.append(dict(first_epoch=first,last_epoch=last,epochs=last-first,serial_seconds=a,parallel_seconds=b,parallel_over_serial=b/a))
sw=[]
for identity,fold in [('seed_20261013',1),('original',2)]:
    history=read(OUT/f'fold_{fold}/{identity}/SW104/validation_history.json')
    selected=[v['seconds'] for v in history if 6<=v['epoch']<=15]
    assert len(selected)==10
    concurrency=read(OUT/f'fold_{fold}/{identity}/SW104/SW104_TIMING.json')['concurrency'] if fold==1 else 'safe overlap; actual peer verified'
    if fold==1:assert concurrency=='solo'
    sw.append(dict(fold=fold,identity=identity,epochs=[6,15],mean_epoch_seconds=statistics.mean(selected),total_epoch_seconds=sum(selected),concurrency=concurrency))
state=read(STUDY/'BATCH_STATE.json');jobs=[]
for j in state['active_jobs']:
    p=psutil.Process(j['pid']);assert p.create_time()==j['create_time'] and p.cmdline()==j['command']
    jobs.append(dict(identity=j['identity'],fold=j['fold'],stage=j['stage'],pid=j['pid'],actual_alive=True))
driver=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
result=dict(observed_epoch=time.time(),active_jobs=jobs,SOURCE_epoch_intervals=intervals,SW_epoch_intervals=sw,
    SOURCE_comparison='Same original fold2/source code/batch64/64steps; original and seed13 differ in RNG/learned Q. Observational, not matched-seed replay.',
    SW_comparison='Same stage code; seed13/fold1 proven solo terminal compared with original/fold2 actual overlap. Different seeds/folds, observational not same-data replay. Original/fold1 SW104 was overlapped and is not a solo control.',
    included='SOURCE cumulative timer includes initialization before epoch5, which cancels in intervals; interval includes training, validation, train-only Q probes and checkpoint work. SW records per-epoch wall only. Complete process wall also includes loading/preprocessing/warmup/restore/write.',
    driver_snapshot=driver,maximum_GPU_jobs=2,training_changed=False,
    conclusion='Individual tasks slow during overlap. Overall acceleration is unproven until complete same-work batch wall is measured; do not sum overlapping process times as wall time. No new benchmark/training launched.',
    pending='Both seeds original folds2-4, full-stage terminal timings, native fusion scans and full OOF. Record serial/parallel periods and approval waiting separately in final timing report.')
(STUDY/'PARALLEL_TIMING_SNAPSHOT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
note='\n2026-10-02 observational parallel timing: fold2 SOURCE epoch5->15 serial original '+str(intervals[-1]['serial_seconds'])+'s versus overlapping seed13 '+str(intervals[-1]['parallel_seconds'])+'s (ratio '+str(intervals[-1]['parallel_over_serial'])+'). SW104 epoch6-15 mean seed13/fold1 proven solo '+str(sw[0]['mean_epoch_seconds'])+'s versus original/fold2 actual overlap '+str(sw[1]['mean_epoch_seconds'])+'s. Original/fold1 SW104 was overlapped, not a serial baseline; that interim label corrected by its terminal timing receipt. Different seeds/folds/Q; no controlled same-seed speed claim. Individual time is distinct from complete same-work total wall; overall acceleration remains unproven. No extra GPU benchmark or training change. Evidence PARALLEL_TIMING_SNAPSHOT.json. Pending full batch timings and OOF; preserve max2 with headroom/PERM exclusions.\n'
for p in [STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md']:
    previous=p.read_text(encoding='utf-8')
    previous='\n'.join(line for line in previous.split('\n') if not line.startswith('2026-10-02 observational parallel timing:'))
    p.write_text(previous+note,encoding='utf-8')
print(json.dumps(dict(SOURCE=intervals[-1],SW=sw,active_jobs=jobs)))
