"""Finite V370 maintenance: completed source diagnostics, not learned weights.

Keep every observation epoch and exact Q table, the latest full observation,
exported full Q trace, best/last checkpoints, predictions, input caches and logs.
Only earlier large train-probe arrays are replaced with compact summaries.
Never compact an active or unverified source; this process cannot launch jobs.
"""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import psutil
from fullcv_pair_context_v370 import ROOT,STUDY,OUTPUT,BASE_SEEDS,own_dir,read,write,sha

def compact(identity,fold):
    out=own_dir(identity,fold)/'SOURCE'
    report_path=out/f'fold_{fold}_report.json';timing_path=out/'SOURCE_TIMING.json'
    if not report_path.exists() or not timing_path.exists():return False
    report=read(report_path);timing=read(timing_path)
    if not timing['terminal'] or timing['exit_code'] or psutil.pid_exists(timing['pid']):return False
    assert report['fold']==fold and report['best_checkpoint_predictions_reproduced']
    assert report['all_validation_rows_covered_once']
    required=[out/f'fold_{fold}.pt',out/f'fold_{fold}_last_state.pt',out/f'fold_{fold}_pred.npy',
              out/'prepared_data.pt',out/f'fold_{fold}_conflict_weights.npz']
    if not all(p.is_file() for p in required):return False
    manifest=out/'SOURCE_DIAGNOSTIC_COMPACTION.json'
    if manifest.exists():
        if read(manifest)['status']!='complete':raise RuntimeError('Interrupted maintenance requires explicit inspection')
        return True
    root_scope=OUTPUT.resolve();target=out.resolve()
    assert target.is_relative_to(root_scope) and target.parent.name in BASE_SEEDS
    trace=out/f'fold_{fold}_conflict_weights.npz'
    with np.load(trace) as z:epochs=z['epochs'].copy();weights=z['weights'].copy()
    assert len(epochs)>0 and np.all(np.diff(epochs)>0)
    observations=sorted((out/f'fold_{fold}_dynamics').glob('observation_*.npz'))
    assert len(observations)==len(epochs)
    latest=int(epochs[-1]);preserved={str(p.relative_to(ROOT)):sha(p) for p in required}
    entries=[];freed=0;start=time.perf_counter()
    write(manifest,dict(status='preparing',identity=identity,fold=fold,preserved_sha256=preserved,
          scope='completed new V370 source only; older diagnostic arrays compacted, no checkpoint deleted',latest_full_observation=latest))
    for path in observations:
        assert path.resolve().is_relative_to(target/f'fold_{fold}_dynamics')
        before=path.stat().st_size;before_sha=sha(path)
        with np.load(path) as z:
            epoch=int(z['epoch']);q=z['weights'].copy();index=int(np.flatnonzero(epochs==epoch)[0])
            assert np.array_equal(q,weights[index])
            if epoch==latest:continue
            stats={}
            for key in z.files:
                if key in ('epoch','weights'):continue
                value=z[key];finite=value[np.isfinite(value)]
                stats[key]=dict(shape=list(value.shape),dtype=str(value.dtype),
                    array_sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                    finite_count=int(len(finite)),minimum=float(finite.min()) if len(finite) else None,
                    maximum=float(finite.max()) if len(finite) else None,
                    mean=float(finite.mean(dtype=np.float64)) if len(finite) else None,
                    std=float(finite.std(dtype=np.float64)) if len(finite) else None)
        temporary=path.with_suffix('.npz.compact.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,epoch=epoch,weights=q)
        with np.load(temporary) as z:assert int(z['epoch'])==epoch and np.array_equal(z['weights'],weights[index])
        temporary.replace(path)
        after=path.stat().st_size;freed+=before-after
        entries.append(dict(path=str(path.relative_to(ROOT)),epoch=epoch,bytes_before=before,bytes_after=after,
                            sha256_before=before_sha,sha256_after=sha(path),removed_arrays_summary=stats))
        write(manifest,dict(status='compacting',identity=identity,fold=fold,preserved_sha256=preserved,
              latest_full_observation=latest,entries=entries,bytes_reclaimed=freed))
    for path,digest in preserved.items():assert sha(ROOT/path)==digest,path
    # Q export and final-state probe recovery still have their exact dependencies.
    for path in observations:
        with np.load(path) as z:
            epoch=int(z['epoch']);index=int(np.flatnonzero(epochs==epoch)[0])
            assert np.array_equal(z['weights'],weights[index])
            if epoch==latest:assert all(k in z.files for k in ('nll','conditional_entropy','target_tail','gate_probability','cdf_l1_change'))
    write(manifest,dict(status='complete',identity=identity,fold=fold,preserved_sha256=preserved,
          all_observation_weights_equal_exported_Q=True,latest_full_observation=latest,entries=entries,
          bytes_reclaimed=freed,seconds=time.perf_counter()-start,training_stopped=False,
          checkpoint_deletions=0,old_experiments_touched=False,scope='earlier completed-source diagnostic arrays summarized'))
    note=f'\nV370 {identity}/fold{fold} completed-source diagnostic compaction reclaimed {freed/2**20:.3f}MiB. Every epoch/Q exact to exported trace; latest full probe, best/last weights, predictions/cache/trace/logs SHA unchanged. No active or old outputs changed, no checkpoint deleted. Manifest: {manifest.relative_to(ROOT).as_posix()}.\n'
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as h:h.write(note)
    print('COMPACT_COMPLETED_SOURCE',identity,fold,freed,flush=True)
    return True

def main():
    receipt=STUDY/'MAINTENANCE_LAUNCH.json'
    if receipt.exists():raise RuntimeError('Maintenance already launched; inspect existing process, no duplicate/retry')
    write(receipt,dict(pid=os.getpid(),started_epoch=time.time(),finite_sources=8,interval_seconds=30,
          GPU_work=False,launch_training=False,old_checkpoints_deleted=False))
    pending={(identity,fold) for fold in range(1,5) for identity in BASE_SEEDS}
    while pending:
        state=read(STUDY/'BATCH_STATE.json')
        if state['status'] in ('failed_or_stopped','stopped'):
            write(STUDY/'MAINTENANCE_STATE.json',dict(status='held_after_batch_terminal_failure_or_stop',pending=sorted(pending)));return
        for identity,fold in sorted(pending.copy()):
            if compact(identity,fold):pending.remove((identity,fold))
        write(STUDY/'MAINTENANCE_STATE.json',dict(status='waiting_for_completed_sources' if pending else 'complete',
              pid=os.getpid(),pending=sorted(pending),observed_epoch=time.time(),GPU_work=False))
        if pending:time.sleep(30)

if __name__=='__main__':main()
