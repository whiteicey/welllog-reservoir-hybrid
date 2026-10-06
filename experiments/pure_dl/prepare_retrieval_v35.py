"""Build SW training-label memory with exact whole-query-well exclusion.
No validation/test label is read for memory selection, values or distances.
"""
from pathlib import Path
from datetime import datetime
import hashlib,json,time,traceback
import numpy as np
import torch
from train_dual8222_heads80 import atomic_json
from gpu_job_guard_v2 import check_slot

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/retrieval_density_v35_20260929'

def main():
    started=time.perf_counter();torch.set_num_threads(3)
    if (STUDY/'memory_manifest.json').exists():raise RuntimeError('Prepared memory is immutable')
    print('ADMISSION',check_slot(min_free_gib=2.),flush=True)
    cache=torch.load(ROOT/'results/deep_learning/champion_scaling_v21_20260928/prepared_data.pt',map_location='cpu',weights_only=False)
    train=cache['train_raw'].numpy();n=len(cache['x']);starts=np.asarray(cache['starts'])
    rawwell=np.repeat(np.arange(80,dtype=np.int16),np.diff(starts));fitw=np.unique(rawwell[train])
    assert n==730268 and len(train)==578891 and len(fitw)==64
    valraw=np.flatnonzero(cache['raw_to_valid']>=0)[cache['validation']]
    assert set(fitw).isdisjoint(np.unique(rawwell[valraw]))
    with np.load(ROOT/'results/deep_learning/feature_structure_v20_20260928/side_views.npz') as z:
        amplitude=z['amplitude']
    columns=[7*c+j for c in range(9) for j in (0,2,5)]
    x=amplitude[:,columns].astype(np.float32)
    assert x.shape==(n,27) and np.isfinite(x).all()
    default=cache['default'][:,2].numpy();labels=cache['labels'][:,2].numpy()
    ref=[];counts=[]
    for w in fitw:
        candidates=train[(rawwell[train]==w)&(~default[train])]
        if not len(candidates):counts.append(dict(well=int(w),available=0,selected=0));continue
        take=np.linspace(0,len(candidates)-1,min(64,len(candidates)),dtype=np.int64)
        selected=candidates[take];ref.extend(selected.tolist())
        counts.append(dict(well=int(w),available=len(candidates),selected=len(selected)))
    ref=np.asarray(ref,np.int64);refwell=rawwell[ref]
    assert len(ref)>=32 and np.isin(ref,train).all() and not default[ref].any()
    targets=labels[ref].astype(np.float32);assert np.isfinite(targets).all()
    rng=np.random.default_rng(20261136);permutation=np.arange(len(ref))
    for w in np.unique(refwell):
        rows=np.flatnonzero(refwell==w);permutation[rows]=rng.permutation(rows)
    assert np.array_equal(refwell,refwell[permutation])
    np.savez(STUDY/'memory.npz',features=x[ref],target=targets,shuffled_target=targets[permutation],raw_rows=ref,well=refwell,permutation=permutation)
    torch.backends.cuda.matmul.allow_tf32=False
    memory=torch.as_tensor(x[ref],device='cuda');mw=torch.as_tensor(refwell,device='cuda')
    norm=memory.square().sum(-1);indices=np.empty((n,16),np.int32);distances=np.empty((n,16),np.float16)
    for lo in range(0,n,4096):
        hi=min(lo+4096,n);query=torch.as_tensor(x[lo:hi],device='cuda');qw=torch.as_tensor(rawwell[lo:hi],device='cuda')
        distance=(query.square().sum(-1)[:,None]+norm[None]-2*(query@memory.T)).clamp_min(0)/27.
        distance.masked_fill_(qw[:,None]==mw[None],float('inf'))
        values,ix=distance.topk(16,largest=False,sorted=True)
        assert bool(torch.isfinite(values).all()) and not bool((mw[ix]==qw[:,None]).any())
        indices[lo:hi]=ix.cpu().numpy();distances[lo:hi]=values.cpu().numpy().astype(np.float16)
        if lo%(4096*40)==0:print('INDEX_ROWS',hi,n,flush=True)
    assert not (refwell[indices]==rawwell[:,None]).any()
    np.save(STUDY/'neighbor_ids.npy',indices);np.save(STUDY/'neighbor_distance.npy',distances)
    perquery=np.sort(refwell[indices],axis=1);distinct=1+(perquery[:,1:]!=perquery[:,:-1]).sum(1)
    meta=dict(created_at=datetime.now().astimezone().isoformat(),raw_rows=n,train_rows=len(train),validation_rows=len(valraw),
        fit_wells=fitw.tolist(),validation_wells=np.unique(rawwell[valraw]).tolist(),reference_count=len(ref),reference_counts=counts,
        memory_selection='Up to64 evenly spaced among nondefault training rows in each well; no validation labels',
        descriptor_columns=columns,descriptor='Nine curves current value/mean51/std51; immutable train64 amplitude scaling',
        memory_target='Original training standardized logSW from prepared_data labels',
        candidate_count=16,candidate_rule='Fixed exact27D squared Euclidean/27; exclude the whole query well before Top16',
        whole_query_well_exclusion_verified=True,shuffled_values_preserve_reference_well=True,
        validation_labels_used_for_memory=False,test_data_used=False,
        train_median_distinct_reference_wells=float(np.median(distinct[train])),validation_median_distinct_reference_wells=float(np.median(distinct[valraw])),
        memory_sha256=hashlib.sha256((STUDY/'memory.npz').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.perf_counter()-started,
        feature_source='feature_structure_v20_20260928/side_views.npz',
        caveat='Candidate pool is fixed input distance; only within-pool target-specific attention is learned.')
    atomic_json(STUDY/'memory_manifest.json',meta)
    print('MEMORY_READY',json.dumps({k:v for k,v in meta.items() if k!='reference_counts'}),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        atomic_json(STUDY/'preparation_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
