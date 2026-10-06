"""Input-only frozen source feature preparation; explicit invocation required."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from compact_cached_features_v103 import ROOT,STUDY,SOURCE
from compact_source_features_v102 import Network as FrozenFeatureNetwork
from compact_five_stage_v100 import load_data
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
from train_dual8222_heads80 import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['random','pretrained'],required=True);a=ap.parse_args()
    destination=STUDY/(a.mode+'_features.npy');manifest=STUDY/(a.mode+'_cache.json')
    if destination.exists() or manifest.exists():raise RuntimeError('Existing cache immutable; no implicit regeneration')
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    torch.manual_seed(20261010);torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True);started=time.perf_counter()
    data,master,metadata=load_data('sw_joint')
    model=FrozenFeatureNetwork(master,metadata,a.mode+'_block256').cuda().eval()
    temporary=STUDY/(a.mode+'_features.partial.npy');n=len(data['x'])
    features=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float16,shape=(n,256))
    coverage=np.zeros(n,np.int8);centers=[];anchors=[]
    for lo,hi in zip(data['starts'][:-1],data['starts'][1:]):
        cc=np.arange(lo,hi,128)+64;centers.extend(cc);anchors.extend(np.minimum(cc,hi-1))
    torch.cuda.reset_peak_memory_stats()
    for begin in range(0,len(centers),32):
        cc=torch.tensor(centers[begin:begin+32],device='cuda');anchor=torch.tensor(anchors[begin:begin+32],device='cuda')
        ix,valid=window_indices(cc,data['lower'],data['upper'],length=256,anchors=anchor)
        relative=torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            hidden=model.features(data['x'][ix],relative,valid,data['absolute'][ix])
        rows=ix[:,64:192].cpu().numpy();take=valid[:,64:192].cpu().numpy();dest=rows[take]
        assert len(np.unique(dest))==len(dest) and not coverage[dest].any()
        values=hidden[:,64:192].float().cpu().numpy()[take].astype(np.float16)
        assert np.isfinite(values).all();features[dest]=values;coverage[dest]+=1
        if begin%1024==0:print('CACHE_ROWS',int(coverage.sum()),n,flush=True)
    assert (coverage==1).all();features.flush();del features;os.replace(temporary,destination)
    info=dict(mode=a.mode,complete=True,rows=n,columns=256,dtype='float16',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        source_feature_window=256,stride=128,retained_window_core=[64,192],relative='own native source-window anchor',
        cache_context='fixed well-local source256 windows; differs from moving SW1024-anchor v102 blocks',
        hidden_normalization='fixed per-point LayerNorm128 per expert',source_predictions_used=False,labels_used=False,
        fitted_statistics='none new; original64 training-well source buffers only',
        cache_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),seconds=time.perf_counter()-started,
        peak_MiB=torch.cuda.max_memory_allocated()/2**20,bytes=destination.stat().st_size,
        source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('gen_train_cache_compact_v103.py','compact_cached_features_v103.py','compact_source_features_v102.py')},
        timing_scope='load/preprocessing, frozen inference, coverage checks, disk and hash included; source training and later model training separately counted')
    atomic_json(manifest,info);print('CACHE_COMPLETE',json.dumps(info),flush=True)

if __name__=='__main__':main()
