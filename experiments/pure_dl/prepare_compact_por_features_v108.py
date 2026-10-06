"""Input-only POR inference; all additional feature fitting uses training rows."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,time
import numpy as np
import torch
from torch.nn import functional as F
from compact_por_features_v108 import ROOT,STUDY,TEACHER
from compact_five_stage_v100 import load_data
from compact_mean_scope_v101 import Network
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot
from train_dual8222_heads80 import atomic_json,atomic_array

def main():
    if (STUDY/'feature_manifest.json').exists():raise RuntimeError('Existing preparation immutable')
    torch.set_num_threads(3);print('ADMISSION',check_slot(min_free_gib=3.),flush=True)
    torch.manual_seed(20261010);torch.use_deterministic_algorithms(True,warn_only=True)
    tick=time.perf_counter();data,master,metadata=load_data('por')
    model=Network(master).cuda();saved=torch.load(TEACHER,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);model.eval().requires_grad_(False)
    n=len(data['x']);values=np.zeros((n,4),np.float32);coverage=np.zeros(n,np.int8)
    centers=[];anchors=[]
    for lo,hi in zip(data['starts'][:-1],data['starts'][1:]):
        cc=np.arange(lo,hi,128)+64;centers.extend(cc);anchors.extend(np.minimum(cc,hi-1))
    torch.cuda.reset_peak_memory_stats()
    for begin in range(0,len(centers),32):
        cc=torch.tensor(centers[begin:begin+32],device='cuda');anchor=torch.tensor(anchors[begin:begin+32],device='cuda')
        ix,valid=window_indices(cc,data['lower'],data['upper'],length=256,anchors=anchor)
        relative=torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            gate,raw=model(data['x'][ix],relative,valid,absolute_depth=data['absolute'][ix])
        weights=raw[...,0,:].float().softmax(-1);means=raw[...,1,:].float();sd=F.softplus(raw[...,2,:].float())+.025
        mean=(weights*means).sum(-1)
        variance=(weights*(means.square()+sd.square())).sum(-1)-mean.square()
        entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1)/np.log(weights.shape[-1])
        stats=torch.stack((gate.float().sigmoid(),mean,variance.clamp_min(0),entropy),-1).squeeze(2)
        rows=ix[:,64:192].cpu().numpy();take=valid[:,64:192].cpu().numpy();dest=rows[take]
        assert len(np.unique(dest))==len(dest) and not coverage[dest].any()
        values[dest]=stats[:,64:192].cpu().numpy()[take];coverage[dest]+=1
    assert (coverage==1).all() and np.isfinite(values).all()
    names=master['selected_feature_names'];indices=[]
    for curve in ('GR','DEN','CNL','RT'):
        matches=[i for i,name in enumerate(names) if name==curve]
        if len(matches)!=1:raise RuntimeError(('Need exact original curve column',curve,matches,names))
        indices.extend(matches)
    control=data['x'][:,indices].float().cpu().numpy();train=data['train_raw'].cpu().numpy();normalization={}
    for mode,array in [('curve_control',control),('por_posterior',values)]:
        median=np.median(array[train],axis=0);q=np.percentile(array[train],[25,75],axis=0);scale=np.maximum(q[1]-q[0],1e-4)
        normalized=np.clip((array-median)/scale,-8,8).astype(np.float16)
        assert np.isfinite(normalized).all();atomic_array(STUDY/(mode+'_features.npy'),normalized)
        normalization[mode]=dict(median=median.tolist(),iqr=scale.tolist(),sha256=hashlib.sha256((STUDY/(mode+'_features.npy')).read_bytes()).hexdigest())
    manifest=dict(complete=True,rows=n,columns=4,train_rows=len(train),teacher_sha256=hashlib.sha256(TEACHER.read_bytes()).hexdigest(),
        teacher_stage=2,total_selected_dependency_stages=4,additional_teacher_training=False,SW_labels_used=False,validation_labels_used=False,
        features=['POR default probability','conditional transformed mean','conditional transformed variance','component entropy'],
        variance_convention='Student-t3 equal-variance sd parameter; statistics in transformed target coordinates',
        window=256,stride=128,core=[64,192],coverage_exactly_once=True,normalization_train_rows_only=normalization,
        control_curves=['GR','DEN','CNL','RT'],seconds=time.perf_counter()-tick,peak_MiB=torch.cuda.max_memory_allocated()/2**20,
        timing_scope='data/teacher load, inference, training-row normalization and disk/hash included; historical teacher training separately counted')
    atomic_json(STUDY/'feature_manifest.json',manifest);print('PREPARED',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
