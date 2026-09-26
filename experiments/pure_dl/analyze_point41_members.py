"""Explanatory CUDA decoding of saved paired41 members; no model inference or fitting."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
import train_invariant as base
from por_t3_patch_loss import probability
from gpu_job_guard_v2 import check_slot


def correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:
        return None
    return float(np.corrcoef(a,b)[0,1])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--expected-other-pid',type=int,required=True)
    args=ap.parse_args()
    admission=check_slot()
    assert [job['pid'] for job in admission['other_jobs']]==[args.expected_other_pid], admission
    print('GPU_ADMISSION',json.dumps(admission),flush=True)
    torch.set_num_threads(3)
    source=Path('outputs/pure_dl_paired_point_view_sequence')
    out=Path('outputs/point41_member_analysis');out.mkdir(exist_ok=True)
    started=time.perf_counter()
    checkpoint=torch.load(source/'fold_5.pt',map_location='cpu',weights_only=False)
    rec=base.load_split(base.ROOT.parent/'data','train')
    ys=[frame[base.TS].to_numpy(np.float64) for _,frame in rec]
    valid=[np.isfinite(y).all(1) for y in ys]
    truth=np.concatenate([y[m] for y,m in zip(ys,valid)])
    wells=np.concatenate([np.full(m.sum(),i,np.int32) for i,m in enumerate(valid)])
    folds=list(GroupKFold(5).split(truth,groups=wells))
    rows=folds[4][1]
    starts=np.r_[0,np.cumsum([len(frame) for _,frame in rec])]
    valid_to_raw=np.concatenate([np.flatnonzero(m)+starts[i] for i,m in enumerate(valid)])
    assert len(rec)==80 and truth.shape==(723556,3) and len(rows)==144665
    assert np.array_equal(checkpoint['validation_raw_rows'],valid_to_raw[rows])
    with np.load(source/'fold_5_distributions.npz') as data:
        assert np.array_equal(data['indices'],rows)
        gates=data['member_gate_logits']
        weights=data['member_weight']
        means=data['mixture_mu'];stds=data['mixture_std']
        assert gates.shape==(len(rows),2,3)
        assert weights.shape==(len(rows),2,3,8)
        assert means.shape==stds.shape==(len(rows),3,16)
        assert np.isfinite(gates).all() and np.isfinite(weights).all()
        assert np.isfinite(means).all() and np.isfinite(stds).all() and (stds>0).all()
        assert (weights>=0).all() and np.allclose(weights.sum(-1),1,atol=.002)
        for j,name in enumerate(base.TS):
            assert np.array_equal(data[name+'_centers'],checkpoint['centers'][j])
    predictions=np.empty((len(rows),2,3),np.float64)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for j,name in enumerate(base.TS):
            ct=np.asarray(checkpoint['centers'][j])
            action=np.linspace(ct[0],ct[-1],800) if j==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
            action=np.unique(np.r_[base.DEFAULTS[name],np.clip(action,*base.BOUNDS[name])])
            utility=torch.tensor(base.kernel_matrix(action,np.r_[base.DEFAULTS[name],ct],name).T,device='cuda',dtype=torch.float32)
            edges=torch.tensor(checkpoint['density_edges'][j],device='cuda',dtype=torch.float32)
            for member in range(2):
                component=slice(member*8,(member+1)*8)
                for begin in range(0,len(rows),2048):
                    end=min(begin+2048,len(rows))
                    w=torch.tensor(weights[begin:end,member,j],device='cuda',dtype=torch.float32)
                    w=w/w.sum(-1,keepdim=True)
                    mu=torch.tensor(means[begin:end,j,component],device='cuda',dtype=torch.float32)
                    sd=torch.tensor(stds[begin:end,j,component],device='cuda',dtype=torch.float32)
                    gate=torch.tensor(gates[begin:end,member,j],device='cuda',dtype=torch.float32).sigmoid()
                    cp=probability(j,w,mu,sd,edges)
                    joint=torch.cat([gate[:,None],(1-gate[:,None])*cp],-1)
                    chosen=(joint@utility).argmax(-1).cpu().numpy()
                    predictions[begin:end,member,j]=action[chosen]
            print('DECODED',name,flush=True)
    assert np.isfinite(predictions).all()
    retained=np.load(source/'fold_5_pred.npy')
    assert retained.shape==(len(rows),3) and np.isfinite(retained).all()
    y=truth[rows];default=base.dmask(y)
    diagnostics={}
    transforms=(np.log1p,np.log10,np.log)
    for j,name in enumerate(base.TS):
        use=~default[:,j]
        a=predictions[use,0,j]-y[use,j];b=predictions[use,1,j]-y[use,j]
        pa,pb=predictions[use,0,j],predictions[use,1,j]
        transformed=None
        if j==0 or ((pa>0).all() and (pb>0).all() and (y[use,j]>0).all()):
            f=transforms[j]
            transformed=correlation(f(pa)-f(y[use,j]),f(pb)-f(y[use,j]))
        diagnostics[name]={'nondefault_truth_rows':int(use.sum()),
            'physical_signed_error_correlation':correlation(a,b),
            'physical_absolute_error_correlation':correlation(np.abs(a),np.abs(b)),
            'transformed_signed_error_correlation':transformed,
            'transform':('log1p','log10','ln')[j],
            'exact_action_agreement_all_rows':float((predictions[:,0,j]==predictions[:,1,j]).mean()),
            'exact_action_agreement_nondefault_truth':float((pa==pb).mean())}
    result={'source':str(source),'rows':len(rows),'member0':'full206',
        'member1':'instantaneous41','member_scores':[base.metric(y,predictions[:,i]) for i in range(2)],
        'retained_pair_scores':base.metric(y,retained),'diagnostics':diagnostics,
        'error_definition':'prediction minus original truth; correlations restricted to true nondefault rows per original dmask. Physical signed/absolute and log-coordinate signed errors reported separately; default-valued predictions remain included.',
        'decoder':'original fixed expected-utility actions and conditional bin probability; no fitting or row-wise member selection',
        'quantization_note':'member mu/std/weights saved float16, reconstructed in float32; member scores approximate original unquantized member inference and are explanatory, not a new model score',
        'admission':admission,'wall_seconds':time.perf_counter()-started,
        'peak_gpu_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'batch_size':2048}
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
