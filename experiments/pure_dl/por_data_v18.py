"""POR-only targets and original-score evaluator for v18; no other task loss."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import hashlib
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
import train_invariant as base
from train_dual8222_heads80 import SOURCE, atomic_json, atomic_torch
from multiscale_sequence import window_indices

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/por_architecture_v18_20260928'
CACHE_STUDY=ROOT/'results/deep_learning/latent_fold5_v16_20260928'
REFERENCE=ROOT/'outputs/pure_dl_independent_v8_selected_fold5'


def prepare():
    """Reuse train-only X cache; newly prepare target-only POR arrays in v18."""
    STUDY.mkdir(parents=True,exist_ok=True)
    target_path=STUDY/'target_data.npz'
    if target_path.exists():
        raise RuntimeError('Target cache exists; do not overwrite or refit silently')
    started=time.perf_counter()
    manifest=json.loads((CACHE_STUDY/'data_outer_manifest.json').read_text(encoding='utf-8'))
    assert not manifest['outer_fifth_used_for_fitting']
    with np.load(CACHE_STUDY/'data_outer.npz') as z:
        starts=z['starts'];fit=z['fit_raw'];monitor=z['monitor_raw'];wells=z['wells'];fitmask=z['fitmask'];included=z['included_wells']
    records=base.load_split(ROOT.parent/'data','train')
    assert len(records)==80 and np.array_equal(included,np.arange(80))
    assert np.array_equal(starts,np.r_[0,np.cumsum([len(f) for _,f in records])])
    yy=np.concatenate([f[base.TS].to_numpy(np.float64) for _,f in records])
    truth=yy[:,0].copy();default=base.dmask(yy)[:,0]
    labeled=np.isfinite(yy).all(1)
    assert np.array_equal(np.flatnonzero(labeled&np.isin(wells,manifest['fit_wells'])),fit)
    assert np.array_equal(np.flatnonzero(labeled&np.isin(wells,manifest['monitor_wells'])),monitor)
    assert (len(fit),len(monitor))==(578891,144665)
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    assert np.array_equal(fit,source['training_raw_rows']) and np.array_equal(monitor,source['validation_raw_rows'])
    median=float(source['regression_median'][0]);scale=float(source['regression_scale'][0])
    centers=np.asarray(source['centers'][0],dtype=np.float64).copy()
    edges=np.asarray(source['density_edges'][0],dtype=np.float32).copy()
    assert scale>0 and centers.shape==(256,) and edges.shape==(257,)
    zlabels=np.zeros(len(truth),np.float32)
    zlabels[fit]=(np.log1p(truth[fit])-median)/scale
    train_default=np.zeros(len(truth),bool);train_default[fit]=default[fit]
    numeric=zlabels[fit][~default[fit]]
    rate=float(default[fit].mean())
    meta=dict(target='POR',target_median=median,target_scale=scale,centers=centers,density_edges=edges,
        initial_means=np.quantile(numeric,np.linspace(.05,.95,8)).astype(np.float32),
        training_default_rate=rate,initial_std=float(np.log(2)+.025),
        metadata_source=str(SOURCE.relative_to(ROOT)),initialization='random backbone,zero output weights,train64 POR prior and quantile means')
    with np.load(REFERENCE/'oof.npz') as ref:
        assert np.array_equal(ref['truth'][ref['fold']==4,0],truth[monitor])
        assert np.array_equal(ref['well'][ref['fold']==4],wells[monitor])
    np.savez(target_path,truth=truth,default_truth=default,labels=zlabels,default=train_default)
    atomic_torch(STUDY/'model_metadata.pt',meta)
    out=dict(target='POR',created_at=datetime.now().astimezone().isoformat(),fit_wells=manifest['fit_wells'],
        monitor_wells=manifest['monitor_wells'],fit_points=len(fit),monitor_points=len(monitor),raw_points=len(truth),
        feature_cache=str((CACHE_STUDY/'data_outer.npz').relative_to(ROOT)),
        feature_manifest_sha256=hashlib.sha256((CACHE_STUDY/'data_outer_manifest.json').read_bytes()).hexdigest(),
        feature_statistics_from_original64_only=True,POR_training_labels_only=True,
        target_metadata_source=meta['metadata_source'],source_checkpoint_training_rows_verified=True,
        old_network_weights_loaded_into_candidate=False,old_quality_used=False,Q=1,
        transform='(log1p(POR)-training_median)/training_IQR',target_median=median,target_scale=scale,
        support_min=float(centers[0]),support_max=float(centers[-1]),default_value=float(base.DEFAULTS['POR']),
        training_default_rate=rate,training_nondefault_points=int(len(numeric)),
        original_row_order_verified=True,remaining_folds_started=False,seconds=time.perf_counter()-started)
    atomic_json(STUDY/'data_manifest.json',out)
    print('POR_DATA_READY',json.dumps(out,ensure_ascii=False))


def load_data():
    with np.load(CACHE_STUDY/'data_outer.npz') as z:
        keys=('x','depth','lower','upper','fitmask','fit_raw','monitor_raw','starts','included_wells','wells')
        d={k:z[k] for k in keys}
    with np.load(STUDY/'target_data.npz') as z:
        d.update({k:z[k] for k in z.files})
    for k in ('x','depth','lower','upper','fitmask','fit_raw','labels','default'):
        d[k]=torch.from_numpy(d[k]).cuda()
    assert not bool(d['fitmask'][torch.as_tensor(d['monitor_raw'],device='cuda')].any())
    return d


class Evaluator:
    def __init__(self,data,ids,metadata):
        self.d=data;self.ids=np.asarray(ids,dtype=np.int64)
        self.local=np.full(len(data['truth']),-1,np.int64);self.local[self.ids]=np.arange(len(self.ids))
        selected=set(np.unique(data['wells'][self.ids]).tolist())
        self.centers=[];self.anchors=[]
        for j,w in enumerate(data['included_wells']):
            if int(w) not in selected: continue
            lo,hi=data['starts'][j:j+2];cc=np.arange(lo,hi,128)+64
            self.centers.extend(cc);self.anchors.extend(np.minimum(cc,hi-1))
        ct=np.asarray(metadata['centers'])
        self.actions=np.unique(np.r_[base.DEFAULTS['POR'],np.clip(np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800),*base.BOUNDS['POR'])])
        self.utilities=torch.tensor(base.kernel_matrix(self.actions,np.r_[base.DEFAULTS['POR'],ct],'POR').T,device='cuda',dtype=torch.float32)

    @torch.no_grad()
    def __call__(self,model):
        model.eval();d=self.d
        pred=np.full(len(self.ids),np.nan,np.float64);seen=np.zeros(len(pred),np.int8)
        bces=nlls=0.;nnum=0
        for start in range(0,len(self.centers),32):
            cc=torch.tensor(self.centers[start:start+32],device='cuda')
            anchors=torch.tensor(self.anchors[start:start+32],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],anchors=anchors)
            rel=torch.tanh((d['depth'][ix]-d['depth'][anchors,None])/10)
            with torch.autocast('cuda',dtype=torch.bfloat16):gate,raw=model(d['x'][ix],rel,valid)
            rawids=ix[:,64:192].cpu().numpy();loc=self.local[rawids]
            use=valid[:,64:192].cpu().numpy()&(loc>=0);dest=loc[use]
            assert np.all(seen[dest]==0) and len(np.unique(dest))==len(dest)
            seen[dest]+=1;mask=torch.as_tensor(use,device='cuda')
            g=gate[:,64:192][mask].float();r=raw[:,64:192][mask].float()
            probability=model.masses(r)
            pg=g.sigmoid();mass=torch.cat((pg[:,None],(1-pg[:,None])*probability),dim=-1)
            pred[dest]=self.actions[(mass@self.utilities).argmax(-1).cpu().numpy()]
            ids=rawids[use];dd=torch.as_tensor(d['default_truth'][ids],device='cuda')
            z=torch.as_tensor((np.log1p(d['truth'][ids])-model.target_median)/model.target_scale,device='cuda',dtype=torch.float32)
            bces+=float(F.binary_cross_entropy_with_logits(g,dd.float(),reduction='sum'))
            nlls+=float(model.log_components(r,z)[~dd].sum());nnum+=int((~dd).sum())
        assert (seen==1).all() and np.isfinite(pred).all()
        y=d['truth'][self.ids];dd=d['default_truth'][self.ids]
        scores=100*np.maximum(0,1-np.abs(pred-y)/(.08*(np.abs(y)+1e-5)))
        perwell=[dict(well=int(w),n=int((d['wells'][self.ids]==w).sum()),POR=float(scores[d['wells'][self.ids]==w].mean())) for w in np.unique(d['wells'][self.ids])]
        stats=dict(POR=float(scores.mean()),default_POR=float(scores[dd].mean()),numeric_POR=float(scores[~dd].mean()),
            default_points=int(dd.sum()),numeric_points=int((~dd).sum()),BCE=bces/len(scores),conditional_NLL=nlls/max(nnum,1),per_well=perwell)
        return pred,stats


if __name__=='__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    prepare()
