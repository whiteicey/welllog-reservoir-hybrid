"""Standalone v97 five-fold inference; no training, labels, or constructor IO."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse, contextlib, hashlib, json, sys, time, zipfile
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
sys.path.insert(0,str(ROOT/'package'))
from src.data import INPUT_COLUMNS, read_curve
from src.feature import FeatureBuilder
from src.utils import DEFAULTS, BOUNDS, kernel_matrix, make_rows
from multiscale_sequence import window_indices
from por_t3_patch_loss import probability

TARGETS=('POR','PERM','SW')
CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')

def configure_runtime():
    torch.set_num_threads(3)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False

def load_inputs(path):
    path=Path(path);directory=path/'test' if (path/'test').is_dir() else path
    files=sorted(directory.glob('*.txt'))
    if not files:raise FileNotFoundError(f'No curve TXT files in {directory}')
    return [(p.stem,read_curve(p)) for p in files]

def amplitude_view(frame,meta):
    cols=[]
    for name in CURVES:
        series=frame[name].replace([np.inf,-np.inf],np.nan).interpolate(method='linear',limit_direction='both').fillna(meta['medians'][name])
        values=series.to_numpy(np.float64)
        if name in ('GR','RXO','RT'):values=np.log1p(np.maximum(values,0))
        series=pd.Series(values);cols.append(values)
        cols.extend(series.rolling(w,center=True,min_periods=1).mean().to_numpy() for w in (9,51,201))
        cols.extend(series.rolling(w,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in (9,51,201))
    values=np.stack(cols,-1).astype(np.float32)
    return np.clip((values-np.asarray(meta['center'],np.float32))/np.asarray(meta['scale'],np.float32),-8,8).astype(np.float16)

def action_grid(preprocessing,c):
    ct=np.asarray(preprocessing['centers'][c]);name=TARGETS[c]
    grid=np.linspace(ct[0],ct[-1],800) if c==2 else np.geomspace(max(ct[0]*.9,.001),ct[-1]*1.1,800)
    return np.unique(np.r_[DEFAULTS[name],np.clip(grid,*BOUNDS[name])])

class Predictor:
    def __init__(self,model_dir=None,device='cuda',folds=(1,2,3,4,5)):
        self.device=torch.device(device);self.model_dir=Path(model_dir or ROOT/'models')
        self.summary=json.loads((self.model_dir/'delivery_summary.json').read_text(encoding='utf-8'))
        self.contract=json.loads((self.model_dir/'feature_contract.json').read_text(encoding='utf-8'))
        self.bundles={};self.builders={}
        for f in folds:
            path=self.model_dir/f'fold_{f}.pt';record=next(r for r in self.summary['folds'] if r['fold']==f)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError(f'Changed model: fold{f}')
            # Complete trusted graphs were exported with cloudpickle, preserving
            # trained hooks. Importing source classes does not call constructors.
            bundle=torch.load(path,map_location='cpu',weights_only=False)
            assert bundle['original_fold']==f
            for model in bundle['models']:model.to(self.device).eval().requires_grad_(False)
            self.bundles[f]=bundle
            builder=FeatureBuilder();builder.medians_=bundle['preprocessing']['medians']
            builder.enhanced_feature_names_=self.contract['full_schema'];self.builders[f]=builder
        self.actions=[];self.utilities={}
        for c,name in enumerate(TARGETS):
            lo=min(float(b['preprocessing']['centers'][c][0]) for b in self.bundles.values())
            hi=max(float(b['preprocessing']['centers'][c][-1]) for b in self.bundles.values())
            grid=np.linspace(lo,hi,800) if c==2 else np.geomspace(max(lo*.9,.001),hi*1.1,800)
            self.actions.append(np.unique(np.r_[DEFAULTS[name],np.clip(grid,*BOUNDS[name])]))
        for f,b in self.bundles.items():
            self.utilities[f]=[torch.tensor(kernel_matrix(self.actions[c],np.r_[DEFAULTS[name],b['preprocessing']['centers'][c]],name).T,
                         device=self.device,dtype=torch.float32) for c,name in enumerate(TARGETS)]

    def features(self,frame,f):
        frame=frame.reindex(columns=INPUT_COLUMNS).copy() # Never pass output labels to features.
        p=self.bundles[f]['preprocessing'];builder=self.builders[f]
        base,enhanced=builder._transform_frames(frame);builder.base_feature_names_=list(base.columns)
        if list(enhanced.columns)!=self.contract['full_schema']:raise ValueError('Feature schema changed')
        selected=p['selected_feature_indices']
        if [self.contract['full_schema'][i] for i in selected]!=p['selected_feature_names']:raise ValueError('Selected feature schema changed')
        values=builder.transform(frame).enhanced[:,selected]
        values=np.clip((values-np.asarray(p['feature_median'],np.float32))/np.asarray(p['feature_scale'],np.float32),-8,8).astype(np.float16)
        side=amplitude_view(frame,self.bundles[f]['amplitude'])
        dtype=torch.float16 if self.device.type=='cuda' else torch.float32
        x=torch.as_tensor(values,device=self.device,dtype=dtype)
        amplitude=torch.cat((x,torch.as_tensor(side,device=self.device,dtype=dtype)),-1)
        depth=torch.as_tensor(frame['DEPTH'].to_numpy(np.float32),device=self.device)
        if not torch.isfinite(depth).all():raise ValueError('Nonfinite depth')
        absolute=((depth-p['absolute_depth_median'])/p['absolute_depth_scale']).clamp(-8,8)
        sample=np.minimum(((np.arange(128)+.5)*len(frame)/128).astype(np.int64),len(frame)-1)
        bank=x[torch.as_tensor(sample,device=self.device)][None]
        return x,amplitude,depth,absolute,bank

    @torch.inference_mode()
    def predict_well(self,frame,batch_windows=32):
        n=len(frame)
        if n==0:raise ValueError('Empty well')
        result=np.empty((n,3),np.float64)
        lower=torch.zeros(n,device=self.device,dtype=torch.long);upper=torch.full((n,),n,device=self.device,dtype=torch.long)
        centers=torch.arange(0,n,128,device=self.device)+64;anchors=centers.clamp_max(n-1)
        total=[torch.zeros(n,len(a),device=self.device) for a in self.actions]
        for f,bundle in self.bundles.items():
            x,side,depth,absolute,bank=self.features(frame,f);p=bundle['preprocessing']
            for c,model in enumerate(bundle['models']):
                length=1024 if c==2 else 256;batch=min(batch_windows,16) if c==2 else batch_windows
                lo=length//2-64;hi=lo+128;coverage=np.zeros(n,np.uint8)
                edges=torch.as_tensor(p['density_edges'][c],device=self.device,dtype=torch.float32)
                for start in range(0,len(centers),batch):
                    cc=centers[start:start+batch];aa=anchors[start:start+batch]
                    ix,valid=window_indices(cc,lower,upper,length=length,anchors=aa)
                    relative=torch.tanh((depth[ix]-depth[aa,None])/10)
                    amp=torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else contextlib.nullcontext()
                    with amp:gate,raw=model((side if c<2 else x)[ix],relative,valid,absolute_depth=absolute[ix],well_inputs=bank.expand(len(cc),-1,-1))
                    use=valid[:,lo:hi];rows=ix[:,lo:hi][use];rr=rows.cpu().numpy()
                    assert not coverage[rr].any() and len(np.unique(rr))==len(rr);coverage[rr]+=1
                    gate=gate[:,lo:hi][use][:,0].float().sigmoid();raw=raw[:,lo:hi][use][:,0].float()
                    density=probability(c,raw[:,0].softmax(-1),raw[:,1],F.softplus(raw[:,2])+.025,edges)
                    joint=torch.cat((gate[:,None],(1-gate[:,None])*density),-1)
                    total[c][rows]+=(joint@self.utilities[f][c])/len(self.bundles)
                assert (coverage==1).all()
            del x,side,depth,absolute,bank
        for c in range(3):result[:,c]=self.actions[c][total[c].argmax(-1).cpu().numpy()]
        assert np.isfinite(result).all()
        return result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir','--data-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'result.json')
    ap.add_argument('--model_dir','--model-dir',type=Path,default=ROOT/'models')
    ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--batch_windows','--batch-windows',type=int,default=32)
    a=ap.parse_args()
    if a.output.suffix.lower()!='.json' or a.batch_windows<1:ap.error('Use a JSON output and positive batch_windows')
    configure_runtime();start=time.perf_counter()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    predictor=Predictor(a.model_dir,device);results=[];rows=0
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    for well,frame in load_inputs(a.data_dir):
        pred=predictor.predict_well(frame,a.batch_windows)
        results.append(dict(logId=well,predictions=make_rows(frame['DEPTH'].to_numpy(),pred,well)))
        rows+=len(frame);print('PREDICTED',well,len(frame),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(resultData=results),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with zipfile.ZipFile(a.output.parent/'result.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.write(a.output,'result.json')
    report=dict(wells=len(results),rows=rows,seconds=time.perf_counter()-start,device=device,folds=list(predictor.bundles),
                aggregation='equal five-fold expected utilities; original256 bins and800 actions; stride128',
                official_oof=predictor.summary['official_oof_scores'],test_accuracy='unknown; test labels unavailable',
                peak_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None)
    a.output.with_suffix('.inference.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('INFERENCE_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
