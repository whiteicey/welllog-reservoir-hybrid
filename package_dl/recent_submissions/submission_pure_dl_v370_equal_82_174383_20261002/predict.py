"""Standalone two-seed/five-fold inference; complete trusted graphs, no constructor IO."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,contextlib,hashlib,json,sys,time,zipfile
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
sys.path.insert(0,str(ROOT/'package'))
from src.data import INPUT_COLUMNS,read_curve
from src.feature import FeatureBuilder
from src.utils import DEFAULTS,BOUNDS,kernel_matrix,make_rows
from multiscale_sequence import window_indices
TARGETS=('POR','PERM','SW');CURVES=('GR','PE','SP','CAL','AC','DEN','CNL','RXO','RT')

def configure_runtime():
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False

def load_inputs(path):
    path=Path(path);directory=path/'test' if (path/'test').is_dir() else path
    files=sorted(directory.glob('*.txt'))
    if not files:raise FileNotFoundError(f'No curve TXT files in {directory}')
    return [(p.stem,read_curve(p)) for p in files]

def amplitude_view(frame,meta):
    cols=[]
    for name in CURVES:
        s=frame[name].replace([np.inf,-np.inf],np.nan).interpolate(method='linear',limit_direction='both').fillna(meta['medians'][name])
        v=s.to_numpy(np.float64)
        if name in ('GR','RXO','RT'):v=np.log1p(np.maximum(v,0))
        s=pd.Series(v);cols.append(v)
        cols.extend(s.rolling(w,center=True,min_periods=1).mean().to_numpy() for w in (9,51,201))
        cols.extend(s.rolling(w,center=True,min_periods=1).std(ddof=0).fillna(0).to_numpy() for w in (9,51,201))
    return np.clip((np.stack(cols,-1).astype(np.float32)-np.asarray(meta['center'],np.float32))/np.asarray(meta['scale'],np.float32),-8,8).astype(np.float16)

class Predictor:
    def __init__(self,model_dir=None,device='cuda',folds=(1,2,3,4,5),weight_policy='fixed'):
        self.device=torch.device(device);self.model_dir=Path(model_dir or ROOT/'models');self.weight_policy=weight_policy
        assert weight_policy in ('diagnostic','fixed')
        self.summary=json.loads((self.model_dir/'delivery_summary.json').read_text(encoding='utf-8'))
        self.contract=json.loads((self.model_dir/'feature_contract.json').read_text(encoding='utf-8'))
        self.bundles={};self.builders={}
        for f in folds:
            path=self.model_dir/f'fold_{f}.pt';record=next(r for r in self.summary['folds'] if r['fold']==f)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed model '+str(f))
            b=torch.load(path,map_location='cpu',weights_only=False);assert b['original_fold']==f
            for pair in b['pairs']:
                for model in pair['models']:model.to(self.device).eval().requires_grad_(False)
            self.bundles[f]=b
            builder=FeatureBuilder();builder.medians_=b['preprocessing']['medians'];builder.enhanced_feature_names_=self.contract['full_schema'];self.builders[f]=builder
        self.actions=[];self.utilities={}
        for c,t in enumerate(TARGETS):
            lo=min(float(b['preprocessing']['centers'][c][0]) for b in self.bundles.values());hi=max(float(b['preprocessing']['centers'][c][-1]) for b in self.bundles.values())
            grid=np.linspace(lo,hi,800) if c==2 else np.geomspace(max(lo*.9,.001),hi*1.1,800)
            self.actions.append(np.unique(np.r_[DEFAULTS[t],np.clip(grid,*BOUNDS[t])]))
        for f,b in self.bundles.items():
            self.utilities[f]=[torch.tensor(kernel_matrix(self.actions[c],np.r_[DEFAULTS[t],b['preprocessing']['centers'][c]],t).T,device=self.device,dtype=torch.float32) for c,t in enumerate(TARGETS)]

    def features(self,frame,f):
        frame=frame.reindex(columns=INPUT_COLUMNS).copy()
        p=self.bundles[f]['preprocessing'];builder=self.builders[f]
        base,enhanced=builder._transform_frames(frame);builder.base_feature_names_=list(base.columns)
        assert list(enhanced.columns)==self.contract['full_schema']
        selected=p['selected_feature_indices'];assert [self.contract['full_schema'][i] for i in selected]==p['selected_feature_names']
        x=builder.transform(frame).enhanced[:,selected]
        x=np.clip((x-np.asarray(p['feature_median'],np.float32))/np.asarray(p['feature_scale'],np.float32),-8,8).astype(np.float16)
        side=amplitude_view(frame,self.bundles[f]['amplitude']);dtype=torch.float16 if self.device.type=='cuda' else torch.float32
        x=torch.as_tensor(x,device=self.device,dtype=dtype);side=torch.cat((x,torch.as_tensor(side,device=self.device,dtype=dtype)),-1)
        depth=torch.as_tensor(frame['DEPTH'].to_numpy(np.float32),device=self.device);assert torch.isfinite(depth).all()
        absolute=((depth-p['absolute_depth_median'])/p['absolute_depth_scale']).clamp(-8,8)
        sample=np.minimum(((np.arange(128)+.5)*len(frame)/128).astype(np.int64),len(frame)-1)
        bank=x[torch.as_tensor(sample,device=self.device)][None]
        return x,side,depth,absolute,bank

    @torch.inference_mode()
    def predict_records(self,records,batch_windows=32):
        sizes=[len(frame) for _,frame in records];starts=np.r_[0,np.cumsum(sizes)];n=int(starts[-1])
        if not n:raise ValueError('Empty well')
        assert all(size>0 for size in sizes)
        lower=torch.as_tensor(np.repeat(starts[:-1],sizes),device=self.device);upper=torch.as_tensor(np.repeat(starts[1:],sizes),device=self.device)
        centers=torch.as_tensor(np.concatenate([np.arange(lo,hi,128)+64 for lo,hi in zip(starts[:-1],starts[1:])]),device=self.device)
        anchors=torch.as_tensor(np.concatenate([np.minimum(np.arange(lo,hi,128)+64,hi-1) for lo,hi in zip(starts[:-1],starts[1:])]),device=self.device)
        row_well=torch.as_tensor(np.repeat(np.arange(len(records)),sizes),device=self.device)
        totals=[torch.zeros(n,len(a),device=self.device) for a in self.actions]
        for f,b in self.bundles.items():
            features=[self.features(frame,f) for _,frame in records]
            x,side,depth,absolute,bank=[torch.cat([v[k] for v in features],0) for k in range(5)];del features
            for c,t in enumerate(TARGETS):
                length=1024 if c==2 else 256;batch=min(batch_windows,16) if c==2 else batch_windows;lo=length//2-64;hi=lo+128
                w=b['weights'][t] if self.weight_policy=='diagnostic' else {'POR':.5,'PERM':.5,'SW':.5}[t]
                coverage=np.zeros(n,np.uint8)
                for start in range(0,len(centers),batch):
                    cc=centers[start:start+batch];aa=anchors[start:start+batch]
                    ix,valid=window_indices(cc,lower,upper,length=length,anchors=aa);relative=torch.tanh((depth[ix]-depth[aa,None])/10)
                    use=valid[:,lo:hi];rows=ix[:,lo:hi][use];r=rows.cpu().numpy()
                    assert not coverage[r].any() and len(np.unique(r))==len(r);coverage[r]+=1
                    utility=[]
                    for pair in b['pairs']:
                        amp=torch.autocast('cuda',dtype=torch.bfloat16) if self.device.type=='cuda' else contextlib.nullcontext()
                        with amp:g,logmass=pair['models'][c]((side if c<2 else x)[ix],relative,valid,absolute_depth=absolute[ix],well_inputs=bank[row_well[aa]])
                        atom=g[:,lo:hi][use][:,0].float().sigmoid();mass=logmass[:,lo:hi][use][:,0].float().exp()
                        joint=torch.cat((atom[:,None],(1-atom[:,None])*mass),-1);utility.append(joint@self.utilities[f][c])
                    totals[c][rows]+=(utility[0]*float(w)+utility[1]*float(1-w))/len(self.bundles)
                assert (coverage==1).all()
            del x,side,depth,absolute,bank
        result=np.column_stack([a[u.argmax(-1).cpu().numpy()] for a,u in zip(self.actions,totals)])
        assert np.isfinite(result).all()
        return [result[lo:hi] for lo,hi in zip(starts[:-1],starts[1:])]

    def predict_well(self,frame,batch_windows=32):
        return self.predict_records([('well',frame)],batch_windows)[0]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir','--data-dir',type=Path,required=True);ap.add_argument('--output',type=Path,default=ROOT/'result.json')
    ap.add_argument('--model_dir','--model-dir',type=Path,default=ROOT/'models');ap.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
    ap.add_argument('--batch_windows','--batch-windows',type=int,default=32);ap.add_argument('--weight-policy',choices=('diagnostic','fixed'),default='fixed')
    a=ap.parse_args()
    if a.batch_windows<1 or a.output.suffix.lower()!='.json':ap.error('Positive batch size and JSON output required')
    configure_runtime();started=time.perf_counter();device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
    predictor=Predictor(a.model_dir,device,weight_policy=a.weight_policy);results=[];rows=0
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    inputs=load_inputs(a.data_dir);predictions=predictor.predict_records(inputs,a.batch_windows)
    for (well,frame),pred in zip(inputs,predictions):
        results.append(dict(logId=well,predictions=make_rows(frame['DEPTH'].to_numpy(),pred,well)))
        rows+=len(frame);print('PREDICTED',well,len(frame),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(resultData=results),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    with zipfile.ZipFile(a.output.parent/'result.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:z.write(a.output,'result.json')
    report=dict(wells=len(results),rows=rows,seconds=time.perf_counter()-started,device=device,folds=list(predictor.bundles),weight_policy=a.weight_policy,
        aggregation=predictor.summary['aggregation'],oof_scores= predictor.summary['official_oof_scores'] if a.weight_policy=='diagnostic' else predictor.summary['fixed_oof_scores'],
        score_scope=predictor.summary['score_scope'],test_accuracy='unknown; no test labels',peak_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None)
    a.output.with_suffix('.inference.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('INFERENCE_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
