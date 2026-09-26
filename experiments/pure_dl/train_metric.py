"""Pure neural V2: ordered soft-label distributions plus metric-aware utility loss."""
from pathlib import Path
import argparse,json,sys,time,random,gc,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import load_split,TARGET_COLUMNS
from src.feature import FeatureBuilder
from src.model import prep_well,CURVES
from src.utils import kernel_matrix,DEFAULTS,BOUNDS

TS=['POR','PERM','SW'];BINS=[80,60,60];OFF=[0,80,140,200]
FW=[np.log1p,np.log10,lambda x:x];IV=[np.expm1,lambda x:10**x,lambda x:x]

def dmask(y):return np.c_[y[:,0]<=.1,y[:,1]<=.01,y[:,2]>=99.9]
def metric(y,p):
    a=np.c_[np.maximum(0,1-abs(p[:,0]-y[:,0])/(.08*(abs(y[:,0])+1e-5))),
            np.maximum(0,1-abs(np.log10(np.maximum(p[:,1]/np.maximum(y[:,1],1e-5),1e-5)))),
            np.maximum(0,1-abs(p[:,2]-y[:,2])/(.05*(abs(y[:,2])+1e-5)))]
    s=a.mean(0)*100;return dict(zip(TS+['TOTAL'],[*s,float(s@np.array([.3,.35,.35]))]))
def write(p,v):p.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')

class Block(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.norm=nn.LayerNorm(c)
        self.depth=nn.Conv1d(c,c,5,padding=2*d,dilation=d,groups=c)
        self.mix=nn.Sequential(nn.Conv1d(c,2*c,1),nn.GELU(),nn.Conv1d(2*c,c,1))
        self.scale=nn.Parameter(torch.full((1,c,1),.1))
    def forward(self,x):
        h=self.norm(x.transpose(1,2)).transpose(1,2)
        return x+self.scale*self.mix(self.depth(h))
class Network(nn.Module):
    def __init__(self,cin=145,c=96):
        super().__init__()
        self.point=nn.Sequential(nn.Conv1d(cin,128,1),nn.GELU(),nn.Conv1d(128,c,1),nn.GELU())
        self.stem=nn.Conv1d(32,c,7,padding=3)
        self.blocks=nn.Sequential(*(Block(c,d) for d in [1,2,4,8,16,32,1,2,4,8,16,32]))
        self.fuse=nn.Sequential(nn.Conv1d(c*2,c,1),nn.GELU())
        self.state=nn.Conv1d(c,4,1);self.value=nn.Conv1d(c,200,1)
    def forward(self,x):
        h=self.fuse(torch.cat([self.point(x),self.blocks(self.stem(x[:,:32]))],dim=1))
        return self.state(h),self.value(h)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'outputs/pure_dl_metric');ap.add_argument('--epochs',type=int,default=50)
    ap.add_argument('--steps',type=int,default=80);ap.add_argument('--batch',type=int,default=8);ap.add_argument('--crop',type=int,default=1024);ap.add_argument('--seed',type=int,default=20260925)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;assert torch.cuda.is_available()
    start=time.perf_counter();rec=load_split(ROOT.parent/'data','train');ys=[f[TS].to_numpy(np.float64) for _,f in rec];valid=[np.isfinite(y).all(1) for y in ys]
    y=np.concatenate([t[v] for t,v in zip(ys,valid)]);well=np.concatenate([np.full(v.sum(),i,np.int32) for i,v in enumerate(valid)])
    fold=np.full(len(y),-1,np.int8)
    for f,(_,idx) in enumerate(GroupKFold(5).split(y,groups=well)):fold[idx]=f
    wf={i:int(fold[np.flatnonzero(well==i)[0]]) for i in range(80)}
    config={**vars(a),'out':str(a.out),'parameters':sum(p.numel() for p in Network().parameters()),'device':torch.cuda.get_device_name(0),
            'torch':torch.__version__,'normalization':'outer train wells only','decode':'expected official utility, includes default action; no tuned thresholds',
            'architecture':'145-channel point MLP + 32-channel depthwise CNN, 12 dilated residual blocks width96, 4-state and 80/60/60 distribution heads',
            'loss':'state CE + ordered soft-label CE; triangular target widths POR4%, PERM0.1log10, SW2.5%; plus 0.2 negative expected official kernel for nondefault',
            'outer_wells':{str(f):[rec[i][0] for i in range(80) if wf[i]==f] for f in range(5)},'valid_rows':len(y),'test_access':False}
    write(a.out/'config.json',config);print('CONFIG',json.dumps({k:v for k,v in config.items() if k!='outer_wells'}),flush=True)
    pred=np.full_like(y,np.nan);fold_reports=[]
    for f in range(5):
        fstart=time.perf_counter();seed=a.seed+f;np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
        train=[i for i in range(80) if wf[i]!=f];val=[i for i in range(80) if wf[i]==f]
        cache=a.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            pred[fold==f]=np.load(cache);fold_reports.append(json.loads((a.out/f'fold_{f+1}_report.json').read_text()));continue
        print('FOLD',f+1,'building train-only feature statistics',flush=True)
        builder=FeatureBuilder().fit([rec[i][1] for i in train]);bases=[builder.transform(fr).base for _,fr in rec]
        ab=np.concatenate([bases[i] for i in train]);fm=np.median(ab,0);q=np.percentile(ab,[25,75],axis=0);fs=np.maximum(q[1]-q[0],1e-3);del ab
        gs={}
        for c in CURVES:
            v=np.concatenate([rec[i][1][c].dropna().to_numpy() for i in train]);v=np.log10(np.maximum(v,.01)) if c in ['RT','RXO'] else v
            q1,q3=np.percentile(v,[25,75]);gs[c]={'median':float(np.median(v)),'iqr':max(float(q3-q1),1e-3)}
        edges=[];centers=[];yt=y[fold!=f];dm=dmask(yt)
        for j,k in enumerate(BINS):
            v=FW[j](yt[~dm[:,j],j]);ed=np.unique(np.quantile(v,np.linspace(0,1,k+1)))
            assert len(ed)==k+1
            cl=np.clip(np.searchsorted(ed,v,side='right')-1,0,k-1);ct=np.array([np.median(v[cl==i]) for i in range(k)])
            edges.append(ed);centers.append(IV[j](ct))
        xs=[];states=[];labels=[];raw_targets=[]
        for i,(_,fr) in enumerate(rec):
            xr=prep_well(fr,gs);xb=np.clip((bases[i]-fm)/fs,-8,8).T.astype(np.float32)
            xs.append(torch.from_numpy(np.concatenate([xr,xb])).cuda())
            clean=np.where(valid[i][:,None],ys[i],np.array([.1,.01,99.9]));default=dmask(clean)
            s=np.select([clean[:,0]<=.1,clean[:,2]>=99.9,clean[:,1]<=.01],[0,2,3],1);s[~valid[i]]=-1
            states.append(torch.tensor(s,device='cuda',dtype=torch.long));cl=[]
            for j,k in enumerate(BINS):
                values=FW[j](np.maximum(clean[:,j],.01));v=np.clip(np.searchsorted(edges[j],values,side='right')-1,0,k-1);v[default[:,j]|~valid[i]]=-1;cl.append(v)
            labels.append(torch.tensor(np.stack(cl),device='cuda',dtype=torch.long))
            raw_targets.append(torch.tensor(clean.T,device='cuda',dtype=torch.float32))
        center_tensors=[torch.tensor(ct,device='cuda',dtype=torch.float32)[None,:,None] for ct in centers]
        del bases;gc.collect()
        model=Network().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.01)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs*a.steps,eta_min=.000015)
        lengths=np.array([len(ys[i]) for i in train]);prob=lengths/lengths.sum();loss_history=[]
        for ep in range(a.epochs):
            model.train();total=0
            for step in range(a.steps):
                wi=rng.choice(train,a.batch,p=prob);pos=[rng.integers(0,len(ys[i])-a.crop+1) for i in wi]
                x=torch.stack([xs[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                st=torch.stack([states[i][s:s+a.crop] for i,s in zip(wi,pos)])
                cl=torch.stack([labels[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                raw_y=torch.stack([raw_targets[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):so,vo=model(x)
                so=so.float();vo=vo.float();loss=F.cross_entropy(so,st,ignore_index=-1)
                for j in range(3):
                    target=raw_y[:,j:j+1];support=center_tensors[j]
                    if j==1:
                        distance=abs(torch.log10(support.clamp_min(.01))-torch.log10(target.clamp_min(.01)))
                        soft=(1-distance/.1).clamp_min(0);utility=(1-distance).clamp_min(0)
                    else:
                        distance=abs(support-target)/(target.abs()+1e-5)
                        soft=(1-distance/(.04 if j==0 else .025)).clamp_min(0)
                        utility=(1-distance/(.08 if j==0 else .05)).clamp_min(0)
                    fallback=F.one_hot(cl[:,j].clamp_min(0),num_classes=BINS[j]).permute(0,2,1).float()
                    soft=torch.where(soft.sum(1,keepdim=True)>0,soft, fallback)
                    soft=soft/soft.sum(1,keepdim=True).clamp_min(1e-8)
                    lp=vo[:,OFF[j]:OFF[j+1]].log_softmax(1)
                    ll=-(soft*lp).sum(1)-.2*(lp.exp()*utility).sum(1)
                    mask=(cl[:,j]>=0);loss=loss+(ll*mask).sum()/mask.sum().clamp_min(1)/3
                assert torch.isfinite(loss)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();sched.step();total+=float(loss.detach())
            loss_history.append(total/a.steps)
            if ep==0 or (ep+1)%5==0:print(f'fold={f+1} epoch={ep+1}/{a.epochs} loss={loss_history[-1]:.4f} seconds={time.perf_counter()-fstart:.1f}',flush=True)
        model.eval();grids=[];kernels=[]
        for j,t in enumerate(TS):
            sup=centers[j];grid=np.linspace(sup[0],sup[-1],400) if j==2 else np.geomspace(max(sup[0]*.9,.001),sup[-1]*1.1,400)
            grid=np.unique(np.r_[DEFAULTS[t],np.clip(grid,*BOUNDS[t])]);grids.append(grid)
            kernels.append(torch.tensor(kernel_matrix(grid,np.r_[DEFAULTS[t],sup],t).T,dtype=torch.float32,device='cuda'))
        val_state=[];val_prob={t:[] for t in TS}
        for wi in val:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):so,vo=model(xs[wi][None])
            sp=so.float().softmax(1)[0].T;vp=vo.float()[0].T
            val_state.append(sp.cpu().numpy()[valid[wi]].astype(np.float16))
            marg=[sp[:,0],sp[:,0]+sp[:,3],sp[:,0]+sp[:,2]];pp=[]
            for j,t in enumerate(TS):
                cond=vp[:,OFF[j]:OFF[j+1]].softmax(1);p=torch.cat([marg[j][:,None],(1-marg[j][:,None])*cond],1)
                val_prob[t].append(cond.cpu().numpy()[valid[wi]].astype(np.float16))
                with torch.no_grad():best=(p@kernels[j]).argmax(1).cpu().numpy()
                pp.append(grids[j][best])
            pred[well==wi]=np.stack(pp,1)[valid[wi]]
        fr={'fold':f+1,'scores':metric(y[fold==f],pred[fold==f]),'seconds':time.perf_counter()-fstart,'loss':loss_history}
        fold_reports.append(fr);write(a.out/f'fold_{f+1}_report.json',fr);np.save(cache,pred[fold==f])
        dist={'indices':np.flatnonzero(fold==f),'state':np.concatenate(val_state)}
        for j,t in enumerate(TS):dist[t+'_prob']=np.concatenate(val_prob[t]);dist[t+'_centers']=centers[j]
        np.savez_compressed(a.out/f'fold_{f+1}_distributions.npz',**dist)
        torch.save({'state_dict':model.cpu().state_dict(),'feature_median':fm.tolist(),'feature_scale':fs.tolist(),'gstats':gs,'centers':[c.tolist() for c in centers],'medians':builder.medians_,'config':config},a.out/f'fold_{f+1}.pt')
        np.savez_compressed(a.out/'oof.npz',truth=y,pred=pred,fold=fold,well=well)
        print('FOLD_RESULT',json.dumps({k:v for k,v in fr.items() if k!='loss'}),flush=True)
        del model,opt,xs,states,labels,kernels,raw_targets;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    report={'config':config,'oof_scores':metric(y,pred),'folds':fold_reports,'seconds':time.perf_counter()-start,
            'per_well':{rec[i][0]:metric(y[well==i],pred[well==i]) for i in range(80)}}
    write(a.out/'report.json',report);print('FINAL',json.dumps({k:v for k,v in report.items() if k in ['oof_scores','seconds']}),flush=True)

if __name__=='__main__':main()
