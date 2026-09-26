"""Fold-local continuous mixture-density neural regression with auxiliary categories.

No test data, teacher model or held-out-fold choices are used. The static density
support is only a quadrature grid: component means/scales are predicted per point.
"""
from pathlib import Path
import argparse, gc, json, random, sys, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'package'))
from src.data import load_split
from src.feature import FeatureBuilder
from src.model import CURVES, prep_well
from src.utils import DEFAULTS, BOUNDS, kernel_matrix
from train import metric, write, dmask, TS, BINS, OFF, FW

MIX = 8
DENSITY_BINS = 128
REG_FW = [np.log, np.log10, np.log]
REG_IV = [np.exp, lambda z: 10 ** z, np.exp]


class ResidualBlock(nn.Module):
    def __init__(self, c, dilation):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.conv = nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation)
        self.mix = nn.Sequential(nn.GELU(), nn.Conv1d(c, 2*c, 1), nn.GELU(),
                                 nn.Dropout(.05), nn.Conv1d(2*c, c, 1))
        self.scale = nn.Parameter(torch.full((1,c,1), .15))

    def forward(self, x):
        return x + self.scale*self.mix(self.conv(self.norm(x.transpose(1,2)).transpose(1,2)))


class Network(nn.Module):
    def __init__(self, initial_means=None, cin=145, c=80):
        super().__init__()
        self.point = nn.Sequential(nn.Conv1d(cin,160,1), nn.GELU(), nn.Dropout(.05),
                                   nn.Conv1d(160,c,1), nn.GELU())
        self.stem = nn.Conv1d(32,c,7,padding=3)
        self.blocks = nn.Sequential(*(ResidualBlock(c,d) for d in [1,2,4,8,16,32,64,1,2,4]))
        self.fuse = nn.Sequential(nn.Conv1d(2*c,c,1), nn.GELU())
        self.state = nn.Conv1d(c,4,1)
        self.gate = nn.Conv1d(c,3,1)
        self.aux = nn.Conv1d(c,200,1)
        self.density = nn.Conv1d(c,3*3*MIX,1)
        nn.init.normal_(self.density.weight, 0, .005)
        nn.init.zeros_(self.density.bias)
        if initial_means is not None:
            with torch.no_grad():
                bias = self.density.bias.view(3,3,MIX)
                bias[:,1,:].copy_(torch.tensor(initial_means, dtype=bias.dtype))
                bias[:,2,:].fill_(-.7)

    def forward(self,x):
        h = self.fuse(torch.cat([self.point(x),self.blocks(self.stem(x[:,:32]))],1))
        mdn = self.density(h).view(x.shape[0],3,3,MIX,x.shape[-1])
        return self.state(h), self.gate(h), self.aux(h), mdn


def mixture_params(output):
    # Output B, target, parameter, component, depth. Softplus avoids unstable
    # negative standard deviations; the nonzero floor prevents singular fits.
    return output[:,:,0].log_softmax(2), output[:,:,1], F.softplus(output[:,:,2])+.035


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/pure_dl_residual')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--batch', type=int, default=6)
    ap.add_argument('--crop', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=20260925)
    a = ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available()
    start = time.perf_counter()
    rec = load_split(ROOT.parent/'data','train')
    assert len(rec)==80
    ys = [fr[TS].to_numpy(np.float64) for _,fr in rec]
    valid = [np.isfinite(y).all(1) for y in ys]
    y = np.concatenate([t[v] for t,v in zip(ys,valid)])
    well = np.concatenate([np.full(v.sum(),i,np.int32) for i,v in enumerate(valid)])
    assert y.shape==(723556,3)
    fold = np.full(len(y),-1,np.int8)
    for f,(_,ix) in enumerate(GroupKFold(5).split(y,groups=well)): fold[ix]=f
    wf = {i:int(fold[np.flatnonzero(well==i)[0]]) for i in range(80)}
    config = {**vars(a),'out':str(a.out),'parameters':sum(p.numel() for p in Network().parameters()),
              'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
              'architecture':'145-channel MLP + raw32 conventional dilated CNN, width80, 10 residual blocks; 8-component continuous Gaussian mixture per target',
              'loss':'4-state CE + 0.3 independent gate BCE + per-target continuous MDN NLL + 0.15 auxiliary quantile CE',
              'decode':'fixed 50:50 state-marginal and independent gate; conditional density quadrature; expected official utility includes default action',
              'normalization':'fit exclusively on outer train wells', 'test_access':False,
              'selection':'single precommitted configuration, fixed final epoch, no held-out threshold or blend tuning',
              'outer_wells':{str(f):[rec[i][0] for i in range(80) if wf[i]==f] for f in range(5)},
              'valid_rows':len(y), 'density_bins':DENSITY_BINS}
    write(a.out/'config.json',config)
    print('CONFIG',json.dumps({k:v for k,v in config.items() if k!='outer_wells'}),flush=True)
    pred=np.full_like(y,np.nan)
    fold_reports=[]
    for f in range(5):
        fstart=time.perf_counter()
        seed=a.seed+f
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed)
        train=[i for i in range(80) if wf[i]!=f]
        val=[i for i in range(80) if wf[i]==f]
        cache=a.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            pred[fold==f]=np.load(cache)
            fold_reports.append(json.loads((a.out/f'fold_{f+1}_report.json').read_text()))
            continue
        print(f'FOLD {f+1} preparing train-only statistics',flush=True)
        builder=FeatureBuilder().fit([rec[i][1] for i in train])
        bases=[builder.transform(fr).base for _,fr in rec]
        ab=np.concatenate([bases[i] for i in train]);fm=np.median(ab,0)
        q=np.percentile(ab,[25,75],axis=0);fs=np.maximum(q[1]-q[0],1e-3);del ab
        gs={}
        for c in CURVES:
            v=np.concatenate([rec[i][1][c].dropna().to_numpy() for i in train])
            if c in ['RT','RXO']:v=np.log10(np.maximum(v,.01))
            q1,q3=np.percentile(v,[25,75])
            gs[c]={'median':float(np.median(v)),'iqr':max(float(q3-q1),1e-3)}
        yt=y[fold!=f];dm=dmask(yt)
        edges=[];rm=[];rs=[];init=[];density_edges=[];centers=[]
        for j,k in enumerate(BINS):
            v=FW[j](yt[~dm[:,j],j])
            ed=np.unique(np.quantile(v,np.linspace(0,1,k+1)));assert len(ed)==k+1
            edges.append(ed)
            z=REG_FW[j](np.maximum(yt[~dm[:,j],j],1e-6))
            med=np.median(z);scale=max(np.subtract(*np.percentile(z,[75,25])),.1)
            rm.append(med);rs.append(scale)
            zn=(z-med)/scale
            init.append(np.quantile(zn,np.linspace(.05,.95,MIX)))
            # Fixed regular quadrature in the target's transformed coordinate.
            # End tails are captured by infinite CDF edge values.
            lo,hi=np.min(z),np.max(z)
            cz=np.linspace(lo,hi,DENSITY_BINS)
            ez=np.r_[-np.inf,(cz[1:]+cz[:-1])/2,np.inf]
            density_edges.append((ez-med)/scale)
            centers.append(REG_IV[j](cz))
        rm=np.asarray(rm);rs=np.asarray(rs)
        xs=[];states=[];gates=[];labels=[];regression=[]
        for i,(_,fr) in enumerate(rec):
            raw=np.clip(prep_well(fr,gs),-12,12)
            xb=np.clip((bases[i]-fm)/fs,-8,8).T.astype(np.float32)
            xs.append(torch.from_numpy(np.concatenate([raw,xb])).cuda())
            clean=np.where(valid[i][:,None],ys[i],np.array([.1,.01,99.9]))
            default=dmask(clean)
            st=np.select([clean[:,0]<=.1,clean[:,2]>=99.9,clean[:,1]<=.01],[0,2,3],1)
            st[~valid[i]]=-1
            states.append(torch.tensor(st,device='cuda',dtype=torch.long))
            gate=default.T.astype(np.float32);gate[:,~valid[i]]=-1
            gates.append(torch.tensor(gate,device='cuda'))
            cl=[];rz=[]
            for j,k in enumerate(BINS):
                vv=FW[j](np.maximum(clean[:,j],.01))
                cc=np.clip(np.searchsorted(edges[j],vv,side='right')-1,0,k-1)
                cc[default[:,j]|~valid[i]]=-1;cl.append(cc)
                zz=(REG_FW[j](np.maximum(clean[:,j],1e-6))-rm[j])/rs[j]
                rz.append(zz)
            labels.append(torch.tensor(np.stack(cl),device='cuda',dtype=torch.long))
            regression.append(torch.tensor(np.stack(rz),device='cuda',dtype=torch.float32))
        del bases;gc.collect()
        model=Network(np.stack(init)).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.005)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs*a.steps,eta_min=.000015)
        lengths=np.array([len(ys[i]) for i in train]);sample_prob=lengths/lengths.sum()
        history=[]
        torch.cuda.reset_peak_memory_stats()
        for ep in range(a.epochs):
            model.train();total=0
            for step in range(a.steps):
                wi=rng.choice(train,a.batch,p=sample_prob)
                pos=[rng.integers(0,len(ys[i])-a.crop+1) for i in wi]
                x=torch.stack([xs[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                st=torch.stack([states[i][s:s+a.crop] for i,s in zip(wi,pos)])
                ga=torch.stack([gates[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                cl=torch.stack([labels[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                z=torch.stack([regression[i][:,s:s+a.crop] for i,s in zip(wi,pos)])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):so,go,ao,mo=model(x)
                so=so.float();go=go.float();ao=ao.float();mo=mo.float()
                logw,mu,sigma=mixture_params(mo)
                loss=F.cross_entropy(so,st,ignore_index=-1)
                gl=F.binary_cross_entropy_with_logits(go,ga.clamp_min(0),reduction='none')
                loss=loss+.3*(gl*(ga>=0)).sum()/(ga>=0).sum().clamp_min(1)
                for j in range(3):
                    active=cl[:,j]>=0
                    zg=z[:,j,None,:]
                    logp=logw[:,j]-.5*((zg-mu[:,j])/sigma[:,j])**2-sigma[:,j].log()-.5*np.log(2*np.pi)
                    nll=-torch.logsumexp(logp,dim=1)
                    ce=F.cross_entropy(ao[:,OFF[j]:OFF[j+1]],cl[:,j],ignore_index=-1,reduction='none')
                    loss=loss+((nll*active).sum()+.15*ce.sum())/active.sum().clamp_min(1)/3
                assert torch.isfinite(loss)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2)
                opt.step();sched.step();total+=float(loss.detach())
            history.append(total/a.steps)
            if ep==0 or (ep+1)%5==0:
                print(f'fold={f+1} epoch={ep+1}/{a.epochs} loss={history[-1]:.4f} seconds={time.perf_counter()-fstart:.1f} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}',flush=True)
        model.eval();grids=[];kernels=[]
        for j,t in enumerate(TS):
            sup=centers[j]
            grid=np.linspace(sup[0],sup[-1],500) if j==2 else np.geomspace(max(sup[0]*.9,.001),sup[-1]*1.1,500)
            grid=np.unique(np.r_[DEFAULTS[t],np.clip(grid,*BOUNDS[t])]);grids.append(grid)
            kernels.append(torch.tensor(kernel_matrix(grid,np.r_[DEFAULTS[t],sup],t).T,dtype=torch.float32,device='cuda'))
        pack={'indices':np.flatnonzero(fold==f)}
        saved_state=[];saved_gate=[];saved_density=[];saved_w=[];saved_mu=[];saved_sd=[]
        for wi in val:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):so,go,ao,mo=model(xs[wi][None])
            sp=so.float().softmax(1)[0].T
            state_marg=torch.stack([sp[:,0],sp[:,0]+sp[:,3],sp[:,0]+sp[:,2]],1)
            gate=.5*(state_marg+go.float().sigmoid()[0].T)
            with torch.no_grad():logw,mu,sd=mixture_params(mo.float())
            weight=logw.exp()[0].permute(2,0,1);mu=mu[0].permute(2,0,1);sd=sd[0].permute(2,0,1)
            pp=[];pds=[]
            for j,t in enumerate(TS):
                ez=torch.tensor(density_edges[j],device='cuda',dtype=torch.float32)
                zedge=(ez[None,:,None]-mu[:,j,None,:])/sd[:,j,None,:]
                cdf=.5*(1+torch.erf(zedge/np.sqrt(2)))
                masses=(cdf[:,1:,:]-cdf[:,:-1,:]).clamp_min(0)
                cp=(masses*weight[:,j,None,:]).sum(-1)
                cp=cp/cp.sum(1,keepdim=True).clamp_min(1e-12)
                probability=torch.cat([gate[:,j,None],(1-gate[:,j,None])*cp],1)
                with torch.no_grad():best=(probability@kernels[j]).argmax(1).cpu().numpy()
                pp.append(grids[j][best]);pds.append(cp.cpu().numpy().astype(np.float16)[valid[wi]])
            pred[well==wi]=np.stack(pp,1)[valid[wi]]
            saved_state.append(sp.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_gate.append(gate.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_density.append(np.stack(pds,1))
            saved_w.append(weight.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_mu.append(mu.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_sd.append(sd.cpu().numpy().astype(np.float16)[valid[wi]])
        pack.update(state=np.concatenate(saved_state),gate=np.concatenate(saved_gate),
                    mixture_weight=np.concatenate(saved_w),mixture_mu=np.concatenate(saved_mu),mixture_std=np.concatenate(saved_sd),
                    regression_median=rm,regression_scale=rs,pred=pred[fold==f])
        dens=np.concatenate(saved_density)
        for j,t in enumerate(TS):pack[f'{t}_prob']=dens[:,j];pack[f'{t}_centers']=centers[j]
        np.savez_compressed(a.out/f'fold_{f+1}_distributions.npz',**pack)
        fr={'fold':f+1,'scores':metric(y[fold==f],pred[fold==f]),'seconds':time.perf_counter()-fstart,
            'loss':history,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30}
        fold_reports.append(fr);write(a.out/f'fold_{f+1}_report.json',fr);np.save(cache,pred[fold==f])
        torch.save({'state_dict':model.cpu().state_dict(),'feature_median':fm.tolist(),'feature_scale':fs.tolist(),
                    'gstats':gs,'centers':[ct.tolist() for ct in centers],'density_edges':[ed.tolist() for ed in density_edges],
                    'regression_median':rm.tolist(),'regression_scale':rs.tolist(),'medians':builder.medians_,'config':config},a.out/f'fold_{f+1}.pt')
        np.savez_compressed(a.out/'oof.npz',truth=y,pred=pred,fold=fold,well=well)
        print('FOLD_RESULT',json.dumps({k:v for k,v in fr.items() if k!='loss'}),flush=True)
        del model,opt,xs,states,gates,labels,regression,kernels,pack,dens
        gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    report={'config':config,'oof_scores':metric(y,pred),'folds':fold_reports,'seconds':time.perf_counter()-start,
            'per_well':{rec[i][0]:metric(y[well==i],pred[well==i]) for i in range(80)}}
    write(a.out/'report.json',report)
    print('FINAL',json.dumps({k:v for k,v in report.items() if k in ['oof_scores','seconds']}),flush=True)


if __name__=='__main__':main()
