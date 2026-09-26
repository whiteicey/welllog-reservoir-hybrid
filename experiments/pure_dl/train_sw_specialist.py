"""SW-only multi-resolution neural density model, fixed group-five-fold protocol.

POR/PERM outputs are immutable OOF predictions from the earlier pure-neural MDN,
not labels. Every reported three-target score is explicitly a SW replacement.
"""
from pathlib import Path
import argparse,gc,hashlib,json,random,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'package'))
from src.data import load_split
from src.feature import FeatureBuilder
from src.model import CURVES,prep_well
from src.utils import DEFAULTS,BOUNDS,kernel_matrix
from train import metric,write,TS
from mdn_exact_decode import mixture_utility

MIX=8
AUX_BINS=80
DENSITY_BINS=256


class Block(nn.Module):
    def __init__(self,c,d,k):
        super().__init__()
        self.norm=nn.LayerNorm(c)
        self.conv=nn.Conv1d(c,c,k,padding=(k//2)*d,dilation=d)
        self.ff=nn.Sequential(nn.GELU(),nn.Conv1d(c,2*c,1),nn.GELU(),nn.Dropout(.05),nn.Conv1d(2*c,c,1))
        self.scale=nn.Parameter(torch.full((1,c,1),.15))
    def forward(self,x):
        return x+self.scale*self.ff(self.conv(self.norm(x.transpose(1,2)).transpose(1,2)))


class SWNetwork(nn.Module):
    def __init__(self,initial_means=None):
        super().__init__()
        self.point=nn.Sequential(nn.Conv1d(269,160,1),nn.GELU(),nn.Dropout(.05),nn.Conv1d(160,96,1),nn.GELU())
        self.fine=nn.Sequential(nn.Conv1d(32,64,7,padding=3),*(Block(64,d,3) for d in [1,2,4,8]))
        self.down=nn.Conv1d(64,96,9,stride=8,padding=4)
        self.coarse=nn.Sequential(*(Block(96,d,5) for d in [1,2,4,8]))
        self.decoder=nn.Sequential(nn.Conv1d(160,96,3,padding=1),nn.GELU())
        self.fuse=nn.Sequential(nn.Conv1d(192,96,1),nn.GELU())
        self.gate=nn.Conv1d(96,1,1)
        self.density=nn.Conv1d(96,3*MIX,1)
        self.aux=nn.Conv1d(96,AUX_BINS,1)
        nn.init.normal_(self.density.weight,0,.005);nn.init.zeros_(self.density.bias)
        if initial_means is not None:
            with torch.no_grad():
                b=self.density.bias.view(3,MIX)
                b[1].copy_(torch.as_tensor(initial_means,dtype=b.dtype));b[2].fill_(-.7)
    def forward(self,x):
        fine=self.fine(x[:,:32])
        coarse=self.coarse(self.down(fine))
        up=F.interpolate(coarse,scale_factor=8,mode='linear',align_corners=False)[...,:x.shape[-1]]
        local=self.decoder(torch.cat([fine,up],1))
        h=self.fuse(torch.cat([self.point(x),local],1))
        return self.gate(h)[:,0],self.density(h).view(x.shape[0],3,MIX,x.shape[-1]),self.aux(h)


def mdn_params(output):
    return output[:,0].log_softmax(1),output[:,1],F.softplus(output[:,2])+.035


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'outputs/pure_dl_sw_specialist')
    ap.add_argument('--reference',type=Path,default=ROOT/'outputs/pure_dl_residual/analytic_800/oof.npz')
    ap.add_argument('--epochs',type=int,default=50);ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--crop',type=int,default=2048)
    ap.add_argument('--seed',type=int,default=20260928)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    start=time.perf_counter()
    reference_sha=sha256(a.reference);source_sha=sha256(Path(__file__))
    reference=np.load(a.reference)
    rec=load_split(ROOT.parent/'data','train')
    ys=[fr[TS].to_numpy(np.float64) for _,fr in rec]
    valid=[np.isfinite(y).all(1) for y in ys]
    y=np.concatenate([yy[v] for yy,v in zip(ys,valid)])
    well=np.concatenate([np.full(v.sum(),i,np.int32) for i,v in enumerate(valid)])
    assert len(rec)==80 and y.shape==(723556,3)
    fold=np.full(len(y),-1,np.int8)
    for f,(_,ix) in enumerate(GroupKFold(5).split(y,groups=well)):fold[ix]=f
    assert np.array_equal(y,reference['truth']) and np.array_equal(well,reference['well']) and np.array_equal(fold,reference['fold'])
    fixed_pred=reference['pred'].copy();assert np.isfinite(fixed_pred).all()
    pred=fixed_pred.copy();pred[:,2]=np.nan
    wf={i:int(fold[np.flatnonzero(well==i)[0]]) for i in range(80)}
    config={**vars(a),'out':str(a.out),'reference':str(a.reference),'reference_sha256':reference_sha,'source_sha256':source_sha,
            'parameters':sum(p.numel() for p in SWNetwork().parameters()),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,
            'architecture':'SW-only point MLP269-160-96 + native64-channel CNN + stride8 coarse96-channel CNN; each branch dilations1,2,4,8; coarse receptive field approximately50m',
            'heads':'independent default gate, 8-Gaussian logSW mixture, auxiliary80 quantile classes',
            'loss':'gate BCE + conditional MDN NLL + 0.15 auxiliary CE; all fixed; final epoch only',
            'decode':'analytic expected official utility on800 candidate actions; exact default and clipped-support endpoint atoms; relative continuous epsilon omitted',
            'non_SW_outputs':'unchanged, SHA-locked POR/PERM pure-neural MDN analytic800 OOF predictions; no labels used as predictions',
            'normalization':'outer train wells only; GPU-resident inputs float16','valid_rows':len(y),'test_access':False,
            'selection':'one precommitted50epoch five-fold run, no held-out calibration or best-epoch selection',
            'outer_wells':{str(f):[rec[i][0] for i in range(80) if wf[i]==f] for f in range(5)}}
    write(a.out/'config.json',config)
    print('CONFIG',json.dumps({k:v for k,v in config.items() if k!='outer_wells'}),flush=True)
    reports=[]
    for f in range(5):
        fs_time=time.perf_counter();seed=a.seed+f
        np.random.seed(seed);random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed)
        tr=[i for i in range(80) if wf[i]!=f];va=[i for i in range(80) if wf[i]==f]
        cache=a.out/f'fold_{f+1}_pred.npy'
        if cache.exists():
            pred[fold==f]=np.load(cache);reports.append(json.loads((a.out/f'fold_{f+1}_report.json').read_text()));continue
        print(f'FOLD {f+1} fitting outer-training-only input and target statistics',flush=True)
        builder=FeatureBuilder().fit([rec[i][1] for i in tr])
        feats=[builder.transform(fr).enhanced for _,fr in rec]
        joined=np.concatenate([feats[i] for i in tr]);fm=np.median(joined,0)
        q=np.percentile(joined,[25,75],axis=0);fscale=np.maximum(q[1]-q[0],1e-3);del joined
        assert fm.shape==(237,)
        gs={}
        for c in CURVES:
            vv=np.concatenate([rec[i][1][c].dropna().to_numpy() for i in tr])
            if c in ['RT','RXO']:vv=np.log10(np.maximum(vv,.01))
            q1,q3=np.percentile(vv,[25,75]);gs[c]={'median':float(np.median(vv)),'iqr':max(float(q3-q1),1e-3)}
        sw=y[(fold!=f)&(y[:,2]<99.9),2]
        assert np.all(sw>0)
        z=np.log(sw);rm=float(np.median(z));rs=max(float(np.subtract(*np.percentile(z,[75,25]))),.1)
        init=np.quantile((z-rm)/rs,np.linspace(.05,.95,MIX))
        aux_edges=np.unique(np.quantile(z,np.linspace(0,1,AUX_BINS+1)));assert len(aux_edges)==AUX_BINS+1
        cz=np.linspace(z.min(),z.max(),DENSITY_BINS);centers=np.exp(cz)
        density_edges=(np.r_[-np.inf,(cz[:-1]+cz[1:])/2,np.inf]-rm)/rs
        xs=[];gates=[];classes=[];targets=[]
        for i,(_,fr) in enumerate(rec):
            raw=np.clip(prep_well(fr,gs),-12,12)
            ext=np.clip((feats[i]-fm)/fscale,-8,8).T.astype(np.float32)
            xx=np.concatenate([raw,ext]);assert np.isfinite(xx).all()
            xs.append(torch.tensor(xx,device='cuda',dtype=torch.float16))
            sw_i=np.where(valid[i],ys[i][:,2],99.9)
            ga=(sw_i>=99.9).astype(np.float32);ga[~valid[i]]=-1
            zz=np.log(np.maximum(sw_i,1e-6))
            cl=np.clip(np.searchsorted(aux_edges,zz,side='right')-1,0,AUX_BINS-1)
            cl[(sw_i>=99.9)|~valid[i]]=-1
            gates.append(torch.tensor(ga,device='cuda'))
            classes.append(torch.tensor(cl,device='cuda',dtype=torch.long))
            targets.append(torch.tensor((zz-rm)/rs,device='cuda',dtype=torch.float32))
        del feats;gc.collect()
        model=SWNetwork(init).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0015,weight_decay=.005)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs*a.steps,eta_min=.000015)
        lengths=np.array([len(ys[i]) for i in tr]);sampling=lengths/lengths.sum()
        history=[];torch.cuda.reset_peak_memory_stats()
        for ep in range(a.epochs):
            model.train();total=0
            for step in range(a.steps):
                wi=rng.choice(tr,a.batch,p=sampling)
                starts=[rng.integers(0,len(ys[i])-a.crop+1) for i in wi]
                x=torch.stack([xs[i][:,s:s+a.crop] for i,s in zip(wi,starts)])
                ga=torch.stack([gates[i][s:s+a.crop] for i,s in zip(wi,starts)])
                cl=torch.stack([classes[i][s:s+a.crop] for i,s in zip(wi,starts)])
                zz=torch.stack([targets[i][s:s+a.crop] for i,s in zip(wi,starts)])
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):go,mo,ao=model(x)
                go=go.float();mo=mo.float();ao=ao.float()
                logw,mu,sd=mdn_params(mo)
                logp=logw-.5*((zz[:,None]-mu)/sd)**2-sd.log()-.5*np.log(2*np.pi)
                nll=-torch.logsumexp(logp,dim=1);active=cl>=0
                gl=F.binary_cross_entropy_with_logits(go,ga.clamp_min(0),reduction='none')
                ce=F.cross_entropy(ao,cl,ignore_index=-1,reduction='none')
                loss=(gl*(ga>=0)).sum()/(ga>=0).sum().clamp_min(1)
                loss=loss+((nll*active).sum()+.15*ce.sum())/active.sum().clamp_min(1)
                assert torch.isfinite(loss)
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2)
                opt.step();schedule.step();total+=float(loss.detach())
            history.append(total/a.steps)
            if ep==0 or (ep+1)%5==0:
                print(f'fold={f+1} epoch={ep+1}/{a.epochs} loss={history[-1]:.4f} seconds={time.perf_counter()-fs_time:.1f} peak_GiB={torch.cuda.max_memory_allocated()/2**30:.3f}',flush=True)
        model.eval()
        grid=np.unique(np.r_[DEFAULTS['SW'],np.clip(np.linspace(centers[0],centers[-1],800),*BOUNDS['SW'])])
        default_u=torch.tensor(kernel_matrix(grid,np.array([DEFAULTS['SW']]),'SW')[:,0],device='cuda',dtype=torch.float32)
        saved_gate=[];saved_w=[];saved_mu=[];saved_sd=[];saved_prob=[]
        for wi in va:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):go,mo,ao=model(xs[wi][None])
            with torch.no_grad():logw,mu,sd=mdn_params(mo.float())
            gate=go.float().sigmoid()[0];weight=logw.exp()[0].T;mu=mu[0].T;sd=sd[0].T
            actual_mean=mu*rs+rm;actual_std=sd*rs
            result=np.empty(len(gate),np.float64)
            for pos in range(0,len(gate),128):
                end=min(pos+128,len(gate))
                utility=mixture_utility(grid,weight[pos:end],actual_mean[pos:end],actual_std[pos:end],'SW',support_bounds=(centers[0],centers[-1]))
                utility=gate[pos:end,None]*default_u+(1-gate[pos:end,None])*utility
                assert torch.isfinite(utility).all()
                result[pos:end]=grid[utility.argmax(1).cpu().numpy()]
            pred[well==wi,2]=result[valid[wi]]
            ez=torch.tensor(density_edges,device='cuda',dtype=torch.float32)
            cdf=torch.special.ndtr((ez[None,:,None]-mu[:,None,:])/sd[:,None,:])
            probability=((cdf[:,1:]-cdf[:,:-1]).clamp_min(0)*weight[:,None,:]).sum(-1)
            probability=probability/probability.sum(1,keepdim=True).clamp_min(1e-12)
            saved_gate.append(gate.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_w.append(weight.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_mu.append(mu.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_sd.append(sd.cpu().numpy().astype(np.float16)[valid[wi]])
            saved_prob.append(probability.cpu().numpy().astype(np.float16)[valid[wi]])
        ix=fold==f
        assert np.array_equal(pred[ix,:2],fixed_pred[ix,:2])
        pack={'indices':np.flatnonzero(ix),'SW_gate':np.concatenate(saved_gate),'SW_prob':np.concatenate(saved_prob),'SW_centers':centers,
              'SW_mixture_weight':np.concatenate(saved_w),'SW_mixture_mu':np.concatenate(saved_mu),'SW_mixture_std':np.concatenate(saved_sd),
              'SW_regression_median':rm,'SW_regression_scale':rs,'pred':pred[ix]}
        np.savez_compressed(a.out/f'fold_{f+1}_distributions.npz',**pack)
        fr={'fold':f+1,'scores':metric(y[ix],pred[ix]),'reference_scores':metric(y[ix],fixed_pred[ix]),
            'seconds':time.perf_counter()-fs_time,'loss':history,'peak_gpu_GiB':torch.cuda.max_memory_allocated()/2**30,
            'POR_PERM_exactly_unchanged':True}
        reports.append(fr);write(a.out/f'fold_{f+1}_report.json',fr);np.save(cache,pred[ix])
        torch.save({'state_dict':model.cpu().state_dict(),'feature_median':fm.tolist(),'feature_scale':fscale.tolist(),
                    'gstats':gs,'centers':centers.tolist(),'density_edges':density_edges.tolist(),'regression_median':rm,'regression_scale':rs,
                    'medians':builder.medians_,'aux_edges':aux_edges.tolist(),'config':config},a.out/f'fold_{f+1}.pt')
        np.savez_compressed(a.out/'oof.npz',truth=y,pred=pred,fold=fold,well=well)
        print('FOLD_RESULT',json.dumps({k:v for k,v in fr.items() if k!='loss'}),flush=True)
        del model,opt,xs,gates,classes,targets,pack;gc.collect();torch.cuda.empty_cache()
    assert np.isfinite(pred).all() and np.array_equal(pred[:,:2],fixed_pred[:,:2])
    assert sha256(a.reference)==reference_sha and sha256(Path(__file__))==source_sha
    score=metric(y,pred);baseline=metric(y,fixed_pred)
    report={'config':config,'oof_scores':score,'reference_scores':baseline,'score_delta':{k:score[k]-baseline[k] for k in score},
            'folds':reports,'seconds':time.perf_counter()-start,'POR_PERM_exactly_unchanged':True,
            'per_well':{rec[i][0]:metric(y[well==i],pred[well==i]) for i in range(80)}}
    write(a.out/'report.json',report)
    print('FINAL',json.dumps({k:v for k,v in report.items() if k in ['oof_scores','reference_scores','score_delta','seconds']}),flush=True)


if __name__=='__main__':main()
