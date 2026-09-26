"""Small CNN baseline: fixed-budget, well-grouped 5-fold CV; no test data."""
from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'package'))
from src.data import load_split, INPUT_COLUMNS, TARGET_COLUMNS

COLS = [c for c in INPUT_COLUMNS if c != 'CASE']
DEFAULTS = np.array([.1, .01, 99.9])
LOW = np.array([0., .01, 8.305])
HIGH = np.array([50., 10000., 99.9])


def scores(y, p):
    a = np.empty_like(y, dtype=np.float64)
    a[:, 0] = np.maximum(0, 1-np.abs(p[:, 0]-y[:, 0])/(.08*(np.abs(y[:, 0])+1e-5)))
    a[:, 1] = np.maximum(0, 1-np.abs(np.log10(np.maximum(p[:, 1]/np.maximum(y[:, 1], 1e-5), 1e-5))))
    a[:, 2] = np.maximum(0, 1-np.abs(p[:, 2]-y[:, 2])/(.05*(np.abs(y[:, 2])+1e-5)))
    means = a.mean(0)
    return {**dict(zip(TARGET_COLUMNS, (means*100).tolist())), 'TOTAL': float(means @ np.array([.3,.35,.35])*100)}


def default_mask(y):
    return np.column_stack([y[:, 0] <= .1, y[:, 1] <= .01, y[:, 2] >= 99.9])


def target_transform(y):
    return np.column_stack([np.log1p(np.maximum(y[:, 0], 0)),
                            np.log10(np.maximum(y[:, 1], .01)), y[:, 2]/100.])


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv1d(channels, channels, 5, padding=2*dilation, dilation=dilation),
            nn.GELU(), nn.Conv1d(channels, channels, 5, padding=2*dilation, dilation=dilation))
    def forward(self, x):
        return F.gelu(x + self.layers(x))


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(27, 64, 5, padding=2)
        self.blocks = nn.Sequential(*(Block(64, d) for d in [1,2,4,8]))
        self.head = nn.Conv1d(64, 6, 1)
    def forward(self, x):
        return self.head(self.blocks(F.gelu(self.stem(x))))


class CNNMLP(nn.Module):
    """Parallel pointwise MLP and depth-context CNN with separate gate/value heads."""
    def __init__(self):
        super().__init__()
        self.stem=nn.Conv1d(27,64,5,padding=2)
        self.blocks=nn.Sequential(*(Block(64,d) for d in [1,2,4,8]))
        self.point_mlp=nn.Sequential(nn.Linear(27,128),nn.GELU(),nn.Linear(128,64),nn.GELU())
        self.fusion=nn.Sequential(nn.Conv1d(128,64,1),nn.GELU())
        self.gate_head=nn.Sequential(nn.Conv1d(64,32,1),nn.GELU(),nn.Conv1d(32,3,1))
        self.value_head=nn.Sequential(nn.Conv1d(64,32,1),nn.GELU(),nn.Conv1d(32,3,1))
    def forward(self,x):
        context=self.blocks(F.gelu(self.stem(x)))
        point=self.point_mlp(x.transpose(1,2)).transpose(1,2)
        h=self.fusion(torch.cat([context,point],dim=1))
        return torch.cat([self.gate_head(h),self.value_head(h)],dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=ROOT.parent/'data')
    ap.add_argument('--out', type=Path, default=ROOT/'outputs/simple_cnn_cv')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--steps', type=int, default=80)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=int, default=512)
    ap.add_argument('--seed', type=int, default=20260923)
    ap.add_argument('--anomaly-treatment', action='store_true', help='fixed feature-only spike handling and train-fold tail clipping')
    ap.add_argument('--model',choices=['cnn','cnn_mlp'],default='cnn')
    args = ap.parse_args()
    model_class=SimpleCNN if args.model=='cnn' else CNNMLP
    architecture=('Conv1d(27,64,5), 4 residual blocks (two conv k5, dilations 1,2,4,8), Conv1d(64,6,1)' if args.model=='cnn'
                  else 'Parallel CNN(27->64, four residual blocks) + pointwise MLP(27->128->64); concat 128->64 fusion; separate 64->32->3 gate and regression heads')
    args.out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()
    records = load_split(args.data_dir, 'train')
    raw=[]; ys=[]; valid=[]; groups=[]
    for wi, (_, frame) in enumerate(records):
        a=frame[COLS].to_numpy(np.float32)
        for c in ['GR','RXO','RT']:
            j=COLS.index(c);a[:,j]=np.log1p(a[:,j])
        raw.append(a)
        y=frame[TARGET_COLUMNS].to_numpy(np.float64)
        ys.append(y);v=np.isfinite(y).all(1);valid.append(v)
        groups.append(np.full(v.sum(),wi,np.int32))
    yall=np.concatenate([y[v] for y,v in zip(ys,valid)])
    group=np.concatenate(groups)
    folds=np.full(len(yall),-1,np.int8)
    for f,(_,va) in enumerate(GroupKFold(5).split(yall,groups=group)): folds[va]=f
    well_fold={wi:int(np.unique(folds[group==wi]).item()) for wi in range(len(records))}
    config={**vars(args),'data_dir':str(args.data_dir),'out':str(args.out),'input_columns':COLS,
            'architecture':architecture,
            'parameters':sum(p.numel() for p in model_class().parameters()),
            'input_channels':'13 fold-robust-scaled inputs + 13 missing masks + relative depth',
            'loss':'BCE gate + SmoothL1(beta=0.2) nondefault standardized regression, per-target masked means',
            'thresholds':[.5,.5,.5],'optimizer':'AdamW lr=0.001 weight_decay=0.0001; cosine LR',
            'seed_policy':'seed + fold; one model per fold; fixed final epoch; no outer-fold tuning',
            'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,
            'gpu':torch.cuda.get_device_name(0),'valid_rows':len(yall),'wells':len(records),
            'fold_wells':{str(f):[wid for wi,(wid,_) in enumerate(records) if well_fold[wi]==f] for f in range(5)}}
    (args.out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('CONFIG',json.dumps(config),flush=True)
    pred=np.full_like(yall,np.nan);gates=np.full_like(yall,np.nan);continuous=np.full_like(yall,np.nan)
    reports=[]
    for f in range(5):
        t0=time.perf_counter();seed=args.seed+f
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        rng=np.random.default_rng(seed)
        train=[wi for wi in range(len(records)) if well_fold[wi]!=f]
        val=[wi for wi in range(len(records)) if well_fold[wi]==f]
        assert set(train).isdisjoint(val)
        fitraw=np.concatenate([raw[wi] for wi in train])
        treatment=None;treatment_audit={}
        if args.anomaly_treatment:
            from anomaly import fit_treatment,transform
            treatment=fit_treatment([raw[wi] for wi in train],COLS)
        med=np.nanmedian(fitraw,axis=0);q=np.nanpercentile(fitraw,[25,75],axis=0)
        med=np.nan_to_num(med);scale=np.maximum(np.nan_to_num(q[1]-q[0]),1e-3)
        yt=target_transform(yall[folds!=f]);dm=default_mask(yall[folds!=f])
        tm=np.array([yt[~dm[:,j],j].mean() for j in range(3)])
        ts=np.array([max(yt[~dm[:,j],j].std(),.1) for j in range(3)])
        xs=[]; labels=[]; regs=[]; masks=[]
        for wi,a in enumerate(raw):
            if treatment is not None:
                a,treatment_audit[records[wi][0]]=transform(a,treatment)
            missing=~np.isfinite(a)
            filled=pd.DataFrame(a).interpolate(limit_direction='both').to_numpy()
            filled=np.where(np.isfinite(filled),filled,med)
            z=np.clip((filled-med)/scale,-10,10)
            d=filled[:,0];rel=(d-d.min())/max(float(d.max()-d.min()),1.)
            x=np.column_stack([z,missing.astype(np.float32),rel]).T.astype(np.float32)
            xs.append(torch.from_numpy(x).cuda())
            v=valid[wi];clean=np.where(v[:,None],ys[wi],DEFAULTS)
            dmw=default_mask(clean);r=(target_transform(clean)-tm)/ts
            labels.append(torch.from_numpy(dmw.T.astype(np.float32)).cuda())
            regs.append(torch.from_numpy(r.T.astype(np.float32)).cuda())
            masks.append(torch.from_numpy(np.broadcast_to(v,(3,len(v))).copy()).cuda())
        del fitraw
        model=model_class().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs*args.steps,eta_min=1e-5)
        lengths=np.array([len(raw[w]) for w in train]);prob=lengths/lengths.sum()
        history=[]
        for epoch in range(args.epochs):
            model.train();loss_acc=0.
            for step in range(args.steps):
                chosen=rng.choice(train,size=args.batch,p=prob)
                starts=[rng.integers(0,len(raw[w])-args.crop+1) for w in chosen]
                def batch_from(source):
                    return torch.stack([source[w][:,s:s+args.crop] for w,s in zip(chosen,starts)])
                xb=batch_from(xs);gb=batch_from(labels);rb=batch_from(regs);mb=batch_from(masks)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): output=model(xb)
                output=output.float();cm=mb & (gb<.5)
                lg=F.binary_cross_entropy_with_logits(output[:,:3],gb,reduction='none')
                lr=F.smooth_l1_loss(output[:,3:],rb,reduction='none',beta=.2)
                loss=((lg*mb).sum((0,2))/mb.sum((0,2)).clamp_min(1)).mean()+((lr*cm).sum((0,2))/cm.sum((0,2)).clamp_min(1)).mean()
                assert torch.isfinite(loss), 'Non-finite loss'
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();sched.step()
                loss_acc+=float(loss.detach())
            history.append(loss_acc/args.steps)
            if (epoch+1)%5==0 or epoch==0:
                print(f'fold={f+1}/5 epoch={epoch+1}/{args.epochs} train_loss={history[-1]:.5f} elapsed={time.perf_counter()-t0:.1f}s',flush=True)
        model.eval()
        with torch.no_grad():
            for wi in val:
                with torch.autocast('cuda',dtype=torch.bfloat16): out=model(xs[wi][None])[0].float().cpu().numpy().T
                v=valid[wi];out=out[v];g=1/(1+np.exp(-np.clip(out[:,:3],-50,50)))
                r=out[:,3:]*ts+tm
                c=np.column_stack([np.expm1(np.clip(r[:,0],-20,10)),10**np.clip(r[:,1],-5,6),r[:,2]*100])
                c=np.clip(c,LOW,HIGH);pp=np.where(g>=.5,DEFAULTS,c)
                idx=group==wi;pred[idx]=pp;gates[idx]=g;continuous[idx]=c
        va=folds==f;rep={'fold':f+1,'train_wells':len(train),'validation_wells':len(val),'rows':int(va.sum()),
                         'seconds':time.perf_counter()-t0,'scores':scores(yall[va],pred[va]),'training_loss':history}
        reports.append(rep)
        if treatment is not None:
            (args.out/f'treatment_fold_{f+1}.json').write_text(json.dumps({'fit_wells':[records[w][0] for w in train],
                'validation_wells':[records[w][0] for w in val],'spec':treatment,'audit':treatment_audit},indent=2))
        torch.save({'state_dict':model.cpu().state_dict(),'input_median':med.tolist(),'input_scale':scale.tolist(),
                    'target_mean':tm.tolist(),'target_scale':ts.tolist(),'fold':f,'config':config,'treatment':treatment},args.out/f'fold_{f+1}.pt')
        np.savez_compressed(args.out/'oof.npz',truth=yall,pred=pred,gate=gates,continuous=continuous,fold=folds,well=group)
        (args.out/'progress.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print('FOLD_RESULT',json.dumps(rep|{'training_loss':history[-1]}),flush=True)
        del model,opt,xs,labels,regs,masks;torch.cuda.empty_cache()
    assert np.isfinite(pred).all()
    summary={'config':config,'folds':reports,'oof_scores':scores(yall,pred),
             'fold_total_mean':float(np.mean([r['scores']['TOTAL'] for r in reports])),
             'fold_total_std':float(np.std([r['scores']['TOTAL'] for r in reports])),
             'all_default_baseline':scores(yall,np.broadcast_to(DEFAULTS,yall.shape)),
             'nondefault_scores':{t:scores(yall[~default_mask(yall)[:,j]],pred[~default_mask(yall)[:,j]])[t] for j,t in enumerate(TARGET_COLUMNS)},
             'per_well':{records[wi][0]:scores(yall[group==wi],pred[group==wi]) for wi in range(len(records))},
             'elapsed_seconds':time.perf_counter()-started}
    (args.out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('FINAL',json.dumps({k:v for k,v in summary.items() if k not in ['config','folds','per_well']}),flush=True)


if __name__=='__main__': main()
