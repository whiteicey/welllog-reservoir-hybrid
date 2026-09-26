"""Predeclared equal-weight neural posterior mixture; never fit weights to OOF labels."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from train import ROOT,TS,metric
from src.utils import DEFAULTS,BOUNDS,kernel_matrix

def load_fold(run,f):
    options=[run/f'fold_{f+1}_distributions.npz',run/f'fold_{f+1}_probabilities.npz']
    p=next((p for p in options if p.exists()),None)
    if p is None:raise FileNotFoundError(str(options))
    z=np.load(p);keys=z.files
    idx=z['indices'] if 'indices' in keys else z['row_index']
    if 'gate' in keys:
        gate=z['gate'].astype(np.float32)
    else:
        s=z['state'].astype(np.float32);s/=s.sum(1,keepdims=True)
        gate=np.c_[s[:,0],s[:,0]+s[:,3],s[:,0]+s[:,2]]
    assert gate.shape==(len(idx),3) and np.isfinite(gate).all()
    assert ((gate>=0)&(gate<=1)).all()
    probs={};sup={}
    for t in TS:
        pk=t+'_prob' if t+'_prob' in keys else 'prob_'+t
        ck=t+'_centers' if t+'_centers' in keys else 'centers_'+t
        v=z[pk].astype(np.float32);v/=v.sum(1,keepdims=True).clip(1e-8)
        assert np.isfinite(v).all() and (v>=0).all()
        probs[t]=v;sup[t]=z[ck].astype(np.float64)
    return idx,gate,probs,sup

def load_specialists(runs,f):
    """One corresponding specialist per target, without materializing new NPZs."""
    ids=None;gates=[];probs={};sup={}
    for t,run in zip(TS,runs):
        with np.load(run/f'fold_{f+1}_distributions.npz') as z:
            idx=z['indices']
            if ids is None:ids=idx
            else:assert np.array_equal(ids,idx)
            g=z[t+'_gate'].astype(np.float32).reshape(-1)
            assert len(g)==len(idx) and np.isfinite(g).all() and ((g>=0)&(g<=1)).all()
            gates.append(g)
            v=z[t+'_prob'].astype(np.float32)
            assert np.isfinite(v).all() and (v>=0).all() and (v.sum(1)>0).all()
            probs[t]=v/v.sum(1,keepdims=True)
            sup[t]=z[t+'_centers'].astype(np.float64)
    return ids,np.column_stack(gates),probs,sup

def load_target_fold(run,target,f):
    """A predeclared additional source for exactly one target."""
    j=TS.index(target)
    with np.load(run/f'fold_{f+1}_distributions.npz') as z:
        idx=z['indices']
        g=z[target+'_gate'] if target+'_gate' in z.files else z['gate'][:,j]
        g=np.asarray(g,dtype=np.float32).reshape(-1)
        p=z[target+'_prob'].astype(np.float32)
        sup=z[target+'_centers'].astype(np.float64)
        assert g.shape==(len(idx),) and np.isfinite(g).all() and ((g>=0)&(g<=1)).all()
        assert p.shape==(len(idx),len(sup)) and np.isfinite(p).all() and (p>=0).all() and (p.sum(1)>0).all()
        gate=np.zeros((len(idx),3),np.float32);gate[:,j]=g
        return idx,gate,{target:p/p.sum(1,keepdims=True)},{target:sup}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--specialists',nargs=3,type=Path,metavar=('POR_DIR','PERM_DIR','SW_DIR'))
    ap.add_argument('--target-source',action='append',default=[],metavar='TARGET=DIR')
    a=ap.parse_args();started=time.perf_counter();torch.set_num_threads(4);a.out.mkdir(parents=True,exist_ok=True)
    extra={t:[] for t in TS}
    for value in a.target_source:
        target,sep,path=value.partition('=')
        if not sep or target not in TS or not path:ap.error('--target-source requires POR=DIR, PERM=DIR or SW=DIR')
        extra[target].append(Path(path))
    all_runs=a.runs+(a.specialists or [])+[r for paths in extra.values() for r in paths]
    outputs=[np.load(r/'oof.npz') for r in all_runs];ref=outputs[0]
    for o in outputs[1:]:
        for k in ['truth','well','fold']:assert np.array_equal(ref[k],o[k]),k
    pred=np.full_like(ref['truth'],np.nan);fold_reports=[]
    for f in range(5):
        sources=[load_fold(r,f) for r in a.runs];idx=np.flatnonzero(ref['fold']==f)
        if a.specialists:sources.append(load_specialists(a.specialists,f))
        for s in sources:assert np.array_equal(s[0],idx)
        for j,t in enumerate(TS):
            target_sources=sources+[load_target_fold(run,t,f) for run in extra[t]]
            for s in target_sources:assert np.array_equal(s[0],idx)
            supports=[s[3][t] for s in target_sources]
            lo=min(float(s.min()) for s in supports);hi=max(float(s.max()) for s in supports)
            lo=max(lo,BOUNDS[t][0],1e-5);hi=min(hi,BOUNDS[t][1])
            grid=np.linspace(lo,hi,800) if t=='SW' else np.geomspace(lo,hi,800)
            grid=np.unique(np.r_[DEFAULTS[t],grid])
            ks=[torch.tensor(kernel_matrix(grid,np.r_[DEFAULTS[t],sup],t).T,dtype=torch.float32,device='cuda') for sup in supports]
            for start in range(0,len(idx),4096):
                end=min(start+4096,len(idx));u=torch.zeros((end-start,len(grid)),device='cuda')
                for source,k in zip(target_sources,ks):
                    g=source[1][start:end,j:j+1];p=source[2][t][start:end]
                    full=np.c_[g,(1-g)*p].astype(np.float32)
                    u.add_(torch.from_numpy(full).cuda()@k,alpha=1/len(target_sources))
                pred[idx[start:end],j]=grid[u.argmax(1).cpu().numpy()]
        s=metric(ref['truth'][idx],pred[idx]);fold_reports.append({'fold':f+1,'scores':s});print('FOLD',f+1,s,flush=True)
    assert np.isfinite(pred).all()
    np.savez_compressed(a.out/'oof.npz',truth=ref['truth'],well=ref['well'],fold=ref['fold'],pred=pred)
    count=len(a.runs)+int(bool(a.specialists))
    config=json.loads((a.runs[0]/'config.json').read_text());config['ensemble']={'runs':[str(r) for r in a.runs],'specialists':dict(zip(TS,map(str,a.specialists))) if a.specialists else None,'target_sources':{t:list(map(str,extra[t])) for t in TS},'weights_per_target':{t:[1/(count+len(extra[t]))]*(count+len(extra[t])) for t in TS},'selection':'fixed equal weights, no fitting weights or thresholds against OOF labels; CV is reused for method development'}
    report={'config':config,'oof_scores':metric(ref['truth'],pred),'folds':fold_reports,'seconds':time.perf_counter()-started}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report['oof_scores']),flush=True)

if __name__=='__main__':main()
