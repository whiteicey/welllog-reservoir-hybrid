"""Resumable full-budget V3 five-fold training and conservative NN fusion.

Uses original V3 NN trainer, architecture, budgets and CatBoost configuration.
Preprocessing statistics are fitted on the training wells of each fold.
Recorded fold-specific V3 assembly choices are frozen before this run.
No test files or all-80-well pretrained weights are read.
"""
import argparse,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from catboost import CatBoostClassifier,Pool

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import nn_seq as NN
from harness import quantile_bins,default_mask,acc_vec
from extra_features import well_extra_features
from src.data import load_split,TARGET_COLUMNS
from src.features import FeatureBuilder

TARGETS=['POR','PERM','SW'];DEFAULTS=np.array([.1,.01,99.9]);LOW=np.array([0,.01,8.305]);HIGH=np.array([50,10000,99.9])
FWD={'POR':np.log1p,'PERM':np.log10,'SW':lambda x:x}
INV={'POR':np.expm1,'PERM':lambda x:10**x,'SW':lambda x:x}

def score(y,p):
    vals={t:float(acc_vec(p[:,i],y[:,i],t).mean()*100) for i,t in enumerate(TARGETS)}
    vals['TOTAL']=sum(vals[t]*w for t,w in zip(TARGETS,[.3,.35,.35]));return vals

def atomic_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8');os.replace(tmp,path)

def kernel(grid,sup,t):
    a=grid[:,None];y=sup[None,:]
    if t=='PERM':return np.maximum(0,1-abs(np.log10(np.maximum(a,1e-5))-np.log10(np.maximum(y,1e-5))))
    return np.maximum(0,1-abs(a-y)/((.08 if t=='POR' else .05)*(abs(y)+1e-5)))

def blend(v,n,w):
    out=v.copy()
    for j,t in enumerate(TARGETS):
        active=~default_mask(v[:,j],t)&~default_mask(n[:,j],t)
        mixed=10**((1-w)*np.log10(np.maximum(v[:,j],.01))+w*np.log10(np.maximum(n[:,j],.01))) if t=='PERM' else (1-w)*v[:,j]+w*n[:,j]
        out[:,j]=np.where(active,mixed,v[:,j])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'outputs/v3_retrain_cv');args=ap.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    started=time.time();torch.set_num_threads(4);assert torch.cuda.is_available()
    status={'state':'running','pid':os.getpid(),'started':started,'stage':'prepare'}
    def update(**kw):
        status.update(kw);status['updated']=time.time();atomic_json(out/'status.json',status)
        print('STATUS',json.dumps(status),flush=True)
    update()
    cfg=json.loads((ROOT/'src/final_cfg_v3.json').read_text());reference=json.loads((ROOT/'results/assembly_D_report.json').read_text())
    contract={'source_config':cfg,'normal_python':sys.executable,'torch':torch.__version__,'catboost':__import__('catboost').__version__,
              'gpu':torch.cuda.get_device_name(0),'normalization':'fit on training fold only',
              'decoding':'frozen original per-fold nested_choices, not global final choices',
              'fusion_weights':[0,.05,.1,.2,.3],'model_budget':'14 CatBoost and 5 NN per fold; original iterations and 60x100 NN steps',
              'limitations':'Retraining in different environment and fold-only normalization: not bitwise reproduction of 81.632. Fusion grid is exploratory same-CV analysis.'}
    atomic_json(out/'experiment.json',contract)
    rec=load_split(ROOT.parent/'data','train');ys=[f[TARGET_COLUMNS].to_numpy(np.float64) for _,f in rec]
    valid=[np.isfinite(y).all(1) for y in ys]
    truth=np.concatenate([y[v] for y,v in zip(ys,valid)]);well=np.concatenate([np.full(v.sum(),wi,np.int32) for wi,v in enumerate(valid)])
    from sklearn.model_selection import GroupKFold
    folds=np.full(len(truth),-1,np.int8)
    for f,(_,va) in enumerate(GroupKFold(5).split(truth,groups=well)):folds[va]=f
    nn=np.load(ROOT/'outputs/cnn_mlp_cv/oof.npz')
    for key,arr in [('truth',truth),('well',well),('fold',folds)]:assert np.array_equal(nn[key],arr)
    pred=np.full_like(truth,np.nan)
    for f in range(5):
        fd=out/f'fold_{f+1}';fd.mkdir(exist_ok=True)
        vi=np.flatnonzero(folds==f);trmask=folds!=f;train=[i for i in range(80) if not np.any(well[vi]==i)];val=sorted(set(well[vi].tolist()))
        if (fd/'prediction.npy').exists():
            pred[vi]=np.load(fd/'prediction.npy');update(fold=f+1,stage='reuse completed fold');continue
        update(fold=f+1,stage='fold-only features')
        builder=FeatureBuilder().fit([rec[i][1] for i in train]);mats=[builder.transform(frame) for _,frame in rec]
        base=np.concatenate([m.base[v] for m,v in zip(mats,valid)]);enh=np.concatenate([m.enhanced[v] for m,v in zip(mats,valid)])
        extra=np.concatenate([well_extra_features(frame,builder.medians_).to_numpy()[v] for (_,frame),v in zip(rec,valid)])
        feats={'base':base,'enh':enh,'base+extra':np.c_[base,extra]};del extra
        atomic_json(fd/'preprocessing.json',{'train_wells':[rec[i][0] for i in train],'validation_wells':[rec[i][0] for i in val],'medians':builder.medians_})
        outputs={};centers={}
        tasks=[(kind,t,spec) for kind in ['gates','bins'] for t,spec in cfg[kind].items()]+[('states','states',cfg['states'])]
        for kind,t,spec in tasks:
            key=f'{kind}_{t}';cache=fd/f'{key}.npz'
            if cache.exists():
                q=np.load(cache);outputs[key]=q['prob'];
                if kind=='bins':centers[t]=q['centers']
                continue
            X=feats[spec['features']]
            if kind=='states':mask=trmask;label=np.select([truth[:,0]<=.1,truth[:,2]>=99.9,truth[:,1]<=.01],[0,2,3],1)[mask];k=4
            else:
                ti=TARGETS.index(t);d=default_mask(truth[:,ti],t);mask=trmask if kind=='gates' else trmask&~d
                if kind=='gates':label=d[mask].astype(np.int8);k=2
                else:label,centers[t],_=quantile_bins(FWD[t](truth[mask,ti]),spec['bins']);k=spec['bins']
            total=np.zeros((len(vi),k),np.float64)
            pool=Pool(X[mask],label=label)
            for seed in cfg['cb_seeds']:
                modelpath=fd/f'{key}_seed{seed}.cbm';m=CatBoostClassifier()
                if modelpath.exists():m.load_model(modelpath)
                else:
                    update(fold=f+1,stage=key,seed=seed)
                    offset=spec.get('seed_offset',0) if kind=='gates' else (100 if kind=='bins' else 0)
                    m=CatBoostClassifier(iterations=spec['iterations'],loss_function='Logloss' if kind=='gates' else 'MultiClass',
                        task_type='GPU',devices='0',border_count=128,random_seed=seed+offset,learning_rate=.05 if kind=='states' else spec.get('lr',.04),
                        depth=spec.get('depth',8),l2_leaf_reg=spec.get('l2',3.),bootstrap_type='Bayesian',allow_writing_files=False,
                        gpu_ram_part=.65,thread_count=4,verbose=100)
                    m.fit(pool);m.save_model(modelpath)
                total+=m.predict_proba(X[vi])[:,np.argsort(m.classes_.astype(int))]/len(cfg['cb_seeds']);del m;gc.collect()
            outputs[key]=total.astype(np.float32)
            np.savez_compressed(cache,prob=outputs[key],**({'centers':centers[t]} if kind=='bins' else {}));del pool;gc.collect()
        # Release tree matrices before NN training; retain per-well base for second group.
        del feats,base,enh;gc.collect();torch.cuda.empty_cache()
        gstats=NN.global_stats([rec[i][1] for i in train]);xraw=[NN.prep_well(frame,gstats) for _,frame in rec]
        allbase=np.concatenate([mats[i].base for i in train]);med=np.median(allbase,axis=0);q1,q3=np.percentile(allbase,[25,75],axis=0);sc=np.maximum(q3-q1,1e-3);del allbase
        nnbins=NN.make_bins(ys,train)
        for group in cfg['nns']:
            name=group['name'];cache=fd/f'{name}.npz'
            if cache.exists():
                outputs[name]=dict(np.load(cache));continue
            x=[np.concatenate([xr,np.clip((mat.base.T-med[:,None])/sc[:,None],-8,8).astype(np.float32)]) if group['use_base'] else xr for xr,mat in zip(xraw,mats)]
            D=SimpleNamespace(xtr=x,ytr=ys,n_in=x[0].shape[0])
            params=SimpleNamespace(**group['train_args']);acc={}
            for seed in group['seeds']:
                modelpath=fd/f'{name}_seed{seed}.pt'
                if modelpath.exists():
                    net=NN.WellNet(D.n_in,ch=params.channels,drop=params.dropout,k=params.kernel).cuda();net.load_state_dict(torch.load(modelpath,map_location='cuda',weights_only=True))
                else:
                    update(fold=f+1,stage=name,seed=seed)
                    net=NN.train_fold(D,nnbins,train,params,torch.device('cuda'),seed)
                    torch.save(net.cpu().state_dict(),modelpath);net=net.cuda()
                with torch.no_grad():
                    perwell=[NN.predict_well(net,D.xtr[wi],torch.device('cuda')) for wi in val]
                for ti,t in enumerate(TARGETS):
                    for suffix,index in [('gate',0),('prob',1)]:
                        ar=np.concatenate([rr[t][index][valid[wi]] for wi,rr in zip(val,perwell)])
                        ky=f'{t}_{suffix}';acc[ky]=acc.get(ky,0)+ar/len(group['seeds'])
                del net;gc.collect();torch.cuda.empty_cache()
            for t in TARGETS:acc[f'{t}_centers']=nnbins[t]['centers']
            outputs[name]=acc;np.savez_compressed(cache,**acc)
        update(fold=f+1,stage='metric decoding')
        post=outputs['states_states'];marg={'POR':post[:,0],'PERM':post[:,0]+post[:,3],'SW':post[:,0]+post[:,2]}
        fp=np.zeros((len(vi),3))
        for ti,t in enumerate(TARGETS):
            d=reference[t]['nested_choices'][f];support=INV[t](centers[t]);grid=np.linspace(support[0],support[-1],800) if t=='SW' else np.exp(np.linspace(np.log(support[0]*.9),np.log(support[-1]*1.1),800))
            spec=json.loads((ROOT/'src/assemble_cfg_D.json').read_text())[t]
            gate=np.zeros(len(vi))
            for source,w in zip(spec['gate_sources'],d['gate_w']):
                name=source['name'];g=marg[t] if name=='states' else outputs[name][f'{t}_gate'] if name.startswith('nn') else outputs[f'gates_{t}'][:,1]
                gate+=w*g
            for start in range(0,len(vi),4096):
                stop=min(start+4096,len(vi));util=np.zeros((stop-start,800))
                for source,w in zip(spec['dist_sources'],d['cont_w']):
                    if not w:continue
                    name=source['name']
                    if name.startswith('nn'):p=outputs[name][f'{t}_prob'][start:stop];sup=INV[t](outputs[name][f'{t}_centers'])
                    else:p=outputs[f'bins_{t}'][start:stop];sup=support
                    p=p/p.sum(1,keepdims=True);util+=w*(p@kernel(grid,sup,t).T)
                cont=np.clip(grid[util.argmax(1)],LOW[ti],HIGH[ti]);fp[start:stop,ti]=np.where(gate[start:stop]>=d['threshold'],DEFAULTS[ti],cont)
        pred[vi]=fp;np.save(fd/'prediction.npy',fp);atomic_json(fd/'scores.json',score(truth[vi],fp))
        update(fold=f+1,stage='fold complete',scores=score(truth[vi],fp));del mats,outputs,xraw;gc.collect()
    assert np.isfinite(pred).all()
    np.savez_compressed(out/'v3_oof.npz',pred=pred,truth=truth,well=well,fold=folds)
    variants={}
    for w in contract['fusion_weights']:
        p=blend(pred,nn['pred'],w)
        variants[str(w)]={'scores':score(truth,p),'fold_scores':[score(truth[folds==f],p[folds==f]) for f in range(5)]}
        np.save(out/f'fusion_{w:.2f}.npy',p)
    result={'v3_retrained':score(truth,pred),'cnn_mlp':score(truth,nn['pred']),'fusion':variants,'elapsed_seconds':time.time()-started,'contract':contract}
    atomic_json(out/'report.json',result);update(state='complete',stage='fusion complete',scores=result['v3_retrained']);print('FINAL',json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
