"""Native probability-mixture scan; one target/fold, exact standalone replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import torch
import pair_fullcv_models_v370 as helper
import champion_scaling_v21
from fullcv_pair_context_v370 import ROOT,STUDY,BASE_SEEDS,own_dir,require_authorization,sha,write
from multiscale_sequence import window_indices
from gpu_job_guard_v2 import check_slot

ORIGINAL_LOAD=torch.load;ORIGINAL_MANUAL=torch.manual_seed
SNAPSHOT=[]
for module in list(sys.modules.values()):
    file=getattr(module,'__file__',None)
    if file and Path(file).resolve().is_relative_to(ROOT/'experiments/pure_dl') and module is not helper:
        for attr in ('SOURCE','PARENT','PARENT_SHA'):
            if hasattr(module,attr):SNAPSHOT.append((module,attr,getattr(module,attr)))

def reset():
    torch.load=ORIGINAL_LOAD;torch.manual_seed=ORIGINAL_MANUAL
    for module,attr,value in SNAPSHOT:setattr(module,attr,value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,choices=[1,2,3,4],required=True)
    ap.add_argument('--target',choices=['POR','PERM','SW'],required=True);a=ap.parse_args()
    plan=require_authorization();fold=a.fold;target=a.target;c=['POR','PERM','SW'].index(target)
    stage=['POR341','PERM336','SW351'][c];length=helper.STAGES[stage][1]
    dest=STUDY/f'fold_{fold}';dest.mkdir(exist_ok=True)
    if (dest/(target+'_GPU_REPORT.json')).exists():raise RuntimeError('Completed native scan immutable')
    torch.set_num_threads(3);slot=check_slot(min_free_gib=4.)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    start=time.perf_counter();models=[];traces={}
    for identity in BASE_SEEDS:
        reset();helper.IDENTITY=identity;helper.FOLD=fold;helper.SEED=BASE_SEEDS[identity]+fold-1
        helper.OUT=own_dir(identity,fold);helper.SOURCE=helper.OUT/f'SOURCE/fold_{fold}.pt'
        helper.TRACE.clear();helper.bind_paths()
        data,master,metadata=helper.load_data(stage)
        model=helper.make_model(stage,master,metadata).cuda();model.initialize_priors(data)
        path=helper.OUT/stage/'model_best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict'])
        expected=np.load(path.parent/'channel_pred.npy');del saved;model.eval()
        ev=helper.Evaluator(data,master,c,length,stage)
        models.append((data,model,ev,expected));traces[identity]=list(helper.TRACE)
    reset();d,_,ev,_=models[0];other=models[1]
    assert np.array_equal(ev.actions,other[2].actions)
    assert ev.centers==other[2].centers and ev.anchors==other[2].anchors
    assert np.array_equal(ev.local,other[2].local)
    assert np.array_equal(d['validation'],other[0]['validation'])
    assert np.array_equal(d['truth'],other[0]['truth'])
    for key in ('depth','lower','upper','row_well','absolute','raw_to_valid'):
        x,y=d[key],other[0][key]
        assert torch.equal(x,y) if torch.is_tensor(x) else np.array_equal(x,y),key
    n=len(d['validation']);weights=np.asarray(plan['scan_weights']);batch=min(m[2].batch for m in models)
    indices=np.full((n,len(weights)),65535,np.uint16);native=np.full((n,2),np.nan,np.float64)
    coverage=np.zeros(n,np.int8);lo=length//2-64;hi=lo+128;torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for begin in range(0,len(ev.centers),batch):
            if (helper.OUT.parent.parent/'STOP_BATCH.json').exists():raise RuntimeError('Explicit stop; no automatic scan resumption')
            cc=torch.tensor(ev.centers[begin:begin+batch],device='cuda')
            anchor=torch.tensor(ev.anchors[begin:begin+batch],device='cuda')
            ix,valid=window_indices(cc,d['lower'],d['upper'],length=length,anchors=anchor)
            rows=d['raw_to_valid'][ix[:,lo:hi].cpu().numpy()]
            use=valid[:,lo:hi].cpu().numpy()&(rows>=0);loc=ev.local[rows[use]]
            assert (loc>=0).all() and (coverage[loc]==0).all() and len(np.unique(loc))==len(loc)
            coverage[loc]+=1;take=torch.as_tensor(use,device='cuda');utility=[]
            for j,(data,model,evaluator,_) in enumerate(models):
                rel=torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
                context=data['well_bank'][data['row_well'][anchor]]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    g,logmass=model(data['x'][ix],rel,valid,absolute_depth=data['absolute'][ix],well_inputs=context)
                p=g[:,lo:hi][take][:,0].float().sigmoid()
                mass=logmass[:,lo:hi][take][:,0].float().exp()
                joint=torch.cat((p[:,None],(1-p[:,None])*mass),-1)
                utility.append(joint@evaluator.utilities)
                native[loc,j]=evaluator.actions[utility[-1].argmax(-1).cpu().numpy()]
            for wi,w in enumerate(weights):
                indices[loc,wi]=(utility[0]*float(w)+utility[1]*float(1-w)).argmax(-1).cpu().numpy().astype(np.uint16)
    assert (coverage==1).all() and (indices<len(ev.actions)).all()
    for j,m in enumerate(models):assert np.array_equal(native[:,j],m[3]),'Independent disk replay differs'
    assert np.array_equal(ev.actions[indices[:,0]],native[:,1])
    assert np.array_equal(ev.actions[indices[:,-1]],native[:,0])
    np.savez(dest/(target+'_grids.npz'),indices=indices,actions=ev.actions,weights=weights,native=native,validation=d['validation'])
    write(dest/(target+'_GPU_REPORT.json'),dict(target=target,fold=fold,rows=n,coverage_once=True,
          native_disk_replay_exact=True,endpoints_exact=True,loads=traces,seconds=time.perf_counter()-start,
          peak_MiB=torch.cuda.max_memory_allocated()/2**20,admission=slot,concurrency='serial',
          includes='load/cache/warmup/two native forwards/101-weight grid/write',extra_training_stages=0))
    print('NATIVE_SCAN_COMPLETE',fold,target,n,flush=True)

if __name__=='__main__':main()
