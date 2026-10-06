"""Read-only fixed equal distribution mixture, one channel; no training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['WELLLOG_RUN_SEED']='20261010'
from pathlib import Path
import argparse, hashlib, importlib, json, time
import numpy as np
import torch
from gpu_job_guard_v2 import check_slot
from multiscale_sequence import window_indices
from train import metric

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/original_seed_pair_v363_20261002'
OWN=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261010'
SPECS=(('POR','POR341','por_coupling_conditional_rps_v341','conditional','pure_dl_por_coupling_conditional_rps_v341_fold5/conditional'),
       ('PERM','PERM336','perm_low_rank_cross_v336','cross','pure_dl_perm_low_rank_cross_v336_fold5/cross'),
       ('SW','SW351','sw_dynamic_cross_spectrum_v351','aligned','pure_dl_sw_dynamic_cross_spectrum_v351_fold5/aligned'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--channel',choices=['POR','PERM','SW'],required=True);args=ap.parse_args()
    c=['POR','PERM','SW'].index(args.channel);target,stage,name,mode,old_rel=SPECS[c]
    if (STUDY/(target+'_RESULT.json')).exists():raise RuntimeError('Finished comparison immutable')
    started=time.perf_counter();torch.set_num_threads(3)
    slot=check_slot(min_free_gib=4.);print('ADMISSION',slot,flush=True)
    torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    m=importlib.import_module(name);length=m.STAGES[mode][1]
    old_data,old_master,old_metadata=m.load_data(mode)
    old_model=m.make_model(mode,old_master,old_metadata).cuda();old_model.initialize_priors(old_data)
    old_path=ROOT/'outputs'/old_rel/'model_best.pt'
    old_saved=torch.load(old_path,map_location='cpu',weights_only=False)
    old_model.load_state_dict(old_saved['state_dict']);del old_saved
    old_eval=m.Evaluator(old_data,old_master,c,length,mode)
    from complete_seed_models_v362r1 import bind_paths,load_data,make_model,Evaluator,TRACE
    bind_paths()
    new_data,new_master,new_metadata=load_data(stage)
    new_model=make_model(stage,new_master,new_metadata).cuda();new_model.initialize_priors(new_data)
    new_path=OWN/('SW351_fixed' if stage=='SW351' else stage)/'model_best.pt'
    new_saved=torch.load(new_path,map_location='cpu',weights_only=False)
    new_model.load_state_dict(new_saved['state_dict']);del new_saved
    new_eval=Evaluator(new_data,new_master,c,length,stage)
    assert np.array_equal(old_eval.actions,new_eval.actions)
    assert np.array_equal(old_data['validation'],new_data['validation'])
    assert np.array_equal(old_data['truth'],new_data['truth'])
    assert np.array_equal(old_eval.local,new_eval.local)
    assert old_eval.centers==new_eval.centers and old_eval.anchors==new_eval.anchors
    for k in ('depth','lower','upper','row_well','absolute','raw_to_valid'):
        assert torch.equal(old_data[k],new_data[k]),k
    for model in (old_model,new_model):model.eval()
    n=len(old_data['validation']);out=np.full((n,3),np.nan,np.float64);coverage=np.zeros(n,np.int8)
    lo=length//2-64;hi=lo+128;batch=min(old_eval.batch,new_eval.batch)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for begin in range(0,len(old_eval.centers),batch):
            cc=torch.tensor(old_eval.centers[begin:begin+batch],device='cuda');anchor=torch.tensor(old_eval.anchors[begin:begin+batch],device='cuda')
            ix,valid=window_indices(cc,old_data['lower'],old_data['upper'],length=length,anchors=anchor)
            rows=old_data['raw_to_valid'][ix[:,lo:hi].cpu().numpy()]
            use=valid[:,lo:hi].cpu().numpy()&(rows>=0);dest=old_eval.local[rows[use]]
            assert (dest>=0).all() and (coverage[dest]==0).all() and len(np.unique(dest))==len(dest)
            coverage[dest]+=1;take=torch.as_tensor(use,device='cuda');utilities=[]
            for j,(data,model,evaluator) in enumerate(((old_data,old_model,old_eval),(new_data,new_model,new_eval))):
                rel=torch.tanh((data['depth'][ix]-data['depth'][anchor,None])/10)
                context=data['well_bank'][data['row_well'][anchor]]
                with torch.autocast('cuda',dtype=torch.bfloat16):gate,logmass=model(data['x'][ix],rel,valid,absolute_depth=data['absolute'][ix],well_inputs=context)
                g=gate[:,lo:hi][take][:,0].float().sigmoid();mass=logmass[:,lo:hi][take][:,0].float().exp()
                joint=torch.cat((g[:,None],(1-g[:,None])*mass),-1)
                utilities.append(joint@evaluator.utilities)
                out[dest,j]=evaluator.actions[utilities[-1].argmax(-1).cpu().numpy()]
            out[dest,2]=old_eval.actions[((utilities[0]+utilities[1])*.5).argmax(-1).cpu().numpy()]
    assert (coverage==1).all() and np.isfinite(out).all()
    old_native=np.load(ROOT/'outputs'/old_rel/'channel_pred.npy')
    new_native=np.load(OWN/('SW351_fixed' if stage=='SW351' else stage)/'channel_pred.npy')
    assert np.array_equal(out[:,0],old_native),'Original disk replay differs'
    assert np.array_equal(out[:,1],new_native),'New disk replay differs'
    y=old_data['truth'][old_data['validation']].copy();pred=y.copy();pred[:,c]=out[:,2]
    np.save(STUDY/(target+'_prediction.npy'),out[:,2])
    result=dict(target=target,score=metric(y,pred)[target],weights=[.5,.5],
                method='Equal native probability mixture then fixed official expected-utility argmax; average utility is linear-equivalent, each native support retained',
                actions_identical=True,supports_identical=torch.equal(old_eval.utilities,new_eval.utilities),
                validation_rows=n,coverage_exactly_once=True,old_disk_replay_exact=True,new_disk_replay_exact=True,
                checkpoint_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (old_path,new_path)],
                actual_new_checkpoint_loads=TRACE,seconds=time.perf_counter()-started,
                peak_MiB=torch.cuda.max_memory_allocated()/2**20,admission=slot,
                concurrent_training_continues=True,extra_training_stages=0,screening_fold=5,
                complete_five_folds=False,remaining_folds_started=False)
    (STUDY/(target+'_RESULT.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PAIR_CHANNEL_FINAL',json.dumps(result),flush=True)

if __name__=='__main__':main()
