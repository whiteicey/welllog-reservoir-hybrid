"""Create isolated read-only weight evaluator from verified V363 evaluator."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
old=ROOT/'experiments/pure_dl/evaluate_original_seed_pair_v363r1.py'
text=old.read_text(encoding='utf-8')
def change(a,b):
    global text
    assert text.count(a)==1,(a,text.count(a))
    text=text.replace(a,b)
change("original_seed_pair_v363_20261002'", "original_seed_weight_grid_v364_20261002'")
change("started=time.perf_counter();torch.set_num_threads(3)", "started=time.perf_counter();weights=np.arange(101,dtype=np.float64)/100;torch.set_num_threads(3)")
change("n=len(old_data['validation']);out=np.full((n,3),np.nan,np.float64);coverage=np.zeros(n,np.int8)", "n=len(old_data['validation']);out=np.full((n,2),np.nan,np.float64);indices=np.full((n,len(weights)),65535,np.uint16);coverage=np.zeros(n,np.int8)")
change("out[dest,2]=old_eval.actions[((utilities[0]+utilities[1])*.5).argmax(-1).cpu().numpy()]", "for wi,w in enumerate(weights):\n                indices[dest,wi]=(utilities[0]*float(w)+utilities[1]*float(1-w)).argmax(-1).cpu().numpy().astype(np.uint16)")
change("y=old_data['truth'][old_data['validation']].copy();pred=y.copy();pred[:,c]=out[:,2]", "assert (indices<len(old_eval.actions)).all()\n    assert np.array_equal(old_eval.actions[indices[:,0]],new_native)\n    assert np.array_equal(old_eval.actions[indices[:,-1]],old_native)\n    assert np.array_equal(old_eval.actions[indices[:,50]],np.load(ROOT/'results/deep_learning/original_seed_pair_v363_20261002'/(target+'_prediction.npy')))\n    y=old_data['truth'][old_data['validation']].copy();pred=y.copy();grid_scores=[]\n    for wi,w in enumerate(weights):\n        pred[:,c]=old_eval.actions[indices[:,wi]];grid_scores.append(dict(original_weight=float(w),new_seed_weight=float(1-w),score=metric(y,pred)[target]))")
change("np.save(STUDY/(target+'_prediction.npy'),out[:,2])", "np.savez_compressed(STUDY/(target+'_grid_actions.npz'),weights=weights,actions=old_eval.actions,indices=indices)")
change("result=dict(target=target,score=metric(y,pred)[target],weights=[.5,.5],", "result=dict(target=target,scores_by_weight=grid_scores,weight_grid=weights.tolist(),endpoints_and_half_exact=True,")
change("Equal native probability mixture then fixed official expected-utility argmax; average utility is linear-equivalent, each native support retained", "Common weighted native probability mixture then unchanged official utility argmax; utility weighting linear-equivalent, each native support retained")
dest=ROOT/'experiments/pure_dl/evaluate_original_seed_weights_v364.py';assert not dest.exists()
compile(text,str(dest),'exec');dest.write_text(text,encoding='utf-8')
(STUDY/'PLAN.json').write_text(json.dumps(dict(original_model=80.70612972323806,new_seed=20261010,weight_grid='0:0.01:1',coarse_grid='0:0.05:1',common_weight_all_heads=True,extra_training_stages=0,maximum_GPU_jobs=2,source_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2),encoding='utf-8')
print('Prepared isolated weight evaluator')
