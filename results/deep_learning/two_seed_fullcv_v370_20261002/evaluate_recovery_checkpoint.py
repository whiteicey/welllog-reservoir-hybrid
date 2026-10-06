"""Read-only serial GPU checkpoint recovery verification; never train or resume."""
import os,sys,json,time,random
from pathlib import Path
identity,stage=sys.argv[1:3]
assert (identity,stage) in (('original','PERM336'),('seed_20261013','PERM154'))
os.environ.update(WELLLOG_MODEL_ID=identity,WELLLOG_FOLD='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[3];STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import numpy as np,torch
import pair_fullcv_models_v370 as helper
from gpu_job_guard_v2 import check_slot
from train_pair_fullcv_stage_v370 import frozen_signature
helper.bind_paths();torch.set_num_threads(3);helper._manual_seed(helper.SEED);np.random.seed(helper.SEED);random.seed(helper.SEED)
torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
admission=check_slot(min_free_gib=6.)
start=time.perf_counter();out=helper.OUT/stage
saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False)
data,master,metadata=helper.load_data(stage);model=helper.make_model(stage,master,metadata).cuda();model.initialize_priors(data)
evaluator=helper.Evaluator(data,master,1,helper.STAGES[stage][1],stage)
model.load_state_dict(saved['current']);assert frozen_signature(model)==saved['config']['fixed_signature']
model.load_state_dict(saved['best_state']);torch.cuda.reset_peak_memory_stats();pred,score=evaluator(model)
assert np.array_equal(pred,saved['best_pred']) and abs(score-saved['best'])<1e-9
result=dict(identity=identity,stage=stage,fold=1,last_completed_epoch=saved['epoch'],best_epoch=saved['best_epoch'],
 best_score=score,rows=len(pred),best_prediction_exact=True,current_frozen_state_exact=True,
 GPU_tasks=1,training_steps=0,original_validation_batch=evaluator.batch,
 peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
 seconds=time.perf_counter()-start,admission=admission,checkpoint_written=False,
 scope='read-only sequential best checkpoint replay; not resumed training or a guarantee against future external GPU memory changes')
(STUDY/(identity+'_'+stage+'_RECOVERY_READBACK.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
