"""Read-only failed-stage best checkpoint replay; no training or resume."""
import os, sys, json, time, random, hashlib
from pathlib import Path
os.environ.update(WELLLOG_MODEL_ID='seed_20261013', WELLLOG_FOLD='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
ROOT=Path(__file__).resolve().parents[3]; STUDY=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/pure_dl'))
import numpy as np, torch
import pair_fullcv_models_v370 as helper
from gpu_job_guard_v2 import check_slot
from train_pair_fullcv_stage_v370 import frozen_signature
helper.bind_paths(); torch.set_num_threads(3); helper._manual_seed(helper.SEED)
np.random.seed(helper.SEED); random.seed(helper.SEED)
torch.use_deterministic_algorithms(True,warn_only=True); torch.backends.cudnn.benchmark=False
admission=check_slot(min_free_gib=7.)
stage='SW351'; out=helper.OUT/stage; start=time.perf_counter()
last_path=out/'last_state.pt'; before=hashlib.sha256(last_path.read_bytes()).hexdigest()
saved=torch.load(last_path,map_location='cpu',weights_only=False)
assert saved['epoch']==11 and saved['best_epoch']==3
data,master,metadata=helper.load_data(stage)
model=helper.make_model(stage,master,metadata).cuda(); model.initialize_priors(data)
evaluator=helper.Evaluator(data,master,2,helper.STAGES[stage][1],stage)
model.load_state_dict(saved['current']); assert frozen_signature(model)==saved['config']['fixed_signature']
model.load_state_dict(saved['best_state']); torch.cuda.reset_peak_memory_stats()
pred,score=evaluator(model)
assert len(pred)==144513 and np.array_equal(pred,saved['best_pred']) and abs(score-saved['best'])<1e-9
assert before==hashlib.sha256(last_path.read_bytes()).hexdigest()
result=dict(fold=4,identity='seed_20261013',stage=stage,last_completed_epoch=11,best_epoch=3,
    best_score=score,rows=len(pred),best_prediction_exact=True,current_frozen_state_exact=True,
    last_state_sha256=before,checkpoint_unchanged=True,training_steps=0,GPU_tasks=1,
    original_validation_batch=evaluator.batch,peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
    peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,seconds=time.perf_counter()-start,
    admission=admission,scope='Read-only best checkpoint replay; training recovery not authorized or started')
(STUDY/'SW351_FOLD4_RECOVERY_READBACK.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
