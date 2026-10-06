"""Restore atomic best after explicitly documented low-value training termination."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import torch,json,time,hashlib,psutil,re
from datetime import datetime
from compact_sw_scaled_sam_v123 import ROOT,STUDY,OUT,SOURCE,load_data,make_model
from train_stage_fullcv_v97 import Evaluator,frozen_signature
from train_dual8222_heads80 import atomic_json,atomic_array
from fullcv_resume_v97 import atomic_save
out=OUT/'rms_direction';decision=json.loads((STUDY/'early_stop_decision.json').read_text());assert not psutil.pid_exists(decision['pid'])
started=time.perf_counter();saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);cfg=saved['config'];assert json.loads((out/'config.json').read_text())==cfg
assert cfg['source_sha256']==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
for name,sha in cfg['source_hashes'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==sha
atomic_json(out/'runtime_status.json',dict(status='finalizing_early_stop',pid=os.getpid(),last_training_pid=decision['pid'],epoch=saved['epoch']))
torch.set_num_threads(3);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
stage='rms_direction';data,master,metadata=load_data(stage);model=make_model(stage,master,metadata).cuda();model.load_state_dict(saved['best_state'],strict=True)
pred,score=Evaluator(data,master,2,1024)(model)
import numpy as np
assert np.array_equal(pred,saved['best_pred']) and abs(score-saved['best'])<1e-9
assert frozen_signature(model)==cfg['fixed_signature']
atomic_save(dict(state_dict=saved['best_state'],config=cfg,best_epoch=saved['best_epoch'],target_score=score),out/'model_best.pt');atomic_array(out/'channel_pred.npy',pred)
launch=json.loads((STUDY/'launch.json').read_text(encoding='utf-8-sig'));launch_time=datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',launch['launched'])).timestamp()
reference=json.loads((ROOT/'results/deep_learning/fullcv_v97_20260930/promotion.json').read_text())['scores']['SW']
report=dict(stage=stage,target='SW',target_score=score,initial_score=saved['initial'],against_initial=score-saved['initial'],against_v97_target=score-reference,
    best_epoch=saved['best_epoch'],stopped_epoch=saved['epoch'],validation_rows=len(pred),predictions_reproduced_exactly=True,frozen_parameters_exact=True,
    gradient_groups=saved['gradient'],seconds=saved['seconds']+time.perf_counter()-started,peak_MiB=max(saved['peak'],torch.cuda.max_memory_allocated()/2**20),complete_five_folds=False,
    early_terminated=True,partial_inflight_epoch_discarded=True,early_stop_decision=decision,finalization_seconds=time.perf_counter()-started,launch_to_final_report_wall_seconds=time.time()-launch_time)
atomic_json(out/'report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete_early_terminated',pid=os.getpid(),**report));print('EARLY_FINAL',json.dumps(report),flush=True)
