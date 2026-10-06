"""Replay saved best from the last complete epoch after explicit priority reallocation."""
import json,hashlib,time,os
from pathlib import Path
import numpy as np
import torch,psutil
from sw_ranked_probability_v155 import ROOT,STUDY,OUT,load_data,make_model
from train_stage_fullcv_v97 import Evaluator,frozen_signature
from fullcv_resume_v97 import atomic_save
from train_dual8222_heads80 import atomic_json,atomic_array
from training_awake_guard import TrainingAwake
from gpu_job_guard_v2 import check_slot

def main():
    out=OUT/'local';assert not (out/'report.json').exists()
    state=json.loads((out/'runtime_status.json').read_text(encoding='utf-8-sig'))
    try:
        live=any(Path(x).name=='train_sw_ranked_probability_v155.py' for x in psutil.Process(state['pid']).cmdline())
    except psutil.Error:live=False
    assert not live,'Do not replay beside the same active training'
    print('ADMISSION',check_slot(min_free_gib=4.),flush=True)
    saved=torch.load(out/'last_state.pt',map_location='cpu',weights_only=False);cfg=saved['config']
    for name,digest in cfg['source_hashes'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest
    torch.set_num_threads(3);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True,warn_only=True);torch.backends.cudnn.benchmark=False
    tick=time.perf_counter();data,master,metadata=load_data('local');model=make_model('local',master,metadata).cuda()
    model.initialize_priors(data);model.load_state_dict(saved['best_state'])
    pred,score=Evaluator(data,master,2,1024)(model)
    assert np.array_equal(pred,saved['best_pred']) and abs(score-saved['best'])<1e-9
    assert frozen_signature(model)==cfg['fixed_signature']
    atomic_save(dict(state_dict=saved['best_state'],config=cfg,best_epoch=saved['best_epoch'],target_score=score),out/'model_best.pt');atomic_array(out/'channel_pred.npy',pred)
    report=dict(stage='local',target='SW',target_score=score,initial_score=saved['initial'],against_initial=score-saved['initial'],
        best_epoch=saved['best_epoch'],stopped_epoch=saved['epoch'],validation_rows=len(pred),predictions_reproduced_exactly=True,
        frozen_parameters_exact=True,gradient_groups=saved['gradient'],seconds=saved['seconds'],peak_MiB=saved['peak'],
        complete_five_folds=False,early_terminated=True,stop_reason='User prioritizes POR augmentation and PERM alternatives; budget reallocation, not a convergence conclusion',
        incomplete_epoch_discarded=True,last_state_sha256=hashlib.sha256((out/'last_state.pt').read_bytes()).hexdigest(),
        replay_seconds=time.perf_counter()-tick,replayed_from_saved_best=True)
    atomic_json(out/'report.json',report);atomic_json(out/'runtime_status.json',dict(status='complete',pid=os.getpid(),**report))
    print(json.dumps(report),flush=True)
if __name__=='__main__':
    with TrainingAwake():main()
