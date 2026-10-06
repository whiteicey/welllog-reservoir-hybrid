"""One predefined fixed-checkpoint ablation, no training or threshold search."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from perm_joint_state_share_v300 import ROOT,STUDY,PARENT,load_data,Network,Evaluator
from gpu_job_guard_v2 import check_slot
from train_dual8222_heads80 import atomic_json,atomic_array
import train_invariant as base

def signature(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('gate_patch.'):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def main():
    destination=STUDY/'state_pruned_assembled.json'
    if destination.exists():raise RuntimeError('Completed ablation immutable')
    started=time.perf_counter();torch.set_num_threads(3)
    print('ADMISSION',check_slot(min_free_gib=4.),flush=True)
    atomic_json(STUDY/'state_prune_runtime_status.json',dict(status='running',pid=os.getpid(),variant_count=1))
    torch.manual_seed(20261010);np.random.seed(20261010)
    torch.use_deterministic_algorithms(True,warn_only=True)
    data,master,metadata=load_data('joint_gate')
    checkpoint=ROOT/'outputs/pure_dl_perm_joint_state_share_v300_fold5/joint_gate/model_best.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['best_epoch']==29
    model=Network(master,metadata,'joint_gate').cuda();model.initialize_priors(data)
    model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    before=signature(model)
    with torch.no_grad():model.gate_patch.weight.zero_();model.gate_patch.bias.zero_()
    assert signature(model)==before
    evaluator=Evaluator(data,master,1,256,'joint_gate')
    values,score=evaluator(model)
    por=ROOT/'outputs/pure_dl_por_direct_share_v290_fold5/raw/channel_pred.npy'
    sw=ROOT/'outputs/pure_dl_sw_raw_well_condition_v297_fold5/well/channel_pred.npy'
    pred=np.column_stack((np.load(por),values,np.load(sw)))
    assert pred.shape==(144665,3) and np.isfinite(pred).all()
    truth=data['truth'][data['validation']];scores=base.metric(truth,pred)
    assert abs(score-scores['PERM'])<1e-9
    mask=base.dmask(truth)[:,1]
    strata=dict(default_PERM=base.metric(truth[mask],pred[mask])['PERM'],continuous_PERM=base.metric(truth[~mask],pred[~mask])['PERM'])
    result=dict(scores=scores,strata=strata,parent_full_PERM=float(saved['target_score']),gain_vs_parent_PERM=score-float(saved['target_score']),gain_vs_retained291=score-89.5155547932572,screening_fold=5,validation_rows=144665,complete_five_folds=False,remaining_folds_started=False,strict_808_qualified=scores['TOTAL']>80.8,total_training_stages=8,added_head_stages=dict(POR=1,PERM=1,SW=1),variant_count=1,no_parameter_scan=True,classification='fixed best29 checkpoint module ablation; not separately trained or pruned-score-selected',zero_only_new_state_delta=True,non_state_weights_signature_unchanged=True,parent_checkpoint=str(checkpoint),parent_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_checkpoint_SHA256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),eval_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seconds=time.perf_counter()-started,peak_MiB=torch.cuda.max_memory_allocated()/2**20)
    atomic_array(STUDY/'state_pruned_channel_pred.npy',values);atomic_array(STUDY/'state_pruned_eight_stage_pred.npy',pred)
    atomic_json(destination,result);atomic_json(STUDY/'state_prune_runtime_status.json',dict(status='complete',pid=os.getpid(),**result))
    note=f"\n\n## Fixed-checkpoint inference state pruning\nActual PERM{score:.12f}, versusfull300{result['gain_vs_parent_PERM']:+.12f}/retained291{result['gain_vs_retained291']:+.12f}; default{strata['default_PERM']:.12f}/continuous{strata['continuous_PERM']:.12f}. Actualoptional8 TOTAL{scores['TOTAL']:.12f}, strict808={result['strict_808_qualified']}; onlyzero newstate at fixedjointbest29, allotherweights unchanged, source/diskcheckpoint hashes preserved. ONE predefined ablation, no stateweight sweep/roworacle/newphase, not optimized for prunedscore. Do not equatepruninganalysis with independent remainingfold verification. GPUeval{result['seconds']:.3f}s incldata/init/diskload/eval/CPUcomposition/record, peak{result['peak_MiB']:.3f}MiB; not trainingtiming.\n"
    for path in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):
        with path.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
