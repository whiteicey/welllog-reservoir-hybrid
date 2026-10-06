"""One actual fold1 input check plus wrong-fold/Q rejection; no training."""
import json,gc,time
from datetime import datetime
import numpy as np
import torch
from fullcv_context_v97 import ROOT,STUDY,original_population,validate_metadata_source,prepare_fold_data,bind_fold_quality,require_training_context,add_well_bank,sha,array_sha
from train_dual8222_heads80 import SOURCE,atomic_json
from run_retrieval_smooth_v36 import jobs,append

def main():
    if jobs():raise RuntimeError('Training is active')
    if (STUDY/'context_diagnostic.json').exists():raise RuntimeError('Completed diagnostic is immutable')
    tick=time.perf_counter();torch.set_num_threads(3);pop=original_population()
    fifth=torch.load(SOURCE,map_location='cpu',weights_only=False)
    try:validate_metadata_source(fifth,pop,0)
    except ValueError as e:wrong_fold=str(e)
    else:raise AssertionError('Fifth source accepted for original fold1')
    file=ROOT/'outputs/pure_dl_paired_calibrated_view_sequence/fold_1.pt'
    master=torch.load(file,map_location='cpu',weights_only=False)
    # Only preprocessing metadata is consumed. This different architecture's
    # state_dict is never loaded into a predictor or used for candidate scores.
    torch.cuda.reset_peak_memory_stats();data=prepare_fold_data(master,0,pop)
    try:require_training_context(data)
    except ValueError as e:pending_q=str(e)
    else:raise AssertionError('Pending Q allowed training')
    try:bind_fold_quality(data,master,SOURCE.parent/'fold_5_conflict_weights.npz')
    except ValueError as e:wrong_q=str(e)
    else:raise AssertionError('Fifth Q accepted for original fold1')
    meta=add_well_bank(data);rows=data['population_rows'];assert meta['bank_manifest']['train_wells']==rows['train_wells']
    val_raw=torch.as_tensor(rows['validation_raw'],device='cuda');assert not bool(data['labeled'][val_raw].any()) and bool((data['labels'][val_raw]==0).all())
    result=dict(original_fold=1,training_performed=False,scoring_performed=False,metadata_only_source=str(file.relative_to(ROOT)),metadata_source_sha256=sha(file),
        architecture_weights_used=False,preprocessing_refit_matches_fold1_metadata=True,train_wells=len(rows['train_wells']),heldout_wells=len(rows['heldout_wells']),
        training_rows=len(rows['training']),validation_rows=len(rows['validation']),feature_width=data['x'].shape[-1],validation_labels_not_allocated_for_loss=True,
        quality_ready=False,pending_Q_blocks_training=True,fifth_weights_metadata_rejected=wrong_fold,fifth_Q_rejected=wrong_q,pending_Q_error=pending_q,
        training_raw_sha256=array_sha(rows['training_raw']),validation_raw_sha256=array_sha(rows['validation_raw']),bank_manifest=meta['bank_manifest'],
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        seconds=time.perf_counter()-tick,scope='raw input loading/train-only feature refit/GPU data and bank allocation/rejection checks included; no training, inference or score',finished_at=datetime.now().astimezone().isoformat())
    atomic_json(STUDY/'context_diagnostic.json',result)
    note='## 按折数据接口实际诊断\n原第一折578935训练/144621验证、64/16井，原206特征重新拟合与既有该折预处理元数据一致；仅消费其元数据，未把旧calibrated模型作为当前候选来源。验证标签未分配到损失，wholewell reference只读64训练输入。第五预处理和第五Q均被拒绝，missing Q禁止训练；本批无训练/推理/新分数。耗时'+str(result['seconds'])+'秒，峰值allocated '+str(result['peak_allocated_MiB'])+'MiB。\n'
    for p in (STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(p,'## 按折数据接口实际诊断',note)
    print(json.dumps({k:v for k,v in result.items() if k!='bank_manifest'},ensure_ascii=False),flush=True)
    del data,meta,pop,master,fifth;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
