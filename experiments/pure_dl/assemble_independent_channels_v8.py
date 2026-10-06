"""Choose one completed model per target and replay three disjoint full models.

No learned ensemble, no per-row/per-well choice, no training. All six predefined
candidates must finish before selection. Replays use saved private checkpoints.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import json,sys,time
import numpy as np
import torch
from independent_channels_v8 import Network,TARGETS
from train_independent_channels_v8 import Evaluator,REFERENCE
from train_dual8222_heads80 import prepare_data,SOURCE,atomic_json,atomic_array
from launch_independent_channels_v8 import ROOT,STUDY,RUNS,PREFIX,trainers
from gpu_job_guard_v2 import check_slot
import train_invariant as base

DEST=ROOT/'outputs/pure_dl_independent_v8_selected_fold5'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    assert not trainers(),'Wait for existing experiment processes, no extra job'
    if DEST.exists():raise RuntimeError('Final assembly is immutable')
    reports={}
    for key in RUNS:
        path=ROOT/'outputs'/(PREFIX+key+'80_fold5')/'fold_5_report.json'
        if not path.exists():raise RuntimeError('Incomplete planned candidate: '+key)
        report=json.loads(path.read_text());assert report['all144665target_predictions_restored_exactly']
        reports[key]=report
    choices=[]
    for c,target in enumerate(TARGETS):
        keys=[key for key,(t,_) in RUNS.items() if t==target]
        key=max(keys,key=lambda k:(reports[k]['target_score'],-reports[k]['trainable_parameters']))
        choices.append({'target':target,'key':key,'architecture':reports[key]['architecture'],
             'target_score':reports[key]['target_score'],'best_epoch':reports[key]['best_epoch'],
             'checkpoint':'outputs/'+PREFIX+key+'80_fold5/model_best.pt'})
    print('FIXED_CHOICES',json.dumps(choices),flush=True)
    print('GPU_ADMISSION',check_slot(min_free_gib=3.),flush=True)
    start=time.perf_counter();torch.set_num_threads(3);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    master=torch.load(SOURCE,map_location='cpu',weights_only=False)
    previous=torch.load(REFERENCE/'readout_checkpoint.pt',map_location='cpu',weights_only=False)
    models=[];pointers=[]
    for choice in choices:
        cp=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False)
        assert cp['config']['target']==choice['target'] and cp['config']['architecture']==choice['architecture']
        model=Network(master,previous,choice['target'],choice['architecture'])
        model.load_state_dict(cp['state_dict'],strict=True);model.eval().requires_grad_(False)
        ptr={p.data_ptr() for p in model.parameters()}
        assert all(not(ptr&other) for other in pointers),'Private models unexpectedly share parameter storage'
        pointers.append(ptr);models.append(model);del cp
    data=prepare_data(master);reference=np.load(REFERENCE/'fold_5_pred.npy')
    combined=np.empty_like(reference);timings=[]
    for c,(choice,model) in enumerate(zip(choices,models)):
        model.cuda();ev=Evaluator(data,master,c,reference);torch.cuda.synchronize();begin=time.perf_counter()
        pred,score,_=ev(model);torch.cuda.synchronize();seconds=time.perf_counter()-begin
        expected=np.load(ROOT/'outputs'/(PREFIX+choice['key']+'80_fold5')/'channel_pred.npy')
        assert np.array_equal(pred,expected),'Reloaded target predictions differ'
        assert abs(score-choice['target_score'])<1e-10
        combined[:,c]=pred;timings.append({'target':choice['target'],'seconds_full_validation':seconds})
        model.cpu();del ev
    truth=data['truth'][data['validation']];scores=base.metric(truth,combined);prior=base.metric(truth,reference)
    wells=data['wells'][data['validation']]
    perwell=[]
    for well in np.unique(wells):
        mask=wells==well;a=base.metric(truth[mask],combined[mask]);b=base.metric(truth[mask],reference[mask])
        perwell.append({'well_index':int(well),'n':int(mask.sum()),'TOTAL_delta':a['TOTAL']-b['TOTAL'],
                        'target_deltas':{c:a[c]-b[c] for c in TARGETS}})
    n=np.array([x['n'] for x in perwell]);d=np.array([x['TOTAL_delta'] for x in perwell])
    ix=np.random.default_rng(20260928).integers(0,len(n),(10000,len(n)))
    boot=(n[ix]*d[ix]).sum(1)/n[ix].sum(1)
    report={'fold':5,'screening_fold':5,'scores':scores,'reference_scores':prior,'delta_reference':scores['TOTAL']-prior['TOTAL'],
        'choices':choices,'three_distinct_full_model_checkpoints':True,'no_shared_parameter_storage':True,
        'all144665rows_reloaded_exactly':True,'original_row_order_verified':True,
        'private_backbone_branches':sum(reports[x['key']]['private_backbone_branches'] for x in choices),
        'total_model_parameters':sum(reports[x['key']]['trainable_parameters'] for x in choices),
        'selected_backbone_changes':{x['target']:reports[x['key']]['selected_backbone_parameter_change_L2'] for x in choices},
        'well_wins':sum(x['TOTAL_delta']>1e-12 for x in perwell),'well_losses':sum(x['TOTAL_delta']< -1e-12 for x in perwell),
        'per_well':perwell,'exploratory_selection_unadjusted_95':np.quantile(boot,[.025,.975]).tolist(),
        'timing':timings,'setup_and_three_replays_seconds':time.perf_counter()-start,
        'score_scope':'actual three independent target models on original full fifth fold',
        'complete_five_folds':False,'remaining_folds_started':False,'screening_threshold':80.5,'screening_passed':scores['TOTAL']>80.5,
        'development_selection':True,'new_shared_training_control_run':False,
        'limitations':'Repeated fold5selection and pretrained multitarget initialization; no independent test. Update scope and total compute differ from old frozen model.',
        'finished_at':datetime.now().astimezone().isoformat()}
    DEST.mkdir()
    atomic_array(DEST/'fold_5_pred.npy',combined)
    full=np.full((len(data['truth']),3),np.nan);full[data['validation']]=combined
    np.savez_compressed(DEST/'oof.npz',truth=data['truth'],pred=full,well=data['wells'],fold=data['fold'])
    atomic_json(DEST/'fold_5_report.json',report);atomic_json(STUDY/'assembled_report.json',report)
    manifest={'choices':choices,'scores':scores,'model_module':'independent_channels_v8',
        'preprocessing_source':str(SOURCE.relative_to(ROOT)),'constructor_reference':str((REFERENCE/'readout_checkpoint.pt').relative_to(ROOT)),
        'training_shares_backbone':False,'inference_shares_backbone':False,'old_delivery_unchanged':True,'extra_folds':False}
    atomic_json(DEST/'model_manifest.json',manifest);atomic_json(STUDY/'CURRENT_RECIPE.json',manifest)
    atomic_json(DEST/'runtime_status.json',{'status':'complete','atlas_current_trial':False})
    (DEST/'AGENTS.md').write_text('# 约束\n固定三模型预测组合，不得逐点/逐井切换模型。维护PROJECTS.md记录已做、待做、计划、目标和完成后事项；旧交付和原数据不变，无自动补折。\n',encoding='utf-8')
    (DEST/'PROJECTS.md').write_text('# 进度\n已完成六项独立训练后的按目标选型、三份私有模型重载与原第五折共同评价。\n未做：原其余四折、独立测试和训练逐位可复现验证。\n目标：完整OOF>82，尚未完成。\n达成后：固定模型和预处理生成新交付；当前不覆盖旧版本。\n',encoding='utf-8')
    (DEST/'README.md').write_text('# 三通道独立模型的实际组合\n\n'+json.dumps(manifest,ensure_ascii=False,indent=2)+'\n\n各目标读取自己的完整检查点，不复用运行时骨干；预处理和模型构造仍依赖记录的原来源。当前为开发集选型，未补折。\n',encoding='utf-8')
    cur=json.loads((ROOT/'docs/deep_learning/atlas/curation.json').read_text(encoding='utf-8'))
    cur['nodes'][DEST.name]={'title':'三通道独立模型·固定选型','summary':'每个目标独立完整骨干与检查点，按目标固定选择单/双视图模型。',
        'conclusion':f"真实三模型共同恢复TOTAL={scores['TOTAL']:.9f}，相对旧参考{scores['TOTAL']-prior['TOTAL']:+.9f}；原第五折开发选择，未补折。",'stage':'loss','highlight':True}
    for x in choices:
        src=PREFIX+x['key']+'80_fold5'
        cur['edges'].append({'source':src,'target':DEST.name,'type':'evolution','note':'固定整条'+x['target']+'由该私有模型预测，非按行切换。','evidence':'outputs/'+DEST.name+'/fold_5_report.json'})
    atomic_json(ROOT/'docs/deep_learning/atlas/curation.json',cur)
    note='\n\n## 独立模型共同恢复完成\n'+json.dumps({'scores':scores,'choices':choices,'no_shared_backbone':True,'well_wins':report['well_wins'],'well_losses':report['well_losses']},ensure_ascii=False)+'\n独立三模型实际预测已验证，补折/交付保持暂停；目标完整OOF>82未完成。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    print('ASSEMBLED',json.dumps({k:v for k,v in report.items() if k!='per_well'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
