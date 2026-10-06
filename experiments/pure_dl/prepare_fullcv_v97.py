"""Freeze original splits and verified fifth predictions; trace actual builders."""
import argparse,gc,hashlib,importlib,json,time
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import numpy as np
import torch
from fullcv_context_v97 import ROOT,STUDY,RECIPE,original_population,fold_rows,array_sha,sha
from train_dual8222_heads80 import SOURCE,atomic_json
from train_invariant import metric
from run_retrieval_smooth_v36 import append,jobs

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def trace_builders(recipe):
    master=torch.load(SOURCE,map_location='cpu',weights_only=False);events=[];json_events=[];original_load=torch.load;original_text=Path.read_text
    metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')};metadata['reference_inputs']=torch.zeros(128,206)
    # This is constructor dependency tracing, not inference: no bank/labels used.
    for choice in recipe['choices']:
        module=importlib.import_module(choice['model_module']);target=choice['target'];mode=choice['mode']
        def tracked_load(path,*args,**kwargs):
            result=original_load(path,*args,**kwargs)
            if isinstance(path,(str,Path)):
                file=Path(path).resolve();cfg=result.get('config',{}) if isinstance(result,dict) else {}
                events.append(dict(target=target,path=str(file.relative_to(ROOT)),sha256=sha(file),bytes=file.stat().st_size,
                    checkpoint_keys=list(result) if isinstance(result,dict) else [],best_epoch=result.get('best_epoch') if isinstance(result,dict) else None,
                    config=cfg,contains_training_raw_rows=isinstance(result,dict) and 'training_raw_rows' in result,
                    training_raw_sha256=array_sha(result['training_raw_rows']) if isinstance(result,dict) and 'training_raw_rows' in result else None))
            return result
        def tracked_text(path,*args,**kwargs):
            result=original_text(path,*args,**kwargs)
            if path.suffix.lower()=='.json':json_events.append(dict(target=target,path=str(path.resolve().relative_to(ROOT)),sha256=sha(path)))
            return result
        with patch('torch.load',tracked_load),patch.object(Path,'read_text',tracked_text):
            if target=='SW':model=module.Network(mode,metadata);cp=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False);model.load_state_dict(cp['state_dict'],strict=True)
            else:model=module.Network(target,master,modes=(mode,));cp=torch.load(ROOT/choice['checkpoint'],map_location='cpu',weights_only=False);model.heads[mode].load_state_dict(cp['delta_state'],strict=True)
        if hasattr(model,'close'):model.close()
        del model,cp;gc.collect()
    unique={}
    for event in events:
        key=event['path'];item=unique.setdefault(key,event|dict(targets=[],loads=0));item['loads']+=1
        if event['target'] not in item['targets']:item['targets'].append(event['target'])
    return dict(checkpoints=list(unique.values()),json_dependencies=list({(r['target'],r['path']):r for r in json_events}.values()),
        master=dict(path=str(SOURCE.relative_to(ROOT)),sha256=sha(SOURCE)),constructor_only=True,training_performed=False,forward_or_scoring_performed=False,
        placeholder_bank='zeros used only to allocate a nonpersistent buffer during constructor tracing; never forwarded, trained, saved as model or scored')

def main():
    if jobs():raise RuntimeError('Active training; preparation is a separate explicit batch')
    if (STUDY/'promotion.json').exists():raise RuntimeError('Prepared promotion immutable')
    tick=time.perf_counter();recipe=read(RECIPE);study=RECIPE.parent;artifact=read(study/'artifact_finalize.json')
    assert artifact['strict805_passed'] and artifact['raw_disk_replay_exact'] and not artifact['complete_five_folds']
    pop=original_population();pred=np.load(study/'assembled_pred.npy');fifth=fold_rows(pop,4)
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        assert np.array_equal(pop['truth'],z['truth']) and np.array_equal(pop['wells'],z['well']) and np.array_equal(pop['fold'],z['fold'])
    scores=metric(pop['truth'][fifth['validation']],pred);assert scores['TOTAL']>80.5
    assert all(abs(scores[k]-v)<1e-9 for k,v in recipe['scores'].items())
    assert sha(ROOT/recipe['choices'][2]['checkpoint'])==artifact['checkpoint_sha256']
    STUDY.mkdir(exist_ok=True);trace=trace_builders(recipe);atomic_json(STUDY/'dependency_trace.json',trace)
    folds=[]
    for f in range(5):
        rows=fold_rows(pop,f);folds.append(dict(fold=f+1,train_wells=rows['train_wells'],heldout_wells=rows['heldout_wells'],training_rows=len(rows['training']),validation_rows=len(rows['validation']),
            training_raw_sha256=array_sha(rows['training_raw']),validation_raw_sha256=array_sha(rows['validation_raw']),
            source_status='verified_fifth_only' if f==4 else 'requires_fold_local_sources_and_training',predictions_available=f==4))
    partial=np.full_like(pop['truth'],np.nan);partial[fifth['validation']]=pred
    np.savez_compressed(STUDY/'oof_pending.npz',truth=pop['truth'],pred=partial,fold=pop['fold'],well=pop['wells'],valid_to_raw=pop['valid_to_raw'])
    protocol=dict(selected_recipe=str(RECIPE.relative_to(ROOT)),selected_recipe_sha256=sha(RECIPE),data_population_sha256=dict(truth=array_sha(pop['truth']),wells=array_sha(pop['wells']),fold=array_sha(pop['fold']),raw_rows=array_sha(pop['valid_to_raw'])),
        selection='fixed recipe chosen on original fifth; no per-row/per-well choice or OOF-fit model selection',
        root_seed=20261006,per_fold_seed='root_seed + original zero-based fold index',budget=dict(maximum_epochs=80,patience=18,minimum_epochs=10,batch=64,steps=64),
        source_rule='all learned sources/preprocessing/Q must use own fold training wells; fifth weights forbidden on folds1..4',
        quality_rule='last train-only root conflict observation strictly before each source-selected best epoch, frozen for downstream; fifth observation50 preserved',
        checkpoints_reusable_across_folds=False,completed_folds=[5],pending_folds=[1,2,3,4],folds=folds)
    atomic_json(STUDY/'protocol.json',protocol)
    promotion=dict(passed=True,screening_fold=5,strict_threshold=80.5,scores=scores,rows=len(pred),recipe_sha256=sha(RECIPE),fifth_prediction_sha256=sha(study/'assembled_pred.npy'),
        partial_oof_sha256=sha(STUDY/'oof_pending.npz'),source_trace_sha256=sha(STUDY/'dependency_trace.json'),protocol_sha256=sha(STUDY/'protocol.json'),
        complete_five_folds=False,remaining_folds_started=False,training_launch_authorized=False,remaining_training_implementation_complete=False,
        seconds=time.perf_counter()-tick,finished_at=datetime.now().astimezone().isoformat())
    atomic_json(STUDY/'promotion.json',promotion)
    lines=['# v97 补折准备实际记录','',f"第五资格TOTAL={scores['TOTAL']:.12f}，原144665点；原总体723556点/80井/五折标签与既有正式人口逐项一致。",'',
        '探索重点：将已筛选通过的完整多阶段独立三通道配方移到原其他折；只做来源和数据接口准备，未训练/未补折。',
        '实际内容与验证：三个当前builder CPU构造并严格加载选定state，捕获实际checkpoint/JSON路径与SHA、配置和共享关系；SW零bank仅用于分配非持久buffer，无前向/保存/评分。原始train数据重建五折，实际第五缓存进入完整OOF占位，其余四折保持NaN；不计算部分OOF TOTAL。',
        f"捕获{len(trace['checkpoints'])}个唯一构造权重依赖；配置见dependency_trace.json。所有是第五来源，只可用于第五；缺少的1~4需各折重新训练，不拿已有不同配方的完整五折权重替代。",'',
        '|折|训练点|验证点|实际状态|','|---|---:|---:|---|']
    for f in folds:lines.append(f"|{f['fold']}|{f['training_rows']}|{f['validation_rows']}|{f['source_status']}|")
    lines+=['','收益：尚无新模型分数或完整OOF收益；只是复用已核验第五80.504197。',
        '不继续：直接改fold数字但加载第五预处理/权重/Q会造成跨折泄漏，禁止。原calibrated75/global75完整五折源不等于本配方local-context80，不能默默替代。',
        '已做：冻结第五资格/配方哈希，原五折人口清单，实际构造依赖，OOF占位；计划：显式折号/路径的数据与Q绑定，按依赖阶段实现训练入口；未做：完整剩余阶段入口、其他四折训练、完整OOF>82。',
        '此前AGENTS明确补折暂缓：不得在persistent goal下自动启动；保持该边界，准备不启动。目标后完整五折官方评价/独立确认/新目录交付，旧交付不改。',
        '耗时：'+str(promotion['seconds'])+'秒，CPU原数据/构造追踪/哈希/保存计入；无GPU训练，无并发速度宣称。']
    text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(text,encoding='utf-8');append(STUDY/'README.md','## 实际准备结果','## 实际准备结果\n'+text);(STUDY/'PROJECTS.md').write_text(text,encoding='utf-8')
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md'):append(p,'## v97补折来源准备','## v97补折来源准备\n'+text)
    append(ROOT/'EXPLORATIONS.md','fullcv_v97_20260930','- [v97 当前配方补折准备](results/deep_learning/fullcv_v97_20260930/README.md)：第五通过资格，原完整五折人口/实际构造依赖已记录；其余四折未训练，完整OOF仍未验证。')
    print(json.dumps(dict(scores=scores,dependencies=len(trace['checkpoints']),folds=folds,seconds=promotion['seconds']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
