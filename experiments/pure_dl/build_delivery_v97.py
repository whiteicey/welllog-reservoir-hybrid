"""Export the verified fixed v97 recipe, without training or old delivery edits."""
from pathlib import Path
import ast, hashlib, importlib, json, shutil, sys
import cloudpickle
import numpy as np
import torch
from fullcv_models_v97 import make_model, best_path, root_path, fold_dir, Chosen
from train_dual8222_heads80 import SOURCE
from train_invariant import feature_contract

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT.parent/'submission_pure_dl_v97_81_891019_20260930'
STUDY=ROOT/'results/deep_learning/fullcv_v97_20260930'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    torch.set_num_threads(3)
    if DEST.exists():raise RuntimeError('New delivery directory already exists')
    DEST.mkdir();(DEST/'models').mkdir()
    proof=json.loads((STUDY/'OOF_VERIFICATION.json').read_text())
    assert proof['complete_five_folds'] and proof['all_four_disk_replays_passed']
    assert sha(STUDY/'oof_progress.npz')==proof['oof_sha256']
    recipe=json.loads((ROOT/'results/deep_learning/sw_factor_sam_v96_20260930/CURRENT_RECIPE.json').read_text())
    source_files={Path(__file__).name,'train_stage_fullcv_v97.py','fullcv_models_v97.py','prepare_fold_fullcv_v97.py',
                  'train_paired_local_context_router_sequence.py','verify_fullcv_v97.py','assemble_sources_fullcv_v97.py'}
    records=[]
    for f in range(1,6):
        cp_path=root_path(f) if f<5 else SOURCE
        master=torch.load(cp_path,map_location='cpu',weights_only=False)
        if f<5:
            cache=torch.load(fold_dir(f)/'data.pt',map_location='cpu',weights_only=False)
            metadata=cache['metadata'];del cache
            amplitude=json.loads((fold_dir(f)/'data_manifest.json').read_text())
            amplitude=dict(medians=amplitude['amplitude_medians'],center=amplitude['amplitude_center'],scale=amplitude['amplitude_scale'])
        else:
            # This bank is inference-only and unused in well-conditioned mode;
            # retain the actual source-fold input reference for faithful export.
            from champion_scaling_v21 import STUDY as DATA_STUDY
            data=torch.load(DATA_STUDY/'prepared_data.pt',map_location='cpu',weights_only=False)
            starts=data['starts'];sample=np.stack([lo+np.minimum(((np.arange(128)+.5)*(hi-lo)/128).astype(np.int64),hi-lo-1) for lo,hi in zip(starts[:-1],starts[1:])])
            train_wells=np.unique(data['wells'][data['fold']!=4])
            metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')}
            metadata.update(reference_inputs=data['x'][torch.as_tensor(sample[train_wells])].mean(0),
                            sw_target_median=float(master['regression_median'][2]),sw_target_scale=float(master['regression_scale'][2]))
            del data
            am=json.loads((ROOT/'results/deep_learning/feature_structure_v20_20260928/feature_manifest.json').read_text())
            amplitude=dict(medians=am['imputation_medians'],center=am['scalers']['amplitude']['median'],scale=am['scalers']['amplitude']['scale'])
        models=[];dependencies=[]
        for c,stage in enumerate(('por_anchored','perm_split','sw_density')):
            if f<5:
                model,spec,trace=make_model(f,stage,master,metadata)
                selected=best_path(f,stage)
                saved=torch.load(selected,map_location='cpu',weights_only=False)
                if c<2:model.model.heads[spec[2]].load_state_dict(saved['delta_state'],strict=True)
                else:model.model.load_state_dict(saved['state_dict'],strict=True)
            else:
                choice=recipe['choices'][c]
                module=importlib.import_module(choice['model_module'])
                if c<2:
                    raw=module.Network(choice['target'],master,modes=(choice['mode'],))
                    model=Chosen(raw,c,mode=choice['mode'])
                else:model=Chosen(module.Network(choice['mode'],metadata),c,bank=True)
                selected=ROOT/choice['checkpoint'];saved=torch.load(selected,map_location='cpu',weights_only=False)
                if c<2:model.model.heads[choice['mode']].load_state_dict(saved['delta_state'],strict=True)
                else:model.model.load_state_dict(saved['state_dict'],strict=True)
                trace=[]
            model.cpu().eval().requires_grad_(False)
            models.append(model)
            dependencies.append(dict(target=('POR','PERM','SW')[c],checkpoint=str(selected.relative_to(ROOT)),
                                     sha256=sha(selected),best_epoch=saved['best_epoch'],target_score=saved['target_score'],construction_trace=trace))
            del saved
        preprocess={k:master[k] for k in ('medians','selected_feature_indices','selected_feature_names','feature_median','feature_scale',
            'regression_median','regression_scale','absolute_depth_median','absolute_depth_scale','centers','density_edges')}
        path=DEST/'models'/f'fold_{f}.pt'
        # The complete inference graph includes local capture hooks and bound
        # attention methods; cloudpickle preserves them without constructor IO.
        torch.save(dict(format='v97_complete_inference_graph',original_fold=f,models=models,preprocessing=preprocess,
                        amplitude=amplitude,source_models=dependencies),path,pickle_module=cloudpickle)
        reloaded=torch.load(path,map_location='cpu',weights_only=False)
        assert all(set(a.state_dict())==set(b.state_dict()) and all(torch.equal(v,b.state_dict()[n]) for n,v in a.state_dict().items())
                   for a,b in zip(models,reloaded['models']))
        records.append(dict(fold=f,file=str(path.relative_to(DEST)),sha256=sha(path),bytes=path.stat().st_size,
                            root_source=str(cp_path.relative_to(ROOT)),root_sha256=sha(cp_path),targets=dependencies))
        print('EXPORTED',f,path.stat().st_size,flush=True)
        del models,reloaded,master
    # Include immutable local dependencies of imported architecture classes,
    # plus the fixed training/replay entry points as source evidence.
    directory=ROOT/'experiments/pure_dl'
    source_files|={Path(m.__file__).name for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                   and Path(m.__file__).parent==directory and Path(m.__file__).suffix=='.py'}
    pending=list(source_files)
    while pending:
        name=pending.pop();p=directory/name
        if not p.exists():continue
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            imports=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for imp in imports:
                local=imp+'.py'
                if (directory/local).exists() and local not in source_files:source_files.add(local);pending.append(local)
    for name in sorted(source_files):
        p=directory/name
        if p.exists():
            dest=DEST/'experiments/pure_dl'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copytree(ROOT/'package/src',DEST/'package/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('FINAL_REPORT.json','OOF_VERIFICATION.json','protocol.json','training_entry_manifest.json','OVERLAP_TIMING.json'):
        shutil.copy2(STUDY/name,DEST/'models'/name)
    write(DEST/'models/feature_contract.json',feature_contract())
    write(DEST/'models/delivery_summary.json',dict(version='v97',folds=records,official_oof_scores=proof['oof_scores'],
            complete_five_folds=True,training_stages='source pretraining plus 18 downstream stages; no simplification claimed',
            source_oof_sha256=proof['oof_sha256'],aggregation='equal five-fold expected utilities before action selection',test_accuracy='unknown'))
    with np.load(STUDY/'oof_progress.npz') as z:
        np.savez_compressed(DEST/'models/oof_predictions.npz',pred=z['pred'],fold=z['fold'],well=z['well'])
    runtime={}
    import platform
    runtime['python']=platform.python_version()
    for n in ('torch','numpy','pandas','scipy','sklearn','cloudpickle'):
        runtime[n]=importlib.import_module(n).__version__
    write(DEST/'runtime_versions.json',runtime)
    print('EXPORT_COMPLETE',json.dumps(dict(directory=str(DEST),model_bytes=sum(r['bytes'] for r in records),source_files=len(source_files))),flush=True)

if __name__=='__main__':main()
