"""CPU export of the explicitly requested foldwise-tuned two-seed delivery."""
from pathlib import Path
import argparse,ast,hashlib,importlib,json,os,shutil,sys
import cloudpickle
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/two_seed_fullcv_v370_20261002'
DEST=ROOT.parent/'submission_pure_dl_v370_82_194651_20261002'
TARGETS=('POR','PERM','SW')
SPECS=(('por_coupling_conditional_rps_v341','conditional'),('perm_low_rank_cross_v336','cross'),('sw_dynamic_cross_spectrum_v351','aligned'))

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    torch.set_num_threads(3)
    ap=argparse.ArgumentParser();ap.add_argument('--resume-build',action='store_true');args=ap.parse_args()
    if DEST.exists():
        assert args.resume_build and (DEST/'AGENTS.md').is_file()
        assert not (DEST/'submission_code.zip').exists() and not (DEST/'models/delivery_summary.json').exists(),'Finalized delivery immutable'
    else:assert not args.resume_build
    final=read(STUDY/'FINAL.json')
    assert read(STUDY/'FINAL_CANONICAL_OOF_AUDIT.json')['coverage_once']
    assert read(STUDY/'FINAL_REMAINING_TRAINING_AUDIT.json')['actual_checkpoint_loads_scoped_and_exact']
    DEST.mkdir(exist_ok=True);(DEST/'models').mkdir(exist_ok=True)
    (DEST/'AGENTS.md').write_text('# 工作目标与约束\n\n交付 V370 两种子五折 submission。仅推理，不训练；不读取测试标签；不按行或真值类别选模型。82.194651是逐折逐头调参诊断，固定比例OOF82.167418。每模型8阶段、每折配对16历史阶段。旧交付不覆盖。维护PROJECTS.md，记录完成、计划、未完成、目标和后续可选工作，每批更新。\n',encoding='utf-8')
    (DEST/'PROJECTS.md').write_text('# 目标与进度\n\n目标：打包用户指定82.194651版本及完整测试集结果。\n\n已完成：创建独立新目录。\n计划：导出10套推理图、原配比、可移植CLI；核验完整OOF和独立解压测试推理。\n未完成：模型导出、预测与ZIP验证。\n后续可选：固定比例版本对照；本交付不继续训练。\n',encoding='utf-8')
    modules=[importlib.import_module(n) for n,_ in SPECS]
    snapshots=[]
    directory=ROOT/'experiments/pure_dl'
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and Path(f).resolve().parent==directory:
            for name in ('SOURCE','PARENT','PARENT_SHA'):
                if hasattr(m,name):snapshots.append((m,name,getattr(m,name)))
    oldsource=ROOT/'outputs/pure_dl_paired_local_context_router80_sequence/fold_5.pt'
    legacy={
      oldsource:'SOURCE',
      ROOT/'outputs/pure_dl_compact_por_mixed_family_v113_fold5/t3_cauchy/model_best.pt':'POR113',
      ROOT/'outputs/pure_dl_por_ranked_probability_v151_fold5/joint_rps/model_best.pt':'POR151',
      ROOT/'outputs/pure_dl_perm_ranked_probability_v154_fold5/perm_conditional/model_best.pt':'PERM154',
      ROOT/'outputs/pure_dl_compact_frequency_v104_fold5/local/model_best.pt':'SW104'}
    fifth=read(ROOT/'results/deep_learning/pairwise_complete_models_v368_20261002/AUGMENT_PLAN.json')
    fifthmodels={m['id']:m for m in fifth['models']}
    records=[]
    for fold in range(1,6):
        pairs=[];preprocessing=None;amplitude=None;dependencies=[]
        for identity in ('original','seed_20261013'):
            for m,n,v in snapshots:setattr(m,n,v)
            if fold<5:
                own=ROOT/f'outputs/pure_dl_two_seed_fullcv_v370/fold_{fold}'/identity
                source=own/f'SOURCE/fold_{fold}.pt'
                heads={t:own/s/'model_best.pt' for t,s in zip(TARGETS,('POR341','PERM336','SW351'))}
            elif identity=='seed_20261013':
                own=ROOT/'outputs/pure_dl_complete_seeds_v362_fold5/seed_20261013'
                source=own/'SOURCE/fold_5.pt'
                heads={t:Path(fifthmodels[identity]['heads'][t]['checkpoint']) for t in TARGETS}
            else:
                own=None;source=oldsource
                heads={t:Path(fifthmodels[identity]['heads'][t]['checkpoint']) for t in TARGETS}
            replacements={p.resolve():(source if s=='SOURCE' else own/s/'model_best.pt').resolve()
                          for p,s in legacy.items()} if own else {}
            for m,n,v in snapshots:
                if isinstance(v,Path) and v.resolve() in replacements:setattr(m,n,replacements[v.resolve()])
            for m,_,_ in snapshots:
                parent=getattr(m,'PARENT',None)
                if hasattr(m,'PARENT_SHA') and isinstance(parent,Path) and parent.exists():m.PARENT_SHA=sha(parent)
            master=torch.load(source,map_location='cpu',weights_only=False)
            savedSW=torch.load(heads['SW'],map_location='cpu',weights_only=False)
            metadata={k:master[k] for k in ('centers','density_edges','selected_feature_names')}
            metadata.update(reference_inputs=savedSW['state_dict']['raw_reference_inputs'].detach().clone(),
                            sw_target_median=float(master['regression_median'][2]),sw_target_scale=float(master['regression_scale'][2]))
            models=[];traces=[]
            original_load=torch.load
            def trace_load(p,*a,**kw):
                if isinstance(p,(str,Path)):
                    path=Path(p).resolve();path=replacements.get(path,path)
                    if path.suffix=='.pt' and path.is_relative_to(ROOT/'outputs'):
                        allowed=set(replacements.values())|set(p.resolve() for p in heads.values())|{source.resolve()}
                        if own is None:allowed|={p.resolve() for p in legacy}
                        assert path in allowed,('unexpected constructor checkpoint',path)
                        traces.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
                    return original_load(path,*a,**kw)
                return original_load(p,*a,**kw)
            torch.load=trace_load
            try:
                for c,(module,(_,mode)) in enumerate(zip(modules,SPECS)):
                    model=module.make_model(mode,master,metadata)
                    selected=heads[TARGETS[c]]
                    saved=original_load(selected,map_location='cpu',weights_only=False)
                    model.load_state_dict(saved['state_dict'],strict=True)
                    model.cpu().eval().requires_grad_(False);models.append(model)
                    dependencies.append(dict(identity=identity,target=TARGETS[c],checkpoint=str(selected.relative_to(ROOT)),
                                             sha256=sha(selected),best_epoch=saved['best_epoch'],target_score=saved['target_score']))
            finally:torch.load=original_load
            pp={k:master[k] for k in ('medians','selected_feature_indices','selected_feature_names','feature_median','feature_scale',
                  'regression_median','regression_scale','absolute_depth_median','absolute_depth_scale','centers','density_edges')}
            if preprocessing is None:preprocessing=pp
            else:assert all(preprocessing[k]==pp[k] if k=='medians' else np.array_equal(preprocessing[k],pp[k]) for k in pp),'The selected paired predictors must share fold preprocessing'
            if fold<5:
                am=read(own.parent/'amplitude_manifest.json')
                amp=dict(medians=am['imputation_medians'],center=am['center'],scale=am['scale'])
            else:
                am=read(ROOT/'results/deep_learning/feature_structure_v20_20260928/feature_manifest.json')
                amp=dict(medians=am['imputation_medians'],center=am['scalers']['amplitude']['median'],scale=am['scalers']['amplitude']['scale'])
            if amplitude is None:amplitude=amp
            else:assert amplitude==amp
            pairs.append(dict(identity=identity,models=models,source_sha256=sha(source),construction_trace=traces))
            del master,savedSW,saved
        weights={t:final['folds'][fold-1]['per_head_best_weights'][t]['weight_original'] for t in TARGETS}
        path=DEST/'models'/f'fold_{fold}.pt'
        torch.save(dict(format='v370_two_seed_complete_inference_graph',original_fold=fold,pairs=pairs,
                        preprocessing=preprocessing,amplitude=amplitude,weights=weights),path,pickle_module=cloudpickle)
        reloaded=torch.load(path,map_location='cpu',weights_only=False)
        for a,b in zip(pairs,reloaded['pairs']):
            for x,y in zip(a['models'],b['models']):
                assert set(x.state_dict())==set(y.state_dict())
                assert all(torch.equal(v,y.state_dict()[n]) for n,v in x.state_dict().items())
        records.append(dict(fold=fold,file=path.relative_to(DEST).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
                            weights=weights,source_models=dependencies))
        print('EXPORTED',fold,path.stat().st_size,flush=True)
        del pairs,reloaded,models
    for m,n,v in snapshots:setattr(m,n,v)
    # Archive transitive local imports of architecture classes and frozen trainers.
    names={Path(m.__file__).name for m in list(sys.modules.values()) if getattr(m,'__file__',None)
           and Path(m.__file__).resolve().parent==directory and Path(m.__file__).suffix=='.py'}
    names|={'train_pair_fullcv_source_v370.py','train_pair_fullcv_stage_v370.py','fullcv_pair_context_v370.py',
            'pair_fullcv_models_v370.py','evaluate_pair_fullcv_v370.py','record_pair_fullcv_v370.py',Path(__file__).name}
    pending=list(names)
    while pending:
        p=directory/pending.pop()
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            imports=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in imports:
                filename=name+'.py'
                if (directory/filename).exists() and filename not in names:names.add(filename);pending.append(filename)
    for name in names:
        dest=DEST/'experiments/pure_dl'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/name,dest)
    shutil.copytree(ROOT/'package/src',DEST/'package/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    from train_invariant import feature_contract
    write(DEST/'models/feature_contract.json',feature_contract())
    for name in ('FINAL.json','FINAL_SUMMARY.md','FINAL_CANONICAL_OOF_AUDIT.json','FINAL_REMAINING_TRAINING_AUDIT.json','FINAL_SOURCE_LINEAGE_REVIEW.json','PLAN.json'):
        shutil.copy2(STUDY/name,DEST/'models'/name)
    write(DEST/'models/delivery_summary.json',dict(version='v370_foldwise_tuned',folds=records,
        official_oof_scores=final['scores']['diagnostic'],fixed_oof_scores=final['scores']['fixed'],complete_five_folds=True,
        score_scope='Each fold labels selected head ratios; tuning diagnostic, not unbiased validation',
        stages_per_model=8,historical_stages_per_pair=16,test_accuracy='unknown',
        aggregation='within each fold pair mix joint predictive distributions by per-head ratio, then equal five-fold expected utilities before argmax'))
    import platform
    runtime=dict(python=platform.python_version())
    for name in ('torch','numpy','pandas','scipy','sklearn','cloudpickle','psutil'):runtime[name]=importlib.import_module(name).__version__
    write(DEST/'runtime_versions.json',runtime)
    versions={'torch':'torch','numpy':'numpy','pandas':'pandas','scipy':'scipy','sklearn':'scikit-learn','cloudpickle':'cloudpickle','psutil':'psutil'}
    (DEST/'requirements.txt').write_text('\n'.join(f"{v}=={runtime[k]}" for k,v in versions.items())+'\n',encoding='utf-8')
    shutil.copy2(ROOT/'experiments/pure_dl/delivery_predict_v370.py',DEST/'predict.py')
    print('EXPORT_COMPLETE',str(DEST),len(names),flush=True)

if __name__=='__main__':main()
