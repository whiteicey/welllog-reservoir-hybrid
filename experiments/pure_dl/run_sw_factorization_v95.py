"""Actual original-fold5 evaluation of a fixed two-source SW probability law."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from datetime import datetime
import gc,json,time
import numpy as np
import torch
from sw_factorization_v95 import ROOT,STUDY,PREVIOUS,GATE,DENSITY,MODES,Network,Evaluator,load_data,reference_prediction,digest
from finish_sw_sam_v93 import source_proof
from finish_all_heads_v26 import diagnostics,native
from train_dual8222_heads80 import atomic_json,atomic_array
from run_retrieval_smooth_v36 import jobs,append
import train_invariant as base

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    if jobs():raise RuntimeError('A GPU training process is active')
    if (STUDY/'assembled_report.json').exists():raise RuntimeError('Completed study is immutable')
    started=datetime.now().astimezone();tick=time.perf_counter()
    old=read(PREVIOUS/'CURRENT_RECIPE.json');reference=reference_prediction()
    gatecfg=read(GATE.parent/'config.json');densitycfg=read(DENSITY.parent/'config.json')
    for cfg in (gatecfg,densitycfg):source_proof(cfg)
    for k in ('quality','input_features','training_rows','validation_rows','bank_manifest','target_normalization','scoring_kernel_sha256'):
        assert gatecfg[k]==densitycfg[k],k
    assert digest(GATE)==read(PREVIOUS/'assembled_report.json')['verification'][0]['sha256']
    sources={p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name('sw_factorization_v95.py'))}
    torch.set_num_threads(3);torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True,warn_only=True)
    data,metadata=load_data();truth=data['truth'][data['validation']];wells=data['wells'][data['validation']]
    with np.load(ROOT/'outputs/pure_dl_independent_v8_selected_fold5/oof.npz') as z:
        assert np.array_equal(truth,z['truth'][z['fold']==4]) and np.array_equal(wells,z['well'][z['fold']==4])
    refs=base.metric(truth,reference);assert all(abs(refs[k]-v)<1e-9 for k,v in old['scores'].items())
    atomic_json(STUDY/'input_context_manifest.json',metadata['bank_manifest'])
    rows=[];predictions={};timings=[]
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_factorization_v95_SW_{mode}_fold5'
        if out.exists():raise FileExistsError(out)
        out.mkdir()
        for name in ('AGENTS.md','README.md','PROJECTS.md'):(out/name).write_text((STUDY/name).read_text(encoding='utf-8'),encoding='utf-8')
        mtick=time.perf_counter();torch.cuda.reset_peak_memory_stats();model=Network(mode,metadata).cuda()
        cfg=dict(mode=mode,target='SW',screening_fold=5,model_module='sw_factorization_v95',
            total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=0,training_performed=False,
            same_capacity_control=True,source_assignment_predefined=True,
            atom_checkpoint=str(GATE.relative_to(ROOT)),atom_sha256=digest(GATE),
            density_checkpoint=str((GATE if mode=='same' else DENSITY).relative_to(ROOT)),density_sha256=digest(GATE if mode=='same' else DENSITY),
            law='P(default)=sigmoid(g_SAM); P(numeric)=(1-P(default))*Gaussian_density_fixed_source',
            prediction_voting=False,row_wise_model_selection=False,validation_labels_in_forward=False,
            only_same_training_fold_weights=True,source_hashes=sources,
            input_features=206,window_length=1024,quality=data['qinfo'],bank_manifest=metadata['bank_manifest'],
            target_normalization=gatecfg['target_normalization'],scoring_kernel_sha256=gatecfg['scoring_kernel_sha256'],
            decoder='unchanged physical256 support/800 actions+default; exact original probability/utility argmax',
            source_selection='fixed r100 atom in both arms; r100 or v82a100 conditional density, chosen before v95 results',
            complete_five_folds=False,remaining_folds_started=False)
        assert cfg['total_parameters']==1678962 and not any(p.requires_grad for p in model.parameters())
        atomic_json(out/'config.json',cfg)
        pred,score,scores=Evaluator(data,metadata,2,reference)(model)
        if mode=='same':assert np.array_equal(pred,reference[:,2]),'Equal-source control must reproduce adopted r100 exactly'
        diag=diagnostics(truth,reference,pred,wells,'SW');assert abs(score-diag['scores']['SW'])<1e-9
        rep=native(dict(mode=mode,target='SW',screening_fold=5,target_score=score,reference_replacement_scores=scores,
            diagnostics=diag,training_performed=False,best_epoch=None,source_epochs=dict(atom=30,density=30 if mode=='same' else 25),
            all144665_raw_disk_evaluation=True,same_source_reproduction_exact=bool(mode=='same'),
            strict805_qualified=bool(scores['TOTAL']>80.5),seconds=time.perf_counter()-mtick,
            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
            complete_five_folds=False,remaining_folds_started=False,finished_at=datetime.now().astimezone().isoformat()))
        atomic_array(out/'channel_pred.npy',pred);atomic_json(out/'fold_5_report.json',rep)
        atomic_json(out/'runtime_status.json',dict(status='complete',mode=mode,training_performed=False))
        rows.append(rep);predictions[mode]=pred;timings.append(dict(mode=mode,seconds=rep['seconds'],peak_allocated_MiB=rep['peak_allocated_MiB'],peak_reserved_MiB=rep['peak_reserved_MiB']))
        print(json.dumps(dict(mode=mode,SW=score,TOTAL=scores['TOTAL'],delta=diag['channel_delta']),ensure_ascii=False),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    cross=next(r for r in rows if r['mode']=='separate');adopt=bool(cross['target_score']>refs['SW']+1e-9)
    combined=reference.copy();choices=[dict(c) for c in old['choices']]
    if adopt:
        combined[:,2]=predictions['separate']
        choices[2].update(key='v95_SW_separate',mode='separate',model_module='sw_factorization_v95',checkpoint=None,
            source_checkpoints=dict(atom=str(GATE.relative_to(ROOT)),density=str(DENSITY.relative_to(ROOT))),
            target_score=cross['target_score'],best_epoch=None,parent_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)),
            builder='load_data; Network(separate,metadata) loads two frozen fold-local weights; original Evaluator; no label-dependent source choice')
    scores=base.metric(truth,combined);assert np.array_equal(combined[:,:2],reference[:,:2])
    result=native(dict(reference_scores=refs,selected_scores=scores,adopted=adopt,selected_mode='separate' if adopt else None,
        candidates=rows,screening_fold=5,rows=144665,strict805_passed=bool(scores['TOTAL']>80.5),
        complete_five_folds=False,remaining_folds_started=False,same_capacity=True,sources_preserved=True,
        wall_seconds=time.perf_counter()-tick,started_at=started.isoformat(),finished_at=datetime.now().astimezone().isoformat()))
    atomic_json(STUDY/'assembled_report.json',result);atomic_array(STUDY/'assembled_pred.npy',combined)
    recipe=dict(old);recipe.update(scores=native(scores),choices=choices,reference_recipe=str((PREVIOUS/'CURRENT_RECIPE.json').relative_to(ROOT)))
    atomic_json(STUDY/'CURRENT_RECIPE.json',recipe)
    atomic_json(STUDY/'timing.json',native(dict(candidates=timings,total_wall_seconds=result['wall_seconds'],actual_gpu_process_concurrency=1,
        scope='train-only cache/input bank loading, both frozen two-network original raw-input evaluations and persistence included; no training/warmup exclusion; authoring/atlas separate')))
    text='\n'.join(['# v95 固定概率分解结果','',f"原第五TOTAL {refs['TOTAL']:.12f} → {scores['TOTAL']:.12f}；采用={result['selected_mode']}。",
        '', '|固定密度来源|SW|TOTAL|对r100 SW|','|---|---:|---:|---:|']+
        [f"|{r['mode']}|{r['target_score']:.12f}|{r['reference_replacement_scores']['TOTAL']:.12f}|{r['diagnostics']['channel_delta']:+.12f}|" for r in rows]+
        ['', '探索重点：r100改善默认而数值退步，预先固定r100默认atom+自身/旧v82条件密度的两项同1678962容量消融；所有权重冻结，未训练。',
        '实际验证：两个fold-local来源/源码/训练侧预处理/Q/原评分hash，原144665点原始输入评价，同来源重复分支与r100逐点相同。原概率公式和动作argmax，模型前向无验证标签、按点挑模型或预测投票。PERM/POR原两列固定。开发折机制消融不等于独立泛化或完整OOF。',
        '收益：separate相对same SW='+str(cross['diagnostics']['channel_delta'])+'；默认/数值/井诊断='+json.dumps(native(cross['diagnostics']),ensure_ascii=False),
        '不继续：若负，关闭这两个固定来源直接拼接的结构；不扫多个来源或比例。旧v84r1是旧gate/私有密度热启动训练best0负结果，本批新SAM gate+已有固定density零更新对照，结论范围不同。',
        '已做：固定两项结构、原始输入评价、实际三列、来源保留。未做：新训练、其他折、独立井泛化。第五严格>80.5且有收益后才考虑补折，原完整OOF>82未完成，不自动启动。目标后独立确认与新目录交付。',
        '成本：'+json.dumps(read(STUDY/'timing.json'),ensure_ascii=False),
        '未新增/复制checkpoint；两来源仍是当前/正控制依赖，保留。没有负权重大文件需要删除。'])+'\n'
    (STUDY/'FINAL.md').write_text(text,encoding='utf-8');(STUDY/'PROJECTS.md').write_text(text,encoding='utf-8');append(STUDY/'README.md','## 最终结果','## 最终结果\n'+text)
    for mode in MODES:
        out=ROOT/f'outputs/pure_dl_sw_factorization_v95_SW_{mode}_fold5';(out/'PROJECTS.md').write_text(text,encoding='utf-8');append(out/'README.md','## 最终结果','## 最终结果\n'+text)
    note=f"## v95固定概率分解完成\n原第五TOTAL{scores['TOTAL']:.12f}，采用={result['selected_mode']}；两同容量固定来源均已原144665输入评价，同来源control与r100精确一致。未训练/其他折，完整OOF>82未完成。配方results/deep_learning/sw_factorization_v95_20260930/CURRENT_RECIPE.json。\n"
    for p in (ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):append(p,'## v95固定概率分解完成',note)
    for p in (ROOT/'AGENTS.md',ROOT.parent/'AGENTS.md'):append(p,'## v95当前研究配方',note.replace('## v95固定概率分解完成','## v95当前研究配方'))
    append(ROOT/'EXPLORATIONS.md','v95完成：','- v95完成：[结果](results/deep_learning/sw_factorization_v95_20260930/FINAL.md)；'+note)
    print(json.dumps(dict(scores=native(scores),adopted=adopt,wall_seconds=result['wall_seconds']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
