"""Retain demonstrated local gains, remove negative weights, and record closure."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from sw_interaction_v56 import ROOT,STUDY,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from train_dual8222_heads80 import atomic_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    assert not jobs()
    result=read(STUDY/'assembled_report.json');timing=read(STUDY/'timing.json')
    delta=result['multiplicative_minus_additive']['channel_delta']
    retained=set()
    if result['adopted']:retained.add(result['selected_mode'])
    if delta>0:retained.add('multiplicative')
    verification=[];pending=set(retained);tick=time.perf_counter()
    if result['new_target_full_disk_replay'] and result['selected_mode'] in pending:
        mode=result['selected_mode'];out=ROOT/f'outputs/pure_dl_sw_interaction_v56_SW_{mode}_fold5'
        report=read(out/'fold_5_report.json')
        verification.append(dict(mode=mode,rows=result['rows'],exact_disk_replay=True,score=report['target_score'],
            replacement_scores=result['selected_scores'],proof_source='assembled_report.json: new_target_full_disk_replay',
            checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
        pending.remove(mode)
    if pending:
        torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
        data,meta=load_data();reference=reference_prediction()
        for mode in sorted(pending):
            out=ROOT/f'outputs/pure_dl_sw_interaction_v56_SW_{mode}_fold5'
            saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
            model=Network(mode,meta).cuda();model.load_state_dict(saved['state_dict'],strict=True)
            p,score,scores=Evaluator(data,meta,2,reference)(model)
            assert np.array_equal(p,np.load(out/'channel_pred.npy'))
            assert abs(score-read(out/'fold_5_report.json')['target_score'])<1e-9
            verification.append(dict(mode=mode,rows=len(p),exact_disk_replay=True,score=score,
                replacement_scores=scores,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            del model,saved
    if retained:
        atomic_json(STUDY/'retained_verification.json',dict(candidates=verification,seconds=time.perf_counter()-tick,
            unchanged_finalizer_replay_reused=bool(retained-pending)))
    deleted=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_sw_interaction_v56_SW_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve()
        assert read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                p=(out/name).resolve();assert p.parent==out and p.name==name
                if p.exists():
                    item=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    p.unlink();deleted.append(item)
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,
            reason='positive local matched gain or adopted' if mode in retained else 'below current SW and no positive tested interaction mechanism',
            no_automatic_resume=True))
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained),training_active=False))
    rows=[]
    for item in result['candidates']:
        r=item['report'];rows.append(f"{item['mode']}：SW {r['target_score']:.12f}，最佳第{r['best_epoch']}轮，停止第{r['stopped_epoch']}轮（{r['stop_reason']}）。")
    note='## v56最终结果与保留边界\n'+'\n'.join(rows)
    note+=f"\n乘法-加法SW={delta:+.12f}；当前SW={result['reference_scores']['SW']:.12f}，实际组合TOTAL={result['selected_scores']['TOTAL']:.12f}。新采用={result['adopted']}。"
    if delta>0:
        note+='\n乘性交互有匹配局部收益，保留其最佳及完整状态；不将其低于现有SW的绝对成绩写成新冠军。该收益来自反复使用的开发折，未完成独立井确认。'
    else:
        note+='\n乘性交互无匹配收益，停止当前分支宽度/层数扫描；只限当前模块和训练配置，不泛化所有门控网络无效。'
    note+=f"\n两组同839481参数、相同完整初始参数与144665点先验；四个内部FF只差加法或乘法，同公共LayerNorm，SW输出全重置；全网络只用SW标签与Gaussian损失。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建={bool(verification)}。不重复未改密度公式检查。"
    note+=f"\n单项耗时分别{[round(t['seconds'],3) for t in timing['candidates']]}秒；双并发总墙钟{timing['first_launch_to_last_completion_seconds']:.3f}秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB={[round(t['peak_allocated_MiB'],2) for t in timing['candidates']]}，代码与最终重建另计。"
    note+=f"\n已删除{len(deleted)}个本轮负检查点，共{sum(x['bytes'] for x in deleted)}字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。"
    note+='\n第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。'
    for p in (STUDY/'FINAL.md',STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append(p,'## v56最终结果与保留边界',note)
    append(ROOT.parent/'AGENTS.md','## v56完成状态',f"## v56完成状态\n当前TOTAL={result['selected_scores']['TOTAL']:.12f}；乘法-加法SW={delta:+.12f}，保留{sorted(retained)}，加法或乘法配置是否采用以assembled_report为准；仅开发第五折结果，非完整OOF收益。配方sw_interaction_v56_20260929/CURRENT_RECIPE.json，无其他折/队列/交付修改。")
    append(ROOT/'EXPLORATIONS.md','v56完成：',f"- v56完成：乘法-加法SW={delta:+.12f}；TOTAL={result['selected_scores']['TOTAL']:.12f}，保留{sorted(retained)}。见[结果](results/deep_learning/sw_interaction_v56_20260929/FINAL.md)。")
    print(json.dumps(dict(retained=sorted(retained),verification=verification,deleted_count=len(deleted),deleted_bytes=sum(x['bytes'] for x in deleted)),ensure_ascii=False))

if __name__=='__main__':main()
