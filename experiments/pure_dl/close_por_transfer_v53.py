"""Retain demonstrated local gains, remove negative weights, and record closure."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from por_transfer_v53 import ROOT,STUDY,MODES,Network,Evaluator,load_data,reference_prediction
from run_retrieval_smooth_v36 import jobs,append
from train_dual8222_heads80 import atomic_json

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    assert not jobs()
    result=read(STUDY/'assembled_report.json');timing=read(STUDY/'timing.json')
    delta=result['transfer_minus_scratch']['channel_delta']
    retained=set()
    if result['adopted']:retained.add(result['selected_mode'])
    if delta>0:retained.add('sw_transfer')
    verification=[]
    if retained:
        tick=time.perf_counter();torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
        data,meta=load_data();reference=reference_prediction()
        for mode in sorted(retained):
            out=ROOT/f'outputs/pure_dl_por_transfer_v53_POR_{mode}_fold5'
            saved=torch.load(out/'model_best.pt',map_location='cpu',weights_only=False)
            model=Network(mode,meta).cuda();model.load_state_dict(saved['state_dict'],strict=True)
            p,score,scores=Evaluator(data,meta,0,reference)(model)
            assert np.array_equal(p,np.load(out/'channel_pred.npy'))
            assert abs(score-read(out/'fold_5_report.json')['target_score'])<1e-9
            verification.append(dict(mode=mode,rows=len(p),exact_disk_replay=True,score=score,
                replacement_scores=scores,checkpoint_sha256=hashlib.sha256((out/'model_best.pt').read_bytes()).hexdigest()))
            del model,saved
        atomic_json(STUDY/'retained_verification.json',dict(candidates=verification,seconds=time.perf_counter()-tick))
    deleted=[]
    for mode in MODES:
        out=(ROOT/f'outputs/pure_dl_por_transfer_v53_POR_{mode}_fold5').resolve()
        assert out.parent==(ROOT/'outputs').resolve()
        assert read(out/'runtime_status.json')['status']=='complete'
        if mode not in retained:
            for name in ('model_best.pt','last_state.pt'):
                p=(out/name).resolve();assert p.parent==out and p.name==name
                if p.exists():
                    item=dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    p.unlink();deleted.append(item)
        atomic_json(out/'candidate_status.json',dict(retained=mode in retained,
            reason='positive local matched gain or adopted' if mode in retained else 'below current POR and no positive tested transfer mechanism',
            no_automatic_resume=True))
    atomic_json(STUDY/'deletion_manifest.json',dict(deleted=deleted,bytes=sum(x['bytes'] for x in deleted),retained=sorted(retained),training_active=False))
    rows=[]
    for item in result['candidates']:
        r=item['report'];rows.append(f"{item['mode']}：POR {r['target_score']:.12f}，最佳第{r['best_epoch']}轮，停止第{r['stopped_epoch']}轮（{r['stop_reason']}）。")
    note='## v53最终结果与保留边界\n'+'\n'.join(rows)
    note+=f"\n迁移-随机POR={delta:+.12f}；当前POR={result['reference_scores']['POR']:.12f}，实际组合TOTAL={result['selected_scores']['TOTAL']:.12f}。新采用={result['adopted']}。"
    if delta>0:
        note+='\n迁移有匹配局部收益，保留迁移最佳及完整状态；不将其低于现有POR的绝对成绩写成新冠军。该收益来自反复使用的开发折，未完成独立井确认。'
    else:
        note+='\n迁移无匹配收益，关闭当前SW到POR来源的冻结层/LR扫描；不否定v50已测得的PERM局部收益。'
    note+=f"\n两组同836921参数、相同初始POR头/144665点先验、不同初始骨干；全骨干只用POR标签与Student-t3损失。训练梯度、来源哈希、最佳恢复和真实三列评分已核验；保留候选磁盘重建={bool(verification)}。不重复未改密度公式检查。"
    note+=f"\n单项耗时分别{[round(t['seconds'],3) for t in timing['candidates']]}秒；双并发总墙钟{timing['first_launch_to_last_completion_seconds']:.3f}秒，包含加载/输入库/预热/启动间隔；每任务峰值MiB={[round(t['peak_allocated_MiB'],2) for t in timing['candidates']]}，代码与最终重建另计。"
    note+=f"\n已删除{len(deleted)}个本轮负检查点，共{sum(x['bytes'] for x in deleted)}字节；保留源码、配置、曲线、预测与哈希清单，当前模型及既有正候选/交付不变。无训练队列，无其他折。"
    note+='\n第五折严格>80.5和完整原五折OOF>82仍未完成。后续只携带已证实局部收益探索独立机制；新实验尚未启动，不计为收益。目标完成后可做独立井确认及新目录交付。'
    for p in (STUDY/'FINAL.md',STUDY/'README.md',STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md'):
        append(p,'## v53最终结果与保留边界',note)
    append(ROOT.parent/'AGENTS.md','## v53完成状态',f"## v53完成状态\n当前TOTAL={result['selected_scores']['TOTAL']:.12f}；迁移-随机POR={delta:+.12f}，保留{sorted(retained)}，仅局部匹配机制收益不等于新冠军。配方por_transfer_v53_20260929/CURRENT_RECIPE.json，无其他折/队列/交付修改。")
    append(ROOT/'EXPLORATIONS.md','v53完成：',f"- v53完成：迁移-随机POR={delta:+.12f}；TOTAL={result['selected_scores']['TOTAL']:.12f}，保留{sorted(retained)}。见[结果](results/deep_learning/por_transfer_v53_20260929/FINAL.md)。")
    print(json.dumps(dict(retained=sorted(retained),verification=verification,deleted_count=len(deleted),deleted_bytes=sum(x['bytes'] for x in deleted)),ensure_ascii=False))

if __name__=='__main__':main()
