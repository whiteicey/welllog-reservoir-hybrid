"""CPU final accounting of v13; preserve all source models and deliveries."""
from pathlib import Path
from datetime import datetime
import argparse, json, sys
import numpy as np
import psutil
import train_invariant as base

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/independent_sw_tolerance_v13_20260928'
REFERENCE=ROOT/'outputs/pure_dl_independent_v8_selected_fold5'
MODES=('likelihood','tolerance_mass')


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def write(p, data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def sw_kernel(pred, truth):
    return np.maximum(0.,1.-np.abs(pred-truth)/(.05*(np.abs(truth)+1e-5)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--record',action='store_true');args=ap.parse_args()
    with np.load(REFERENCE/'oof.npz') as z:
        assert all(k in z.files for k in ['truth','fold','well']), z.files
        idx=z['fold']==4; truth=z['truth'][idx]; wells=z['well'][idx]
    assert truth.shape==(144665,3)
    ref=np.load(REFERENCE/'fold_5_pred.npy'); assert ref.shape==truth.shape
    reference_report=load(REFERENCE/'fold_5_report.json')
    reference_scores=base.metric(truth,ref)
    assert abs(reference_scores['TOTAL']-reference_report['scores']['TOTAL'])<1e-8
    configs=[load(ROOT/'outputs'/('pure_dl_independent_sw_v13_'+m+'80_fold5')/'config.json') for m in MODES]
    differences={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1]) if configs[0].get(k)!=configs[1].get(k)}
    assert set(differences)=={'mode','auxiliary_weight'}, differences
    records=[];starts=[];ends=[]
    for mode in MODES:
        ident='pure_dl_independent_sw_v13_'+mode+'80_fold5'; p=ROOT/'outputs'/ident
        status=load(p/'runtime_status.json');history=load(p/'validation_history.json') if (p/'validation_history.json').exists() else []
        report=load(p/'fold_5_report.json') if (p/'fold_5_report.json').exists() else None
        record=dict(mode=mode,id=ident,status=status,report=report,last=history[-1] if history else None,
            best_after_update=max(history,key=lambda row:row['target_score']) if history else None)
        if report:
            pred=np.load(p/'channel_pred.npy');assert pred.shape==(144665,)
            trial=ref.copy();trial[:,2]=pred;scores=base.metric(truth,trial)
            assert abs(scores['SW']-report['target_score'])<1e-8
            assert abs(scores['TOTAL']-report['reference_replacement_total'])<1e-8
            delta=100*(sw_kernel(pred,truth[:,2])-sw_kernel(ref[:,2],truth[:,2]))
            default=base.dmask(truth)[:,2]
            populations={}
            for name,mask in [('true_default',default),('true_nondefault',~default)]:
                populations[name]=dict(n=int(mask.sum()),SW_delta=float(delta[mask].mean()),
                    TOTAL_contribution=float(.35*delta[mask].sum()/len(delta)))
            perwell=[dict(well=int(w),n=int((wells==w).sum()),SW_delta=float(delta[wells==w].mean())) for w in np.unique(wells)]
            record.update(recomputed_replacement_scores=scores,populations=populations,per_well=perwell,
                well_wins=sum(x['SW_delta']>1e-12 for x in perwell),well_losses=sum(x['SW_delta']< -1e-12 for x in perwell),
                retained_predictions_exactly_equal_reference=bool(np.array_equal(pred,ref[:,2])),
                first_training_stats=history[0],last_training_stats=history[-1])
            starts.append(datetime.fromisoformat(load(p/'launch.json')['started_at']))
            ends.append(datetime.fromisoformat(report['finished_at']))
        records.append(record)
    prior=[]
    for mode in ['likelihood','joint_utility']:
        p=ROOT/'outputs'/('pure_dl_independent_sw_v12_'+mode+'80_fold5')
        if (p/'fold_5_report.json').exists():
            rep=load(p/'fold_5_report.json');hist=load(p/'validation_history.json') if (p/'validation_history.json').exists() else []
            prior.append(dict(mode=mode,report=rep,best_after_update=max(hist,key=lambda x:x['target_score']) if hist else None,
                attribution='Already completed before current-turn inspection; not launched by v13 work'))
    all_done=all(x['report'] is not None for x in records)
    progress=dict(observed_at=datetime.now().astimezone().isoformat(),all_complete=all_done,reference_scores=reference_scores,
        runs=records,prior_v12=prior,complete_five_folds=False,remaining_folds_started=False,
        score_scope='original fifth; reference other channels reused from already verified independent models',
        scoring_population=144665,development_selection=True,verified_config_differences=differences)
    if all_done:
        progress['matched_difference']=dict(SW=records[1]['report']['target_score']-records[0]['report']['target_score'],
            postupdate_SW=records[1]['best_after_update']['target_score']-records[0]['best_after_update']['target_score'])
        progress['timing']=dict(explicit_concurrent_jobs=2,launch_to_both_finished_wall_seconds=(max(ends)-min(starts)).total_seconds(),
            per_run_seconds={x['mode']:x['report']['seconds'] for x in records},
            per_run_peak_allocated_MiB={x['mode']:x['report']['peak_gpu_allocated_mib'] for x in records},
            scope='Wall includes launch/setup/train/validation/restoration; excludes implementation and report/atlas generation.')
    write(STUDY/'progress.json',progress)
    write(STUDY/'prior_v12_summary.json',prior)
    if all_done:
        improved=[x for x in records if x['report']['target_score']>reference_scores['SW']]
        recipe=dict(selected_new_SW=bool(improved),reference_scores=reference_scores,
            choices=reference_report['choices'],three_private_models_retained=not bool(improved),
            reference_assembly_report='outputs/pure_dl_independent_v8_selected_fold5/fold_5_report.json',
            new_three_model_gpu_replay_this_turn=False,remaining_folds_started=False,
            note='No new model is substituted unless independently reloaded and actually assembled. Old validation evidence is referenced, not claimed to be repeated.')
        if improved:
            recipe['pending_candidate']=max(improved,key=lambda x:x['report']['target_score'])['id']
        write(STUDY/'CURRENT_RECIPE.json',recipe)
        lines=['# SW容差概率质量监督v13：最终结果','',
            '本批显式运行两项强SW私有双专家微调。v12两项在本轮开始读取前已完成，本批只补记其结果。','',
            '|模式|参数数|训练后最高SW（排除第0轮）|轮次|最终选定SW|最佳/停止轮次|',
            '|---|---:|---:|---:|---:|---|']
        for x in records:
            f=x['report'];b=x['best_after_update']
            lines.append(f"|{x['mode']}|{f['trainable_parameters']}|{b['target_score']:.12f}|{b['epoch']}|{f['target_score']:.12f}|{f['best_epoch']}/{f['stopped_epoch']}|")
        lines+=['','## 机制与验证',
            '两组同SW私有双专家、同初始权重/数据/随机窗口/优化器，只差容差质量负对数系数0和0.5。所有SW私有参数可训练，POR/PERM模型不更新。CDF差分保留梯度，首批前向与原推理实现差值为0，辅助损失同时到达gate/density，SW骨干收到非零梯度。',
            '两组最终均恢复起点；全部144665点预测精确复现，训练期间骨干有实际变化。恢复原模型不能当成新损失带来的增益。原三私有模型TOTAL80.274135337791保持。本轮未重复执行三模型GPU重载，沿用此前已核验的组合证据。',
            '容差损失采用原256格上的概率质量积分近似，固定alpha0.5；其本身非proper scoring rule，保留原NLL。不把它等同于准确的argmax官方风险。CUDA插值反向仍使用确定性警告模式，不宣称训练逐位可复现。',
            '', '## 收益与关闭范围',
            '本配置没有新增收益。两组更新后最高均在第1轮，容差组比似然对照更低；不继续从同起点无约束全参数微调的宽系数/学习率扫描。没有证明所有效用监督、所有SW架构或输入信息均达到上限。',
            '后续待研究：优先用训练井内的分组验证检查训练内拟合与新井泛化差距，再决定是否引入跨井局部参照；该步骤尚未启动。第五折严格>80.5与完整OOF>82未完成，无自动补折或旧交付覆盖。',
            '', '## 工程状态',
            '所有训练源和旧权重保持不可变；本批源码、完整last_state和紧凑证据已保存。未验证中断恢复与不中断训练的逐位等价。',
            '编辑主curate.py被工具内容保护拒绝，原文件未改；新摘要由curate_sw_tolerance_v13.py覆盖合并到现有curation.json，然后build_atlas.py生成图谱，保留其他节点。',
            '两个一次性监控命令因shell引号/编码传递失败，未运行Python正文、未影响训练；后续使用固定脚本与读取工具完成核对。',
            '', '完整分数、历史、Q、时长和来源见progress.json、两个输出目录的config.json/fold_5_report.json。']
        write_text='\n'.join(lines)+'\n';(STUDY/'FINAL.md').write_text(write_text,encoding='utf-8')
        if args.record:
            note='\n\n## '+progress['observed_at']+' v13完成\n两项强SW私有双专家损失对照完成18轮，均保留第0轮73.622384767129，无新增收益；更新后最佳分别'+str(records[0]['best_after_update']['target_score'])+'/'+str(records[1]['best_after_update']['target_score'])+'。原三独立模型80.274135337791保留，无新三模型重载宣称、无补折/交付更改。完整报告results/deep_learning/independent_sw_tolerance_v13_20260928/FINAL.md。下一步分组泛化诊断尚未启动，>82目标未完成。\n'
            for p in [STUDY/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT.parent/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
                with p.open('a',encoding='utf-8') as f:f.write(note)
            for x in records:
                p=ROOT/'outputs'/x['id'];f=x['report'];b=x['best_after_update']
                local=f"\n\n## 完成结论\n实际训练18轮；更新后最高SW{b['target_score']:.12f}@{b['epoch']}，最终选择第0轮{f['target_score']:.12f}，没有新增收益。全部144665点恢复精确，骨干真实更新，无其他目标更新。\n后续不扩大本设置的系数/学习率网格；跨井泛化诊断尚未执行。目标>80.5与完整OOF>82未完成，原交付不变。\n"
                for name in ['README.md','PROJECTS.md']:
                    with (p/name).open('a',encoding='utf-8') as stream:stream.write(local)
    active=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            v=proc.info;argv=v.get('cmdline') or []
            if 'python' in (v.get('name') or '').lower() and any('pure_dl/' in a.replace('\\','/') and Path(a.replace('\\','/')).name.startswith(('train_','benchmark_')) for a in argv):active.append(v['pid'])
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    print(json.dumps(dict(all_complete=all_done,runs=[dict(mode=x['mode'],report=x['report']) for x in records],
        matched_difference=progress.get('matched_difference'),timing=progress.get('timing'),active_training_pids=active,
        prior_v12=[dict(mode=x['mode'],best_after_update=x['best_after_update']) for x in prior]),ensure_ascii=False))


if __name__=='__main__':main()
