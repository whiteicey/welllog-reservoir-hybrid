"""Finish six-model evidence after actual three-model replay; no new training."""
from pathlib import Path
from datetime import datetime
import json,sys
import torch
from launch_independent_channels_v8 import ROOT,STUDY,RUNS,PREFIX,trainers
from train_independent_channels_v8 import channel_objective


def verify_single_target_loss():
    checks=[]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260928)
        for c in range(3):
            gate=torch.randn(2,5,1,requires_grad=True);raw=torch.randn(2,5,1,3,16,requires_grad=True)
            truth=torch.randn(2,5,3);default=torch.rand(2,5,3)>.5
            mask=torch.rand(2,5)>.2;quality=torch.rand(2,5)*.9+.1
            loss=channel_objective((gate,raw),truth,default,mask,quality,c)[0]
            grads=torch.autograd.grad(loss,(gate,raw))
            changed_y=truth.clone();changed_d=default.clone()
            for other in range(3):
                if other!=c:changed_y[:,:,other]=1234.;changed_d[:,:,other]=~changed_d[:,:,other]
            g2=gate.detach().clone().requires_grad_(True);r2=raw.detach().clone().requires_grad_(True)
            loss2=channel_objective((g2,r2),changed_y,changed_d,mask,quality,c)[0]
            grads2=torch.autograd.grad(loss2,(g2,r2))
            assert torch.equal(loss,loss2) and all(torch.equal(a,b) for a,b in zip(grads,grads2))
            checks.append({'target_index':c,'other_labels_changed':True,'loss_unchanged':True,'output_gradients_unchanged':True})
    return {'scope':'synthetic CPU contract check of actual training objective','checks':checks,'new_training':False}


def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(2)
    assert not trainers()
    report=json.loads((STUDY/'assembled_report.json').read_text(encoding='utf-8'))
    rows=[];intervals=[]
    for key in RUNS:
        folder=ROOT/'outputs'/(PREFIX+key+'80_fold5')
        final=json.loads((folder/'fold_5_report.json').read_text())
        hist=json.loads((folder/'validation_history.json').read_text())
        peak=max(hist,key=lambda h:h['target_score'])
        cfg=json.loads((folder/'config.json').read_text())
        launch=json.loads((folder/'launch.json').read_text())
        intervals.append((datetime.fromisoformat(launch['started_at']),datetime.fromisoformat(final['finished_at'])))
        rows.append({'key':key,'target':final['target'],'architecture':final['architecture'],
            'initial':final['initial_target_score'],'trained_peak':peak['target_score'],'trained_peak_epoch':peak['epoch'],
            'retained':final['target_score'],'best_epoch':final['best_epoch'],'stopped_epoch':final['stopped_epoch'],
            'delta_reference':final['delta_shared_reference'],'parameters':final['trainable_parameters'],
            'last_backbone_change_L2':final['last_backbone_parameter_change_L2'],'selected_backbone_change_L2':final['selected_backbone_parameter_change_L2'],
            'peak_gpu_allocated_mib':final['peak_gpu_allocated_mib'],'seconds':final['seconds'],
            'training_bitwise_reproducibility':cfg['bitwise_training_reproducibility_claimed']})
    contract=verify_single_target_loss()
    (STUDY/'training_isolation_check.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    live=max_live=0
    for _,change in sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals],key=lambda x:(x[0],x[1])):
        live+=change;max_live=max(max_live,live)
    timing={'successful_training_batch_span_seconds':(max(b for a,b in intervals)-min(a for a,b in intervals)).total_seconds(),
        'max_concurrent_successful_trainers':max_live,'scope':'includes successful setup/validation and explicit launch gaps; initial failed attempts and final replays separate'}
    assert max_live<=2
    outcome={'observed_at':datetime.now().astimezone().isoformat(),'runs':rows,'scores':report['scores'],
        'delta_reference':report['delta_reference'],'accuracy_improved':report['delta_reference']>1e-10,
        'no_shared_backbone_verified':report['no_shared_parameter_storage'],'timing':timing,
        'original_delivery_unchanged':True,'remaining_folds_started':False,'full_oof_goal_met':False}
    (STUDY/'FINAL_EVIDENCE.json').write_text(json.dumps(outcome,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 三通道独立模型：本轮结论','',outcome['observed_at'],'',
        '六项训练及三份完整模型的独立重载已经完成。每个模型只用自身目标损失更新私有骨干，训练与推理均不共享参数。初始化仍来自旧多任务预训练，未做从零独立训练。','',
        '|目标|结构|初始通道分|训练后最高分（不含0轮）|最终保留分|最佳/停止轮次|相对旧参考|',
        '|---|---|---:|---:|---:|---|---:|']
    for r in rows:
        lines.append(f"|{r['target']}|{r['architecture']}|{r['initial']:.6f}|{r['trained_peak']:.6f}@{r['trained_peak_epoch']}|{r['retained']:.6f}|{r['best_epoch']}/{r['stopped_epoch']}|{r['delta_reference']:+.6f}|")
    lines += ['', '## 实际三模型组合',json.dumps({k:report[k] for k in ['scores','delta_reference','choices','private_backbone_branches','total_model_parameters','well_wins','well_losses']},ensure_ascii=False,indent=2),
        '只有这一组合的TOTAL来自三份独立模型实际预测。各模型的reference_replacement_total只是单目标替换诊断，未用于伪造本轮组合收益。',
        '', '## 已证实的实现边界',
        '每项训练均取得非零骨干梯度且末轮骨干权重发生变化。最终模型分别拥有完整checkpoint。三个模型同时创建后的参数存储不相交，实际重载输出逐元素等于保存预测，原144665行顺序保持。',
        'CPU合同检查使用训练同一损失函数：改变其他两个通道的数值和默认标签，不改变当前目标损失或输出梯度。此检查证明实现隔离，不证明单任务训练泛化更强。',
        '初次sys导入遗漏和随后CUDA插值确定性反向限制已修复，失败attempt1/2保存；都发生在第一次optimizer.step之前。成功训练使用固定种子、deterministic warn_only，故不宣称训练逐位可复现。固定前向重载已独立验证。',
        '', '## 结论适用范围']
    if report['delta_reference']<=1e-10:
        lines += ['当前六项预训练后单任务微调没有提高组合精度；独立三模型接口已实现，但不应仅为相同精度把旧交付替换为更昂贵的重复骨干。',
            '若最佳epoch为0，最终保留的是初始函数，必须与更新权重后的最高分分开。这个结果只限制本轮预训练/结构/损失/学习率配置，不能推出独立模型普遍无效。']
    else:
        lines += ['独立模型组合在当前第五折取得了上表增益；参数与总训练预算均高于旧冻结方案，不能把增益单独归因为去除跨目标梯度冲突。']
    lines += ['单视图和双视图都使用多尺度卷积与已有注意力模型家族；尚未做TCN/GRU/Transformer等完全不同骨干的从零独立训练。',
        '后续优先分别检验目标专用的训练目标、或与旧预训练不同的完整骨干，并建立匹配对照；不从本轮阴性结果直接继续缩小学习率做无界微调。',
        '当前不启动剩余四折、不覆盖已有交付。第五折反复用于开发，需跨折或新井证据才能确认泛化；完整OOF>82仍未完成。',
        '', '## 成本与复现',json.dumps(timing,ensure_ascii=False,indent=2),
        '独立模型结构与初始化见METHOD.md、各config；训练/恢复/失败见对应输出目录；固定三模型见outputs/pure_dl_independent_v8_selected_fold5/model_manifest.json。',
        '最终推理需要3至6条私有骨干，不能把模型数量变化直接换算为实测部署速度。现有报告的推理耗时只覆盖本机原第五折，未比较其他设备。']
    (STUDY/'FINAL.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state='# 进度\n\n目标：各目标独立模型与实际预测验证。\n\n已完成：六项单/双视图全骨干微调、首次骨干梯度、最终恢复、三模型不共享参数及重载检查、其他目标标签不影响损失的合同检查。\n\n当前TOTAL='+str(report['scores']['TOTAL'])+'，相对参考='+str(report['delta_reference'])+'。\n\n未完成：从零独立训练、其他模型家族、匹配总预算共享训练对照、跨折泛化与完整OOF>82、训练逐位复现。\n\n后续：根据目标误差选择下一项机制，本批不再添加模型；不自动补折。\n\n目标达成后：固定三个模型与预处理，独立验证成本与新井表现，再创建新交付目录。\n'
    (STUDY/'PROJECTS.md').write_text(state,encoding='utf-8')
    note='\n\n## '+outcome['observed_at']+' 独立通道v8完成\n六项已完成并共同重载；TOTAL='+str(report['scores']['TOTAL'])+'，相对旧参考='+str(report['delta_reference'])+'。模型之间无共享参数，骨干实际训练。初始化仍来自旧多任务预训练，未比较其他独立从零骨干。完整总结results/deep_learning/independent_channels_v8_20260928/FINAL.md；无自动补折/旧交付更改。\n'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        with p.open('a',encoding='utf-8') as f:f.write(note)
    for key in RUNS:
        p=ROOT/'outputs'/(PREFIX+key+'80_fold5')/'PROJECTS.md'
        with p.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps(outcome,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
