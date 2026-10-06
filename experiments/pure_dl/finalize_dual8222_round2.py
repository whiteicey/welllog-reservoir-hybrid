"""Finalize recorded evidence after all requested bounded runs and fixed replay."""
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'results/deep_learning/dual8222_round2_20260928'
HERE=ROOT/'docs/deep_learning/atlas'
SELECTED='pure_dl_dual8222_selected_round2_fold5'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    diagnosis=json.loads((STUDY/'saved_prediction_attribution.json').read_text(encoding='utf-8'))
    final=json.loads((ROOT/'outputs'/SELECTED/'fold_5_report.json').read_text(encoding='utf-8'))
    assert all(row['complete'] for row in diagnosis['runs'].values())
    names={'pointwise':'上一轮逐点非线性','local5':'上一轮局部深度头','linear':'独立原线性头',
           'cross80':'两专家联合读出','gate_only':'只更新门控','density_only':'只更新连续分布'}
    source=diagnosis['source_scores']
    now=datetime.now().astimezone().isoformat(timespec='seconds')
    lines=['# 双专家80.22独立输出头：本轮优化结论','', '记录时间：'+now,'',
        '本轮新增四项训练，分两批、每批最多两项并发；均使用保留的80.221861双专家、原固定第五折、原单覆盖解码。历史模型和已交付包保持不变。',
        '', '## 已完成结果','', '|方案|POR|PERM|SW|TOTAL|相对80.221861|','|---|---:|---:|---:|---:|---:|',
        f"|原双专家|{source['POR']:.6f}|{source['PERM']:.6f}|{source['SW']:.6f}|{source['TOTAL']:.9f}|—|"]
    for arm,name in names.items():
        row=diagnosis['runs'][arm];s=row['scores']
        lines.append(f"|{name}|{s['POR']:.6f}|{s['PERM']:.6f}|{s['SW']:.6f}|{s['TOTAL']:.9f}|{row['delta_source']:+.9f}|")
    s=final['scores']
    lines += [f"|固定三通道组合|{s['POR']:.6f}|{s['PERM']:.6f}|{s['SW']:.6f}|{s['TOTAL']:.9f}|{final['delta_source']:+.9f}|",'',
        '所有新增训练的首个预测与来源逐元素一致，最终三个已选头的共同恢复由各自报告验证。固定组合再次通过共享两骨干的实际推理，未逐点/逐井切换来源。',
        '', '## 固定通道配方']
    for choice in final['choices']:
        lines.append('- '+choice['target']+'：'+choice['arm']+'，选定轮次'+str(choice['epoch'])+'，整条通道分数'+str(choice['score']))
    lines += ['', '## 本轮学到什么',
        '- 线性-only对照已解释前轮几乎全部改善。逐点非线性相对线性'+f"{diagnosis['pairwise']['pointwise_minus_linear']['TOTAL_delta']:+.9f}"+
          '，局部头相对线性'+f"{diagnosis['pairwise']['local5_minus_linear']['TOTAL_delta']:+.9f}"+'。不再把约0.03的总增益全归给新增非线性容量。',
        '- 两专家联合读出相对线性'+f"{diagnosis['pairwise']['cross80_minus_linear']['TOTAL_delta']:+.9f}"+'；该小差异尚不足以证明稳定更优，不继续无界扩大宽度/核长。',
        '- 默认/非默认分组诊断发现收益与损失抵消。因素分离对照保持前向结构，仅改变门控或密度参数的更新范围；参数数目差异是消融本身。',
        '- 门控和条件密度通过内部专家份额与效用决策联系。密度-only保持门控logit不变，但最终输出默认值的数量仍可改变。',
        '- 当前完整五折最佳仍81.713233；本批第五折反复参与选模，不构成独立测试或跨井稳定泛化证明。',
        '', '## 下一步（尚未启动）',
        '优先针对有收益目标做更细的连续分布读出：混合权重、位置、尺度三组参数分开控制，或加约束保留原分布的尺度。每次只改一项，与当前最简读出匹配比较。PERM/SW保留本次实测最好的设置，不强行共同扩大头结构。',
        '第五折严格>80.5且明确收益后才讨论晋级，补折仍须另行恢复授权。没有自动启动新训练、旧ABD、测试伪标签或完整五折队列。',
        '', '## 时间口径',
        '以下为各进程从启动准备到最终恢复的报告秒数，包含部分并发，不把它们相加冒充单次总墙钟或速度提升。']
    for arm in ['linear','cross80','gate_only','density_only']:
        r=diagnosis['runs'][arm]
        lines.append(f"- {arm}: {r['seconds']:.3f}s；训练停止轮次{r['stopped_epochs']}。")
    lines += ['', '## 证据索引',
        '- `saved_prediction_attribution.json`：原144665点重算、每井及标签人口分解、线性匹配差值。',
        '- `progress.json`：实际状态和来源、初始/反向/最终恢复记录。',
        '- 各`outputs/pure_dl_dual8222_v2_*`与`v3_*`的`fold_5_report.json`和`heads_best.pt`。',
        '- `outputs/'+SELECTED+'/fold_5_report.json`与`selected_heads.pt`：固定通道共同恢复。',
        '- `trainer_derivation.json`和`factor_trainer_derivation.json`：新训练器的精确修改记录，旧源文件不变。']
    text='\n'.join(lines)+'\n'
    (STUDY/'FINAL.md').write_text(text,encoding='utf-8')
    summary=f"\n\n## {now} 四项优化与固定通道组合已完成\n本轮linear/cross80/gate_only/density_only全部自然结束并共同恢复。最终固定组合TOTAL={s['TOTAL']:.12f}，相对原80.221861={final['delta_source']:+.12f}；配方={final['choices']}。没有达到独立泛化目标，不启动补折或改交付。更细分布读出优化未启动。总结见results/deep_learning/dual8222_round2_20260928/FINAL.md。\n"
    for path in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',STUDY/'PROJECTS.md',HERE/'PROJECTS.md']:
        with path.open('a',encoding='utf-8') as stream:stream.write(summary)
    for arm in ['linear','cross80','gate_only','density_only']:
        folder=ROOT/'outputs'/diagnosis['runs'][arm]['folder']
        with (folder/'README.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n本轮完整比较与后续边界见 `../../results/deep_learning/dual8222_round2_20260928/FINAL.md`；本目录最新状态为已完成。\n')
        with (folder/'PROJECTS.md').open('a',encoding='utf-8') as stream:stream.write(summary)
    curated=json.loads((HERE/'curation.json').read_text(encoding='utf-8'))
    curated['nodes'][SELECTED]={'title':'双专家·按目标固定读出组合','summary':'从已完成候选为POR/PERM/SW各选固定读出，原两骨干只算一次，原解码不变。',
        'conclusion':f"完成共同恢复：{s['TOTAL']:.9f}；源模型差{final['delta_source']:+.9f}。开发集选模，未补折；下一步只在有证据的输出子结构上探索。",'stage':'loss','highlight':True}
    for choice in final['choices']:
        a=choice['arm']
        source='pure_dl_paired_local_context_router80_sequence' if a=='source' else diagnosis['runs'][a]['folder']
        if not any(e['source']==source and e['target']==SELECTED for e in curated['edges']):
            curated['edges'].append({'source':source,'target':SELECTED,'type':'evolution',
                'note':'固定整条目标读出的来源；实际共同推理已验证，非按行选择。','evidence':'outputs/'+SELECTED+'/fold_5_report.json'})
    (HERE/'curation.json').write_text(json.dumps(curated,ensure_ascii=False,indent=2),encoding='utf-8')
    print(text)

if __name__=='__main__':main()
