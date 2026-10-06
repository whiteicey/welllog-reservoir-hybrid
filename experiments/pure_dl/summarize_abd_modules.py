"""Compact report for the predeclared ABD internal-module interventions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/pure_dl_abd_backbone_module_ablation'
NAMES = {'local_attention':'patch 内局部注意力','global_attention':'patch 间全局注意力',
         'whole_hierarchy':'整个 patch 层级分支','point_mlp':'空 point 分支（数值对照）',
         'bottleneck_residuals':'瓶颈残差堆叠','unet_skips':'U-Net 三层跳连'}


def main():
    report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    assert report['weights_unchanged'] and report['all_validation_rows_covered_exactly_four_times']
    rows=report['arms']; baseline=rows[0]
    assert len(rows)==13 and baseline['arm']=='baseline'
    lines=['# A+B+D 双骨干内部模块消融','',
           f"同口径起点：ABD **{baseline['scores']['TOTAL']:.8f}**；A 单独 **{baseline['solo_scores']['TOTAL']:.8f}**。历史采纳基线80.4091518634；重分批浮点差异见report，不宣称逐元素相同。",'',
           '**这是固定权重的推理干预，不是重新训练后的模块因果增益。** B、D保持原状，只操作A的一个骨干一个模块；关闭模块后的内部路由仍正常根据输入和新后验运算。','',
           '| A 内被干预的骨干 | 关闭模块 | A 单独 | A变化 | ABD融合 | ABD变化 | 井级融合变好/变差/相同 |',
           '|---|---|---:|---:|---:|---:|---|']
    compact=[]
    for row in rows[1:]:
        differences=[w['scores']['TOTAL']-b['scores']['TOTAL'] for w,b in zip(row['per_well'],baseline['per_well'])]
        assert [w['well_id'] for w in row['per_well']]==[w['well_id'] for w in baseline['per_well']]
        wins=sum(d>1e-10 for d in differences);losses=sum(d< -1e-10 for d in differences)
        title='完整206视图' if row['member']==0 else '校准68视图'
        lines.append(f"|{title}|{NAMES[row['module']]}|{row['solo_scores']['TOTAL']:.6f}|{row['delta_solo']:+.6f}|{row['scores']['TOTAL']:.6f}|{row['delta_fused']:+.6f}|{wins}/{losses}/{16-wins-losses}|")
        compact.append({**{k:row[k] for k in ['arm','member','module','scores','solo_scores','delta_fused','delta_solo']},
                        'wells_positive':wins,'wells_negative':losses,
                        'channel_deltas':{j:row['scores'][j]-baseline['scores'][j] for j in ['POR','PERM','SW']}})
    lines+=['','## 三个目标通道的融合变化','',
            '|骨干|关闭模块|POR变化|PERM变化|SW变化|','|---|---|---:|---:|---:|']
    for row in compact:
        d=row['channel_deltas'];title='206' if row['member']==0 else '68'
        lines.append(f"|{title}|{NAMES[row['module']]}|{d['POR']:+.6f}|{d['PERM']:+.6f}|{d['SW']:+.6f}|")
    lines+=['','## 解释边界和下一步','',
            '当前模型的继承链已通过blur_no_point_sequence把独立逐点MLP替换为EmptyPoint（输出0个通道）。两项point实验实际是空操作，微小分数差只用于数值稳定性对照，不能据此宣称MLP无用。瓶颈是UngatedBlock堆叠，原学习门控也已移除；本轮旁路的是整个残差加工。',
            '下降说明当前训练好的参数组合依赖该模块；置零也引入内部表示分布变化。不能把下降幅度直接称为该模块从头训练的净收益，更不能将各项差值相加。',
            '对强依赖模块，应进行同初始化/同训练预算的有无模块训练对照；对无明显依赖模块，可优先做删除后训练的简化实验。B/D固定，每轮按ABD融合评分选checkpoint，保留A单独分数作诊断。',
            '原206/68视图和后验路由已经显示互补，但这轮只干预A，不证明B内部同名模块有同等作用。第五折反复参与开发，未完成独立全五折验证。','',
            f"13组总墙钟{report['seconds']:.2f}秒，包含预处理、冻结缓存和推理；峰值GPU分配{report['peak_gpu_allocated_mib']:.2f}MiB。实际并发与逐项耗时见report。"]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'baseline':baseline['scores'],'solo_baseline':baseline['solo_scores'],
            'results':compact,'reference_check':report['reference_check']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(compact,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
