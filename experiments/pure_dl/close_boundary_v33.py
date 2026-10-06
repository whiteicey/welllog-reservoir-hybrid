"""Record aggregate loss budget and finish v33 without new training."""
from pathlib import Path
from datetime import datetime
import json
from run_boundary_v33 import ROOT,STUDY,TARGETS,append_once,jobs

def main():
    diagnostic=json.loads((STUDY/'diagnostic.json').read_text(encoding='utf-8'))
    result=json.loads((STUDY/'assembled_report.json').read_text(encoding='utf-8'))
    assert result['all_four_complete'] and not jobs()
    total_loss=100-diagnostic['scores']['TOTAL'];budget=[]
    for group in ['near_0_5','middle_6_25','far_gt25']:
        rows=[x for x in diagnostic['groups'] if x['group']==group and x['population']=='all']
        value=sum(x['TOTAL_loss_contribution'] for x in rows)
        budget.append(dict(group=group,n=rows[0]['n'],population_fraction=rows[0]['n']/144665,
            TOTAL_loss=value,fraction_of_total_loss=value/total_loss))
    assert abs(sum(x['TOTAL_loss'] for x in budget)-total_loss)<1e-9
    (STUDY/'loss_budget.json').write_text(json.dumps(dict(score=diagnostic['scores']['TOTAL'],total_loss=total_loss,groups=budget),indent=2),encoding='utf-8')
    marker='## 总失分预算：困难程度与影响范围要分开'
    lines=[marker,'',
        '近边界非默认点的条件分数很低，但其数量有限。仅观察条件难度会夸大这条优化方向的整体影响；不能将边界关联直接解释为当前主瓶颈。',
        '|范围|验证点数|对TOTAL的失分|占全部失分|','|---|---:|---:|---:|']
    for row in budget:lines.append(f"|{row['group']}|{row['n']}|{row['TOTAL_loss']:.9f}|{100*row['fraction_of_total_loss']:.3f}%|")
    lines+=['','距输入边界超过25点的区域承担87.92%的当前失分。这里的25点约为2.5米；边界只是这一次固定输入变化定义，不是完整地层边界集合。',
        '将近边界点理想修复的0.409747分仅为该区域失分的算术上限，不能视为任何模型可以实现的收益。',
        '本轮四个小头没有超过基线，且真实主损失仍在远离这些强输入边界的区域。因此暂停该阈值/硬截断/小头路线；后续可先按非默认数值范围及输入重复模式分层定位稳定高损失群，不再直接增加边界模型复杂度。该后续诊断尚未执行。']
    for p in [STUDY/'FINAL.md',STUDY/'README.md']:
        append_once(p,marker,'\n'.join(lines))
    progress='''# v33 已完成

## 目标
基于80.336513当前预测器，诊断输入边界并验证边界截断统计是否提供增量。

## 已完成
九曲线输入边界阈值只由64训练井拟合，原行序和评分已复现；默认/非默认、逐井、总失分预算均已计算。POR与SW分别完成fixed/segmented两项，共四项；每项11344参数，全部第18轮自然早停、选第0轮。冻结状态和训练内最佳恢复通过。最终三列实际评分80.336513436816，与原预测完全相同，未采用新头，未重复未变化模型的全量磁盘推理。

## 结论与停止方向
边界附近的非默认点难度高，但±5点仅占总失分2.08%，>25点占87.92%。截断51点统计未取得收益，不追加阈值、窗口和宽度搜索。保持原v31配方和交付。

## 未完成与后续
未进行不同种子、新井、标签错位诊断或数值群组诊断；均未启动。其他原折仍暂停，完整OOF>82未完成。后续应先检查当前主要失分群的特征和分布，按证据决定新试验。

## 完成后维护
记录FINAL.md、loss_budget.json、assembled_report.json、CURRENT_RECIPE.json、训练配置/最佳更新后分数和状态；增量图谱脚本curate_boundary_v33.py。原训练来源、权重、标签、交付均未修改。
'''
    (STUDY/'PROJECTS.md').write_text(progress,encoding='utf-8')
    for t in TARGETS:
        out=ROOT/'outputs'/f'pure_dl_boundary_v33_{t}_fold5'
        (out/'PROJECTS.md').write_text(progress,encoding='utf-8')
        append_once(out/'README.md',marker,'\n'.join(lines))
    note='## v33最终结论\n四项已完成，当前TOTAL80.336513436816不变，无活动训练。输入近边界非默认样本很难但总体贡献有限：近±5点失分占2.08%，远>25点占87.92%。不继续硬截断统计扫描。报告results/deep_learning/boundary_context_v33_20260929/FINAL.md。未补其他折、未修改旧交付。'
    for p in [ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md',ROOT/'docs/deep_learning/atlas/PROJECTS.md']:
        append_once(p,'## v33最终结论',note)
    for p in [ROOT.parent/'AGENTS.md',ROOT/'AGENTS.md']:
        append_once(p,'## v33当前研究配方','## v33当前研究配方\n四项边界统计对照未获新增收益，保持TOTAL80.336513436816及原v31权重；配方见results/deep_learning/boundary_context_v33_20260929/CURRENT_RECIPE.json。近边界困难不等于总损失集中，暂停本次边界侧统计扫描，不自动补折或修改交付。')
    notes=json.loads((STUDY/'execution_notes.json').read_text(encoding='utf-8'))
    notes.update(closeout_time=datetime.now().astimezone().isoformat(),active_jobs=jobs(),loss_budget_verified=True)
    (STUDY/'execution_notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
    print('LOSS_BUDGET',json.dumps(budget,ensure_ascii=False))
    print('FINAL_SCORE',result['selected_scores']['TOTAL'],'ACTIVE',jobs())

if __name__=='__main__':main()
