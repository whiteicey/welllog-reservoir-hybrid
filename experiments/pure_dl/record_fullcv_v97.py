"""Update experiment records and atlas after a completed fullCV batch."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from fullcv_context_v97 import ROOT, STUDY
from fullcv_models_v97 import STAGES, fold_dir, root_dir


def main():
    rows=[]
    for f in (1,2,3,4):
        rdir=root_dir(f);hpath=rdir/f'fold_{f}_validation_history.json'
        h=json.loads(hpath.read_text()) if hpath.exists() else []
        root_report=json.loads((rdir/f'fold_{f}_report.json').read_text()) if (rdir/f'fold_{f}_report.json').exists() else None
        stages=[stage for stage,*_ in STAGES if (fold_dir(f)/stage/'report.json').exists()]
        assembled=json.loads((fold_dir(f)/'fold_report.json').read_text()) if (fold_dir(f)/'fold_report.json').exists() else None
        rows.append(dict(fold=f,root_last_epoch=h[-1]['epoch'] if h else None,
            root_best_total=max((r['official_total'] for r in h),default=None),source_complete=bool(root_report),
            completed_stages=stages,stage_count=len(STAGES),assembled=assembled))
    snapshot=dict(recorded_at=datetime.now().astimezone().isoformat(),folds=rows,fifth_verified_total=80.50419742447689,
        fifth_reused=True,complete_five_folds=all(r['assembled'] for r in rows))
    (STUDY/'batch_snapshot.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    text='# v97 已授权补齐原其余四折\n\n目标：固定v96配方原分组五折官方OOF>82。第五144665点TOTAL80.504197424477已核验并复用；各折全部学习依赖及统计/Q只能来自本折64训练井。\n\n'
    text+='已做：明确授权留档、来源/人口/哈希、按折预处理/Q绑定、原子可恢复root与18个有效训练阶段入口、固定私有提取CPU逐权重等价证明、按原行序OOF汇总入口。串行控制器逐项执行，失败停止且不自动重试；仅显式恢复未完成状态。\n\n'
    text+='|原折|骨干最近轮|骨干最佳TOTAL（非最终配方）|有效阶段完成|最终三通道|\n|---|---:|---:|---:|---|\n'
    for r in rows:
        text+=f"|{r['fold']}|{r['root_last_epoch']}|{r['root_best_total']}|{len(r['completed_stages'])}/{len(STAGES)}|{r['assembled']['scores'] if r['assembled'] else '未完成'}|\n"
    text+='|5|历史已完成|不混同当前来源骨干|配方已完成|80.504197424477；已核验复用|\n\n'
    final=STUDY/'FINAL_REPORT.json'
    if final.exists():text+='完整实际OOF：'+json.dumps(json.loads(final.read_text())['oof_scores'],ensure_ascii=False)+'。\n\n'
    else:text+='未完成：其余四折全部阶段与最终实际三通道、完整723556点OOF；尚无完整收益，不能用骨干单折成绩认定OOF>82。\n\n'
    text+='计划：固定18阶段依赖按本折最佳checkpoint推进；单模型目标独立选模，完成每折再真实合并；最终按折1~5排序和原行序报告。达成后：在新目录交付代码/模型/预测与来源验证，历史交付不覆盖。\n\n'
    text+='不继续：历史负对照/参数探索/旧队列不重跑，第五权重/拟合元数据/Q不得进入其他折。v8 POR/PERM历史selected epoch0经逐权重完全一致，固定为私有提取阶段，省去无效微调；不按新折成绩选择是否恢复该旧搜索。\n\n'
    text+='耗时规则：每stage与root记录真实seconds（含该项加载/验证/checkpoint），总墙钟由controller授权批次另计，串行并发1、上限2；不把阶段总和当并发加速。\n'
    (STUDY/'PROJECTS.md').write_text(text,encoding='utf-8')
    (STUDY/'PROGRESS.md').write_text(text,encoding='utf-8')
    for p in (ROOT.parent/'PROJECTS.md',ROOT/'PROJECTS.md'):
        with p.open('a',encoding='utf-8') as stream:stream.write('\n\n'+text)
    for script in ('curate.py','build_atlas.py'):
        subprocess.run([sys.executable,str(ROOT/'docs/deep_learning/atlas'/script)],cwd=ROOT,check=True)
    print('FULLCV_RECORDS_UPDATED',json.dumps(snapshot))


if __name__=='__main__':main()
